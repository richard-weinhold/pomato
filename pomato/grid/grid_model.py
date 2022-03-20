
import logging
import datetime as dt
import itertools
import numpy as np
import pandas as pd
import types
from pathlib import Path

import pomato
import pomato.tools as tools

class GridModel():
    """GridRepresentation of POMATO, represents the network in the market model.

    The GridRepresentation creates a grid representation to be used in the market model based on the
    chosen options. This module acts as a combinator of the data and grid modules
    and allow to easily change grid representation for the market model.

    Its main feature is a minimal nodal/zonal N-1 grid representation which is achieved through a
    redundancy removal algorithm based on the "Clarkson" algorithm.

    For a more comprehensive documentation note that the ptdf matrix, regardless if with/without
    contingencies or representing nodal/zonal sensitivities, is denoted as matrix A in this module and
    the line limits as vector b. Therefore the resulting power flow problem is simply written in the
    form of a linear problem Ax <= b. The redundancy reduction algorithm finds the smallest set of
    constraints, essential set, that fully defines the problem. The indices, i.e. the rows in A or
    cbco's, are called essential indices.

    The the class attributes are divided into core attributes, with *grid_representation* as the
    mein outout/result and additional attributes that are used to facilitate the redundancy removal
    algorithm.

    Parameters
    ----------
    wdir : pathlib.Path
        Working directory
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class with processed input data.
    grid : :class:`~pomato.grid.GridTopology`
       An instance of the GridModel class.
    options : dict
        The options from POMATO main module.

    Attributes
    ----------
    wdir, julia_dir : pathlib.Path
        Working directory, Sub-directory for temporary files related with the
        redundancy removal algorithm.
    options : dict
        The options from DataManagement persist in the InputProcessing.
    grid : :class:`~pomato.grid.GridTopology`
        Instance of the GridTopology class. Provides functionality to create N-0 and (filtered) N-1 ptdf.
    data : :class:`~pomato.data.DataManagement`
       Instance of the DataManagement class with processed input data.
    grid_representation : types.SimpleNamespace
        Containing the grid representation to be used in the market model and the determination of
        the economic dispatch. Depends on the chosen configuration in the options file.
    julia_instance : :class:`~pomato.tools.JuliaDaemon`
        Julia process that is initialized when used the first time and then kept to be able to
        easily re-run the redundancy algorithm without restarting a julia process.
    """

    def __init__(self, wdir, grid, data, option):
        # Import Logger
        self.logger = logging.getLogger('log.pomato.grid.GridTopology')
        self.logger.info("Initializing the GridTopology....")

        self.options = option
        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])
        self.julia_dir = wdir.joinpath("data_temp/julia_files")

        # Core attributes
        self.grid = grid
        self.data = data
        self.grid_representation = types.SimpleNamespace(
                option=None,
                multiple_slack=None,
                slack_zones=None,
                lines=pd.DataFrame(),
                grid=pd.DataFrame(),
                contingency_groups={},
                redispatch_grid=pd.DataFrame(),
                ntc=pd.DataFrame()
            )
        self.julia_instance = None
    
    def _start_julia_daemon(self):
        self.julia_instance = tools.JuliaDaemon(
            self.logger, self.wdir, self.package_dir, "redundancy_removal", self.options["solver"]["name"])

    def create_grid_representation(self, flowbased_parameters=None):
        """Create grid representation based on model type.

        Options are: uniform, ntc, opf, scopf, fbmc.

        *grid_representation* contains the following:
            - *option*: The chosen option of grid representation.
            - *multiple_slack*: Bool indicator if there are multiple slacks.
            - *slack_zones*: dict to map each node to a slack/reference node.
            - *grid*: DataFrame including the ptdf for each line/outage,
              depending on chosen option, including line capacities and
              regional information.
            - *redispatch_grid*: DataFrame including the ptdf for the redispatch.
              As default this is nodal but could also be an N-1 ptdf, similar to
              *grid*.
            - *ntc*: DataFrame with the zonal commercial exchange capacities.

        All values are set according the chosen option and might remain empty.
        If flow based parameters are supplied, the grid representation will be 
        generated accordingly. 

        Parameters
        ----------
        flowbased_parameters : optional, pandas.DataFrame
            Flowbased parameters, derived using :class:`~pomato.fbmc.FBMCModule`

        """
        # Data Structure of grid_representation dict
        self.grid_representation.option = self.options["type"]
        self.grid_representation.multiple_slack = self.grid.multiple_slack
        self.grid_representation.slack_zones = self.grid.slack_zones()
        self.grid_representation.lines = self.grid.lines.copy()
        self.grid_representation.lines.loc[:, "capacity"] *= self.options["grid"]["long_term_rating_factor"]

        if isinstance(flowbased_parameters, pd.DataFrame):
            self.process_flowbased_grid_representation(flowbased_parameters)
        elif self.options["type"] == "ntc":
            self.process_ntc()
            self.grid_representation.grid = pd.DataFrame()
            self.grid_representation.contingency_groups = {}
        elif self.options["type"] == "opf":
            self.grid_representation.grid = self.create_nodal_grid_parameters()
        elif self.options["type"] == "scopf":
            self.grid_representation.contingency_groups = self.grid.contingency_groups
            self.grid_representation.grid = self.create_scopf_grid_parameters()
        elif self.options["type"] == "uniform":
            self.grid_representation.grid = pd.DataFrame()
            self.grid_representation.ntc = pd.DataFrame()
            self.grid_representation.contingency_groups = {}
        else:
            raise AttributeError("Please Choose a valid type")
        
        if self.options["redispatch"]["include"]:
            self.add_redispatch_grid()
        else:
            self.grid.redispatch_grid = pd.DataFrame()
    
    def process_flowbased_grid_representation(self, flowbased_parameters):
        """Process grid information for flow based grid representation.
        
        The flow based parameters reflect a zonal ptdf, including contingencies,
        for a preselected set of critical branches under critical outages (cbco)
        and available capacities (RAM) depending on the chosen basecase. 
        Therefore the optimization type has to be *fbmc*, for the model to 
        account for a zonal, timedependant ptdf.

        Parameters
        ----------
        flowbased_parameters : pandas.DataFrame
            Flowbased parameters, derived using :class:`~pomato.fbmc.FBMCModule`

        """
        self.options["type"] = "fbmc"
        self.grid_representation.option = "fbmc"
        self.grid_representation.grid = flowbased_parameters
        self.grid_representation.contingency_groups = self.grid.contingency_groups
        self.process_ntc()

    def create_nodal_grid_parameters(self):
        """Process grid information for nodal N-0 representation.

        Here *grid_representation.grid* consists of the N-0 ptdf.

        There is the option to try to reduce this ptdf, however the number of
        redundant constraints is expected to be very low.
        """
            
        grid_option = self.options["grid"]
        nodal_network = pd.DataFrame(columns=self.grid.nodes.index, data=self.grid.ptdf)
        nodal_network["ram"] = self.grid.lines.capacity.values*self.options["grid"]["long_term_rating_factor"]
        
        nodal_network["cb"] = list(self.grid.lines.index)
        nodal_network["co"] = ["basecase" for i in range(0, len(self.grid.lines.index))]
        nodal_network = nodal_network[["cb", "co", "ram"] + list(self.grid.nodes.index)]

        if grid_option["redundancy_removal_option"] == "nodal_redundancy_removal":
            nodal_injection_limits = self.create_nodal_injection_limits()

            cbco_index = self.clarkson_algorithm(A=nodal_network.loc[:, self.grid.nodes.index].values, 
                                                 b=nodal_network.loc[:, "ram"].values, 
                                                 x_bounds=nodal_injection_limits)
            nodal_network = self.return_cbco(nodal_network, cbco_index)

        return nodal_network
           
    def add_redispatch_grid(self):
        """Add nodal N-0 grid representation as redispatch grid.
        
        Depending on the optional argument *grid_representation.redispatch_grid* 
        will reflect N-0 or N-1, i.e. including contingencies, grid representation.
        """
        if self.options["grid"]["include_contingencies_redispatch"]:
            self.grid_representation.redispatch_grid = self.create_scopf_grid_parameters()
        else:
            self.grid_representation.redispatch_grid = self.create_nodal_grid_parameters()

    def create_scopf_grid_parameters(self):
        """Process grid information for nodal N-1 representation.

        Based on chosen sensitivity and GSK the return of
        :meth:`~pomato.grid.create_cnec_data` runs the redundancy removal
        algorithm to reduce the number of constraints to a minimal set. The
        redundancy removal algorithm can take long to conclude, e.g. about
        2 hours for the DE case study which comprises of ~450 nodes and ~1000
        lines.
        Therefore is useful to keep the resulting file with if the relevant
        cbco's and just read it in when needed. This is done by specifying the
        cbco fil in *options["grid"]["precalc_filename"]*.

        There are multiple options to pick, where one is the obvious best
        *conditional_redundancy_removal*. This runs the redundancy removal algorithm including
        bounds on the nodal injections, which are calculated based on
        installed capacity and availability/load timeseries. The other options
        are: *redundancy_removal* redundancy removal without bounds on nodal injections
        and "save" saving the relevant files for the RedundancyRemoval
        algorithm so that it can be run separately from the python POMATO.

        """
        A, b, cbco_info = self.create_cnec_data(self.options["grid"]["sensitivity"],
                                                self.options["grid"]["preprocess"])

        if self.options["grid"]["precalc_filename"]:
            try:
                filename = self.options["grid"]["precalc_filename"]
                self.logger.info("Using cbco indices from pre-calc: %s", filename)
                precalc_cbco = pd.read_csv(self.julia_dir.joinpath(f"cbco_data/{filename}.csv"),
                                           delimiter=',')
                if len(precalc_cbco.columns) > 1:
                    condition = cbco_info[["cb", "co"]].apply(tuple, axis=1) \
                                    .isin(precalc_cbco[["cb", "co"]].apply(tuple, axis=1))
                    cbco_index = list(cbco_info.reset_index().index[condition])
                    self.logger.info("Number of CBCOs from pre-calc: %s", str(len(cbco_index)))
                else:
                    cbco_index = list(precalc_cbco.constraints.values)
                    self.logger.info("Number of CBCOs from pre-calc: %s", str(len(cbco_index)))

            except FileNotFoundError:
                self.logger.warning("FileNotFound: No Precalc available")
                self.logger.warning("Running with full N-1 representation (subject to the lodf filter)")
                cbco_index = cbco_index = list(range(0, len(b)))

        else:
            # 3 valid args supported for redundancy_removal_option:
            # clarkson, conditional_redundancy_removal, full (default)
            if self.options["grid"]["redundancy_removal_option"] == "full":
                cbco_index = list(range(len(b)))
            elif self.options["grid"]["redundancy_removal_option"] == "conditional_redundancy_removal":
                cbco_index = self.clarkson_algorithm(
                    A=A, b=b, x_bounds=self.create_nodal_injection_limits())
            elif self.options["grid"]["redundancy_removal_option"] == "redundancy_removal":
                cbco_index = self.clarkson_algorithm(A=A, b=b)
            elif self.options["grid"]["redundancy_removal_option"] == "save":
                cbco_index = list(range(len(b)))
                self.write_cbco_info(self.julia_dir.joinpath("cbco_data"), "py_save", 
                                     A=A, b=b, Ab_info=cbco_info, 
                                     x_bounds=self.create_nodal_injection_limits())
            else:
                raise AttributeError("No valid redundancy_removal_option set!")

        cbco_nodal_network = self.return_cbco(cbco_info, cbco_index)
        return cbco_nodal_network

    def create_cnec_data(self, sensitivity=5e-2, preprocess=False, gsk=None):
        """Create all relevant N-1 PTDFs in the form of Ax<b (PTDF x < ram).

        This uses the method :meth:`~pomato.grid.create_filtered_n_1_ptdf` to
        generate a filtered ptdf matrix, including outages with a higher impact
        of the argument *sensitivity*.

        Parameters
        ----------
        sensitivity : float, optional
            The sensitivity defines the threshold from which outages are
            considered critical. A outage that can impact the lineflow,
            relative to its maximum capacity, more than the sensitivity is
            considered critical.
        preprocess : bool, optional
            Performing a light preprocessing by removing duplicate constraints.
        gsk : np.ndarray, optional
            When gsk is an argument, this method creates a zonal ptdf matrix
            with it.

        Returns
        -------
        A : np.ndarray
            ptdf matrix, nodal or zonal depending of *gsk* argument, containing
            all lines under outages with significant impact.
        b : np.ndarray
            Line capacities of the cbco in A.
        info : pd.DataFrame
            DataFrame containing the ptdf, ram and information which cbco each
            row corresponds to.

        """
        n_1_ptdf = self.grid.create_filtered_n_1_ptdf(
            sensitivity=sensitivity, 
            short_term_rating_factor=self.options["grid"]["short_term_rating_factor"],
            long_term_rating_factor=self.options["grid"]["long_term_rating_factor"],
            )
        # Processing: Rounding, remove duplicates and 0...0 rows
        if preprocess:
            self.logger.info("Preprocessing Ab...")
            _, idx = np.unique(n_1_ptdf[list(self.grid.nodes.index) + ["ram"]].round(decimals=6).values,
                               axis=0, return_index=True)

            n_1_ptdf = n_1_ptdf.loc[np.sort(idx), :]

        A = n_1_ptdf.loc[:, self.grid.nodes.index].values
        b = n_1_ptdf["ram"].values
        
        if isinstance(gsk, np.ndarray):  # replace nodal ptdf by zonal ptdf
            A = np.dot(A, gsk)
            n_1_ptdf = pd.concat((n_1_ptdf.loc[:, ["cb", "co", "ram"]],
                              pd.DataFrame(columns=self.data.zones.index,
                                           data=A)), axis=1)
        return A, b, n_1_ptdf

    def write_cbco_info(self, folder, suffix, **kwargs):
        """Write cbco information to disk to run the redundancy removal algorithm.

        Parameters
        ----------
        folder : pathlib.Path
            Save file to the specified folder.
        suffix : str
            A suffix for each file, to make it recognizable.
        """
        self.logger.info("Saving A, b...")
        
        for data in [d for d in ["x_bounds", "I"] if d not in kwargs]:
            kwargs[data] = np.array([])

        for data in kwargs:
            self.logger.info("Saving %s to disk...", data)
            if isinstance(kwargs[data], np.ndarray):
                np.savetxt(folder.joinpath(f"{data}_{suffix}.csv"),
                           np.asarray(kwargs[data]), delimiter=",")

            elif isinstance(kwargs[data], pd.DataFrame):
                kwargs[data].to_csv(str(folder.joinpath(f'{data}.csv')), index_label='index')

        self.logger.info("Saved everything to folder: \n %s", str(folder))

    def create_nodal_injection_limits(self):
        """Create nodal injection limits.

        For each node the nodal injection limits depend on the installed
        capacity and availability/load timeseries. Additionally, each node can
        have a slack variables/infeasibility variables, DC-line injections and
        storage charge/discharge.

        Because a nodal injection can impact a line positively/negatively
        depending on the (arbitrary) definition of the incidence matrix,
        only the max(positive bound, abs(negative bound)) is considered.

        TODO Rewrite without for n in nodes loop (takes 7s for the ieee network and 8760t)

        Returns
        -------
        nodal_injection_limits : np.ndarray
            Contains the abs maximum power injected/load at each node.

        """
        infeasibility_upperbound = self.options["infeasibility"]["electricity"]["bound"]
        nodal_injection_limits = []
        demand_el = self.data.demand_el.copy()
        net_export = self.data.net_export[self.data.net_export.net_export > 0].copy()

        for node in self.data.nodes.index:
            plant_types = self.options["plant_types"]

            condition_plant_node = self.data.plants.node == node

            condition_storage = condition_plant_node & \
                                (self.data.plants.plant_type.isin(plant_types["es"]))
            condition_el_heat = condition_plant_node & \
                                (self.data.plants.plant_type.isin(plant_types["ph"]))

            max_dc_inj = self.data.dclines.capacity[(self.data.dclines.node_i == node) |
                                                   (self.data.dclines.node_j == node)].sum()
            
            condition_nex = net_export.node == node
            nex_max = max(0, net_export.loc[condition_nex, "net_export"].max())
            nex_min = -min(0, net_export.loc[condition_nex, "net_export"].min())
            
            condition_demand = demand_el.node == node
            upper = max(self.data.plants.g_max[condition_plant_node].sum()
                        - demand_el.loc[condition_demand, "demand_el"].min()
                        + nex_max
                        + max_dc_inj
                        + infeasibility_upperbound
                        , 0)

            lower = max(demand_el.loc[condition_demand, "demand_el"].max()
                        + self.data.plants.g_max[condition_storage].sum()
                        + self.data.plants.g_max[condition_el_heat].sum()
                        + nex_min
                        + max_dc_inj
                        + infeasibility_upperbound, 0)

            nodal_injection_limits.append(max(upper, lower))

        nodal_injection_limits = np.array(nodal_injection_limits).reshape(len(nodal_injection_limits), 1)
        return nodal_injection_limits

    def clarkson_algorithm(self, args={"file_suffix": "py"}, **kwargs):
        """Run the redundancy removal algorithm.

        The redundancy removal algorithm is run by writing the necessary data
        to disk with "_py" suffix, starting a julia instance and running the
        algorithm. After (successful) completion the resulting file with the
        non-redundant cbco indices is read and returned.

        Returns
        -------
        cbco : list
            List of the essential indices, i.e. the indices of the non-redundant
            cbco's.
        """

        self.write_cbco_info(self.julia_dir.joinpath("cbco_data"), "py", **kwargs)

        if not self.julia_instance:
            self.julia_instance = tools.JuliaDaemon(
                self.logger, self.wdir, self.package_dir, "redundancy_removal", self.options["solver"]["name"]
            )
        if not self.julia_instance.is_alive:
            self.julia_instance = tools.JuliaDaemon(
                self.logger, self.wdir, self.package_dir, "redundancy_removal", self.options["solver"]["name"]
            )

        t_start = dt.datetime.now()
        self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))
        self.julia_instance.run(args=args)
        t_end = dt.datetime.now()
        self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")

        if self.julia_instance.solved:
            file = tools.newest_file_folder(self.julia_dir.joinpath("cbco_data"), keyword="cbco")
            self.logger.info("cbco list save for later use to data_output: \n%s", file.stem + ".csv")
            cbco = list(pd.read_csv(file, delimiter=',').constraints.values)
        else:
            self.logger.critical("Error in Julia code")
            cbco = None
        return cbco

    def return_cbco(self, cbco_info, cbco_index):
        """Return only the cbco's of the info attribute DataFrame.

        Returns
        -------
        cbco_info : DataFrame
            Slice of the full info attribute, containing filtered contingency ptdfs,
            based on the cbco indices resulting from the redundancy removal algorithm.

        """
        cbco_info = cbco_info.iloc[cbco_index].copy()
        cbco_info.loc[:, "index"] = cbco_info.cb + "_" + cbco_info.co
        cbco_info = cbco_info.set_index("index")
        return cbco_info

    def process_ntc(self):
        """Process grid information for NTC representation.

        This only includes assigning ntc data. However if no data is available, dummy
        data is generated.
        """
        if self.data.ntc.empty:
            self.grid_representation.ntc = self.create_ntc()
        else:
            self.grid_representation.ntc = self.data.ntc

    def create_ntc(self, default_ntc=1e5):
        """Create NTC data.

        The ntc generated in this method are high (10.000) or zero. This is useful
        to limit commercial exchange to connected zones or when the model uses a
        simplified line representation.
        
        Parameters
        ----------
        default_ntc : float, optional
            NTC value, defaults to 1e5. 
        Returns
        -------
        ntc : pd.DataFrame
            DataFrame that contains dummy ntc which are the *default_ntc* when market areas are 
            connected and zero otherwise.  
        """

        tmp = []
        for from_zone, to_zone in itertools.combinations(set(self.data.nodes.zone), 2):
            lines = []

            from_nodes = self.data.nodes.index[self.data.nodes.zone == from_zone]
            to_nodes = self.data.nodes.index[self.data.nodes.zone == to_zone]

            condition_i_from = self.data.lines.node_i.isin(from_nodes)
            condition_j_to = self.data.lines.node_j.isin(to_nodes)

            condition_i_to = self.data.lines.node_i.isin(to_nodes)
            condition_j_from = self.data.lines.node_j.isin(from_nodes)

            lines += list(self.data.lines.index[condition_i_from & condition_j_to])
            lines += list(self.data.lines.index[condition_i_to & condition_j_from])

            dclines = []
            condition_i_from = self.data.dclines.node_i.isin(from_nodes)
            condition_j_to = self.data.dclines.node_j.isin(to_nodes)

            condition_i_to = self.data.dclines.node_i.isin(to_nodes)
            condition_j_from = self.data.dclines.node_j.isin(from_nodes)

            dclines += list(self.data.dclines.index[condition_i_from & condition_j_to])
            dclines += list(self.data.dclines.index[condition_i_to & condition_j_from])

            if lines or dclines:
                tmp.append([from_zone, to_zone, default_ntc])
                tmp.append([to_zone, from_zone, default_ntc])
            else:
                tmp.append([from_zone, to_zone, 0])
                tmp.append([to_zone, from_zone, 0])

        return pd.DataFrame(tmp, columns=["zone_i", "zone_j", "ntc"])

