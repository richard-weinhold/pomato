import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progress
progress.HIDE_CURSOR, progress.SHOW_CURSOR = '', ''

import pomato
import pomato.tools as tools
from pomato.grid import GridModel
from progress.bar import Bar

class FBMCModule():
    """The FBMC module calculates FB paramerters based on a suitable market result.

    Flow based market coupling derives commercial exchange capacities for a day-ahead 
    market clearing from a forecasted market results. These day ahead capacities are 
    reported in a zonal PTDF matrix, representing the impact of changes in zonal net-position (NEX)
    on network elements under contingencies, together with the remaining capacity on these 
    network elements (RAM, remaining available margin).

    These are denoted as flow based paramters and represent the main output of this module. 


    Parameters
    ----------
    wdir : pathlib.Path
        Working directory
    grid_object : :class:`~pomato.grid.GridTopology`
        An instance of the DataManagement class with processed input data.
    data : :class:`~pomato.data.DataManagement`
        An instance of the DataManagement class with processed input data.
    options : dict
        The options from POMATO main module.

    """
    def __init__(self, wdir, grid, data, options):
        self.logger = logging.getLogger('log.pomato.fbmc.FBMCModule')
        self.logger.info("Initializing the FBMCModule....")

        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])

        self.options = options
        self.grid = grid
        self.data = data

        # Attributes that are calculated once, as they only depend on the topology and are 
        # independent from the basecase. 
        self.custom_cbco = None
        self.flowbased_region = None
        self.nodal_fbmc_ptdf = None
        self.fbmc_data = None

    def calculate_parameters(self, custom_cbco=None, flowbased_region=None):
        """Calculate basic, non result specific, paramters.

        Parameters
        ----------
        custom_cbco : pd.DataFrame, optional
            Specify the list of CBCOs considered for the FB paramters, by default None which 
            will cause CBCOs chosen based on a zone-to-zone sensitivity.
        flowbased_region : list, optional
            List of countries for which the FB parameters are calculated, defaults to all zones.
        """        
        self.custom_cbco = custom_cbco
        if not flowbased_region:
            self.flowbased_region = list(self.data.zones.index)
        self.nodal_fbmc_ptdf, self.fbmc_data = self.create_base_fbmc_parameters()

    def create_dynamic_gsk(self, basecase, timestep):
        """Returns GSK based on the included basecase.

        As the dynamic GSK depends on the generation schedule, it needs 
        the timestep as input arguments
        
        Parameters
        ----------
        basecase : :class:`~pomato.data.Result`
            The basecase, that is used for the calculation of the FB paramters.
        timestep : string
            Timestep, for which the GSK is calculated
        
        Returns
        -------
        gsk : Array
            A Zone X Node array that will yield a zonal ptdf when multiplied
            with the nodal ptdf. 
        """
        
        gsk = pd.DataFrame(index=self.grid.nodes.index)
        plant_types = self.data.options["plant_types"]
        condition = (~self.data.plants.plant_type.isin(plant_types["ts"])) & \
                    (~self.data.plants.plant_type.isin(plant_types["es"]))
        
        gen = basecase.G.copy()
        gen = gen[(gen.t == timestep)&(gen.p.isin(self.data.plants.index[condition]))]
        gen.loc[:, "n"] = self.data.plants.loc[gen.p, "node"].values
        for zone in self.data.zones.index:
            gsk[zone] = 0
            nodes_in_zone = self.grid.nodes.index[self.grid.nodes.zone == zone]
            tmp = gen[gen.n.isin(nodes_in_zone)].groupby("n").sum().copy()
            tmp.loc[:, "G"] /= tmp.G.max()
            gsk.loc[tmp.index, zone] = tmp.G.values
        
        return gsk.values
        
    def create_gsk(self, option="flat"):
        """Returns static GSK.

        Input options are: 
            - *flat*: for equal participation of each node to the NEX
            - *gmax*: for weighted participation proportional to the installed
              capacity of conventional generation. 
        
        Conventional generation is defined as generation which is not of plant type
        *es* or *ts*. 

        Parameters
        ----------
        option : str, optional
            Options are *flat* or *gmax*, defaults to flat. 
        
        Returns
        -------
        gsk : Array
            A Zone X Node array that will yield a zonal ptdf when multiplied
            with the nodal ptdf. 
        """        
        gsk = pd.DataFrame(index=self.grid.nodes.index)

        plant_types = self.data.options["plant_types"]
        condition = (~self.data.plants.plant_type.isin(plant_types["ts"])) & \
                    (~self.data.plants.plant_type.isin(plant_types["es"]))
        
        gmax_per_node = self.data.plants.loc[condition, ["g_max", "node"]] \
                        .groupby("node").sum()

        for zone in self.data.zones.index:
            nodes_in_zone = self.grid.nodes.index[self.grid.nodes.zone == zone]
            gsk[zone] = 0
            gmax_in_zone = gmax_per_node[gmax_per_node.index.isin(nodes_in_zone)]
            if option == "gmax":
                if not gmax_in_zone.empty:
                    gsk_value = gmax_in_zone.g_max/gmax_in_zone.values.sum()
                    gsk.loc[gsk.index.isin(gmax_in_zone.index), zone] = gsk_value
                else:
                    gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)

            elif option == "flat":
                gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)

        return gsk.values

    def return_critical_branches(self, threshold=5e-2, gsk_strategy="gmax", 
                                 only_crossborder=False):
        """Returns Critical branches based on Zone-to-Zone ptdf.
        
        In the calculation of FB parameters it makes sense to use only lines
        to constrain commercial exchange, that are actually affected by it. Otherwise,
        lines would be part of the FB domain that have no sensitivity towards the commercial 
        exchange. 
            
        Lines are selected based on the Zone-to-Zone PTDF values. This is done to 
        remove the slack dependencies of the nodal PTDF. The Zonal PTDF is calculated 
        using the defined GSK strategy. 

        In addition to the CBs, crossborder lines are part of the set. 

        Parameters
        ----------
        threshold : float, optional
            Zone-to-Zone PTDF threshold, defaults to 5%. 
        gsk_strategy : str, optional
            GSK strategy, defaults to *gmax*.
        only_crossborder : str, optional 
            Only consider cross-border lines as critical branches, defaults to False. 
        Returns
        -------
        CBs : list
            List of all critical branches, including all cross border lines. 
        
        """
        self.logger.info("List of CBs is generated from zone-to-zone PTDFs with:")
        self.logger.info("GSK Strategy: %s, Threshold: %d percent", gsk_strategy, threshold*100)

        if not only_crossborder:
            gsk = self.create_gsk(gsk_strategy)
            zonal_ptdf = np.dot(self.grid.ptdf, gsk)
            zonal_ptdf_df = pd.DataFrame(index=self.grid.lines.index,
                                            columns=self.data.zones.index,
                                            data=zonal_ptdf)

            z2z_ptdf_df = pd.DataFrame(index=self.grid.lines.index)
            for zone in self.flowbased_region:
                for z_zone in self.flowbased_region:
                    z2z_ptdf_df["-".join([zone, z_zone])] = zonal_ptdf_df[zone] - zonal_ptdf_df[z_zone]

            critical_branches = list(z2z_ptdf_df.index[np.any(z2z_ptdf_df.abs() > threshold, axis=1)])
        else: 
            critical_branches = []

        condition_cross_border = self.grid.nodes.zone[self.grid.lines.node_i].values != \
                                 self.grid.nodes.zone[self.grid.lines.node_j].values

        condition_fb_region = self.grid.nodes.zone[self.grid.lines.node_i].isin(self.flowbased_region).values & \
                         self.grid.nodes.zone[self.grid.lines.node_j].isin(self.flowbased_region).values

        cross_border_lines = list(self.grid.lines.index[condition_cross_border&condition_fb_region])
        
        all_cbs = list(set(critical_branches + cross_border_lines))
        self.logger.info("Number of Critical Branches: %d, Number of Cross Border lines: %d, Total Number of CBs: %d",
                          len(critical_branches), len(cross_border_lines), len(all_cbs))

        return all_cbs

    def create_base_fbmc_parameters(self, lodf_sensitivity=10e-2):
        """Create the nodal ptdf for the FB paramters.
        
        Which lines are considered critical and their critical outages are independent 
        from the basecase. Therefore the ptdf for all CBCOs are obtained once and used throughout 
        this method.
        
        The process is similar to :meth:`~pomato.grid.GridTopology.create_filtered_n_1_ptdf`, 
        for a list of CBs, that are obtained in :meth:`~return_critical_branches`,
        outages are selected based on a lodf threshold. The method returns the nodal ptdf Nodes x CBCOs 
        matrix and a pd.DataFrame with all additional information. 
        
        Parameters
        ----------
        lodf_sensitivity : float, optional
            The sensitivity defines the threshold from which outages are
            considered critical. A outage that can impact the lineflow,
            relative to its maximum capacity, more than the sensitivity is
            considered critical.
        
        Returns
        -------
        nodal_fbmc_ptdf : numpy.array
            nodal PTDF for each CBCO
        fbmc_data : pandas.DataFrame
            The ptdf, together with all information, like CBCO, capacity, nodes.
        """

        if isinstance(self.custom_cbco, pd.DataFrame):
            base_cb = list(self.custom_cbco.cb[self.custom_cbco.co == "basecase"])

            index_position = [self.grid.lines.index.get_loc(line) for line in base_cb]
            base_ptdf = self.grid.ptdf[index_position, :]
            full_ptdf = [base_ptdf, -base_ptdf]
            label_lines = list(base_cb)+list(base_cb)
            label_outages = ["basecase" for line in label_lines]

            self.custom_cbco = self.custom_cbco[~(self.custom_cbco.co == "basecase")]
            select_lines = []
            select_outages = {}
            for line in self.custom_cbco.cb.unique():
                select_lines.append(line)
                select_outages[line] = list(self.custom_cbco.co[self.custom_cbco.cb == line])

        else:
            self.grid.lines["cb"] = False
            critical_branches = self.return_critical_branches(threshold=self.options["grid"]["sensitivity"])
            self.grid.lines.loc[self.grid.lines.index.isin(critical_branches), "cb"] = True

            select_lines = self.grid.lines.index[(self.grid.lines["cb"])&(self.grid.lines.contingency)]
            select_outages = {}
            for line in select_lines:
                select_outages[line] = list(self.grid.lodf_filter(line, lodf_sensitivity))

            index_position = [self.grid.lines.index.get_loc(line) for line in select_lines]
            base_ptdf = self.grid.ptdf[index_position, :]
            full_ptdf = [base_ptdf, -base_ptdf]
            label_lines = list(select_lines)+list(select_lines)
            label_outages = ["basecase" for line in label_lines]

        for line in select_lines:
            outages = select_outages[line]
            tmp_ptdf = np.vstack([self.grid.create_n_1_ptdf_cbco(line, out) for out in outages])
            full_ptdf.extend([tmp_ptdf, -1*tmp_ptdf])
            label_lines.extend([line for i in range(0, 2*len(outages))])
            label_outages.extend(outages*2)

        nodal_fbmc_ptdf = np.concatenate(full_ptdf)
        nodal_fbmc_ptdf = nodal_fbmc_ptdf.reshape(len(label_lines), len(list(self.grid.nodes.index)))

        fbmc_data = pd.DataFrame(columns=list(self.data.zones.index))
        fbmc_data["cb"] = label_lines
        fbmc_data["co"] = label_outages

        return nodal_fbmc_ptdf, fbmc_data

    def create_flowbased_ptdf(self, gsk_strategy, timestep, basecase):
        """Create Zonal PTDF, reference flows and RAM based on the basecase and GSK.

        For a specific timestep, the reference flow is calculated as the basecase flow 
        minus the flow resulting from the DA market:
        
        F_ref = F_basecase - F_DA

        Where F_DA = PTDF*GSK*NEX, derives from the net position in the basecase.
        The RAM therefore represents the line capacity minus F_ref and additional security 
        margins line FRM/FAV. 

        Depending on how the CBCOs are selected and the basecase is cleared, negative RAMs
        are possible, however for the purpose of market clearing, these have to positive. 

        This either indicates an error in the calculation or the need for relaxation via the 
        minRAM option. 

        Parameters
        ----------
        gsk_strategy : str
            GSK strategy, which will be generation by :meth:`~create_gsk()`. 
        timestep : str
            Timestep for which the paramters are calculated for.
        basecase : :class:`~pomato.data.Results`
            Market resultsfrom which the FB paramters are deducted. 

        Returns
        -------
        fbmc_domain_data, pd.DataFrame
            Based on the precalculated attribute *fbmc_data*, the domain parameters are calculated 
            for a specific timestep and GSK. This is returned, including the zonal ptdf and ram. 
        """
        domain_data = self.fbmc_data.copy()
        # optional frm/fav margin todo
        frm_fav = pd.DataFrame(index=domain_data.cb.unique())
        frm_fav["value"] = self.grid.lines.capacity[frm_fav.index]*(1 - self.options["grid"]["capacity_multiplier"])

        # F Ref Basecase: The actual flow in the basecase
        # On all CBs under COs
        injection = basecase.INJ.INJ[basecase.INJ.t == timestep].values
        f_ref_base_case = np.dot(self.nodal_fbmc_ptdf, injection)

        if gsk_strategy == "dynamic":
            gsk = self.create_dynamic_gsk(basecase, timestep)
        else:
            gsk = self.create_gsk(gsk_strategy)

        # F Day Ahead: The Flow on each CBCO, without injection from the market 
        # coupling. The market is "removed" by removing the NEX via a zonal ptdf and GSK
        net_position = basecase.net_position()
        zonal_fbmc_ptdf = np.dot(self.nodal_fbmc_ptdf, gsk)
        f_da = np.dot(zonal_fbmc_ptdf, net_position.loc[timestep].values)

        # F_ref without the DA market = F Ref Basecase - F DA
        f_ref_nonmarket = f_ref_base_case - f_da

        # RAMs
        ram = (self.grid.lines.capacity[domain_data.cb] 
               - frm_fav.value[domain_data.cb] 
               - f_ref_nonmarket)

        minram = self.grid.lines.capacity[domain_data.cb] * self.options["grid"]["minram"] 
        ram[ram < minram] = minram[ram < minram]

        ram = ram.values.reshape(len(ram), 1)

        if any(ram < 0):
            self.logger.warning("Number of RAMs below: [0 - %d, 10 - %d, 100 - %d, 1000 - %d]", 
                                sum(ram<0), sum(ram<10), sum(ram<100), sum(ram<1000))

        domain_data[list(self.data.zones.index)] = zonal_fbmc_ptdf
        domain_data["ram"] = ram
        domain_data["timestep"] = timestep
        domain_data["gsk_strategy"] = gsk_strategy
        domain_data = domain_data[["cb", "co", "ram", "timestep", "gsk_strategy"] + list(list(self.data.zones.index))]
        return domain_data

    def create_flowbased_parameters(self, basecase, gsk_strategy="gmax", reduce=False):
        """Create Flow-Based Paramters.

        Creates the FB Paramters for the basecase arguments. Optional arguments are 
        the GSK that is used during the process and whether the constraints are reduced
        to a minimal set.  
        
        Parameters
        ----------
        basecase : :class:`~pomato.data.Results`
            Market resultsfrom which the FB paramters are deducted. 
        gsk_strategy : str, optional
            GSK strategy, defaults to "gmax". 
        reduce : bool, optional
            Runs the RedundancyRemoval for each timestep 
        
        Returns
        -------
        fb_paramters : pandas.DataFrame
            Flow Based Parameters which are a zonal ptdf for each and ram, depending
            on the reference flows derived from the basecase for each timestep. 
        """
        domain_data = {}
        if not all([self.custom_cbco, self.flowbased_region, self.nodal_fbmc_ptdf, self.fbmc_data]):
                    self.calculate_parameters()

        grid_model = GridModel(self.wdir, self.grid, self.data, self.data.options)
        if reduce:
            grid_model._start_julia_daemon()
        grid_model.options["type"] = "cbco_zonal"
        grid_model.options["grid"]["cbco_option"] = "clarkson"

        self.logger.info("Generating FB parameters for each timestep...")
        self.logger.info("Enforcing minRAM of %s%%", str(round(self.options["grid"]["minram"]*100)))

        bar = Bar('Processing FB Parameters...', max=len(basecase.model_horizon), 
                  check_tty=False, hide_cursor=True)
        for timestep in basecase.model_horizon:
            domain_data[timestep] = self.create_flowbased_ptdf(gsk_strategy, timestep, basecase)
            bar.next()
        bar.finish()

        cbco_info =  pd.concat([domain_data[t] for t in domain_data.keys()], ignore_index=True)
        if reduce:
            cbco_index = grid_model.clarkson_algorithm(args={"fbmc_domain": True}, Ab_info=cbco_info)
        else:
            cbco_index = list(range(0, len(cbco_info)))

        fb_paramters = grid_model.return_cbco(cbco_info, cbco_index)
        fb_paramters.set_index(fb_paramters.cb + "_" + fb_paramters.co, inplace=True)

        if reduce:
            grid_model.julia_instance.join()
            grid_model.julia_instance.julia_instance = None

        return fb_paramters
