import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pomato
from pomato.grid import GridModel

class FBMCModule():
    """The FBMC module calculates FB parameters based on a suitable market result.

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

        self.grid_model = GridModel(self.wdir, self.grid, self.data, self.data.options)

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
            if tmp.G.sum() > 0:
                tmp.loc[:, "G"] /= tmp.G.sum()
                gsk.loc[tmp.index, zone] = tmp.G.values
            else:
                 gsk.loc[nodes_in_zone, zone] = 1/len(nodes_in_zone)
        return gsk.values
        
    def create_gsk(self, option="gmax"):
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
                if not (gmax_in_zone.empty or gmax_in_zone.g_max.sum() == 0):
                    gsk_value = gmax_in_zone.g_max/gmax_in_zone.values.sum()
                    gsk.loc[gsk.index.isin(gmax_in_zone.index), zone] = gsk_value
                else:
                    gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)

            elif option == "flat":
                gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)
        
        return gsk.values

    def return_critical_branches(self, threshold=5e-2, gsk_strategy="gmax", flowbased_region=None):
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
        flowbased_region : list-like, optional,
            Specify a subset of zones that compose the flow based region. Default to all zones. 
        Returns
        -------
        CBs : list
            List of all critical branches, including all cross border lines. 
        
        """

        if not flowbased_region:
            flowbased_region = list(self.data.zones.index)
        
        if not self.options["fbmc"]["only_crossborder"]:
            gsk = self.create_gsk(gsk_strategy)
            zonal_ptdf = np.dot(self.grid.ptdf, gsk)
            zonal_ptdf_df = pd.DataFrame(index=self.grid.lines.index,
                                         columns=self.data.zones.index,
                                         data=zonal_ptdf)

            z2z_ptdf_df = pd.DataFrame(index=self.grid.lines.index)
            for zone in flowbased_region:
                for z_zone in flowbased_region:
                    z2z_ptdf_df["-".join([zone, z_zone])] = zonal_ptdf_df[zone] - zonal_ptdf_df[z_zone]

            critical_branches = list(z2z_ptdf_df.index[np.any(z2z_ptdf_df.abs() > threshold, axis=1)])
        else: 
            critical_branches = []

        condition_cross_border = self.grid.nodes.zone[self.grid.lines.node_i].values != \
                                 self.grid.nodes.zone[self.grid.lines.node_j].values

        condition_fb_region = self.grid.nodes.zone[self.grid.lines.node_i].isin(flowbased_region).values & \
            self.grid.nodes.zone[self.grid.lines.node_j].isin(flowbased_region).values
        cross_border_lines = list(self.grid.lines.index[condition_cross_border&condition_fb_region])
        
        all_cbs = list(set(critical_branches + cross_border_lines))
        self.logger.info("Number of Critical Branches: %d, Number of Cross Border lines: %d, Total Number of CBs: %d",
                          len(critical_branches), len(cross_border_lines), len(all_cbs))

        return all_cbs

    def create_base_fbmc_parameters(self, critical_branches, lodf_sensitivity=10e-2):
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
        critical_branches : list-like, 
            Lines considered critical network elements, which are considered for the FB parameters.  
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
        self.grid.lines["cb"] = False
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
   
    def create_flowbased_parameters(self, basecase, timesteps=None):
        """Create Flow-Based Paramters.

        Creates the FB Paramters for the supplied basecase. Optional arguments are 
        the GSK that is used during the process and whether the constraints are reduced
        to a minimal set.  

        For a specific timestep or over the full model horizon, the reference flow is calculated 
        as the basecase flow minus the flow resulting from the DA market:
        
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
        basecase : :class:`~pomato.data.Results`
            Market resultsfrom which the FB paramters are deducted. 

        Returns
        -------
        fb_parameters : pandas.DataFrame
            Flow Based Parameters which are a zonal ptdf for each and ram, depending
            on the reference flows derived from the basecase for each timestep. 
        """

        # Check arguments and options 
        if not timesteps:
            timesteps = basecase.model_horizon
        cne_sensitivity = self.options["fbmc"]["cne_sensitivity"]
        lodf_sensitivity = self.options["fbmc"]["lodf_sensitivity"]
        gsk_strategy = self.options["fbmc"]["gsk"]

        if not self.options["fbmc"]["flowbased_region"]:
            flowbased_region = list(self.data.zones.index)
            self.options["fbmc"]["flowbased_region"] = flowbased_region
        else:
            flowbased_region = self.options["fbmc"]["flowbased_region"]

        self.logger.info("CBs are selected from zone-to-zone PTDFs with %d%% threshold", cne_sensitivity*100)
        critical_branches = self.return_critical_branches(cne_sensitivity, flowbased_region=flowbased_region)
        self.logger.info("COs are selected from nodal PTDFs with %d%% threshold", lodf_sensitivity*100)
        nodal_fbmc_ptdf, fbmc_data = self.create_base_fbmc_parameters(critical_branches, lodf_sensitivity)

        if (self.options["grid"]["precalc_filename"]) and (not self.options["fbmc"]["precalc_filename"]):
            cbco = self.grid_model.create_cbco_nodal_grid_parameters()
            condition = fbmc_data[["cb", "co"]].apply(tuple, axis=1).isin(cbco[["cb", "co"]].apply(tuple, axis=1).values).values
            nodal_fbmc_ptdf, fbmc_data = nodal_fbmc_ptdf[condition | (fbmc_data.co == "basecase") , :], \
                                                         fbmc_data.loc[condition | (fbmc_data.co == "basecase") , :]

        inj = basecase.INJ[basecase.INJ.t.isin(timesteps)].pivot(index="t", columns="n", values="INJ")
        inj = inj.loc[timesteps, basecase.data.nodes.index]
        f_ref_base_case = np.dot(nodal_fbmc_ptdf, inj.T)

        frm_fav = self.grid.lines.capacity[fbmc_data.cb].values*self.options["fbmc"]["frm"]
        nex = basecase.net_position().loc[timesteps, :]

        self.logger.info("Calculating zonal ptdf using %s gsk strategy.", gsk_strategy)
        if gsk_strategy == "dynamic":
            zonal_fbmc_ptdf = {timestep: np.dot(nodal_fbmc_ptdf, self.create_dynamic_gsk(basecase, timestep)) for timestep in timesteps}
            f_da = np.vstack([np.dot(zonal_fbmc_ptdf[timestep], nex.loc[timestep, :]) for timestep in timesteps]).T
        else:
            zonal_fbmc_ptdf_tmp = np.dot(nodal_fbmc_ptdf, self.create_gsk(gsk_strategy))
            zonal_fbmc_ptdf = {timestep: zonal_fbmc_ptdf_tmp for timestep in timesteps}
            f_da = np.dot(zonal_fbmc_ptdf_tmp, nex.values.T)
        
        f_ref_nonmarket = f_ref_base_case - f_da
        ram = (self.grid.lines.capacity[fbmc_data.cb].values - frm_fav - f_ref_nonmarket.T).T
        minram = (self.grid.lines.capacity[fbmc_data.cb] * self.options["fbmc"]["minram"]).values.reshape(len(f_ref_base_case), 1)
        self.logger.info("Applying minRAM of %d%% on %d CBCOs", self.options["fbmc"]["minram"]*100, (ram < minram).any(axis=1).sum())
        for j in range(0, ram.shape[0]):
            ram[j, ram[j, :] < minram[j]] = minram[j]

        domain_data = {}
        for t, timestep in enumerate(timesteps):
            fb_parameters = fbmc_data.copy()
            fb_parameters[list(self.data.zones.index)] = zonal_fbmc_ptdf[timestep]
            fb_parameters["ram"] = ram[:, t]
            fb_parameters["timestep"] = timestep
            fb_parameters["gsk_strategy"] = gsk_strategy
            fb_parameters = fb_parameters[["cb", "co", "ram", "timestep", "gsk_strategy"] + list(list(self.data.zones.index))]
            domain_data[timestep] = fb_parameters

        fb_parameters =  pd.concat([domain_data[timestep] for timestep in timesteps], ignore_index=True)
        
        if self.options["fbmc"]["enforce_ntc_domain"]:
            fb_parameters = self.enforce_ntc_domain(fb_parameters)

        if self.options["fbmc"]["reduce"]:
            if self.options["fbmc"]["precalc_filename"]:
                name = self.options["fbmc"]["precalc_filename"]
                if Path(name).is_file():
                    filename = Path(name)
                if Path(name).with_suffix('.csv').is_file():
                    filename = Path(name).with_suffix('.csv')
                elif self.grid_model.wdir.joinpath(f"data_temp/julia_files/cbco_data/{name}.csv").is_file():
                    filename = self.wdir.joinpath(f"data_temp/julia_files/cbco_data/{name}.csv")
                else:
                    raise FileNotFoundError("No precalculated list of CNECs found")
                precalc_cbco = pd.read_csv(filename, delimiter=',')
                cbco_index = list(precalc_cbco.constraints.values)
            else:
                cbco_index = self.grid_model.clarkson_algorithm(args={"fbmc_domain": True}, 
                                                            Ab_info=fb_parameters)   
            fb_parameters = fb_parameters.loc[cbco_index, :]
            
        fb_parameters.set_index(fb_parameters.cb + "_" + fb_parameters.co, inplace=True)
        return fb_parameters

    def enforce_ntc_domain(self, fb_parameters):
        """Remove enforce domain to include NTC values"""

        if self.data.ntc.empty:
            self.logger.error("NTCs not in data.")
            return fb_parameters


        self.logger.info("Enforcing FB-parameters to include NTC domain.")
        A = fb_parameters.loc[:, self.data.zones.index].values
        b = fb_parameters.loc[:, "ram"].values.reshape(len(A), 1)

        fb_region = self.options["fbmc"]["flowbased_region"]
        zones = list(self.data.zones.index)

        ntc = self.data.ntc[(self.data.ntc.ntc > 0)].copy()
        condition_fb_region = (ntc.zone_i.isin(fb_region))&(ntc.zone_j.isin(fb_region))
        ntc.set_index(["zone_i", "zone_j"], inplace=True)

        vertices_ntc_domain = int(np.math.factorial(len(ntc.index)) / np.math.factorial(len(fb_region))  / 
            np.math.factorial(len(ntc.index) - len(fb_region))) 
        self.logger.info("Including %s vertices of the NTC domains", vertices_ntc_domain)
        if vertices_ntc_domain > 1e7:
            self.logger.error("Too many dimension to consider (combination(ntc, FB Region).")
            return fb_parameters

        points = []
        for exchange in itertools.combinations(ntc.loc[condition_fb_region.values].index, len(fb_region)):
            tmp = np.zeros((len(zones), 1))
            for (f,t) in exchange:
                tmp[zones.index(f)] += ntc.loc[(f,t), "ntc"]
                tmp[zones.index(t)] -= ntc.loc[(f,t), "ntc"]
            points.append(tmp)
        
        condition = []
        for p in points:
            condition.append(np.dot(A, p).reshape(len(A), 1) <= b)

        return fb_parameters.loc[np.hstack(condition).all(axis=1), :]
