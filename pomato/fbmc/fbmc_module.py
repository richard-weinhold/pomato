import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pomato
import pomato.tools as tools
from pomato.grid import GridModel

class FBMCModule():
    """Class to do all calculations in connection with cbco calculation

    Parameters
    ----------
    wdir : pathlib.Path
        Working directory
    grid_object : :class:`~pomato.grid.GridTopology`
        An instance of the DataManagement class with processed input data.
    data : :class:`~pomato.data.DataManagement`
        An instance of the DataManagement class with processed input data.
    options : dict
        The options from POMATO main method persist in the CBCOModule.
    custom_cbco : pd.DataFrame, optional
        Specify the list of CBCOs considered for the FB paramters, by default None which 
        will cause CBCOs chosen based on a zone-to-zone sensitivity.
    flowbased_region : list, optional
        List of countries for which the FB parameters are calculated, defaults to all zones.
    """
    def __init__(self, wdir, grid_object, data, options, custom_cbco=None, flowbased_region=None):

        self.logger = logging.getLogger('Log.MarketModel.FBMCModule')
        self.logger.info("Initializing the FBMCModule....")

        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])

        self.options = options
        self.grid = grid_object
        self.nodes = grid_object.nodes
        self.lines = grid_object.lines
        if not flowbased_region:
            self.flowbased_region = list(data.zones.index)

        self.custom_cbco = custom_cbco
        self.data = data
        self.nodal_fbmc_ptdf, self.domain_info = self.create_fbmc_info()

        self.logger.info("FBMCModule  Initialized!")

    def create_dynamic_gsk(self, basecase, timestep):
        """Returns GSK based on the included basecase."""
        
        gsk = pd.DataFrame(index=self.nodes.index)
        plant_types = self.data.options["optimization"]["plant_types"]
        condition = (~self.data.plants.plant_type.isin(plant_types["ts"])) & \
                    (~self.data.plants.plant_type.isin(plant_types["es"]))
        
        gen = basecase.G.copy()
        gen = gen[(gen.t == timestep)&(gen.p.isin(self.data.plants.index[condition]))]
        gen.loc[:, "n"] = self.data.plants.loc[gen.p, "node"].values
        for zone in self.data.zones.index:
            gsk[zone] = 0
            nodes_in_zone = self.nodes.index[self.nodes.zone == zone]
            tmp = gen[gen.n.isin(nodes_in_zone)].groupby("n").sum().copy()
            tmp.loc[:, "G"] /= tmp.G.max()
            gsk.loc[tmp.index, zone] = tmp.G.values
        
        return gsk.values
        
    def create_gsk(self, option="flat"):
        """returns GSK, either flat or gmax"""
        gsk = pd.DataFrame(index=self.nodes.index)

        plant_types = self.data.options["optimization"]["plant_types"]
        condition = (~self.data.plants.plant_type.isin(plant_types["ts"])) & \
                    (~self.data.plants.plant_type.isin(plant_types["es"]))
        
        gmax_per_node = self.data.plants.loc[condition, ["g_max", "node"]] \
                        .groupby("node").sum()

        for zone in self.data.zones.index:
            nodes_in_zone = self.nodes.index[self.nodes.zone == zone]
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

    def return_critical_branches(self, threshold=1e-2, gsk_strategy="gmax"):

        self.logger.info("List of CBs is generated from zone-to-zone PTDFs with:")
        self.logger.info("GSK Strategy: %s, Threshold: %d percent", gsk_strategy, threshold*100)

        gsk = self.create_gsk(gsk_strategy)
        zonal_ptdf = np.dot(self.grid.ptdf, gsk)
        zonal_ptdf_df = pd.DataFrame(index=self.lines.index,
                                     columns=self.data.zones.index,
                                     data=zonal_ptdf)

        z2z_ptdf_df = pd.DataFrame(index=self.lines.index)
        for zone in self.flowbased_region:
            for z_zone in self.flowbased_region:
                z2z_ptdf_df["-".join([zone, z_zone])] = zonal_ptdf_df[zone] - zonal_ptdf_df[z_zone]

        critical_branches = list(z2z_ptdf_df.index[np.any(z2z_ptdf_df.abs() > threshold, axis=1)])

        condition_cross_border = self.nodes.zone[self.lines.node_i].values != \
                                 self.nodes.zone[self.lines.node_j].values

        condition_fb_region = self.nodes.zone[self.lines.node_i].isin(self.flowbased_region).values & \
                         self.nodes.zone[self.lines.node_j].isin(self.flowbased_region).values

        cross_border_lines = list(self.lines.index[condition_cross_border&condition_fb_region])
        total_cbs = list(set(critical_branches + cross_border_lines))

        self.logger.info("Number of Critical Branches: %d, Number of Cross Border lines: %d, Total Number of CBs: %d",
                          len(critical_branches), len(cross_border_lines), len(total_cbs))

        return total_cbs

    def create_fbmc_info(self, lodf_sensitivity=10e-2):
        """
        create ptdf, determine CBs
        """

        if isinstance(self.custom_cbco, pd.DataFrame):
            base_cb = list(self.custom_cbco.cb[self.custom_cbco.co == "basecase"])

            index_position = [self.lines.index.get_loc(line) for line in base_cb]
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
            self.lines["cb"] = False
            critical_branches = self.return_critical_branches(threshold=self.options["grid"]["sensitivity"])
            self.lines.loc[self.lines.index.isin(critical_branches), "cb"] = True

            select_lines = self.lines.index[(self.lines["cb"])&(self.lines.contingency)]
            select_outages = {}
            for line in select_lines:
                select_outages[line] = list(self.grid.lodf_filter(line, lodf_sensitivity))

            index_position = [self.lines.index.get_loc(line) for line in select_lines]
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
        nodal_fbmc_ptdf = nodal_fbmc_ptdf.reshape(len(label_lines), len(list(self.nodes.index)))

        domain_info = pd.DataFrame(columns=list(self.data.zones.index))
        domain_info["cb"] = label_lines
        domain_info["co"] = label_outages

        return nodal_fbmc_ptdf, domain_info

    def create_flowbased_ptdf(self, gsk_strategy, timestep, basecase):
        """
        Create Zonal ptdf -> creates both positive and negative line
        restrictions or ram. Depending on flow != 0
        """

        self.logger.info("Creating zonal Ab for timestep %s", timestep)
        # Calculate zonal ptdf based on ram -> (if current flow is 0 the
        # zonal ptdf is based on overall
        # available line capacity (l_max)), ram is calculated for every n-1
        # ptdf matrix to ensure n-1 security constrained FB Domain
        # The right side of the equation has to be positive

        frm_fav = pd.DataFrame(index=self.domain_info.cb.unique())
        frm_fav["value"] = self.lines.maxflow[frm_fav.index]*0

        injection = basecase.INJ.INJ[basecase.INJ.t == timestep].values

        f_ref_base_case = np.dot(self.nodal_fbmc_ptdf, injection)
        if gsk_strategy == "dynamic":
            gsk = self.create_dynamic_gsk(basecase, timestep)
        else:
            gsk = self.create_gsk(gsk_strategy)
        zonal_fbmc_ptdf = np.dot(self.nodal_fbmc_ptdf, gsk)

        # F Day Ahead (should include LTNs)
        net_position = basecase.net_position() * 1
        # net_position.loc[:, ~net_position.columns.isin(self.flowbased_region)] = 0

        f_da = np.dot(zonal_fbmc_ptdf, net_position.loc[timestep].values)
        # f_ref_nonmarket = f_ref_base_case
        f_ref_nonmarket = f_ref_base_case - f_da

        # capacity_multiplier = basecase.data.options["grid"]["capacity_multiplier"]

        ram = (self.lines.maxflow[self.domain_info.cb] 
               - frm_fav.value[self.domain_info.cb] 
               - f_ref_nonmarket)

        self.logger.info("Applying minRAM at %i percent of line capacity", 
                         int(self.options["grid"]["minram"]*100))
        minram = self.lines.maxflow[self.domain_info.cb] * self.options["grid"]["minram"] 
        ram[ram < minram] = minram[ram < minram]

        ram = ram.values.reshape(len(ram), 1)

        if any(ram < 0):
            self.logger.warning("Number of RAMs below: [0 - %d, 10 - %d, 100 - %d, 1000 - %d]", 
                                sum(ram<0), sum(ram<10), sum(ram<100), sum(ram<1000))
            # ram[ram <= 0] = 0.1

        self.domain_info[list(self.data.zones.index)] = zonal_fbmc_ptdf
        self.domain_info["ram"] = ram
        self.domain_info["timestep"] = timestep
        self.domain_info["gsk_strategy"] = gsk_strategy
        self.domain_info = self.domain_info[["cb", "co", "ram", "timestep", "gsk_strategy"] + list(list(self.data.zones.index))]
        # self.logger.info("Done!")
        return zonal_fbmc_ptdf, ram

    def create_flowbased_parameters(self, basecase, gsk_strategy="gmax", reduce=True):
        
        domain_data = {}
        grid_model = GridModel(self.wdir, self.grid, self.data, self.data.options)
        
        if reduce:
            grid_model._start_julia_daemon()
            
        grid_model.options["optimization"]["type"] = "cbco_zonal"
        grid_model.options["grid"]["cbco_option"] = "clarkson"

        for timestep in basecase.INJ.t.unique():
            self.create_flowbased_ptdf(gsk_strategy, timestep, basecase)
            domain_data[timestep] = self.domain_info.copy()

        cbco_info =  pd.concat([domain_data[t] for t in domain_data.keys()], ignore_index=True)
        
        if reduce:
            cbco_index = grid_model.clarkson_algorithm(args={"fbmc_domain": True}, Ab_info=cbco_info)
        else:
            cbco_index = list(range(0, len(cbco_info)))

        fbmc_rep = grid_model.return_cbco(cbco_info, cbco_index)
        fbmc_rep.set_index(fbmc_rep.cb + "_" + fbmc_rep.co, inplace=True)
 
        if reduce:
            grid_model.julia_instance.join()
            grid_model.julia_instance.julia_instance = None

        return fbmc_rep
