import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


class InputProcessing(object):
    """Data Woker Class"""
    def __init__(self, data):
        self.logger = logging.getLogger('Log.MarketModel.DataManagement.InputData')
        self.data = data

        self.options = data.options
        self.logger.info("Processing Input Data...")
        if self.data.data_attributes["source"] == "mpc_casefile":
            self.data.demand_h = pd.DataFrame(columns=self.data.data_structure["demand_h"].attributes[1:])
            self.data.availability = pd.DataFrame(columns=self.data.data_structure["availability"].attributes[1:])
            self.data.inflows = pd.DataFrame(columns=self.data.data_structure["inflows"].attributes[1:])
            self.data.net_position = pd.DataFrame(columns=self.data.data_structure["net_position"].attributes[1:])

        elif self.data.data_attributes["source"] == "xls":
            self.process_demand()
            self.efficiency()
            self.marginal_costs()
            self.process_net_position()
            self.process_inflows()

        if "availability" in self.options["data"]["process"]:
                self.process_availability()
        if "opsd" in self.options["data"]["process"]:
            self.process_opsd_net_export()
            
        if "net_export" in self.options["data"]["process"]:
            self.process_net_export()

        if self.options["data"]["unique_mc"]:
            self.unique_mc()

        if self.options["data"]["d2cf_data"]:
            self.process_d2cf_data()

        if not self.data.data_attributes["data"]["dclines"]:
            self.data.dclines = pd.DataFrame(columns=['node_i', 'node_j', 'name_i', 'name_j', 'maxflow'])

        if self.options["data"]["all_lines_cb"]:
            self.data.lines.cb = True

        self._check_data()

    def process_demand(self):
        """ Process Demand data"""
        tmp = self.data.demand_el.pivot(index="timestep", columns="node", values="demand_el").fillna(0)
        for node in self.data.nodes.index[~self.data.nodes.index.isin(tmp.columns)]:
            tmp[node] = 0
        self.data.demand_el = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="demand_el")

        if self.data.demand_h.empty:
            self.data.demand_h = pd.DataFrame(columns=self.data.data_structure["demand_h"].attributes[1:])

    def process_net_position(self):
        if self.data.net_position.empty:
            net_position_columns = self.data.data_structure["net_position"].attributes.values[1:]
            self.data.net_position = pd.DataFrame(columns=net_position_columns)
            self.data.net_position["timestep"] = self.data.demand_el.timestep.unique()

        tmp = self.data.net_position.pivot(index="timestep", columns="zone", values="net_position").fillna(0)
        for zone in self.data.zones.index:
            if not zone in tmp.columns:
                tmp[zone] = 0
        self.data.net_position = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="net_position")

    def process_inflows(self):
        if self.data.inflows.empty:
            inflows_columns = self.data.data_structure["inflows"].attributes.values[1:]
            self.data.inflows = pd.DataFrame(columns=inflows_columns)
            self.data.inflows["timestep"] = self.data.demand_el.timestep.unique()

        tmp = self.data.inflows.pivot(index="timestep", columns="plant", values="inflow").fillna(0)
        condition = self.data.plants.plant_type.isin(self.options["optimization"]["plant_types"]["es"])
        for es_plant in self.data.plants.index[condition]:
            if not es_plant in tmp.columns:
                tmp[es_plant] = 0
        self.data.inflows = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="inflow")
        

    def process_net_export(self):

        tmp = self.data.net_export.pivot(index="timestep", columns="node", values="net_export").fillna(0)
        for node in self.data.nodes.index[~self.data.nodes.index.isin(tmp.columns)]:
            tmp[node] = 0
        self.data.net_export = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="net_export")

    def process_opsd_net_export(self):
        """
        Process net Export so that it can be used on the nodal energy balance in the market model as a parameter!
        net_export - Zonal (Export-Import) from nodes not within the model's market clearing area as Timeseries devided on the connected nodes in that zone
        raw data is zonal in a from->to format, where the from nodes are not within the model's market clearing area
        if no data is available all exports are 0
        """
        if self.data.net_export.empty:
            self.data.net_export = pd.DataFrame(index=self.data.demand_el.index,
                                                columns=self.data.nodes.index,
                                                data=0)
        else:
            net_export_raw = self.data.net_export.copy()
            self.data.net_export = pd.DataFrame(index=self.data.demand_el.timestep.unique())
            for zones in set([(from_zone,to_zone) for (from_zone,to_zone) in zip(net_export_raw.from_zone, net_export_raw.to_zone)]):
                nodes = []
                nodes_from_zones = self.data.nodes.index[self.data.nodes.zone == zones[0]]
                nodes_to_zones = self.data.nodes.index[self.data.nodes.zone == zones[1]]

                nodes.extend(list(self.data.lines.node_i[(self.data.lines.node_i.isin(nodes_from_zones))& \
                                                         (self.data.lines.node_j.isin(nodes_to_zones))]))

                nodes.extend(list(self.data.lines.node_j[(self.data.lines.node_i.isin(nodes_to_zones))& \
                                                         (self.data.lines.node_j.isin(nodes_from_zones))]))

                nodes.extend(list(self.data.dclines.node_i[(self.data.dclines.node_i.isin(nodes_from_zones))& \
                                                           (self.data.dclines.node_j.isin(nodes_to_zones))]))

                nodes.extend(list(self.data.dclines.node_j[(self.data.dclines.node_i.isin(nodes_to_zones))& \
                                                           (self.data.dclines.node_j.isin(nodes_from_zones))]))
                nodes = list(set(nodes))
                for node in nodes:
                    tmp = net_export_raw[(net_export_raw["from_zone"] == zones[0]) & \
                                         (net_export_raw["to_zone"] == zones[1])].copy()

                    tmp.loc[:, "export"] = tmp.export/len(nodes)
                    tmp = tmp[["timestep", "export"]].rename(columns={"export": node})
                    if not node in self.data.net_export.columns:
                        self.data.net_export = pd.merge(self.data.net_export, tmp, how="left",
                                                        left_index=True, right_on="timestep").set_index("timestep")
                    else:
                        self.logger.warning("node %s with multiple net export timeseries", node)
            ## Fill NaNs with
            if any(self.data.net_export.isna().any(axis=0)):
                self.logger.warning("Net export contains NaNs, set NaNs to 0")
                self.data.net_export.fillna(0, inplace=True)

            for n in list(set(self.data.nodes.index) - set(self.data.net_export.columns)):
                self.data.net_export[n] = 0
        
        self.data.net_export = self.data.net_export.stack().reset_index()
        self.data.net_export.columns = self.data.data_structure["net_export"].attributes[1:]

    def process_availability(self):
        """Process availability so there is a timeseries per plant"""
        self.data.availability = pd.DataFrame(index=self.data.demand_el.timestep.unique())
        """Calculate the availability for generation that relies on timeseries"""
        ts_type = self.options["optimization"]["plant_types"]["ts"]
        for elm in self.data.plants.index[self.data.plants.plant_type.isin(ts_type)]:
            ts_zone = self.data.timeseries.zone == self.data.nodes.zone[self.data.plants.node[elm]]
            self.data.availability[elm] = self.data.timeseries[self.data.plants.tech[elm]][ts_zone].values
        
        self.data.availability = self.data.availability.stack().reset_index()
        self.data.availability.columns = self.data.data_structure["availability"].attributes[1:]

    def efficiency(self):
        """Calculate the efficiency for plants that dont have it maunually set"""
        tmp = self.data.plants[['tech', 'fuel', 'plant_type']][self.data.plants.eta.isnull()]
        tmp = pd.merge(tmp, self.data.tech[['tech', 'fuel', 'eta']],
                       how='left', on=['tech', 'fuel']).set_index(tmp.index)
        self.data.plants.loc[self.data.plants.eta.isnull(), "eta"] = tmp.eta
        ## If there are still data.plants without efficiency
        self.data.plants.loc[self.data.plants.eta.isnull(), "eta"] = self.options["data"]["default_efficiency"]

    def marginal_costs(self):
        """Calculate the marginal costs for plants that don't have it manually set"""
        co2_price = self.options["data"]["co2_price"]
        tmp_costs = pd.merge(self.data.fuel,
                              self.data.tech[['fuel', 'tech', "plant_type", 'variable_om']],
                              how='left', left_index=True, right_on=['fuel'])

        if not "mc_el" in self.data.plants.columns:
            self.data.plants["mc_el"] = np.nan        
        if not "mc_heat" in self.data.plants.columns:
            self.data.plants["mc_heat"] = np.nan

        condition_mc_el = self.data.plants.mc_el.isnull()
        condition_mc_heat = self.data.plants.mc_heat.isnull()
        ## Plants with el or chp
        condition_el = (self.data.plants.g_max > 0)&(self.data.plants.h_max >= 0)

        tmp_plants = self.data.plants[['mc_el', "mc_heat", 'fuel', 'tech', 'eta', "plant_type"]][condition_mc_el&condition_el]
        tmp_plants = pd.merge(tmp_plants, tmp_costs, how='left', on=['tech', 'fuel', "plant_type"])
        tmp_plants.mc_el = tmp_plants.fuel_price / tmp_plants.eta + tmp_plants.variable_om + tmp_plants.co2_content * co2_price

        self.data.plants.loc[condition_mc_el&condition_el, "mc_el"] = tmp_plants.mc_el.values
        self.data.plants.loc[condition_mc_heat&condition_el, "mc_heat"] = 0

        ## Plants with he only
        condition_he = (self.data.plants.g_max == 0)&(self.data.plants.h_max > 0)
        tmp_plants = self.data.plants[['mc_el', "mc_heat", 'fuel', 'tech', 'eta']][condition_mc_heat&condition_he]
        tmp_plants = pd.merge(tmp_plants, tmp_costs, how='left', on=['tech', 'fuel'])
        tmp_plants.mc_heat = tmp_plants.fuel_price / tmp_plants.eta + tmp_plants.variable_om + tmp_plants.co2_content * co2_price
        self.data.plants.loc[condition_mc_heat&condition_he, "mc_el"] = 0
        self.data.plants.loc[condition_mc_heat&condition_he, "mc_heat"] = tmp_plants.mc_heat.values

        if len(self.data.plants.mc_el[self.data.plants.mc_el.isnull()]) > 0:
            default_value = self.options["data"]["default_mc"]
            self.data.logger.info(f"Number of Plants without marginal costs for electricity: \
                                  {len(self.data.plants.mc[self.data.plants.mc_el.isnull()])} \
                                  -> set to: {default_value}")
            self.data.plants.loc[self.data.plants.mc_el.isnull(), "mc_el"] = default_value


        if len(self.data.plants.mc_heat[self.data.plants.mc_heat.isnull()]) > 0:
            default_value = self.options["data"]["default_mc"]
            self.data.logger.info(f"Number of Plants without marginal costs for heat: \
                                  {len(self.data.plants.mc[self.data.plants.mc_heat.isnull()])} \
                                  -> set to: 0")
            self.data.plants.loc[self.data.plants.mc_heat.isnull(), "mc_heat"] = 0


    def unique_mc(self):
        """make mc's unique by adding a small amount - this helps the solver"""
        for marginal_cost in self.data.plants.mc_el:
            self.data.plants.loc[self.data.plants.mc_el == marginal_cost, "mc"] = \
            self.data.plants.mc_el[self.data.plants.mc_el == marginal_cost] + \
            [int(x)*1E-4 for x in range(0, len(self.data.plants.mc_el[self.data.plants.mc_el == marginal_cost]))]
    
    def process_d2cf_data(self):
        """Process D2CF Data:
        - Fill NAs
        - Define Critical Branches (CB) and CNECs
        - Convert Export Data to Nodal and Zonal Export Values
        - Deduct Export from demand on Non-Cwe Zones
        """
        self.logger.info("Processing D2CF data....")
        self.data.reference_flows.fillna(0, inplace=True)

        ### Check Reference Flows
        for line in self.data.reference_flows.columns:
           if any(np.subtract(self.data.lines.maxflow[line], np.abs(self.data.reference_flows[line])).values < 0):
               self.logger.warning(f"Reference Flow on line {line} larger than maxflow")

        #custom rules
        self.data.reference_flows.l969 = -self.data.reference_flows.l969
        self.data.reference_flows.l1693 = -self.data.reference_flows.l1693
        self.data.reference_flows.l1685 = -self.data.reference_flows.l1685
        self.data.reference_flows.l402 = -self.data.reference_flows.l402

        self.data.frm_fav.fillna(0, inplace=True)
        self.data.lines["cnec"] = False
        self.data.lines["cb"] = False
        condition = self.data.lines.index.isin(self.data.reference_flows.columns)
        self.data.lines.loc[condition, "cnec"]= True
        self.data.lines.loc[condition, "cb"]= True

        condition_cross_border = self.data.nodes.zone[self.data.lines.node_i].values != \
                                    self.data.nodes.zone[self.data.lines.node_j].values

        cwe = ["DE", "FR", "BE", "NL", "LU"]
        condition_cwe = self.data.nodes.zone[self.data.lines.node_i].isin(cwe).values & \
                        self.data.nodes.zone[self.data.lines.node_j].isin(cwe).values

        # self.data.lines["cb"][condition_cross_border] = True
        self.data.lines.loc[condition_cross_border & condition_cwe, "cb"] = True

        self.data.logger.info(f"Number of CNECs in Dataset: {len(self.data.lines[self.data.lines.cnec])}")
        self.data.logger.info(f"Number of CBs in Dataset: {len(self.data.lines[self.data.lines.cb])}")

        # net position 0 for all zones not in the data set
        for zone in self.data.zones.index.difference(self.data.net_position.columns):#
            self.data.net_position[zone] = 0


    def _clean_names(self):
        """
        Julia does not play well with "-" in plant names
        GAMS Does not like special characters
        use this function to find and replace corresponding chars
        """
        self.logger.info("Cleaning Names...")
        # replace the follwing chars
        char_dict = {"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
                     "å": "a", "Å": "A", "-": "_", "/": "_"}

        # Replace in index of the DataFrame
        to_check = ['plants', 'nodes', 'lines', 'heatareas']
        for attr in to_check:
            if attr in list(self.data.__dict__.keys()):
                for i in char_dict:
                    print(attr)
                    self.data.__dict__[attr].index = self.__dict__[attr].index.str.replace(i, char_dict[i])
        # replace in the dataframe
        try:
            self.data.plants.heatarea.replace(char_dict,
                                              regex=True,
                                              inplace=True)
            # self.nodes.replace(char_dict, regex=True, inplace=True)
            # self.lines.replace(char_dict, regex=True, inplace=True)
        except:
            pass

    def line_susceptance(self):
        """Calculate the efficiency for plants that dont have it maunually set"""
        tmp = self.lines[['length', 'type', 'b']][self.lines.b.isnull()]
        tmp.b = self.lines.length/(self.lines.type)
        self.lines.b[self.lines.b.isnull()] = tmp.b

    def _check_data(self):
        """ checks if dataset contaisn NaNs"""
        self.logger.info("Checking Data...")

        data_nan = {}
        for i, df_name in enumerate(self.data.data_attributes["data"]):
            tmp_df = getattr(self.data, df_name)
            for col in tmp_df.columns:
                if not tmp_df[col][tmp_df[col].isnull()].empty:
                    data_nan[i] = {"df_name": df_name, "col": col}
                    self.logger.warning("DataFrame " + df_name +
                                     " contains NaN in column " + col)
        return data_nan