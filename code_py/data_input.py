import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


class InputProcessing(object):
    """Data Woker Class"""
    def __init__(self, data, options):
        self.logger = logging.getLogger('Log.MarketModel.DataManagement.InputData')
        self.data = data

        self.options = options
        self.logger.info("Processing Input Data...")
        if self.data.data_attributes["source"] == "mpc_casefile":
            pass

        elif self.data.data_attributes["source"] == "xls":
            self.process_demand()
            # self.efficiency()
        #     self.marginal_costs()
        # self.process_availability()
        # self.process_net_export()
        self.process_net_position()
        self.process_inflows()

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
        if self.data.demand_h.empty:
            self.data.demand_h = pd.DataFrame(index=self.data.demand_el.index)

        for node in self.data.nodes.index[~self.data.nodes.index.isin(self.data.demand_el.columns)]:
            self.logger.warning("Node %s not in demand_el, including with demand of 0", node)
            self.data.demand_el[node] = 0

    def process_net_position(self):
        for zone in self.data.zones.index.difference(self.data.net_position.columns):#
            self.data.net_position[zone] = 0

    def process_inflows(self):
        condition = self.data.plants.plant_type.isin(self.options["optimization"]["plant_types"]["es"])
        for es_plant in self.data.plants.index[condition]:
            if not es_plant in self.data.inflows:
                self.data.inflows[es_plant] = 0

    def process_net_export(self):
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
            self.data.net_export = pd.DataFrame(index=self.data.demand_el.index)
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

    def process_availability(self):
        """Process availability so there is a timeseries per plant"""
        self.data.availability = pd.DataFrame(index=self.data.demand_el.index)
        """Calculate the availability for generation that relies on timeseries"""
        ts_tech = self.options["optimization"]["plant_types"]["ts"]
        for elm in self.data.plants.index[self.data.plants.tech.isin(ts_tech)]:
            ts_zone = self.data.timeseries.zone == self.data.nodes.zone[self.data.plants.node[elm]]
            self.data.availability[elm] = self.data.timeseries[self.data.plants.tech[elm]][ts_zone].values

    def efficiency(self):
        """Calculate the efficiency for plants that dont have it maunually set"""
        tmp = self.data.plants[['tech', 'fuel']][self.data.plants.eta.isnull()]
        tmp = pd.merge(tmp, self.data.tech[['tech', 'fuel', 'eta']],
                       how='left', on=['tech', 'fuel']).set_index(tmp.index)
        self.data.plants.loc[self.data.plants.eta.isnull(), "eta"] = tmp.eta
        ## If there are still data.plants without efficiency
        self.data.plants.loc[self.data.plants.eta.isnull(), "eta"] = self.options["data"]["default_efficiency"]

    def marginal_costs(self):
        """Calculate the marginal costs for plants that don't have it manually set"""
        co2_price = self.options["data"]["co2_price"]
        self.data.plants["mc"] = np.nan
        tmp_plants = self.data.plants[['mc', 'fuel', 'tech', 'eta']][self.data.plants.mc.isnull()]
        tmp_costs = pd.merge(self.data.fuel,
                             self.data.tech[['fuel', 'tech', 'variable_om']],
                             how='left', left_on="index", right_on=['fuel'])

        tmp_plants = pd.merge(tmp_plants, tmp_costs, how='left', on=['tech', 'fuel'])
        tmp_plants.mc = tmp_plants.fuel_price / tmp_plants.eta + tmp_plants.variable_om + tmp_plants.co2 * co2_price

        self.data.plants.loc[self.data.plants.mc.isnull(), "mc"] = tmp_plants.mc.values

        if len(self.data.plants.mc[self.data.plants.mc.isnull()]) > 0:
            default_value = self.options["data"]["default_mc"]
            self.data.logger.info(f"Number of Plants without marginal costs: \
                                  {len(self.data.plants.mc[self.data.plants.mc.isnull()])} \
                                  -> set to: {default_value}")
            self.data.plants.loc[self.data.plants.mc.isnull(), "mc"] = default_value

    def unique_mc(self):
        """make mc's unique by adding a small amount - this helps the solver"""
        for marginal_cost in self.data.plants.mc:
            self.data.plants.loc[self.data.plants.mc == marginal_cost, "mc"] = \
            self.data.plants.mc[self.data.plants.mc == marginal_cost] + \
            [int(x)*1E-4 for x in range(0, len(self.data.plants.mc[self.data.plants.mc == marginal_cost]))]

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