import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


class InputProcessing(object):
    """Data Woker Class"""
    def __init__(self, data, set_up=None):
        self.logger = logging.getLogger('Log.MarketModel.DataManagement.InputData')
        self.data = data

        if not set_up:
            self.logger.info("Using default data processing set-up")

            self.set_up = {"unique_mc": True,
                           "round_demand": True,
                           "default_efficiency": 0.5,
                           "default_mc": 200,
                           "co2_price": 20,
                           "all_lines_cb": False,
                           }
        else:
            self.set_up = set_up

        self.logger.info("Processing Input Data...")

        ### Custm Rule :(
        if self.data.data_attributes["source"] == "mpc_casefile":
            pass

        elif self.data.data_attributes["source"] == "xls":

            self.data.zones.set_index("country", inplace=True)

            self.efficiency()
            self.marginal_costs()

        self.process_availability()
        self.process_net_export()

        if self.set_up["unique_mc"]:
            self.unique_mc()

        if all([self.data.data_attributes["data"][d2cf_data] \
                for d2cf_data in ["net_export", "reference_flows", "frm_fav", "net_position"]]):
            self.process_d2cf_data()
        
        if not self.data.data_attributes["data"]["dclines"]:
            self.data.dclines = pd.DataFrame(columns=['node_i', 'node_j', 'name_i', 'name_j', 'maxflow'])
        
        if self.set_up["all_lines_cb"]:
            self.data.lines.cb = True

        self._check_data()


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
                nodes.extend(list(self.data.lines.node_i[(self.data.lines.node_i.isin(self.data.nodes.index[self.data.nodes.zone == zones[0]]))& \
                                                         (self.data.lines.node_j.isin(self.data.nodes.index[self.data.nodes.zone == zones[1]]))]))

                nodes.extend(list(self.data.lines.node_j[(self.data.lines.node_i.isin(self.data.nodes.index[self.data.nodes.zone == zones[1]]))& \
                                                         (self.data.lines.node_j.isin(self.data.nodes.index[self.data.nodes.zone == zones[0]]))]))

                nodes.extend(list(self.data.dclines.node_i[(self.data.dclines.node_i.isin(self.data.nodes.index[self.data.nodes.zone == zones[0]]))& \
                                                           (self.data.dclines.node_j.isin(self.data.nodes.index[self.data.nodes.zone == zones[1]]))]))

                nodes.extend(list(self.data.dclines.node_j[(self.data.dclines.node_i.isin(self.data.nodes.index[self.data.nodes.zone == zones[1]]))& \
                                                           (self.data.dclines.node_j.isin(self.data.nodes.index[self.data.nodes.zone == zones[0]]))]))
                nodes = list(set(nodes))

                for n in nodes:
                    self.data.net_export[n] = net_export_raw.export[(net_export_raw["from_zone"] == zones[0]) & \
                                                                    (net_export_raw["to_zone"] == zones[1])].values/len(nodes)

            for n in list(set(self.data.nodes.index) - set(self.data.net_export.columns)):
                self.data.net_export[n] = 0

    def process_availability(self):
        """Process availability so there is a timeseries per plant"""

        self.data.availability = pd.DataFrame(index=self.data.demand_el.index)
        """Calculate the availability for generation that relies on timeseries"""
        ts_tech = ["wind onshore", "wind offshore", "solar"]
        for elm in self.data.plants.index[self.data.plants.tech.isin(ts_tech)]:
            ts_zone = self.data.timeseries.zone == self.data.nodes.zone[self.data.plants.node[elm]]
            self.data.availability[elm] = self.data.timeseries[self.data.plants.tech[elm]][ts_zone].values

    def efficiency(self):
        """Calculate the efficiency for plants that dont have it maunually set"""
        tmp = self.data.plants[['tech', 'fuel']][self.data.plants.eta.isnull()]
        tmp = pd.merge(tmp, self.data.tech[['tech', 'fuel', 'eta']],
                       how='left', on=['tech', 'fuel']).set_index(tmp.index)
        self.data.plants.eta[self.data.plants.eta.isnull()] = tmp.eta
        ## If there are still data.plants without efficiency
        self.data.plants.eta[self.data.plants.eta.isnull()] = self.set_up["default_efficiency"]

    def marginal_costs(self):
        """Calculate the marginal costs for plants that don't have it manually set"""
        co2_price = self.set_up["co2_price"]
        self.data.plants["mc"] = np.nan
        tmp_plants = self.data.plants[['mc', 'fuel', 'tech', 'eta']][self.data.plants.mc.isnull()]
        tmp_costs = pd.merge(self.data.fuel[["fuel", "fuel_price", "co2"]],
                             self.data.tech[['fuel', 'tech', 'variable_om']],
                             how='left', on=['fuel'])

        tmp_plants = pd.merge(tmp_plants, tmp_costs, how='left', on=['tech', 'fuel'])
        tmp_plants.mc = tmp_plants.fuel_price / tmp_plants.eta + tmp_plants.variable_om + tmp_plants.co2 * co2_price

        self.data.plants.mc[self.data.plants.mc.isnull()] = tmp_plants.mc.values

        if len(self.data.plants.mc[self.data.plants.mc.isnull()]) > 0:
            self.data.logger.info(f"Number of Plants without marginal costs: {len(self.data.plants.mc[self.data.plants.mc.isnull()])} -> set to 200")
            self.data.plants.mc[self.data.plants.mc.isnull()] = self.set_up["default_mc"]

    def unique_mc(self):
        """make mc's unique by adding a small amount - this helps the solver"""
        for marginal_cost in self.data.plants.mc:
            self.data.plants.mc[self.data.plants.mc == marginal_cost] = \
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
        # self.data.reference_flows.l1686 = -self.data.reference_flows.l1686
        # self.data.reference_flows.l2333 = -self.data.reference_flows.l2333

        self.data.frm_fav.fillna(0, inplace=True)
        self.data.lines["cnec"] = False
        self.data.lines["cb"] = False
        self.data.lines["cnec"][self.data.lines.index.isin(self.data.reference_flows.columns)] = True
        self.data.lines["cb"][self.data.lines.index.isin(self.data.reference_flows.columns)] = True
        condition = self.data.nodes.zone[self.data.lines.node_i].values != \
                         self.data.nodes.zone[self.data.lines.node_j].values
        self.data.lines["cb"][condition] = True

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
        ## Heatarea contains NaN, but that's alright
        data_nan = {}
        for i, df_name in enumerate(self.data.data_attributes["data"]):
            tmp_df = getattr(self.data, df_name)
            for col in tmp_df.columns:
                if not tmp_df[col][tmp_df[col].isnull()].empty:
                    data_nan[i] = {"df_name": df_name, "col": col}
                    self.logger.warning("DataFrame " + df_name +
                                     " contains NaN in column " + col)
        return data_nan