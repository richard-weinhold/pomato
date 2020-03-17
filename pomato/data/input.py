"""Input Processing of POMATO, which processes the raw ipunt data into something usefule.

This module could definately be a set of functioons or be completely seperate from the POMATO
package itself. However this would require perfectly formatted input data which does not exists.
So here we are.
"""
# import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import logging

class InputProcessing(object):
    """Process input data to bring the raw data into the right format.

    This module provides a fairly flexible way to modify the raw inout data. While it is desirable
    to have the raw input data in the right format in the first place, it is convenient to have
    methods available to adjust, check and alter data when initializing POMATO.

    Therefore, this module should be modified by the user himself and adjusted to the specific
    needs. In this form this module represents a set-up in which three different data sets can be
    easily used: the IEEE case studies, a data set for CWE based in OPSD and DIW data and another
    non-open dataset which is of very similar structure.

    The way it works now (and this is really not that practical) is that the user specifies in the
    option file which data should be processed, i.e. has an appropriate method available in this
    module.
    Additionally there are general methods that check the consitency/usability of data.

    Parameters
    ----------
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class, including all (raw) input data.

    Attributes
    ----------
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class, including all (raw) input data.
    options : dict
        The options from DataManagement persist in the InputProcessing.
    """

    def __init__(self, data):
        self.logger = logging.getLogger('Log.MarketModel.DataManagement.InputData')
        self.data = data

        self.options = data.options
        self.logger.info("Processing Input Data...")

        if self.options["data"]["data_type"] in ["opsd", "ramses"]:
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

        if not self.data.data_attributes["dclines"]:
            self.data.dclines = pd.DataFrame(columns=['node_i', 'node_j', 'name_i', 'name_j', 'maxflow'])

        self._check_data()

    def process_demand(self):
        """Process demand data.

        This method validatates that each node has a demand. If a node is not included in the raw
        data, a demand of zero is added for each timestep.

        If heat demand is not included an empty table is creted.
        """
        tmp = self.data.demand_el.pivot(index="timestep", columns="node",
                                        values="demand_el").fillna(0)
        for node in self.data.nodes.index[~self.data.nodes.index.isin(tmp.columns)]:
            tmp[node] = 0
        self.data.demand_el = pd.melt(tmp.reset_index(), id_vars=["timestep"],
                                      value_name="demand_el")

        if self.data.demand_h.empty:
            self.data.demand_h = pd.DataFrame(columns=self.data.model_structure["demand_h"].attributes[1:])

    def process_net_position(self):
        """Process net position data.

        For some FBMC calculations it is useful to calibrate/constrain the net position to historic
        values. Therefore, this method validates existing data or provides a table with all-zero for
        each zone and timestep.
        """
        if self.data.net_position.empty:
            net_position_columns = self.data.data_structure["net_position"].attributes.values[1:]
            self.data.net_position = pd.DataFrame(columns=net_position_columns)
            self.data.net_position["timestep"] = self.data.demand_el.timestep.unique()

        tmp = self.data.net_position.pivot(index="timestep", columns="zone",
                                           values="net_position").fillna(0)
        for zone in self.data.zones.index:
            if not zone in tmp.columns:
                tmp[zone] = 0
        self.data.net_position = pd.melt(tmp.reset_index(), id_vars=["timestep"],
                                         value_name="net_position")

    def process_inflows(self):
        """Process inflows to (hydro-) storages.

        If no raw data create an all zero timeseries for all electric storage (plant_type es)
        power plants
        """
        if self.data.inflows.empty:
            inflows_columns = self.data.model_structure["inflows"].attributes.values[1:]
            self.data.inflows = pd.DataFrame(columns=inflows_columns)
            self.data.inflows["timestep"] = self.data.demand_el.timestep.unique()

        tmp = self.data.inflows.pivot(index="timestep", columns="plant", values="inflow").fillna(0)
        condition = self.data.plants.plant_type.isin(self.options["optimization"]["plant_types"]["es"])
        for es_plant in self.data.plants.index[condition]:
            if es_plant not in tmp.columns:
                tmp[es_plant] = 0
        self.data.inflows = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="inflow").dropna()

        
    def process_net_export(self):
        """Process net export timeseries.

        The net export parameter is used on nodes on, or across the border between modeled regions,
        i.e. zones with power plant stack, and those without.

        Thios method makes sure each node has such a timeseries, so that the implementation in
        the market model iself is trivial.

        """
        tmp = self.data.net_export.pivot(index="timestep", columns="node",
                                         values="net_export").fillna(0)
        for node in self.data.nodes.index[~self.data.nodes.index.isin(tmp.columns)]:
            tmp[node] = 0
        self.data.net_export = pd.melt(tmp.reset_index(), id_vars=["timestep"],
                                       value_name="net_export")

    def process_opsd_net_export(self):
        """Process net export timeseries from the OPSD dataset.

        Equivalent to :meth:`~process_net_export`, however since OPSD data includes zone-to-zone
        physical exchange, this parameter is distributed accordingly to all border connections.
        E.g. for DE-PL, OPSD data contains all DE nodes and lines that connect DE-PL, including
        the nodes in PL. Therefore a potential net export DE->PL will accure as a load
        (or negative injection) on both PL nodes weighted equally.

        """
        if self.data.net_export.empty:
            self.data.net_export = pd.DataFrame(index=self.data.demand_el.timestep.unique(),
                                                columns=self.data.nodes.index,
                                                data=0)
        else:
            
            net_export_raw = self.data.net_export.copy()
            self.data.net_export_raw = self.data.net_export.copy()

            self.data.net_export = pd.DataFrame(index=self.data.demand_el.timestep.unique())
            for zones in set([(from_zone,to_zone) \
                              for (from_zone,to_zone) in zip(net_export_raw.from_zone,
                                                             net_export_raw.to_zone)]):
                nodes = []
                nodes_from_zones = self.data.nodes.index[self.data.nodes.zone == zones[0]]
                nodes_to_zones = self.data.nodes.index[self.data.nodes.zone == zones[1]]

                lines = self.data.lines
                dclines = self.data.dclines
                nodes.extend(list(lines.node_i[(lines.node_i.isin(nodes_from_zones))& \
                                               (lines.node_j.isin(nodes_to_zones))]))

                nodes.extend(list(lines.node_j[(lines.node_i.isin(nodes_to_zones))& \
                                               (lines.node_j.isin(nodes_from_zones))]))

                nodes.extend(list(dclines.node_i[(dclines.node_i.isin(nodes_from_zones))& \
                                                 (dclines.node_j.isin(nodes_to_zones))]))

                nodes.extend(list(dclines.node_j[(dclines.node_i.isin(nodes_to_zones))& \
                                                 (dclines.node_j.isin(nodes_from_zones))]))
                nodes = list(set(nodes))
                for node in nodes:
                    tmp = net_export_raw[(net_export_raw["from_zone"] == zones[0]) & \
                                         (net_export_raw["to_zone"] == zones[1])].copy()

                    tmp.loc[:, "export"] = tmp.export/len(nodes)
                    tmp = tmp[["timestep", "export"]].rename(columns={"export": node})
                    if node not in self.data.net_export.columns:
                        self.data.net_export = pd.merge(self.data.net_export,
                                                        tmp, how="left", left_index=True,
                                                        right_on="timestep").set_index("timestep")
                    else:
                        self.logger.warning("node %s with multiple net export timeseries", node)
            # Fill NaNs with
            if any(self.data.net_export.isna().any(axis=0)):
                self.logger.warning("Net export contains NaNs, set NaNs to 0")
                self.data.net_export.fillna(0, inplace=True)

            for n in list(set(self.data.nodes.index) - set(self.data.net_export.columns)):
                self.data.net_export[n] = 0

        self.data.net_export = self.data.net_export.stack().reset_index()
        self.data.net_export.columns = self.data.data_structure["net_export"].attributes[1:]

    def process_availability(self):
        """Process availability so there is a timeseries of capacity factors per plant.

        This is an example for data set specific methods. In this case opsd availability data
        is plant_type (or technology) specific. And the availability get assinged accordingly.
        For other datasets this might be already included plant specific, or with additional
        regional indicators etc....
        """
        self.data.availability = pd.DataFrame(index=self.data.demand_el.timestep.unique())
        """Calculate the availability for generation that relies on timeseries"""
        ts_type = self.options["optimization"]["plant_types"]["ts"]
        plants = self.data.plants
        for elm in plants.index[plants.plant_type.isin(ts_type)]:
            ts_zone = self.data.timeseries.zone == self.data.nodes.zone[plants.node[elm]]
            self.data.availability[elm] = self.data.timeseries[plants.tech[elm]][ts_zone].values

        self.data.availability = self.data.availability.stack().reset_index()
        self.data.availability.columns = self.data.model_structure["availability"].attributes[1:]

    def efficiency(self):
        """Calculate the efficiency for plants that dont have it maunually set.

        Calculates/assigns the efficiency of a power plant based on information in tech/fuel/plant
        tables.

        If none is available eta defaults to the default value set in the option file.
        """
        tmp = self.data.plants[['tech', 'fuel', 'plant_type']][self.data.plants.eta.isnull()]
        tmp = pd.merge(tmp, self.data.tech[['tech', 'fuel', 'eta']],
                       how='left', on=['tech', 'fuel']).set_index(tmp.index)
        self.data.plants.loc[self.data.plants.eta.isnull(), "eta"] = tmp.eta
        ## If there are still data.plants without efficiency
        default_value = self.options["data"]["default_efficiency"]
        self.data.plants.loc[self.data.plants.eta.isnull(), "eta"] = default_value

    def marginal_costs(self):
        """Calculate the marginal costs for plants that don't have it manually set.

        Marginal costs are calculated by: mc = fuel price / eta + o&m + CO2 costs

        If heat plants are included, costs are allocatet to the mc_heat if the plant is heat only
        and to mc_el for conventional and chp plants.

        """
        co2_price = self.options["data"]["co2_price"]
        tmp_costs = pd.merge(self.data.fuel,
                             self.data.tech[['fuel', 'tech', "plant_type", 'variable_om']],
                             how='left', left_index=True, right_on=['fuel'])

        if "mc_el" not in self.data.plants.columns:
            self.data.plants["mc_el"] = np.nan
        if "mc_heat" not in self.data.plants.columns:
            self.data.plants["mc_heat"] = np.nan

        condition_mc_el = self.data.plants.mc_el.isnull()
        condition_mc_heat = self.data.plants.mc_heat.isnull()
        # Plants with el or chp
        condition_el = (self.data.plants.g_max > 0) & (self.data.plants.h_max >= 0)

        columns = ['mc_el', "mc_heat", 'fuel', 'tech', 'eta', "plant_type"]
        tmp_plants = self.data.plants[columns][condition_mc_el & condition_el]
        tmp_plants = pd.merge(tmp_plants, tmp_costs, how='left', on=['tech', 'fuel', "plant_type"])
        tmp_plants.mc_el = tmp_plants.fuel_price / tmp_plants.eta + tmp_plants.variable_om + \
                           tmp_plants.co2_content * co2_price

        self.data.plants.loc[condition_mc_el & condition_el, "mc_el"] = tmp_plants.mc_el.values
        self.data.plants.loc[condition_mc_heat & condition_el, "mc_heat"] = 0

        ## Plants with he only
        condition_he = (self.data.plants.g_max == 0) & (self.data.plants.h_max > 0)
        columns = ['mc_el', "mc_heat", 'fuel', 'tech', 'eta']
        tmp_plants = self.data.plants[columns][condition_mc_heat & condition_he]
        tmp_plants = pd.merge(tmp_plants, tmp_costs, how='left', on=['tech', 'fuel'])
        tmp_plants.mc_heat = tmp_plants.fuel_price / tmp_plants.eta + tmp_plants.variable_om + \
                             tmp_plants.co2_content * co2_price
        self.data.plants.loc[condition_mc_heat & condition_he, "mc_el"] = 0
        self.data.plants.loc[condition_mc_heat & condition_he, "mc_heat"] = tmp_plants.mc_heat.values

        if len(self.data.plants.mc_el[self.data.plants.mc_el.isnull()]) > 0:
            default_value = self.options["data"]["default_mc"]
            self.data.logger.info(f"Number of Plants without marginal costs for electricity: \
                                  {len(self.data.plants.mc_el[self.data.plants.mc_el.isnull()])} \
                                  -> set to: {default_value}")
            self.data.plants.loc[self.data.plants.mc_el.isnull(), "mc_el"] = default_value

        if len(self.data.plants.mc_heat[self.data.plants.mc_heat.isnull()]) > 0:
            default_value = self.options["data"]["default_mc"]
            number_nan = len(self.data.plants.mc_heat[self.data.plants.mc_heat.isnull()])
            self.data.logger.info(f"Number of Plants without marginal costs for heat: \
                                  {number_nan} -> set to: 0")
            self.data.plants.loc[self.data.plants.mc_heat.isnull(), "mc_heat"] = 0

    def unique_mc(self):
        """Make marginal costs unique.

        This is done by adding a small increment multiplied by the number if plants with the
        same mc. This makes the solver find a unique solition (at leat in regards to generation
        scheduel) and is sopposed to have positive effect on solvetime.
        """
        for marginal_cost in self.data.plants.mc_el:
            condition_mc = self.data.plants.mc_el == marginal_cost
            self.data.plants.loc[condition_mc, "mc"] = \
            self.data.plants.mc_el[condition_mc] + \
            [int(x)*1E-4 for x in range(0, len(self.data.plants.mc_el[condition_mc]))]

    def line_susceptance(self):
        """Calculate line susceptance for lines that have none set.

        This is not maintained as the current grid data set includes this parameter. However, this
        Was done with the simple formula b = length/type ~ where type is voltage level. While this
        is technically wrong, it works with linear load flow, as it only relies on the
        conceptual "conductance"/"resistance" of each circuit/line in relation to others.
        """
        tmp = self.lines[['length', 'type', 'b']][self.lines.b.isnull()]
        tmp.b = self.lines.length/(self.lines.type)
        self.lines.b[self.lines.b.isnull()] = tmp.b

    def _clean_names(self):
        """Clean raw data of special character or names that didnt encode well.

        This is becaise Julia/GAMS does not play well with "-" in plant names or special characters.
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
                    self.logger.debug("Replaced characters %s in attribute %s", i, attr)
                    self.data.__dict__[attr].index = self.__dict__[attr] \
                                                         .index.str.replace(i, char_dict[i])
        # replace in the dataframe
        try:
            self.data.plants.heatarea.replace(char_dict,
                                              regex=True,
                                              inplace=True)
            # self.nodes.replace(char_dict, regex=True, inplace=True)
            # self.lines.replace(char_dict, regex=True, inplace=True)
        except:
            pass

    def _check_data(self):
        """Check if dataset contains NaNs."""
        self.logger.info("Checking Data...")

        data_nan = {}
        for i, df_name in enumerate(self.data.data_attributes):
            tmp_df = getattr(self.data, df_name)
            for col in tmp_df.columns:
                if not tmp_df[col][tmp_df[col].isnull()].empty:
                    data_nan[i] = {"df_name": df_name, "col": col}
                    self.logger.warning("DataFrame %s contains NaN in column %s", df_name, col)
        return data_nan
