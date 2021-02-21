"""Results of POMATO."""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import types
import threading
from copy import deepcopy

import pomato.tools as tools
from pomato.visualization.geoplot_functions import line_coordinates
# # pylint: disable-msg=E1101

class Results():
    """Results of POMATO makes market results available to the user.

    The Results module provides an interface between the market result
    and the data itself. All result variables are set as an attribute in a long
    DataFrame with its sets as columns which can be easily accessed
    with *result.VAR*.

    Attributes
    ----------
    data : :class:`~pomato.data.DataManagement`
        An instance of the DataManagement class with the processed input data
        that is the basis of the loaded results.
    grid : :class:`~pomato.grid.GridTopology`
        An instance of the GridTopology class, to provide its functionality to the results.
    result_folder : pathlib.Path
        The data is initialized from a folder containing the result. The Results
        can be initialized after a model run or from stored data.
    result_attributes : dict,
        *result_attributes* covers all variables from the market model,
        along with dual and infeasibility/slack variables and other market
        model specific information which are relevant to the results itself.

    Parameters
    ----------
    data : :class:`~pomato.data.DataManagement`
        An instance of the DataManagement class with the processed input data
        that is the basis of the loaded results.
    grid : :class:`~pomato.grid.GridTopology`,
        An instance of the GridTopology class, to provide its functionality to the
        results.
    result_folder : pathlib.Path
        Folder with the results of the market model in .csv files.
    """

    def __init__(self, data, grid, result_folder):
        self.logger = logging.getLogger('log.pomato.data.DataWorker.Results')
        self.grid = grid
        self.data = data
        self.output_folder = self.data.wdir.joinpath("data_output").joinpath(result_folder.name)

        variables = {variable: False for variable in ["G", "H",
                                                      "D_es", "L_es",
                                                      "D_hs", "L_hs",
                                                      "INJ", "EX",
                                                      "D_ph", "F_DC", "CURT", "Alpha"]}

        dual_variables = {variable: False for variable in ["EB_nodal", "EB_zonal", "EB_heat"]}

        infeasibility_variables = {variable: False
                                   for variable in ["INFEAS_H_POS", "INFEAS_H_NEG",
                                                    "INFEAS_EL_N_POS", "INFEAS_EL_N_NEG"]}

        self.result_attributes = {"variables": variables,
                                  "dual_variables": dual_variables,
                                  "infeasibility_variables": infeasibility_variables,
                                  "model_horizon": None, "source_folder": None, "objective": None,
                                  "is_redispatch_result": False, 
                                  "corresponding_market_result_name": None}

        self.model_horizon = self.result_attributes["model_horizon"]

        self._cached_results = types.SimpleNamespace(n_0_flows=pd.DataFrame(), 
                                                     n_1_flows=pd.DataFrame(),
                                                     generation=pd.DataFrame(),
                                                     demand=pd.DataFrame(),
                                                     result_data=None,
                                                     averaged_result_data=None)

        for var in self.result_attributes["variables"]:
            setattr(self, var, pd.DataFrame())
        for var in self.result_attributes["dual_variables"]:
            setattr(self, var, pd.DataFrame())
        for var in self.result_attributes["infeasibility_variables"]:
            setattr(self, var, pd.DataFrame())

        # Add opt Set-Up to the results attributes
        self.load_results_from_folder(result_folder)
        # Set Redispatch = True if result is a redispatch result 
        if "redispatch" in self.result_attributes["name"]:
            self.result_attributes["title"] += "_redispatch"
            self.result_attributes["is_redispatch_result"] = True
            # Look for different names 
            market_result_names = ["_".join(self.result_attributes["name"].split("_")[:n]) + "_market_results" for n in range(0,4)]
            result_exists = [(name in self.data.results) for name in market_result_names]
            if sum(result_exists) > 0:
                result = [market_result_names[i] for i, exists in enumerate(result_exists) if exists][0]
                self.result_attributes["corresponding_market_result_name"] = result
            else:
                self.logger.warning("Corresponding market result to %s not or with new name instantiated", self.result_attributes["name"])
                self.logger.warning("Manually set market result name in result attributes.")
        # set-up: don't show the graphs when created
        plt.ioff()

    def load_results_from_folder(self, folder):
        """Load results from folder.

        Results are loaded as csv files from results folder with additional
        information inform of the options file that was used to get the results
        as well as the model horizon which is extracted from the INJ variable.

        All variables are set as an attribute of this instance of
        ResultProcessing.

        Parameters
        ----------
        folder : pathlib.Path
            Folder with the results of the market model.
        """
        folder_name = folder.name
        self.logger.info("Loading Results from results folder %s", folder_name)

        for variable_type in ["variables", "dual_variables", "infeasibility_variables"]:
            for var in self.result_attributes[variable_type]:
                try:
                    setattr(self, var, tools.reduce_df_size(pd.read_csv(str(folder.joinpath(f"{var}.csv")))))
                    # setattr(self, var, (pd.read_csv(str(folder.joinpath(f"{var}.csv")))))
                    self.result_attributes[variable_type][var] = True
                except FileNotFoundError:
                    self.logger.warning("%s not in results folder %s", var, folder_name)

        # Set result attributes from result json file or data.option:
        try:
            with open(str(folder.joinpath("misc_results.json")), "r") as jsonfile:
                self.result_attributes["objective"] = json.load(jsonfile)
        except FileNotFoundError:
            self.logger.warning("Results Not Found. This is an Error")
            
        try:
            with open(str(folder.joinpath("optionfile.json")), "r") as jsonfile:
                self.result_attributes = {**self.result_attributes,
                                          **json.load(jsonfile)}
        except FileNotFoundError:
            self.logger.warning("No options file found in result folder, \
                                using data.options")
            self.result_attributes = {**self.result_attributes,
                                      **self.data.options}

        self.result_attributes["source_folder"] = str(folder)
        self.result_attributes["name"] = folder.name
        if not "title" in self.result_attributes:
            self.result_attributes["title"] = self.result_attributes["name"]
        # Model Horizon as attribute
        self.result_attributes["model_horizon"] = list(self.INJ['t'].drop_duplicates().sort_values())
        self.model_horizon = self.result_attributes["model_horizon"]

    def result_data_struct(self):
        """Data struct, as a standart template for result processing.
        
        Returns
        -------
        result_data, types.SimpleNamespace
            Returns empty data struct, with predefined data structure. 
        """        

        return types.SimpleNamespace(
            nodes=self.data.nodes,
            lines=self.data.lines,
            line_coordinates=line_coordinates(self.data.lines.copy(), self.data.nodes.copy()),
            dclines=self.data.dclines,
            dcline_coordinates=line_coordinates(self.data.dclines, self.data.nodes),
            inj=pd.Series(index=self.data.nodes.index, data=0),
            dc_flow= pd.Series(index=self.data.dclines.index, data=0),
            gen=pd.DataFrame(),
            demand=pd.DataFrame(),
            prices=pd.DataFrame(),
            n_0_flow=pd.Series(index=self.data.lines.index, data=0),
            n_1_flow=pd.Series(index=self.data.lines.index, data=0)
            )
    
    def create_result_data(self, force_recalc=False):
        """Creates result data struct from result instance.

        Based on :meth:`~result_data_struct`this method fills the data struct with data and results
        from the market result specified which is an instance of :class:`~pomato.data.Results`.
        This data struct is intended for the generation of visualizations of result in e.g. the
        dynamic geoplot.

        Parameters
        ----------
        market_result : :class:`~pomato.data.Results`
            Market result which gets subsumed into the predefined data struct.
        """
        if not (not isinstance(self._cached_results.result_data, types.SimpleNamespace) or force_recalc):
            self.logger.debug("Returning cached result for result_data.")
            return deepcopy(self._cached_results.result_data)
        
        self.logger.info("Precalculating and caching common results..")
        data_struct = self.result_data_struct()
        data_struct.inj = self.INJ
        data_struct.dc_flow = self.F_DC
        data_struct.gen = self.generation()
        data_struct.demand = self.demand()
        data_struct.n_0_flow = self.n_0_flow()
        data_struct.n_1_flow = self.absolute_max_n_1_flow(sensitivity=0.1)
        data_struct.prices = self.price()
        self._cached_results.result_data = data_struct
        self.logger.info("Done calculating common results.")

        return deepcopy(data_struct)

    def create_averaged_result_data(self, force_recalc=False):
        """Creates averaged result data struct.

        Based on :meth:`~result_data_struct` and  :meth:`~create_result_data` this method fills 
        the data struct with data and results from the market result specified which is an 
        instance of :class:`~pomato.data.Results`. All results are averaged in useful ways. This 
        data struct is intended for the static geoplot, which visualizes the results in 
        average flows, injections, generation and prices. 
        """
        if not (not isinstance(self._cached_results.averaged_result_data, types.SimpleNamespace) or force_recalc):
            self.logger.debug("Returning cached result for averaged_result_data.")
            return deepcopy(self._cached_results.averaged_result_data)

        data_struct = self.create_result_data()

        data_struct.inj = data_struct.inj.groupby("n").mean().reindex(self.grid.nodes.index).INJ
        data_struct.n_0_flow = data_struct.n_0_flow.abs().mean(axis=1)
        data_struct.n_1_flow = data_struct.n_1_flow.abs().mean(axis=1)
        data_struct.dc_flow = data_struct.dc_flow.pivot(index="dc", columns="t", values="F_DC") \
                                .abs().mean(axis=1).reindex(self.data.dclines.index).fillna(0)
        data_struct.prices = data_struct.prices[["n", "marginal"]].groupby("n").mean()
        self._cached_results.averaged_result_data = data_struct
        return data_struct

    def redispatch(self):
        """Return Redispatch.
        Calculates a delta between redispatch,- and market result. 
        Positive delta represents a higher generation after redispatch i.e. positive and negative 
        vice versa. 
        """
        # Find corresponding Market Result
        corresponding_market_result = self.result_attributes["corresponding_market_result_name"]
        if not (self.result_attributes["is_redispatch_result"] and bool(corresponding_market_result)):
            self.logger.warning("Corresponding market result not initialized or found")
            return None

        gen = self.generation()
        # Redispatch Calculation G_redispatch - G_market
        gen = pd.merge(self.data.results[corresponding_market_result].G, gen, on=["p", "t"], 
                       suffixes=("_market", "_redispatch"))
        gen["delta"] = gen["G_redispatch"] - gen["G_market"]
        gen["delta_abs"] = gen["delta"].abs()
        return gen

    def infeasibility(self, drop_zero=True):
        """Return electricity infeasibilities.
        
        Infeasibilities occur when the electricity energy balances cannot be satisfied in the 
        model due to other constraints, like capacity or network constraints. Nodal infeasibilities 
        represent dropped load (positive) or dumped energy (negative).

        Parameters
        ----------
        drop_zero : bool, optional
            If True drop all infeasibility entries with value 0, by default True

        Returns
        -------
        DataFrame
            DataFrame of nodal infeasibilities with columns [t, n, pos, neg].
        """        
        infeasibility = pd.merge(self.data.nodes, self.INFEAS_EL_N_POS, left_index=True, right_on="n")
        infeasibility = pd.merge(infeasibility, self.INFEAS_EL_N_NEG, on=["t", "n"])
        infeasibility = infeasibility.rename(columns={"INFEAS_EL_N_POS": "pos", "INFEAS_EL_N_NEG": "neg"})
        
        if drop_zero:
            return infeasibility[(infeasibility.pos > 0) | (infeasibility.neg > 0)]
        else:
            return infeasibility

    def price(self):
        """Return electricity prices.

        Returns the dual of the energy balances (nodal and zonal). Since
        the model can be cleared with constraints on both simultaneously, the
        resulting nodal price is the sum of the zonal and nodal components.
        The dual is obtained from Julia/JuMP with the dual function and therefore
        multiplied with -1.

        Returns
        -------
        price : DataFrame
            Price DataFrame with columns timestep (t), node (n), zone (z) and
            price (marginal).
        """
        eb_nodal = self.EB_nodal.copy()
        eb_nodal = pd.merge(eb_nodal, self.data.nodes.zone.to_frame(),
                            how="left", left_on="n", right_index=True)
        eb_nodal.loc[abs(eb_nodal.EB_nodal) < 1E-3, "EB_nodal"] = 0

        eb_zonal = self.EB_zonal.copy()
        eb_zonal.loc[abs(eb_zonal.EB_zonal) < 1E-3, "EB_zonal"] = 0

        price = pd.merge(eb_nodal, eb_zonal, how="left",
                         left_on=["t", "zone"], right_on=["t", "z"])

        price["marginal"] = -(price.EB_zonal + price.EB_nodal)
        return price[["t", "n", "z", "marginal"]]

    def net_position(self):
        """Calculate net position for each zone and timestep.

        Returns
        -------
        net_position : DataFrame
            DataFrame with the timesteps as index and zones as columns.
        """
        net_position = pd.DataFrame(index=self.EX.t.unique())
        for zone in self.data.zones.index:
            net_position[zone] = self.EX[self.EX.z == zone].groupby("t").sum() - \
                                 self.EX[self.EX.zz == zone].groupby("t").sum()
        return net_position
    
    def generation(self, force_recalc=False):
        """Generation data.
        
        Returns DataFrame with columns [node, plant_type, g_max, zone, t, p, G]
        """
        if not (self._cached_results.generation.empty or force_recalc):
            self.logger.debug("Returning cached result for generation.")
            return self._cached_results.generation.copy()
        
        gen = pd.merge(self.data.plants[["plant_type", "fuel", "node", "g_max"]],
                        self.G, left_index=True, right_on="p", how="right")
        
        gen["zone"] = self.data.nodes.loc[gen.node, "zone"].values
        if "technology" in self.data.plants.columns:
            gen = pd.merge(gen, self.data.plants[["technology"]], 
                           right_index=True, left_on="p")
        else:
            gen["technology"] = gen.plant_type
        gen = tools.reduce_df_size(gen)
        self._cached_results.generation = gen
        return gen.copy()

    def full_load_hours(self):
        """Returns plant data including full load hours."""        

        gen = self.generation()[["t", "p", "fuel", "technology", "G", "g_max"]].copy()
        ava = self.data.availability.copy()[["timestep", "plant", "availability"]]
        ava.columns = ["t", "p", "availability"]
        
        flh = pd.merge(gen, ava, on=["t", "p"], how="left")
        flh.loc[:, "availability"] = flh.loc[:, "availability"].fillna(1)
        flh["utilization"] = flh.G/(flh.g_max * flh.availability)
        flh["flh"] = flh.G/(gen.g_max)
        return flh.groupby([ "p", "fuel", "technology"], observed=True).mean()[["flh", "utilization"]].reset_index()

    def storage_generation(self):
        """Return storage generation schedules.

        Returns DataFrame with columns [node, plant_type, zone, t, p, G, D_es, L_es]
        """

        es_plant_types = self.data.options["plant_types"]["es"]
        es_plants = self.data.plants.loc[self.data.plants.plant_type.isin(es_plant_types), ["node", "plant_type"]]
        es_plants["zone"] = self.data.nodes.loc[es_plants.node, "zone"].values

        es_gen = pd.merge(es_plants, self.G, left_index=True, right_on="p")
        es_gen = pd.merge(es_gen, self.D_es, on=["p", "t"])
        es_gen = pd.merge(es_gen, self.L_es, on=["p", "t"])
        return es_gen

    def _sort_timesteps(self, column):
        """Helper function to sort timesteps explicitly."""
        order = {timestep: index for index, timestep in enumerate(self.model_horizon)}
        return column.map(order)

    def demand(self, force_recalc=False):
        """Process total nodal demand composed of load and market results of storage/heatpump usage."""
        
        if not (self._cached_results.demand.empty or force_recalc):
            self.logger.debug("Returning cached result for demand.")
            return self._cached_results.demand.copy()

        map_pn = self.data.plants.node.copy().reset_index()
        map_pn.columns = ['p', 'n']
        demand = self.data.demand_el[self.data.demand_el.timestep.isin(self.model_horizon)].copy()
        demand.rename(columns={"node": "n", "timestep": "t"}, inplace=True)
        if not self.D_ph.empty:
            demand_ph = pd.merge(self.D_ph, map_pn[["p", "n"]], 
                                 how="left", on="p").groupby(["n", "t"], as_index=False, observed=True).sum()
            demand = pd.merge(demand, demand_ph[["D_ph", "n", "t"]], how="outer", on=["n", "t"])
        else:
            demand["D_ph"] = 0
        if not self.D_es.empty:
            demand_es = pd.merge(self.D_es, map_pn[["p", "n"]], 
                                 how="left", on="p").groupby(["n", "t"], as_index=False, observed=True).sum()
            demand = pd.merge(demand, demand_es[["D_es", "n", "t"]], how="outer", on=["n", "t"])
        else:
            demand["D_es"] = 0
        
        demand.loc[:, ["demand_el", "D_ph", "D_es"]].fillna(value=0, inplace=True)
        demand["demand"] = demand.demand_el + demand.D_ph + demand.D_es
        demand = demand.sort_values(by='t', key=self._sort_timesteps)
        
        self._cached_results.demand = demand
        return demand.copy()

    # Grid Analytics - Load Flows
    def n_0_flow(self, force_recalc=False):
        """Calculate N-0 Flows.

        Calculates the N-0 power flows on all lines. Optionally just calculate
        for a list/subset of timesteps.

        Parameters
        ----------
        force_recalc : bool, optional
            Power flow results are automatically cached to avoid recalculation.
            This argument forces recalculation e.g. when paramters have been altered. 

        Returns
        -------
        n_0_flows : DataFrame
            N-0 power flows for each line.
        """
        if not (self._cached_results.n_0_flows.empty or force_recalc):
            self.logger.debug("Returning cached result for n_0_flows.")
            return self._cached_results.n_0_flows.copy()

        inj = self.INJ.pivot(index="t", columns="n", values="INJ")
        inj = inj.loc[self.model_horizon, self.data.nodes.index]
        flow = np.dot(self.grid.ptdf, inj.T)
        n_0_flows = pd.DataFrame(index=self.data.lines.index, columns=self.model_horizon, data=flow)

        self._cached_results.n_0_flows = n_0_flows.copy()
        return n_0_flows

    def n_1_flow(self, sensitivity=5e-2, force_recalc=False):
        """N-1 power flows on lines (cb) under outages (co).

        Calculates the power flows on all lines under the outages with significant impact.
        This is calculated with :meth:`~pomato.grid.create_filtered_n_1_ptdf`
        where this is described in greater detail.

        Parameters
        ----------
        force_recalc : bool, optional
            Power flow results are automatically cached to avoid recalculation.
            This argument forces recalculation e.g. when paramters have been altered. 

        sensitivity : float, optional
            The sensitivity defines the threshold from which outages are
            considered critical. An outage that can impact the line flow,
            relative to its maximum capacity, more than the sensitivity is
            considered critical. Defaults to 5%.

        Returns
        -------
        n_1_flows : DataFrame
            Returns Dataframe of N-1 power flows with lines and contingencies
            specified.
        """
        if not (self._cached_results.n_1_flows.empty or force_recalc):
            self.logger.debug("Returning cached result for n_1_flows.")
            return self._cached_results.n_1_flows.copy()
        
        data = self.grid.create_filtered_n_1_ptdf(sensitivity=sensitivity)
        ptdf = data.loc[:, self.data.nodes.index]
        inj = self.INJ.pivot(index="t", columns="n", values="INJ")
        inj = inj.loc[self.model_horizon, self.data.nodes.index]
        flow = np.dot(ptdf, inj.T)
        n_1_flows = pd.DataFrame(columns=self.model_horizon, data=flow)
        n_1_flows["cb"] = data.cb
        n_1_flows["co"] = data.co

        self.logger.info("Done Calculating N-1 Flows")
        n_1_flows = n_1_flows.loc[:, ["cb", "co"] + self.model_horizon]
        self._cached_results.n_1_flows = n_1_flows.copy()
        return n_1_flows

    def absolute_max_n_1_flow(self, sensitivity=0.05):
        """Calculate the absolute max of N-1 Flows.

        This method essentially proviedes a n_1_flow.groupby("cb") yielding the 
        absolute maximum flow, maintaining the directionality of the flow.
        Thanks @https://stackoverflow.com/a/64559655
        
        Parameters
        ----------
        sensitivity : float, optional
            The sensitivity defines the threshold from which outages are
            considered critical. An outage that can impact the line flow,
            relative to its maximum capacity, more than the sensitivity is
            considered critical. Defaults to 5%.

        """

        n_1_flows = self.n_1_flow(sensitivity=sensitivity)
        n_1_flows = n_1_flows.drop("co", axis=1)
        n_1_flow_max = n_1_flows.groupby("cb").max()
        n_1_flow_min = n_1_flows.groupby("cb").min()
        n_1_flows = pd.DataFrame(np.where(n_1_flow_max > -n_1_flow_min, n_1_flow_max, n_1_flow_min),
                                 index=n_1_flow_min.index, columns=n_1_flow_min.columns)

        return n_1_flows.reindex(self.grid.lines.index)

    def overloaded_lines_n_0(self):
        """Calculate overloaded lines (N-0) power.

        Calculates what lines are overloaded, without taking into account
        contingencies. This uses the method :meth:`~n_0_flow()` and compares
        the absolute flows to the maximum capacity.

        Parameters
        ----------
        timesteps : list like, optional
            Subset of model horizon. Defaults to the full model horizon.

        Returns
        -------
        agg_info : DataFrame
            DataFrame that provides the information which line is overloaded,
            how often an overload occurs over the specified timesteps and
            the average loading of the line. Returns an empty DataFrame when
            no line is overloaded.
        n_0_load : DataFrame
            Line loadings for the overloaded lines and considered timesteps.
        """
        flows = self.n_0_flow()
        timesteps = self.model_horizon
        rel_load_array = np.vstack([(abs(flows[t]))/self.data.lines.capacity for t in timesteps]).T
        rel_load = pd.DataFrame(index=flows.index, columns=flows.columns,
                                data=rel_load_array)

        # Only those with overloads (with 1% tolerance)
        n_0_load = rel_load[np.any(rel_load.values > 1.01, axis=1)]

        agg_info = pd.DataFrame(index=n_0_load.index)
        condition = np.any(rel_load.values > 1.01, axis=1)

        overloaded_energy = (n_0_load - 1)
        overloaded_energy[overloaded_energy < 0] = 0
        line_capacities = self.data.lines.loc[condition, "capacity"]
        overloaded_energy = overloaded_energy.multiply(line_capacities, axis=0).sum(axis=1)

        agg_info["# of overloads"] = np.sum(rel_load.values > 1.01, axis=1)[condition]
        agg_info["avg load [%]"] = n_0_load.mean(axis=1)
        agg_info["overloaded energy [GWh]"] = overloaded_energy/1000
        return agg_info, n_0_load

    def overloaded_lines_n_1(self, sensitivity=5e-2):
        """Overloaded lines under contingencies (N-1).

        Uses method :meth:`~n_1_flow()` to obtain N-1 power flows under
        contingencies. Compiles additional information for overloaded lines.
        How often are lines overloaded and under which contingencies its
        average load and whether or not an overload already occurs in the
        base case, meaning the N-0 loading.

        Parameters
        ----------
        timesteps : list like, optional
            Subset of model horizon. Defaults to the full model horizon.
        sensitivity : float, optional
            The sensitivity defines the threshold from which outages are
            considered critical. Am outage that can impact the line flow,
            relative to its maximum capacity, more than the sensitivity is
            considered critical. Defaults to 5%.

        Returns
        -------
        agg_info : DataFrame
            DataFrame that provides the information which line is overloaded,
            how often an overload occurs over the specified timesteps and
            contingencies, average loading of the line. Returns an empty
            DataFrame when no line is overloaded.
        n_1_overload : DataFrame
            Line loadings for the overloaded cbco's and considered timesteps.
        """
        n_1_flow = self.n_1_flow(sensitivity=sensitivity)
        n_1_load = n_1_flow.copy()

        self.logger.info("Processing Flows")

        timesteps = self.model_horizon
        capacity_values = self.grid.lines.capacity[n_1_load.cb].values
        n_1_load.loc[:, timesteps] = n_1_flow.loc[:, timesteps].div(capacity_values, axis=0).abs()

        # 1% overload as tolerance
        n_1_overload = n_1_load[~(n_1_load[timesteps] <= 1.01).all(axis=1)]
        agg_info = n_1_overload[["cb", "co"]].copy()
        agg_info["# of overloads"] = np.sum(n_1_overload[timesteps] > 1, axis=1).values
        agg_info["# of COs"] = 1
        agg_info = agg_info.groupby("cb").sum()
        agg_info["avg load"] = n_1_overload.groupby(by=["cb"]).mean().mean(axis=1).values

        condition = n_1_overload.co == "basecase"
        bool_values = [line in n_1_overload.cb[condition].values for line in agg_info.index]
        agg_info["basecase overload"] = bool_values
        self.logger.info("Done")

        return agg_info, n_1_overload
