import json
import logging
import shutil
import types
from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pomato.tools as tools
from pomato.visualization.geoplot_functions import line_coordinates


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
        self.logger = logging.getLogger('log.pomato.data.Results')
        if not result_folder.is_dir():
            raise FileNotFoundError("result_folder argument is not a folder")

        self.grid = grid
        self.data = data
        self.wdir = self.data.wdir
        self.output_folder = self.wdir.joinpath("data_output").joinpath(result_folder.name)

        variables = {
            variable: False for variable in [
                "G", "H", "D_es", "L_es", "D_hs", "L_hs",
                "INJ", "EX", "D_ph", "F_DC", "CURT", "Alpha", "CC_LINE_MARGIN", 
                "Dump_Water", "COST_G", "COST_H", "COST_EX", "COST_CURT", "COST_CC_LINE_MARGIN",
                "COST_REDISPATCH", "COST_INFEASIBILITY_EL", "COST_INFEASIBILITY_H", "COST_INFEASIBILITY_ES"
            ]
        }

        dual_variables = {variable: False for variable in ["EB_nodal", "EB_zonal", "EB_heat"]}

        infeasibility_variables = {
            variable: False for variable in [
                "INFEASIBILITY_ES", "INFEASIBILITY_H_POS", "INFEASIBILITY_H_NEG",
                "INFEASIBILITY_EL_POS", "INFEASIBILITY_EL_NEG"
            ]
        }

        self.result_attributes = {
            "variables": variables,
            "dual_variables": dual_variables,
            "infeasibility_variables": infeasibility_variables,
            "model_horizon": None, "source_folder": None, "objective": None,
            "is_redispatch_result": False, 
            "corresponding_market_result_name": None
        }

        self.model_horizon = self.result_attributes["model_horizon"]

        self._cached_results = {}
        self._cached_result_structs = types.SimpleNamespace(
            result_data=None,
            averaged_result_data=None,
        )

     # for var in self.result_attributes["variables"]:
        #     setattr(self, var, self.read_cached_result(var))
        # for var in self.result_attributes["dual_variables"]:
        #     setattr(self, var, self.read_cached_result(var))
        # for var in self.result_attributes["infeasibility_variables"]:
        #     setattr(self, var, self.read_cached_result(var))

        # Add opt Set-Up to the results attributes
        self.load_results_from_folder(result_folder)
        # Set Redispatch = True if result is a redispatch result 
        if "redispatch" in self.result_attributes["name"]:
            if not "redispatch" in self.result_attributes["title"]:
                self.result_attributes["title"] += "_redispatch"
            self.result_attributes["is_redispatch_result"] = True
            # Find corresponding market result
            potential_name_1 = self.result_attributes["name"].split("redispatch")[0] + "market_results"
            potential_name_2 = self.result_attributes["name"].split("_redispatch")[0]
            market_result_names = [potential_name_1, potential_name_2]
            result_exists = [(name in self.data.results) for name in market_result_names]
            if sum(result_exists) == 1:
                result = [market_result_names[i] for i, exists in enumerate(result_exists) if exists][0]
                self.result_attributes["corresponding_market_result_name"] = result
            elif sum(result_exists) > 1:
                result = [market_result_names[i] for i, exists in enumerate(result_exists) if exists][0]
                self.result_attributes["corresponding_market_result_name"] = result
                self.logger.warning("Multiple market results to %s found, using %s", 
                                    self.result_attributes["name"], result)
            else:
                self.logger.warning(
                    "Corresponding market result to %s not or with new name instantiated", 
                    self.result_attributes["name"]
                )
                self.logger.warning("Manually set market result name in result attributes.")

    @property
    def G(self):
        return self.read_cached_result("G")
    @property
    def H(self):
        return self.read_cached_result("H")
    @property
    def D_es(self):
        return self.read_cached_result("D_es")
    @property
    def L_es(self):
        return self.read_cached_result("L_es")
    @property
    def D_hs(self):
        return self.read_cached_result("D_hs")
    @property
    def L_hs(self):
        return self.read_cached_result("L_hs")    
    @property
    def Dump_Water(self):
        return self.read_cached_result("Dump_Water")
    @property
    def INJ(self):
        return self.read_cached_result("INJ")
    @property
    def EX(self):
        return self.read_cached_result("EX")
    @property
    def D_ph(self):
        return self.read_cached_result("D_ph")
    @property
    def F_DC(self):
        return self.read_cached_result("F_DC")
    @property
    def CURT(self):
        return self.read_cached_result("CURT")
    @property
    def Alpha(self):
        return self.read_cached_result("Alpha")
    @property
    def CC_LINE_MARGIN(self):
        return self.read_cached_result("CC_LINE_MARGIN")
    @property
    def COST_G(self):
        return self.read_cached_result("COST_G")
    @property
    def COST_H(self):
        return self.read_cached_result("COST_H")
    @property
    def COST_EX(self):
        return self.read_cached_result("COST_EX")
    @property
    def COST_CURT(self):
        return self.read_cached_result("COST_CURT")
    @property
    def COST_REDISPATCH(self):
        return self.read_cached_result("COST_REDISPATCH")
    @property
    def COST_INFEASIBILITY_EL(self):
        return self.read_cached_result("COST_INFEASIBILITY_EL")
    @property
    def COST_INFEASIBILITY_H(self):
        return self.read_cached_result("COST_INFEASIBILITY_H")   
    @property
    def COST_INFEASIBILITY_ES(self):
        return self.read_cached_result("COST_INFEASIBILITY_ES")  
    @property
    def INFEASIBILITY_H_POS(self):
        return self.read_cached_result("INFEASIBILITY_H_POS")
    @property
    def INFEASIBILITY_H_NEG(self):
        return self.read_cached_result("INFEASIBILITY_H_NEG")  
    @property
    def INFEASIBILITY_EL_POS(self):
        return self.read_cached_result("INFEASIBILITY_EL_POS")  
    @property
    def INFEASIBILITY_EL_NEG(self):
        return self.read_cached_result("INFEASIBILITY_EL_NEG")  
    @property
    def INFEASIBILITY_ES(self):
        return self.read_cached_result("INFEASIBILITY_ES")  
    @property
    def EB_nodal(self):
        return self.read_cached_result("EB_nodal")  
    @property
    def EB_zonal(self):
        return self.read_cached_result("EB_zonal")     
    @property
    def COST_CC_LINE_MARGIN(self):
        return self.read_cached_result("COST_CC_LINE_MARGIN")  

    def delete_temporary_files(self):
        """Delete temporary files."""
        folder = self.wdir.joinpath("data_temp/results_cache").joinpath(str(id(self)))
        self.logger.debug("Deleting folder %s", folder)
        shutil.rmtree(folder, ignore_errors=True)
        
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

        if ".zip" in str(folder):
            with ZipFile(folder) as zip_archive:
                tmp_zip_folder = self.wdir.joinpath("data_temp/temp_zip")
                zip_archive.extractall(self.wdir.joinpath("data_temp/temp_zip"))
                folder = self.wdir.joinpath("data_temp/temp_zip")

        for variable_type in ["variables", "dual_variables", "infeasibility_variables"]:
            for var in self.result_attributes[variable_type]:
                try:
                    # setattr(self, var, tools.reduce_df_size(pd.read_csv(str(folder.joinpath(f"{var}.csv")))))
                    self.cache_to_disk(
                        tools.reduce_df_size(pd.read_csv(str(folder.joinpath(f"{var}.csv")))),
                        var
                    )
                    # setattr(self, var, tools.reduce_df_size(pd.read_csv(str(folder.joinpath(f"{var}.csv")))))
                    # setattr(self, var, (pd.read_csv(str(folder.joinpath(f"{var}.csv")))))
                    self.result_attributes[variable_type][var] = True
                except FileNotFoundError:
                    self.logger.warning("%s not in results folder %s", var, str(folder.name))

        # Set result attributes from result json file or data.option:
        try:
            with open(str(folder.joinpath("misc_results.json")), "r") as jsonfile:
                self.result_attributes["objective"] = json.load(jsonfile)
        except FileNotFoundError:
            self.logger.warning("Results Not Found. This is an Error")
            
        try:
            if folder.joinpath("optionfile.json").is_file():
                with open(str(folder.joinpath("optionfile.json")), "r") as jsonfile:
                    self.result_attributes = {
                        **self.result_attributes, **json.load(jsonfile)
                    }
            else:
                with open(str(folder.joinpath("options.json")), "r") as jsonfile:
                    self.result_attributes = {
                        **self.result_attributes, **json.load(jsonfile)
                    }
        except FileNotFoundError:
            self.logger.warning("No option file found in result folder, using data.options")
            self.result_attributes = {**self.result_attributes,
                                      **self.data.options}

        self.result_attributes["source_folder"] = str(folder)
        self.result_attributes["name"] = folder.name
        if not "title" in self.result_attributes or self.result_attributes["title"] == "default":
            self.result_attributes["title"] = self.result_attributes["name"]
        # Model Horizon as attribute
        self.result_attributes["model_horizon"] = list(self.INJ['t'].drop_duplicates().sort_values())
        self.model_horizon = self.result_attributes["model_horizon"]

    def save(self, folder):
        """Save Result to folder"""
        if not folder.is_dir():
            folder.mkdir()
        tools.copytree(self.result_attributes["source_folder"], folder)

    def cache_to_disk(self, df, name):
        """Cache processed results as feather format to relieve memory."""

        folder = self.wdir.joinpath("data_temp/results_cache").joinpath(str(id(self)))
        if not folder.is_dir():
            folder.mkdir() 
        file = folder.joinpath(name)
        df.reset_index().to_feather(file)
        self._cached_results[name] = file

    def read_cached_result(self, name):
        """Read cached processed from disk."""

        if not name in self._cached_results:
            raise ValueError("Result not cached yet.")
        else:
            file = self._cached_results[name]
            df = pd.read_feather(file).set_index("index")
            return df

    # def _clear_cached_results(self):
    #     self._cached_results = {}
    #     self._cached_result_structs = types.SimpleNamespace(
    #         result_data=None,
    #         averaged_result_data=None,
    #     )
    #     folder = self.wdir.joinpath("data_temp/results_cache").joinpath(str(id(self)))
    #     shutil.rmtree(folder, ignore_errors=True)
   
    def create_result_data(self):
        self.generation()
        self.redispatch()
        self.generation()
        self.demand()
        self.n_0_flow()
        self.absolute_max_n_1_flow(sensitivity=0.2)
        self.price()

    # def create_result_data(self, force_recalc=False):
    #     """Creates result data struct from result instance.

    #     Based on :meth:`~result_data_struct`this method fills the data struct with data and results
    #     from the market result specified which is an instance of :class:`~pomato.data.Results`.
    #     This data struct is intended for the generation of visualizations of result in e.g. the
    #     dynamic geoplot.

    #     Parameters
    #     ----------
    #     market_result : :class:`~pomato.data.Results`
    #         Market result which gets subsumed into the predefined data struct.
    #     """
    #     if not (not isinstance(self._cached_result_structs.result_data, types.SimpleNamespace) or force_recalc):
    #         self.logger.debug("Returning cached result for result_data.")
    #         return deepcopy(self._cached_result_structs.result_data)
        
    #     self.logger.info("Precalculating and caching common results..")
    #     data_struct = self.result_data_struct()
    #     data_struct.inj = self.INJ
    #     data_struct.dc_flow = self.F_DC
    #     data_struct.generation = self.generation()
    #     data_struct.demand = self.demand()
    #     data_struct.n_0_flow = self.n_0_flow()
    #     data_struct.n_1_flow = self.absolute_max_n_1_flow(sensitivity=0.2)
    #     data_struct.prices = self.price()
        
    #     self._cached_result_structs.result_data = deepcopy(data_struct)
    #     self.logger.info("Done calculating common results.")

    #     return data_struct

    # def create_averaged_result_data(self, force_recalc=False):
    #     """Creates averaged result data struct.

    #     Based on :meth:`~result_data_struct` and  :meth:`~create_result_data` this method fills 
    #     the data struct with data and results from the market result specified which is an 
    #     instance of :class:`~pomato.data.Results`. All results are averaged in useful ways. This 
    #     data struct is intended for the static geoplot, which visualizes the results in 
    #     average flows, injections, generation and prices. 
    #     """
    #     if not (not isinstance(self._cached_result_structs.averaged_result_data, types.SimpleNamespace) or force_recalc):
    #         self.logger.debug("Returning cached result for averaged_result_data.")
    #         return deepcopy(self._cached_result_structs.averaged_result_data)

    #     data_struct = self.create_result_data()

    #     data_struct.inj = data_struct.inj.groupby("n").mean().reindex(self.grid.nodes.index).INJ
    #     data_struct.n_0_flow = data_struct.n_0_flow.abs().mean(axis=1)
    #     data_struct.n_1_flow = data_struct.n_1_flow.abs().mean(axis=1)
    #     data_struct.dc_flow = data_struct.dc_flow.pivot(index="dc", columns="t", values="F_DC") \
    #                             .abs().mean(axis=1).reindex(self.data.dclines.index).fillna(0)
    #     data_struct.prices = data_struct.prices[["n", "marginal"]].groupby("n").mean()
    #     self._cached_result_structs.averaged_result_data = deepcopy(data_struct)
    #     return data_struct

    def redispatch(self, force_recalc=False):
        """Return Redispatch.
        Calculates a delta between redispatch,- and market result. 
        Positive delta represents a higher generation after redispatch i.e. positive and negative 
        vice versa. 

        Parameters
        ----------
        force_recalc : bool, optional
            Price is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 

        Returns
        -------
        redispatch : DataFrame
            Returns DataFrame with columns [node, plant_type, g_max, zone, t, p, G, delta, delta_abs]

        """
        
        # Find corresponding Market Result
        corresponding_market_result = self.result_attributes["corresponding_market_result_name"]
        if not (self.result_attributes["is_redispatch_result"] and bool(corresponding_market_result)):
            self.logger.warning("Corresponding market result not initialized or found")
            return None

        if not ((not "redispatch" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for redispatch.")
            return self.read_cached_result("redispatch")

        gen = self.generation(force_recalc=force_recalc)
        # Redispatch Calculation G_redispatch - G_market
        gen = pd.merge(self.data.results[corresponding_market_result].G, gen, on=["p", "t"], 
                       suffixes=("_market", "_redispatch"))
        gen["delta"] = gen["G_redispatch"] - gen["G_market"]
        gen["delta_abs"] = gen["delta"].abs()
        gen[["delta_pos", "delta_neg"]] = 0
        gen.loc[gen.delta > 0, "delta_pos"] = gen.loc[gen.delta > 0, "delta"]
        gen.loc[gen.delta < 0, "delta_neg"] = gen.loc[gen.delta < 0, "delta"]

        gen = tools.reduce_df_size(gen)
        self.cache_to_disk(gen, "redispatch")
        return gen

    def redispatch_infeasibility(self, force_recalc=False):
        """Chances in infeasibility variable usage between market and redispatch. 

        """
        # Find corresponding Market Result
        corresponding_market_result = self.result_attributes["corresponding_market_result_name"]
        if not (self.result_attributes["is_redispatch_result"] and bool(corresponding_market_result)):
            self.logger.warning("Corresponding market result not initialized or found")
            return None

        if not ((not "redispatch_infeasibility" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for redispatch_infeasibility.")
            return self.read_cached_result("redispatch_infeasibility")

        infeas = pd.merge(
            self.data.results[corresponding_market_result].infeasibility(),
            self.infeasibility(), how="right",
            on=["t", "n", "zone"], suffixes=("_market", "_redispatch")
        ).reset_index().rename(columns={"n": "node"})

        # sort columns
        cols = ["index", "t", "node", "zone", "pos_market", "neg_market", "pos_redispatch", "neg_redispatch"]
        infeas = infeas[cols]
        infeas.loc[:, infeas.columns[4:]] = infeas.loc[:, infeas.columns[4:]].fillna(0)
        infeas["delta_pos"] = infeas.pos_redispatch - infeas.pos_market
        infeas["delta_neg"] = -(infeas.neg_redispatch - infeas.neg_market)
        infeas["delta_abs"] = -infeas.delta_neg + infeas.delta_pos

        infeas = tools.reduce_df_size(infeas)
        self.cache_to_disk(infeas, "redispatch_infeasibility")
        return infeas

    def zonal_redispatch(self, force_recalc=False):
        """Return Redispatch.
        Calculates a delta between redispatch,- and market result. 
        Positive delta represents a higher generation after redispatch i.e. positive and negative 
        vice versa. 

        Parameters
        ----------
        force_recalc : bool, optional
            Price is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 

        Returns
        -------
        redispatch : DataFrame
            Returns DataFrame with columns [zone, plant_type, g_max, zone, t, G, delta, delta_abs]
        """
        # Find corresponding Market Result
        corresponding_market_result = self.result_attributes["corresponding_market_result_name"]
        if not (self.result_attributes["is_redispatch_result"] and bool(corresponding_market_result)):
            self.logger.warning("Corresponding market result not initialized or found")
            return None
        if not ((not "zonal_redispatch" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for zonal redispatch.")
            return self.read_cached_result("zonal_redispatch")
        gen = self.redispatch()
        cols = ['zone', 'fuel','technology', 't', 'G_redispatch', "G_market", 'delta', 'delta_abs', "delta_pos", "delta_neg"]
        gen_agg = gen[cols].groupby(cols[:-6], observed=True).sum().reset_index()
        self.cache_to_disk(gen_agg, "zonal_redispatch")
        return gen_agg
    
    def infeasibility(self, force_recalc=False):
        """Return electricity infeasibilities.
        
        Infeasibilities occur when the electricity energy balances cannot be satisfied in the 
        model due to other constraints, like capacity or network constraints. Nodal infeasibilities 
        represent dropped load (positive) or dumped energy (negative).

        Parameters
        ----------
        force_recalc : bool, optional
            Infeasibility is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 

        Returns
        -------
        DataFrame
            DataFrame of nodal infeasibilities with columns [t, n, pos, neg].
        """        
        if not ((not "infeasibility" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for infeasibility.")
            return self.read_cached_result("infeasibility")

        infeasibility = pd.merge(self.data.nodes[["zone"]], self.INFEASIBILITY_EL_POS, left_index=True, right_on="n")
        infeasibility = pd.merge(infeasibility, self.INFEASIBILITY_EL_NEG, on=["t", "n"])
        infeasibility = infeasibility.rename(columns={"INFEASIBILITY_EL_POS": "pos", "INFEASIBILITY_EL_NEG": "neg"})
        infeasibility = infeasibility[(infeasibility.pos > 0) | (infeasibility.neg > 0)]
        self.cache_to_disk(infeasibility, "infeasibility")
        return infeasibility

    def price(self, force_recalc=False):
        """Return electricity prices.

        Returns the dual of the energy balances (nodal and zonal). Since
        the model can be cleared with constraints on both simultaneously, the
        resulting nodal price is the sum of the zonal and nodal components.
        The dual is obtained from Julia/JuMP with the dual function and therefore
        multiplied with -1.

        Parameters
        ----------
        force_recalc : bool, optional
            Price is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 

        Returns
        -------
        price : DataFrame
            Price DataFrame with columns timestep (t), node (n), zone (z) and
            price (marginal).
        """
        if not ((not "price" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for price.")
            return self.read_cached_result("price")

        eb_nodal = self.EB_nodal.copy()
        eb_nodal = pd.merge(eb_nodal, self.data.nodes.zone.to_frame(),
                            how="left", left_on="n", right_index=True)
        eb_nodal.loc[abs(eb_nodal.EB_nodal) < 1E-3, "EB_nodal"] = 0
        eb_zonal = self.EB_zonal.copy()
        eb_zonal.loc[abs(eb_zonal.EB_zonal) < 1E-3, "EB_zonal"] = 0
        price = pd.merge(eb_nodal, eb_zonal, how="left",
                         left_on=["t", "zone"], right_on=["t", "z"])
        price["marginal"] = -(price.EB_zonal + price.EB_nodal)

        price = price[["t", "n", "z", "marginal"]]
        price = tools.reduce_df_size(price)

        self.cache_to_disk(price, "price")
        return price

    def net_position(self):
        """Calculate net position for each zone and timestep.

        Returns
        -------
        net_position : DataFrame
            DataFrame with the timesteps as index and zones as columns.
        """
        net_position = pd.DataFrame(index=self.EX.t.unique())
        for zone in self.data.zones.index:
            net_position[zone] = self.EX.loc[self.EX.z == zone, ["t", "EX"]].groupby("t").sum() - \
                                 self.EX.loc[self.EX.zz == zone, ["t", "EX"]].groupby("t").sum()
        return net_position
    
    def generation(self, force_recalc=False):
        """Return generation variable merged to input data.
        
        Parameters
        ----------
        force_recalc : bool, optional
            Generation is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 

        Returns
        -------
        generation : DataFrame
            Returns DataFrame with columns [node, plant_type, g_max, zone, t, p, G]

        """
        if not ((not "generation" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for generation.")
            return self.read_cached_result("generation")
        
        cols = ["plant_type", "fuel", "availability", "g_max", "node"]
        gen = pd.merge(self.data.plants[cols],
                        self.G, left_index=True, right_on="p", how="right")
        
        gen["zone"] = self.data.nodes.loc[gen.node, "zone"].values
        if "technology" in self.data.plants.columns:
            gen = pd.merge(gen, self.data.plants[["technology"]], 
                           right_index=True, left_on="p")
        else:
            gen["technology"] = gen.plant_type
        gen = tools.reduce_df_size(gen)
        self.cache_to_disk(gen, "generation")
        return gen

    def zonal_generation(self, force_recalc=False):
        """Return generation variable merged to input data.
        
        Parameters
        ----------
        force_recalc : bool, optional
            Generation is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 

        Returns
        -------
        generation : DataFrame
            Returns DataFrame with columns ['zone', 'fuel','technology', 'node', 'G']

        """
        if not ((not "zonal_generation" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for zonal generation.")
            return self.read_cached_result("zonal_generation")
        
        gen = self.generation()
        cols = ['zone', 'fuel','technology', 't', 'G']
        gen_agg = gen[cols].groupby(cols[:-1], observed=True).sum().reset_index()
        self.cache_to_disk(gen_agg, "zonal_generation")
        return gen_agg  

    def curtailment(self, force_recalc=False):
        """Return Curtailment merge to input data.
        
        Parameters
        ----------
        force_recalc : bool, optional
            Generation is cached automatically. To enforce recalc, e.g. when explicitly changing
            data set this force_recalc to True. Defaults to False. 
        
        Returns
        -------
        Curtailment : DataFrame
            Returns DataFrame with columns [node, plant_type, g_max, zone, t, p, CURT]
        """
        if not ((not "curtailment" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for curtailment.")
            return self.read_cached_result("curtailment")

        curtailment = pd.merge(self.data.plants[["plant_type", "node", "g_max"]],
                               self.CURT, left_index=True, right_on="p", how="right")
        curtailment["zone"] = self.data.nodes.loc[curtailment.node, "zone"].values

        curtailment = curtailment[curtailment.CURT > 0]
        curtailment = tools.reduce_df_size(curtailment)
        self.cache_to_disk(curtailment, "curtailment")
        return curtailment

    def full_load_hours(self):
        """Returns plant data including full load hours."""        

        gen = self.generation()[["t", "p", "fuel", "technology", "G", "g_max"]].copy()
        ava = self.data.availability.copy()[["timestep", "plant", "availability"]]
        ava.columns = ["t", "p", "availability"]
        
        flh = pd.merge(gen, ava, on=["t", "p"], how="left")
        flh.loc[:, "availability"] = flh.loc[:, "availability"].fillna(1)
        flh["utilization"] = flh.G/(flh.g_max * flh.availability)
        flh["flh"] = flh.G/(gen.g_max)
        cols = ["p", "fuel", "technology", "flh", "utilization"]
        flh = flh[cols].groupby([ "p", "fuel", "technology"], observed=True).mean()[["flh", "utilization"]].reset_index()
        return flh

    def storage_generation(self, force_recalc=False):
        """Return storage generation schedules.

        Returns DataFrame with columns [node, plant_type, zone, t, p, G, D_es, L_es]
        """
        if not ((not "storage_generation" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for storage_generation.")
            return self.read_cached_result("storage_generation")
        es_plant_types = self.result_attributes["plant_types"]["es"]
        es_plants = self.data.plants.loc[self.data.plants.plant_type.isin(es_plant_types), ["node", "plant_type"]]
        es_plants["zone"] = self.data.nodes.loc[es_plants.node, "zone"].values

        es_gen = pd.merge(es_plants, self.G, left_index=True, right_on="p")
        es_gen = pd.merge(es_gen, self.D_es, on=["p", "t"])
        es_gen = pd.merge(es_gen, self.L_es, on=["p", "t"])
        es_gen = tools.reduce_df_size(es_gen)
        self.cache_to_disk(es_gen, "storage_generation")
        return es_gen

    def _sort_timesteps(self, column):
        """Helper function to sort timesteps explicitly."""
        order = {timestep: index for index, timestep in enumerate(self.model_horizon)}
        return column.map(order)

    def demand(self, force_recalc=False):
        """Process total nodal demand composed of load and market results of storage/heatpump usage."""
        
        if not ((not "demand" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for demand.")
            return self.read_cached_result("demand")

        map_pn = self.data.plants.node.copy().reset_index()
        map_pn.columns = ['p', 'n']
        demand = self.data.demand_el[self.data.demand_el.timestep.isin(self.model_horizon)].copy()
        demand.rename(columns={"node": "n", "timestep": "t"}, inplace=True)
        if not self.D_ph.empty:
            demand_ph = pd.merge(self.D_ph, map_pn[["p", "n"]], 
                                 how="left", on="p").groupby(["n", "t"], as_index=False, observed=True).sum()
            demand = pd.merge(demand, demand_ph[["D_ph", "n", "t"]], how="left", on=["n", "t"])
            demand.loc[:, "D_ph"].fillna(0, inplace=True)
        else:
            demand["D_ph"] = 0
        if not self.D_es.empty:
            demand_es = pd.merge(self.D_es, map_pn[["p", "n"]], 
                                 how="left", on="p").groupby(["n", "t"], as_index=False, observed=True).sum()
            demand = pd.merge(demand, demand_es[["D_es", "n", "t"]], how="left", on=["n", "t"])
            demand.loc[:, "D_es"].fillna(0, inplace=True)
        else:
            demand["D_es"] = 0
        demand["demand"] = demand.demand_el + demand.D_ph + demand.D_es
        demand = demand.sort_values(by='t', key=self._sort_timesteps)
        demand = tools.reduce_df_size(demand)

        self.cache_to_disk(demand, "demand")
        return demand

    # Grid Analytics - Load Flows
    def n_0_flow(self, force_recalc=False):
        """Calculate N-0 Flows.

        Calculates the N-0 power flows on all lines. Optionally just calculate
        for a list/subset of timesteps.

        Parameters
        ----------
        force_recalc : bool, optional
            Power flow results are automatically cached to avoid recalculation.
            This argument forces recalculation e.g. when parameters have been altered. 

        Returns
        -------
        n_0_flows : DataFrame
            N-0 power flows for each line.
        """
        if not ((not "n_0_flows" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for n_0_flows.")
            return self.read_cached_result("n_0_flows")

        inj = self.INJ.pivot(index="t", columns="n", values="INJ")
        inj = inj.loc[self.model_horizon, self.data.nodes.index]
        flow = np.dot(self.grid.ptdf, inj.T)
        n_0_flows = pd.DataFrame(index=self.data.lines.index, columns=self.model_horizon, data=flow)
        n_0_flows.index.name = 'index'
        self.cache_to_disk(n_0_flows, "n_0_flows")
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
            This argument forces recalculation e.g. when parameters have been altered. 

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
        if not ((not "n_1_flows" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for n_1_flows.")
            return self.read_cached_result("n_1_flows")
        
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
        n_1_flows.index.name = 'index'
        self.cache_to_disk(n_1_flows, "n_1_flows")
        return n_1_flows

    def absolute_max_n_1_flow(self, sensitivity=0.05, force_recalc=False):
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
        if not ((not "abs_max_n_1_flows" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached result for redispatch.")
            return self.read_cached_result("abs_max_n_1_flows")

        n_1_flows = self.n_1_flow(sensitivity=sensitivity)
        n_1_flows = n_1_flows.drop("co", axis=1)
        n_1_flow_max = n_1_flows.groupby("cb").max()
        n_1_flow_min = n_1_flows.groupby("cb").min()
        n_1_flows = pd.DataFrame(np.where(n_1_flow_max > -n_1_flow_min, n_1_flow_max, n_1_flow_min),
                                 index=n_1_flow_min.index, columns=n_1_flow_min.columns)

        n_1_flows = n_1_flows.reindex(self.grid.lines.index)
        self.cache_to_disk(n_1_flows, "abs_max_n_1_flows")
        return n_1_flows

    def overloaded_lines_n_0(self, force_recalc=False):
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

        if not ((not "overloaded_n_0" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached results.")
            return (self.read_cached_result("n_0_info"), self.read_cached_result("n_0_overload"))

        flows = self.n_0_flow()
        timesteps = self.model_horizon

        lt_linerating = self.data.lines.capacity * self.result_attributes["grid"]["long_term_rating_factor"]
        rel_load_array = np.vstack([(abs(flows[t]))/lt_linerating for t in timesteps]).T
        rel_load = pd.DataFrame(index=flows.index, columns=flows.columns,
                                data=rel_load_array)

        # Only those with overloads (with 1% tolerance)
        n_0_load = rel_load[np.any(rel_load.values > 1.01, axis=1)]

        n_0_info = pd.DataFrame(index=n_0_load.index)
        condition = np.any(rel_load.values > 1.01, axis=1)

        overloaded_energy = (n_0_load - 1)
        overloaded_energy[overloaded_energy < 0] = 0
        line_capacities = self.data.lines.loc[condition, "capacity"]
        overloaded_energy = overloaded_energy.multiply(line_capacities, axis=0).sum(axis=1)

        n_0_info["# of overloads"] = np.sum(rel_load.values > 1.01, axis=1)[condition]
        n_0_info["avg load [%]"] = n_0_load.mean(axis=1)
        n_0_info["overloaded energy [GWh]"] = overloaded_energy/1000
        
        self.cache_to_disk(n_0_load, "n_0_overload")
        self.cache_to_disk(n_0_info, "n_0_info")
        return n_0_info, n_0_load

    def overloaded_lines_n_1(self, sensitivity=5e-2, force_recalc=False):
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
        if not ((not "overloaded_n_0" in self._cached_results) or force_recalc):
            self.logger.debug("Returning cached results.")
            return (self.read_cached_result("n_1_info"), self.read_cached_result("n_1_overload"))

        n_1_flow = self.n_1_flow(sensitivity=sensitivity)
        n_1_load = n_1_flow.copy()

        self.logger.info("Processing Flows")

        timesteps = self.model_horizon

        capacity_values = self.grid.lines.capacity[n_1_load.cb].values * self.result_attributes["grid"]["short_term_rating_factor"]
        n_1_load.loc[:, timesteps] = n_1_flow.loc[:, timesteps].div(capacity_values, axis=0).abs()

        # 1% overload as tolerance
        n_1_overload = n_1_load[~(n_1_load[timesteps] <= 1.01).all(axis=1)]
        n_1_info = n_1_overload[["cb", "co"]].copy()
        n_1_info["# of overloads"] = np.sum(n_1_overload[timesteps] > 1, axis=1).values
        n_1_info["# of COs"] = 1
        n_1_info = n_1_info[["cb", "# of COs"]].groupby("cb").sum()
        n_1_info["avg load"] = n_1_overload.loc[:, ~n_1_overload.columns.isin(['co'])].groupby(by=["cb"]).mean().mean(axis=1).values

        condition = n_1_overload.co == "basecase"
        bool_values = [line in n_1_overload.cb[condition].values for line in n_1_info.index]
        n_1_info["basecase overload"] = bool_values
        self.logger.info("Done")

        self.cache_to_disk(n_1_overload, "n_1_overload")
        self.cache_to_disk(n_1_info, "n_1_info")
        return n_1_info, n_1_overload
