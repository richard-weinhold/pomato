import sys
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt


class ResultProcessing(object):
    """Data Woker Class"""
    def __init__(self, data, opt_folder, opt_setup):
        self.logger = logging.getLogger('Log.MarketModel.DataManagement.ResultData')
        self.data = data

        self.result_folder = self.data.wdir.joinpath("data_output").joinpath(str(opt_folder).split("\\")[-1])
        if not self.result_folder.is_dir():
            self.result_folder.mkdir()

        for var in self.data.result_attributes["variables"]:
            setattr(self, var, pd.DataFrame())
        for var in self.data.result_attributes["dual_variables"]:
            setattr(self, var, pd.DataFrame())
        for var in self.data.result_attributes["infeasibility_variables"]:
            setattr(self, var, pd.DataFrame())

        # Add opt Set-Up to the results attributes
        self.data.result_attributes = {**self.data.result_attributes, **opt_setup}
        self.data.result_attributes["source"] = opt_folder
        self.load_results_from_jlfolder(opt_folder)

        # set-up: dont show the graphs when created
        plt.ioff()

    def load_results_from_jlfolder(self, folder):
        folder_name = str(folder).split("\\")[-1]
        self.logger.info(f"Loading Results from results folder {folder_name}")

        for variable_type in ["variables", "dual_variables", "infeasibility_variables"]:
            for var in self.data.result_attributes[variable_type]:
                try:
                    setattr(self, var, pd.read_json(str(folder.joinpath(f"{var}.json")),
                                                    orient="index").sort_index())
                    self.data.result_attributes[variable_type][var] = True
                except:
                    self.logger.warning(f"{var} not in results folder {folder_name}")

        ## Manual setting of attributes:
        with open(str(folder.joinpath("misc_result.json")), "r") as jsonfile:
            data = json.load(jsonfile)
        self.data.result_attributes["objective"] = data["Objective Value"]

    def commercial_exchange(self, t):
        exchange = self.EX[(self.EX.t == t)][["EX", "z", "zz"]]
        exchange.columns = ["values", "from_zone", "to_zone"]
        exchange = exchange.pivot(values="values", index="from_zone", columns="to_zone")
        return exchange

    def net_position(self):
        net_position = pd.DataFrame(index=self.EX.t.unique())
        for zone in self.data.zones.index:
            net_position[zone] = self.EX[self.EX.z == zone].groupby("t").sum() - self.EX[self.EX.zz == zone].groupby("t").sum()
        return net_position

    def check_infeasibilities(self):
        """
        checks for infeasiblity variables in electricity/heat energy balance
        and line infeasibility variables
        returns nothing
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for infeasibilities in self.data.result_attributes["infeasibility_variables"]:
            tmp = getattr(self, infeasibilities)
            for col in tmp.select_dtypes(include=numerics):
                if any(tmp[col] > 1e-3):
                    self.logger.warning(f"Infeasibilites in {col}")

    def default_plots(self, show_plot=False):
        """Set of Standard Plots"""
#        self = mato.data.results
        if show_plot:
            plt.ion()

        generation = pd.merge(self.G, self.data.plants[["node", "fuel", "tech"]],
                              how="left", left_on="p", right_index=True)
        generation = pd.merge(generation, self.data.nodes.zone.to_frame(),
                              how="left", left_on="node", right_index=True)
        model_horizon = list(np.unique(generation.t.values))

        # By Fuel
        fig, ax = plt.subplots()
        g_by_fuel = generation.groupby(["t", "fuel"], as_index=False).sum()
        g_by_fuel.pivot(index="t", columns="fuel", values="G").plot.area(ax=ax, xticks=[x for x in range(0, len(model_horizon))], figsize=(20,10), rot=45)
        ax.legend(loc='upper right')
        ax.margins(x=0)
        fig.savefig(str(self.result_folder.joinpath("gen_fuel.png")))

        # Aggregated example
        fig, ax = plt.subplots()
        g_by_fuel_agg = g_by_fuel.groupby("fuel").sum().plot.pie(ax=ax, y="G", figsize=(20,20),)
        ax.legend(loc='upper right')
        ax.margins(x=0)
        fig.savefig(str(self.result_folder.joinpath("gen_fuel_pichart.png")))

        # By Tech
        fig, ax = plt.subplots()
        g_by_tech = generation.groupby(["t", "tech"], as_index=False).sum()
        g_by_tech.pivot(index="t", columns="tech", values="G").plot.area(ax=ax, xticks=[x for x in range(0, len(model_horizon))], figsize=(20,10), rot=45)
        ax.legend(loc='upper right')
        ax.margins(x=0)
        fig.savefig(str(self.result_folder.joinpath("gen_tech.png")))

        # Renewables generation
        fig, ax = plt.subplots()
        res_gen = generation[generation.fuel.isin(["sun", "wind"])].groupby(["t", "fuel"], as_index=False).sum()
        res_gen.pivot(index="t", columns="fuel", values="G").plot(ax=ax, xticks=[x for x in range(0, len(model_horizon))], figsize=(20,10), rot=45)
        ax.legend(loc='upper right')
        ax.margins(x=0)
        fig.savefig(str(self.result_folder.joinpath("gen_res.png")))

        # Storage Generation, Demand and LEvel
        fig, ax = plt.subplots()
        stor_d = self.D_es.groupby(["t"], as_index=True).sum()
        stor_l = self.L_es.groupby(["t"], as_index=True).sum()
        stor_tech = ["reservoir", "psp"]
        stor_g = generation[generation.tech.isin(stor_tech)].groupby(["t"], as_index=True).sum()
        pd.concat([stor_d, stor_l, stor_g], axis=1).plot(ax=ax, xticks=[x for x in range(0, len(model_horizon))], figsize=(20,10), rot=45)
        ax.legend(loc='upper right')
        ax.margins(x=0)
        fig.savefig(str(self.result_folder.joinpath("storage.png")))

        # Close all Figures
        fig.clf()
