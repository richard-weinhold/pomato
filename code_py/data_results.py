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

        self.grid = None

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
        self.data.result_attributes["model_horizon"] = list(self.INJ.t.unique())



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

    # def courtailment(self):

    #     res_fuels = ["sun", "wind", "water", "biomass"]

    def default_plots(self, show_plot=False):
        """Set of Standard Plots"""
        # self = mato.data.results
        if show_plot:
            plt.ion()

        generation = pd.merge(self.G, self.data.plants[["node", "fuel", "tech"]],
                              how="left", left_on="p", right_index=True)
        generation = pd.merge(generation, self.data.nodes.zone.to_frame(),
                              how="left", left_on="node", right_index=True)
        model_horizon = self.data.result_attributes["model_horizon"]

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

    ######
    # Grid Analytics
    # - Load Flows

    def n_0_flow(self, timesteps):
        if not self.grid:
            self.logger.error("Grid Model not available!")
            return None
        n_0_flows = pd.DataFrame(index=self.data.lines.index)
        for t in timesteps:
            n_0_flows[t] = np.dot(self.grid.ptdf, self.INJ.INJ[self.INJ.t == t].values)
        return n_0_flows

    def n_1_flow(self, timesteps, lines, outages):
        """Line flows on lines (cb) under outages (co)
           input lines/outages list of line indices
           timesteps list of timestepts
           output DF[lines, outages, timesteps]
        """

        if not self.grid:
            self.logger.error("Grid Model not available!")
            return None
        if not all([x in self.data.lines.index for x in lines + outages]):
            self.logger.error("Not all CBs/COs are indices of lines!")
            return None
        n_1_flows = pd.DataFrame([[l, o] for l in lines for o in outages],
                                 columns=["lines", "outages"])
        ptdf = np.vstack([self.grid.create_n_1_ptdf_cbco(l,o) for l in lines for o in outages])
        for t in timesteps:
            n_1_flows[t] = np.dot(ptdf, self.INJ.INJ[self.INJ.t == t].values)
        return n_1_flows

    def overloaded_lines_n_0(self, timesteps=None):
        """
        Information about N-0 (over) Lineflows
        returns a DataFrame with respective info
        and timeseries of overloaded lines
        """
        if not timesteps:
            # if not specifie use full model horizon
            timesteps = self.data.result_attributes["model_horizon"]

        flows = self.n_0_flow(timesteps)
        relative_load = pd.DataFrame(index=flows.index, columns=flows.columns,
                                     data=np.vstack([(abs(flows[t]))/self.data.lines.maxflow for t in timesteps]).T)

        # Only those with over loadings
        n_0_load = relative_load[np.any(relative_load.values>1, axis=1)]

        return_df = pd.DataFrame(index=n_0_load.index)
        return_df["# of overloads"] = np.sum(relative_load.values>1, axis=1)[np.any(relative_load.values>1, axis=1)]
        return_df["avg load"] = n_0_load.mean(axis=1)
  
        return return_df, n_0_load

    def overloaded_lines_n_1(self, timesteps=None):

        if not timesteps:
            # if not specifie use full model horizon
            timesteps = self.data.result_attributes["model_horizon"]

        n_1_load = pd.DataFrame(columns=["lines", "outages"] + timesteps)
        for l, line in enumerate(self.data.lines.index):

            ## Print Progress-Bar
            sys.stdout.flush()
            sys.stdout.write("\r[%-35s] %d%% done!" % \
                            ('='*int(l*35/len(self.data.lines.index)),
                             int(l*101/len(self.data.lines.index))))
            
            flow = self.n_1_flow(timesteps, [line], list(self.data.lines.index))

            relative_load = pd.DataFrame(columns=timesteps, index=self.data.lines.index,
                                         data=np.vstack([(abs(flow[t])/self.data.lines.maxflow[line]) for t in timesteps]).T)

            if np.any(relative_load.values>1):
                # overloaded_n_1_load = relative load on line with overloadings
                overloaded_n_1_load = relative_load[np.any(relative_load.values>1, axis=1)]
                overloaded_n_1_load["lines"] = line
                n_1_load = n_1_load.append(overloaded_n_1_load.rename_axis('outages').reset_index(),
                                           sort=True, ignore_index=True)

            if sys.getsizeof(n_1_load)/(8*1E9) > 1:
                # break if size of n_1_load is larger than 1GB
                sys.stdout.write("\n")
                sys.stdout.write("\n")
                self.logger.warning("Return DF is too large!")
                return pd.DataFrame(), n_1_load
                break

        tmp_df = n_1_load[["lines", "outages"]].copy()
        tmp_df["# of overloads"] = np.sum(n_1_load[timesteps] > 1, axis=1)
        tmp_df["# of COs"] = 1
        tmp_df["avg load"] = n_1_load[timesteps].mean(axis=1)

        return_df = tmp_df[["# of overloads", "# of COs", "lines"]].groupby("lines").sum()
        return_df[["avg load"]] = tmp_df[["avg load", "lines"]].groupby("lines").mean()

        sys.stdout.write("\n")
        return return_df, n_1_load


