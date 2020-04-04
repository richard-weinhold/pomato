"""Bokeh Plot Interface."""

import json
import logging
import re
import subprocess
import threading

import pandas as pd
import psutil

import pomato.tools as tools
from pomato.visualization.plot import create_static_plot


class BokehPlot():
    """Interface market data and the creation of a geographic plot.

    There are two option to create a bokeh plot: static and dynamic.

    A static plot will generate a plot based on the provided market result
    with limited interaction. It is either a signle timestep or an average
    load for a timeseires.
    A dynamic plot displays the full market result with possibilitiy to
    change between timestept, N-0 and N-2 line flows and inspection of
    generation for each or group of nodes. This requires a running
    bokeh server process and a bit more setup, which is why this module exists.

    Attributes
    ----------
    wdir, bokeh_dir : pathlib.Path
        Working directory, bokeh_directory, used to store temporary data.
    bokeh_type : str
        Indicating type of plot. Static or dynamic.
    bokeh_server : subprocess
        Subprocess running the bokeh server.
    bokeh_thread : thread
        Spawns a seprate thread that contains the bokeh server, this way
        the model remains repsonsive.
    """

    def __init__(self, wdir, bokeh_type="static"):
        # Impoort Logger
        self.logger = logging.getLogger('Log.MarketModel.BokehPlot')

        self.wdir = wdir
        self.bokeh_dir = wdir.joinpath("data_temp/bokeh_files")
        # Make sure all folders exist
        tools.create_folder_structure(self.wdir, self.logger)

        self.bokeh_type = bokeh_type
        # attributes required for dynamic bokeh plot
        self.bokeh_server = None
        self.bokeh_thread = None
        self._bokeh_pid = None

    def create_empty_static_plot(self, data):
        """Create a geo plot without injection or line flows"""

        inj = pd.Series(index=data.nodes.index, data=0).values
        flow_n_0 = pd.Series(index=data.lines.index, data=0)
        flow_n_1 = pd.Series(index=data.lines.index, data=0)
        f_dc = pd.Series(index=data.dclines.index, data=0)

        create_static_plot(data.lines, data.nodes, data.dclines,
                           inj, flow_n_0, flow_n_1, f_dc)

    def create_static_plot(self, market_results):
        """Create static bokeh plot of the market results.

        Creates a fairly interactive geographic plot of the provided market
        results. It creates either a single plot if one market result is provided
        or multiples if the provided dict contrains more.
        If the dict contains two market results, that correspond to naming of
        the redispatch model, it will create a plot for the market result and
        another including the redispatched generation units.

        Parameters
        ----------
        market_results : dict of :class:`~pomato.data.DataProcessing`
            The *result* attribute of the DataManagement is provided to be
            plotted.
        """
        if len(market_results) == 1:
            market_result = list(market_results)[0]
            folder = market_results[market_result].result_attributes["source"]
            self.logger.info(
                "initializing bokeh plot with from folder: %s", str(folder))
            plots = [(market_results[market_result], None)]

        elif (len(market_results) == 2) and any(["redispatch" in result for result in market_results]):
            redisp_result = market_results[next(
                r for r in list(market_results) if "redispatch" in r)]
            market_result = market_results[next(
                r for r in list(market_results) if "market_result" in r)]
            gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "node"]],
                           market_result.G, left_index=True, right_on="p")
            gen = pd.merge(gen, redisp_result.G, on=[
                           "p", "t"], suffixes=("_market", "_redispatch"))
            gen["delta"] = gen["G_redispatch"] - gen["G_market"]
            gen["delta_abs"] = gen["delta"].abs()
            plots = [(market_result, None), (redisp_result, gen), ]

        else:
            plots = []
            for result in list(market_results):
                plots.append((market_results[result], None))

        for plot_result, gen in plots:
            n_0 = plot_result.n_0_flow()
            n_1 = plot_result.n_1_flow()
            n_1.loc[:, n_0.columns] = n_1.loc[:, n_0.columns].abs()
            n_1 = n_1.groupby("cb").max(
            ).loc[plot_result.data.lines.index, n_0.columns]
            n_0_rel = n_0.divide(plot_result.data.lines.maxflow, axis=0).abs()
            n_1_rel = n_1.divide(plot_result.data.lines.maxflow, axis=0).abs()
            inj = plot_result.INJ.groupby("n").mean(
            ).loc[plot_result.data.nodes.index].values
            flow_n_0 = pd.DataFrame(index=plot_result.data.lines.index)
            flow_n_0 = n_0_rel.mean(axis=1).multiply(
                plot_result.data.lines.maxflow)
            flow_n_1 = pd.DataFrame(index=plot_result.data.lines.index)
            flow_n_1 = n_1_rel.mean(axis=1).multiply(
                plot_result.data.lines.maxflow)
            f_dc = plot_result.F_DC.pivot(index="dc", columns="t", values="F_DC") \
                .abs().mean(axis=1).reindex(plot_result.data.dclines.index).fillna(0)

            create_static_plot(plot_result.data.lines,
                               plot_result.data.nodes,
                               plot_result.data.dclines,
                               inj, flow_n_0, flow_n_1, f_dc,
                               redispatch=gen, option=0,
                               title=plot_result.result_attributes["source"].name)

    def add_market_result(self, market_result, name):
        """Create data set for bokeh plot from julia market_result-object
        with the associated grid model
        """
        if not market_result.grid:
            self.logger.error("Grid Model not in Results Object!")
            raise Exception('Grid Model not in Results Object!')

        self.logger.info("Processing market model data...")

        data_path = self.bokeh_dir.joinpath("market_result").joinpath(name)
        if not data_path.is_dir():
            data_path.mkdir()

        # market_result.data.zones.to_csv(str(folder.joinpath('zones.csv')), index_label='index')
        # market_result.data.tech.to_csv(str(folder.joinpath('tech.csv')))
        pd.DataFrame(index=market_result.grid.lines.index,
                     columns=market_result.grid.lines.index,
                     data=market_result.grid.lodf).to_csv(str(data_path.joinpath('lodf.csv')),
                                                          index_label='index')
        market_result.data.fuel.to_csv(str(data_path.joinpath('fuel.csv')),
                                       index_label='index')
        market_result.data.dclines.to_csv(str(data_path.joinpath('dclines.csv')),
                                          index_label='index')
        market_result.grid.nodes.to_csv(str(data_path.joinpath('nodes.csv')),
                                        index_label='index')
        market_result.grid.lines.to_csv(str(data_path.joinpath('lines.csv')))
        market_result.F_DC.to_csv(str(data_path.joinpath('F_DC.csv')))
        market_result.INJ.to_csv(str(data_path.joinpath('INJ.csv')))

        n_0_flows, n_1_flows = self.process_grid_data(market_result)
        n_1_flows.to_csv(str(data_path.joinpath('n_1_flows.csv')))
        n_0_flows.to_csv(str(data_path.joinpath('n_0_flows.csv')))

        generation_by_fuel = self.process_generation_data(market_result)
        generation_by_fuel.to_csv(
            str(data_path.joinpath('g_by_fuel.csv')), index_label='index')

        demand = self.process_demand_data(market_result)
        demand.to_csv(str(data_path.joinpath('demand.csv')),
                      index_label='index')

        t_first = market_result.data.result_attributes["model_horizon"][0]
        t_last = market_result.data.result_attributes["model_horizon"][-1]

        # convert to int, bc of the slider widget
        t_dict = {"t_first": int(re.search(r'\d+', t_first).group()),
                  "t_last": int(re.search(r'\d+', t_last).group())}
        with open(data_path.joinpath('t.json'), 'w') as time_frame:
            json.dump(t_dict, time_frame)

        self.logger.info("Market Results %s successfully initialized!", name)

    def process_grid_data(self, market_result):
        """precalculatting the line flows for the bokeh plot"""
        self.logger.info(
            "Precalculatting line flows and saving them to file...")
        n_0_flows = market_result.n_0_flow()
        n_1_flows = market_result.n_1_flow()

        # convert n_1 flows to the max of N-1 flows
        time = market_result.data.result_attributes["model_horizon"]
        n_1_flows = n_1_flows.drop("co", axis=1)
        n_1_flows[time] = n_1_flows[time].abs()
        n_1_flows = n_1_flows.groupby("cb").max().reset_index()

        self.logger.info("Done!")
        return n_0_flows, n_1_flows.set_index("cb").reindex(market_result.grid.lines.index)

    def process_generation_data(self, market_result):
        """Prepare Generation Data for Bokeh Plot"""
        generation = market_result.G
        # Save relevant variables from market result
        generation = pd.merge(generation, market_result.data.plants[["node", "fuel"]],
                              how="left", left_on="p", right_index=True)
        generation_by_fuel = generation.groupby(
            ["t", "fuel", "node"], as_index=False).sum()
        return generation_by_fuel

    def process_demand_data(self, market_result):
        """Process total nodal demand composed of load data and market results of storage/heatpump usage."""

        map_pn = market_result.data.plants.node.reset_index()
        map_pn.columns = ['p', 'n']

        demand = market_result.data.demand_el[market_result.data.demand_el.node.isin(
            market_result.data.nodes.index)].copy()
        demand.rename(columns={"node": "n", "timestep": "t",
                               "demand_el": "d_el"}, inplace=True)
        demand_d = market_result.D_d
        demand_ph = market_result.D_ph
        demand_es = market_result.D_es

        if not demand_d.empty:
            demand_d = pd.merge(
                demand_d, map_pn[["p", "n"]], how="left", left_on="d", right_on="p")
            demand_d = demand_d.groupby(["n", "t"], as_index=False).sum()
        else:
            demand_d = pd.DataFrame(columns=["d", "t", "D_d"])
            demand_d = pd.merge(
                demand_d, map_pn[["p", "n"]], how="left", left_on="d", right_on="p")

        if not demand_ph.empty:
            demand_ph = pd.merge(
                demand_ph, map_pn[["p", "n"]], how="left", on="p")
            demand_ph = demand_ph.groupby(["n", "t"], as_index=False).sum()
        else:
            demand_ph = pd.DataFrame(columns=["p", "t", "D_ph"])
            demand_ph = pd.merge(
                demand_ph, map_pn[["p", "n"]], how="left", on="p")

        if not demand_es.empty:
            demand_es = pd.merge(
                demand_es, map_pn[["p", "n"]], how="left", on="p")
            demand_es = demand_es.groupby(["n", "t"], as_index=False).sum()
        else:
            demand_es = pd.DataFrame(columns=["p", "t", "D_es"])
            demand_es = pd.merge(
                demand_es, map_pn[["p", "n"]], how="left", on="p")

        demand = pd.merge(
            demand, demand_d[["D_d", "n", "t"]], how="outer", on=["n", "t"])
        demand = pd.merge(
            demand, demand_ph[["D_ph", "n", "t"]], how="outer", on=["n", "t"])
        demand = pd.merge(
            demand, demand_es[["D_es", "n", "t"]], how="outer", on=["n", "t"])
        demand.fillna(value=0, inplace=True)
        demand["d_total"] = demand.d_el + \
            demand.D_d + demand.D_ph + demand.D_es

        return demand[["n", "t", "d_total"]]

    def _output_reader(self, proc):
        """Print stdout to console."""
        for line in iter(proc.stdout.readline, b''):
            bokeh_output = '{0}'.format(line.decode('utf-8')).strip()
            self.logger.info('bokeh: %s', bokeh_output)

            # listen to output and save bokeh pid
            if "Starting Bokeh server with process id" in bokeh_output:
                self._bokeh_pid = int(bokeh_output.split()[-1])
                print("HERE:", self._bokeh_pid)
            # listen to output and stop server if websocket is closed
            kill_keywords = ['code=1001', 'WebSocket connection closed']
            if any(k in bokeh_output for k in kill_keywords):
                self.stop_server()

    def start_server(self):
        """Run the Bokeh server via command Line."""
        self.logger.info(
            "Starting Bokeh Server - Close Browser Window to Terminate")
        args_list = ["bokeh", "serve", "--show",
                     str(self.wdir.joinpath("code_py/bokeh_plot.py")),
                     "--args", str(self.bokeh_dir)]

        self.bokeh_server = subprocess.Popen(args_list,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             shell=False)

        self.bokeh_thread = threading.Thread(target=self._output_reader,
                                             args=(self.bokeh_server,))
        self.bokeh_thread.start()

    def stop_server(self):
        """Stop bokeh server."""
        process = psutil.Process(self._bokeh_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
        self.bokeh_thread.join()
