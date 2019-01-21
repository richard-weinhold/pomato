"""
BOKEH Plot Interface -
populate Bokeh instace data for multiple model runs
start and stop bokeh plot through threading
"""

import re
import logging
import pickle
import json
import subprocess
import threading
import datetime as dt
import pandas as pd
import psutil
import tools


class BokehPlot(object):
    """interface market data and bokeh plot, init all data then run the server from cmd"""
    def __init__(self, wdir):
        # Impoort Logger
        self.logger = logging.getLogger('Log.MarketModel.BokehPlot')

        self.wdir = wdir
        self.bokeh_dir = wdir.joinpath("data_temp/bokeh_files")
        # Make sure all folders exist
        tools.create_folder_structure(self.wdir, self.logger)

        # predefine attributes
        self.bokeh_server = None
        self.bokeh_thread = None
        self.bokeh_pid = None

    def add_market_result(self, market_result, name):
        """create data set for bokeh plot from julia market_result-object with the associated grid model"""
        if not market_result.grid:
            self.logger.error("Grid Model not in Results Object!")
            raise

        self.logger.info("Processing market model data...")

        data_path = self.bokeh_dir.joinpath("market_result").joinpath(name)
        if not data_path.is_dir():
            data_path.mkdir()

        # market_result.data.zones.to_csv(str(folder.joinpath('zones.csv')), index_label='index')
        # market_result.data.tech.to_csv(str(folder.joinpath('tech.csv')))
        market_result.data.fuel.to_csv(str(data_path.joinpath('fuel.csv')), index_label='index')
        market_result.data.dclines.to_csv(str(data_path.joinpath('dclines.csv')), index_label='index')
        market_result.grid.nodes.to_csv(str(data_path.joinpath('nodes.csv')), index_label='index')
        market_result.grid.lines.to_csv(str(data_path.joinpath('lines.csv')))
        market_result.F_DC.to_csv(str(data_path.joinpath('F_DC.csv')))
        market_result.INJ.to_csv(str(data_path.joinpath('INJ.csv')))
        
        n_0_flows, n_1_flows = self.process_grid_data(market_result)
        n_1_flows.to_csv(str(data_path.joinpath('n_1_flows.csv')))
        n_0_flows.to_csv(str(data_path.joinpath('n_0_flows.csv')))

        generation_by_fuel = self.process_generation_data(market_result)
        generation_by_fuel.to_csv(str(data_path.joinpath('g_by_fuel.csv')), index_label='index')

        demand = self.process_demand_data(market_result)
        demand.to_csv(str(data_path.joinpath('demand.csv')), index_label='index')
        

        t_first = market_result.data.result_attributes["model_horizon"][0]
        t_last = market_result.data.result_attributes["model_horizon"][-1]

        # convert to int, bc of the slider widget
        t_dict = {"t_first": int(re.search(r'\d+', t_first).group()),
                  "t_last": int(re.search(r'\d+', t_last).group())}
        with open(data_path.joinpath('t.json'), 'w') as time_frame:
            json.dump(t_dict, time_frame)

        self.logger.info(f"Market Results {name} successfully initialized!")

    def process_grid_data(self, market_result):
        """precalculatting the line flows for the bokeh plot"""
        self.logger.info("Precalculatting line flows and saving them to file...")
        n_0_flows = market_result.n_0_flow()
        n_1_flows = market_result.n_1_flow()

        # convert n_1 flows to the max of N-1 flows
        time = market_result.data.result_attributes["model_horizon"]
        n_1_max = n_1_flows.copy()
        n_1_max[time] = n_1_max[time].abs()
        n_1_max = n_1_max.groupby("cb").max().reset_index()
        n_1_flows = pd.merge(n_1_max[["cb", "co"]],
                             n_1_flows, on=["cb", "co"],
                             how="left").drop("co", axis=1)

        self.logger.info("Done!")
        return n_0_flows, n_1_flows.set_index("cb").reindex(market_result.grid.lines.index)
    
    def process_generation_data(self, market_result):

        generation = market_result.G
        ## Save relevant variables from market result
        generation = pd.merge(generation, market_result.data.plants[["node", "fuel"]],
                              how="left", left_on="p", right_index=True)
        generation_by_fuel = generation.groupby(["t", "fuel", "node"], as_index=False).sum()
        return generation_by_fuel

    def process_demand_data(self, market_result):
        """bring the data from julia/gams in the right format and store it"""
        
        map_pn = market_result.data.plants.node.reset_index()
        map_pn.columns = ['p', 'n']

        demand = market_result.data.demand_el.unstack().reset_index()
        demand.columns = ["n", "t", "d_el"]
        demand_d = market_result.D_d
        demand_ph = market_result.D_ph
        demand_es = market_result.D_es

        if not demand_d.empty:
            demand_d = pd.merge(demand_d, map_pn[["p", "n"]], how="left", left_on="d", right_on="p")
            demand_d = demand_d.groupby(["n", "t"], as_index=False).sum()
        else:
            demand_d = pd.DataFrame(columns=["d", "t", "D_d"])
            demand_d = pd.merge(demand_d, map_pn[["p", "n"]], how="left", left_on="d", right_on="p")

        if not demand_ph.empty:
            demand_ph = pd.merge(demand_ph, map_pn[["p", "n"]], how="left", left_on="ph",
                                 right_on="p")
            demand_ph = demand_ph.groupby(["n", "t"], as_index=False).sum()
        else:
            demand_ph = pd.DataFrame(columns=["ph", "t", "D_ph"])
            demand_ph = pd.merge(demand_ph, map_pn[["p", "n"]], how="left", left_on="ph",
                                 right_on="p")

        if not demand_es.empty:
            demand_es = pd.merge(demand_es, map_pn[["p", "n"]], how="left", left_on="es",
                                 right_on="p")
            demand_es = demand_es.groupby(["n", "t"], as_index=False).sum()
        else:
            demand_es = pd.DataFrame(columns=["es", "t", "D_es"])
            demand_es = pd.merge(demand_es, map_pn[["p", "n"]], how="left", left_on="es",
                                 right_on="p")

        demand = pd.merge(demand, demand_d[["D_d", "n", "t"]], how="outer", on=["n", "t"])
        demand = pd.merge(demand, demand_ph[["D_ph", "n", "t"]], how="outer", on=["n", "t"])
        demand = pd.merge(demand, demand_es[["D_es", "n", "t"]], how="outer", on=["n", "t"])
        demand.fillna(value=0, inplace=True)
        demand["d_total"] = demand.d_el + demand.D_d + demand.D_ph + demand.D_es
        
        return demand[["n", "t", "d_total"]]

    def output_reader(self, proc):
        """helper function to print stdout to console"""
        for line in iter(proc.stdout.readline, b''):
            bokeh_output = '{0}'.format(line.decode('utf-8')).strip()
            self.logger.info('bokeh: ' + bokeh_output)

            # listen to output and save bokeh pid
            if "Starting Bokeh server with process id" in bokeh_output:
                self.bokeh_pid = int(bokeh_output.split()[-1])
                print("HERE:", self.bokeh_pid)
            # listen to output and stop server if websocket is closed
            kill_keywords = ['code=1001', 'WebSocket connection closed']
            if any(k in bokeh_output for k in kill_keywords):
                self.stop_server()

    def start_server(self):
        """Run the Bokeh server via command Line"""
        self.logger.info("Starting Bokeh Server - Close Browser Window to Terminate")
        args_list = ["bokeh", "serve", "--show", str(self.wdir.joinpath("code_py/bokeh_plot.py")), "--args",
                     str(self.bokeh_dir)]

        self.bokeh_server = subprocess.Popen(args_list,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             shell=False
                                             )

        self.bokeh_thread = threading.Thread(target=self.output_reader,
                                             args=(self.bokeh_server,))
        self.bokeh_thread.start()

    def stop_server(self):
        """ stop bokeh server"""
        process = psutil.Process(self.bokeh_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
        self.bokeh_thread.join()
