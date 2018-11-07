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

class BokehPlot(object):
    """interface market data and bokeh plot, init all data then run the server from cmd"""
    def __init__(self, wdir, data):
        # Impoort Logger
        self.logger = logging.getLogger('Log.MarketModel.BokehPlot')

        self.wdir = wdir
        self.bokeh_dir = wdir.joinpath("data_temp/bokeh_files")
        ## Store Base Data
        self.create_folders()
        data.plants.to_csv(str(self.bokeh_dir.joinpath('data').joinpath('plants.csv')), index_label='index')
        data.nodes.to_csv(str(self.bokeh_dir.joinpath('data').joinpath('nodes.csv')), index_label='index')
        data.zones.to_csv(str(self.bokeh_dir.joinpath('data').joinpath('zones.csv')), index_label='index')
        data.tech.to_csv(str(self.bokeh_dir.joinpath('data').joinpath('tech.csv')))
        data.fuel.to_csv(str(self.bokeh_dir.joinpath('data').joinpath('fuel.csv')), index_label='index')
        data.dclines.to_csv(str(self.bokeh_dir.joinpath('data').joinpath('dclines.csv')), index_label='index')
        
        # Make data available to methods
        self.plants = data.plants

        # predefine attributes
        self.bokeh_server = None
        self.bokeh_thread = None
        self.bokeh_pid = None

    def create_folders(self):
        """ create folders for bokeh interface"""
        if not self.wdir.joinpath("data_temp/bokeh_files").is_dir():
            self.wdir.joinpath("data_temp/bokeh_files").mkdir()
        if not self.bokeh_dir.joinpath("market_result").is_dir():
            self.bokeh_dir.joinpath("market_result").mkdir()
        if not self.bokeh_dir.joinpath("data").is_dir():
            self.bokeh_dir.joinpath("data").mkdir()

    def add_market_result(self, market_result, grid_model, name):
        """create data set for bokeh plot from julia market_result-object with the associated grid model"""
        generation = market_result.return_results("G")

        demand = market_result.return_results("d_el")
        demand_d = market_result.return_results("D_d")
        demand_ph = market_result.return_results("D_ph")
        demand_es = market_result.return_results("D_es")

        t_first = market_result.model_horizon[0]
        t_last = market_result.model_horizon[-1]
        # convert to int, bc of the slider widget
        t_dict = {"t_first": int(re.search(r'\d+', t_first).group()),
                  "t_last": int(re.search(r'\d+', t_last).group())}

        inj = market_result.return_results("INJ")
        f_dc = market_result.return_results("F_DC")

        data_path = self.bokeh_dir.joinpath("market_result").joinpath(name)
        if not data_path.is_dir():
            data_path.mkdir()

        self.process_market_data(generation, 
                                 demand, demand_d, demand_ph, demand_es,
                                 t_dict, inj, f_dc, data_path, name)

        self.process_grid_data(grid_model, inj, data_path)

    def process_grid_data(self, grid_model, inj, data_path):
        """precalculatting the line flows for the bokeh plot"""
        self.logger.info("Precalculatting line flows and saving them to file...")
        n_1_flows = grid_model.n_1_flows_timeseries(inj)
        n_0_flows = grid_model.n_0_flows_timeseries(inj)
        n_1_flows.to_csv(str(data_path.joinpath('n_1_flows.csv')))
        n_0_flows.to_csv(str(data_path.joinpath('n_0_flows.csv')))
        grid_model.lines.to_csv(str(data_path.joinpath('lines.csv')))
        self.logger.info("Done!")

    def process_market_data(self, generation, demand, demand_d, demand_ph, demand_es,
                            t_dict, inj, f_dc, data_path, name):
        """bring the data from julia/gams in the right format and store it"""
        self.logger.info("Processing market model data...")

        ## Save relevant variables from market result
        generation = pd.merge(generation, self.plants[["node", "fuel"]],
                              how="left", left_on="p", right_index=True)
        g_by_fuel = generation.groupby(["t", "fuel", "node"], as_index=False).sum()
#        g_by_fuel = g_by_fuel.drop("p", axis=1)

        map_pn = self.plants.node.reset_index()
        map_pn.columns = ['p', 'n']

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
        demand = demand[["n", "t", "d_total"]]

        demand.to_csv(str(data_path.joinpath('demand.csv')), index_label='index')
        g_by_fuel.to_csv(str(data_path.joinpath('g_by_fuel.csv')), index_label='index')
        f_dc.to_csv(str(data_path.joinpath('F_DC.csv')))
        inj.to_csv(str(data_path.joinpath('INJ.csv')))

        with open(data_path.joinpath('t.json'), 'w') as time_frame:
            json.dump(t_dict, time_frame)
        self.logger.info("Done!")

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
