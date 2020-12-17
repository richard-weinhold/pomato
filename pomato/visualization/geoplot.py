"""GroPlot Interface."""

import json
import logging
import re
import subprocess
import threading
from pathlib import Path
import types

import pandas as pd
import psutil
from bokeh.io import save, show

import pomato
import pomato.tools as tools
from pomato.visualization import Visualization
from pomato.visualization.geoplot_static import create_static_plot

class GeoPlot(Visualization):
    """GeoPlot utilizes Bokeh to create an interactive geo plot..

    The GeoPlot module inherits result processing capabilities of Visualization module and in 
    combination with methods of the :class:`~pomato.data.Results` instances creates geo plots.
    
    There are two option to create a bokeh plot: static and dynamic:
        - A static plot will generate a plot based on the provided market result
          with limited interaction. It is either a single timestep or an average
          load for a timeseires.
        - A dynamic plot displays the full market result with possibility to
          change between timestep, N-0 and N-1 line flows and inspection of
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
        the model remains responsive.
    """

    def __init__(self, wdir, data):
        self.logger = logging.getLogger('Log.MarketModel.BokehPlot')
        super().__init__(wdir, data)
        self.bokeh_dir = self.wdir.joinpath("data_temp/bokeh_files")

        # attributes required for dynamic bokeh plot
        self.bokeh_server = None
        self.bokeh_thread = None
        self._bokeh_pid = None

        # make the static plot persistent once created. 
        self.static_plot = None

    def show_plot(self):
        """Show Plot"""
        show(self.static_plot)

    def save_plot(self, filename):
        """Save plot as html file"""
        save(self.static_plot, filename=filename)

    def create_empty_static_plot(self):
        """Create a geo plot without injection or line flows"""
        self.static_plot = create_static_plot(self.result_data_struct(), self.logger, flow_option=2)

    def create_static_plot(self, market_result_name=None, show_prices=False, price_range=None,
                           show_redispatch=True, title=None, plot_dimensions=(700, 800), flow_option=0):
        """Create static bokeh plot of the market results.

        Creates a fairly interactive geographic plot of the provided market
        results. If no market result is supplied the method will check if only one result is 
        instantiated or if only one redispatch result is instantatiated. 
        For redispatch results, the corresponding market results has to be instantiated to visualize 
        redispatch. 

        Parameters
        ----------
        market_result : string, optional
            Name of an instantiated market result.
        show_redispatch : bool, optional
            Include redispatch, this requires the redispatch and market results to be instantiated, by 
            default False
        show_prices : bool, optional
            Include a visual representation of the locational marginal prices, by default False.
        price_range : tuple, optional
            Limit the depcited price range to a subset. This can be useful when only a certain range is of 
            interest. By default None
        flow_option : int, optional
            Lines are colored based on N-0 flows (flow_option = 0), N-1 flows (flow_option = 1) and 
            voltage levels (flow_option = 2), by default 0.
        title : string, optional
            Title depcited on the top the of geo plot, by default None
        plot_dimensions : tuple, optional
            Dimensions of the plot in pixels as a tuple (width, hight), by default (700, 800)
        """
        
        if not market_result_name:
            # One results is instantiated, plot this one.
            if len(self.data.results) == 1:
                market_result = self.data.results[list(self.data.results)[0]]
            # Two results are instantiated, if this is a marketresult/redispatch pair
            # plot the redispatch one
            elif sum(["redispatch" in r for r in self.data.results]) == 1:
                result_name = next(r for r in list(self.data.results) if "redispatch" in r)
                market_result = self.data.results[result_name]
            else:
                raise ValueError("Specify one specific result to plot.")
        else:
            try:
                market_result = self.data.results[market_result_name]
            except KeyError:
                self.logger.error("Specified results not available.")

        geoplot_data_struct = self.create_averaged_result_data(market_result)
        is_redispatch_result = market_result.result_attributes["is_redispatch_result"]
        corresponding_market_result = market_result.result_attributes["corresponding_market_result_name"]
        if is_redispatch_result and bool(corresponding_market_result):
            geoplot_data_struct.gen = market_result.redispatch()
        else: 
            show_redispatch=False

        self.static_plot = create_static_plot(geoplot_data_struct, self.logger,
                                              show_redispatch=show_redispatch, 
                                              show_prices=show_prices,
                                              price_range=price_range,
                                              flow_option=flow_option,
                                              title=title, 
                                              plot_dimensions=plot_dimensions)

    def add_market_result(self, market_result, name):
        """Create data set for bokeh plot from market_result instance."""
        if not market_result.grid:
            self.logger.error("Grid Model not in Results Object!")
            raise Exception('Grid Model not in Results Object!')

        self.logger.info("Processing market model data...")

        data_path = self.bokeh_dir.joinpath("market_result").joinpath(name)
        if not data_path.is_dir():
            data_path.mkdir()
        
        # General GeoPlot data
        geoplot_data_struct = self.create_result_data(market_result)
        #Saving to disk 
        geoplot_data_struct.dclines.to_csv(str(data_path.joinpath('dclines.csv')),
                                          index_label='index')
        geoplot_data_struct.nodes.to_csv(str(data_path.joinpath('nodes.csv')),
                                        index_label='index')
        geoplot_data_struct.lines.to_csv(str(data_path.joinpath('lines.csv')))
        geoplot_data_struct.dc_flow.to_csv(str(data_path.joinpath('f_dc.csv')))
        geoplot_data_struct.inj.to_csv(str(data_path.joinpath('inj.csv')))
        geoplot_data_struct.n_1_flow.to_csv(str(data_path.joinpath('n_1_flows.csv')))
        geoplot_data_struct.n_0_flow.to_csv(str(data_path.joinpath('n_0_flows.csv')))
        gen_by_fuel = geoplot_data_struct.gen.groupby(["t", "fuel", "node"], as_index=False).sum()
        gen_by_fuel.to_csv(str(data_path.joinpath('g_by_fuel.csv')), index_label='index')

        geoplot_data_struct.demand[["n", "t", "demand"]].to_csv(str(data_path.joinpath('demand.csv')),
                                                                index_label='index')

        lodf_data = pd.DataFrame(index=market_result.grid.lines.index,
                                 columns=market_result.grid.lines.index,
                                 data=market_result.grid.lodf)
        lodf_data.to_csv(str(data_path.joinpath('lodf.csv')), index_label='index')

        t_first = market_result.result_attributes["model_horizon"][0]
        t_last = market_result.result_attributes["model_horizon"][-1]
        # convert to int, to be used in the slider widget
        t_dict = {"t_first": int(re.search(r'\d+', t_first).group()),
                  "t_last": int(re.search(r'\d+', t_last).group())}
        with open(data_path.joinpath('t.json'), 'w') as time_frame:
            json.dump(t_dict, time_frame)
        self.logger.info("Market Results %s successfully initialized!", name)

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
                     str(self.package_dir.joinpath("visualization/geoplot_dynamic.py"))]

        self.bokeh_server = subprocess.Popen(args_list,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             shell=False,
                                             cwd=self.bokeh_dir)

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
