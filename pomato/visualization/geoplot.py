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
from pomato.visualization.geoplot_static import create_static_plot

class GeoPlot():
    """Interface market data and the creation of a geographic plot.

    There are two option to create a bokeh plot: static and dynamic.

    A static plot will generate a plot based on the provided market result
    with limited interaction. It is either a single timestep or an average
    load for a timeseires.
    A dynamic plot displays the full market result with possibility to
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
        # Impoort Logger
        self.logger = logging.getLogger('Log.MarketModel.BokehPlot')

        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])
        self.bokeh_dir = wdir.joinpath("data_temp/bokeh_files")
        self.data = data
        
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
        self.static_plot = create_static_plot(self.geoplot_data_struct(), flow_option=2)

    def create_static_plot(self, market_result_name=None, show_prices=True, price_range=None,
                           show_redispatch=False, title=None, plot_dimensions=(700, 800), flow_option=0):
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
                market_result = self.data.results[self.data.results.keys()[0]]
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

        geoplot_data_struct = self.create_mean_geoplot_data(market_result)
        # If the result is a redispatch result and the corresponding market result is instantiated
        # calculate redispatch by the delta between market and redispatch. 
        is_redispatch_result = market_result.result_attributes["is_redispatch_result"]
        corresponding_market_result = market_result.result_attributes["corresponding_market_result_name"]
        if is_redispatch_result and bool(corresponding_market_result):
            # Redispatch Calculation G_redispatch - G_market
            geoplot_data_struct.gen = pd.merge(self.data.results[corresponding_market_result].G, 
                                               geoplot_data_struct.gen, 
                                               on=["p", "t"], 
                                               suffixes=("_market", "_redispatch"))
            geoplot_data_struct.gen["delta"] = geoplot_data_struct.gen["G_redispatch"] - geoplot_data_struct.gen["G_market"]
            geoplot_data_struct.gen["delta_abs"] = geoplot_data_struct.gen["delta"].abs()
        else: 
            show_redispatch=False

        self.static_plot = create_static_plot(geoplot_data_struct, self.logger,
                                              show_redispatch=show_redispatch, 
                                              show_prices=show_prices,
                                              price_range=price_range,
                                              flow_option=flow_option,
                                              title=title, 
                                              plot_dimensions=plot_dimensions)

    def geoplot_data_struct(self):
        """Data struct for geoplot data.
        
        Returns
        -------
        geoplot_data, types.SimpleNamespace
            Returns empty data struct, with predefined data structure. 
        """        

        return types.SimpleNamespace(nodes=self.data.nodes,
                                     lines=self.data.lines,
                                     dclines=self.data.dclines,
                                     inj=pd.Series(index=self.data.nodes.index, data=0),
                                     dc_flow= pd.Series(index=self.data.dclines.index, data=0),
                                     gen=pd.DataFrame(),
                                     demand=pd.DataFrame(),
                                     prices=pd.DataFrame(),
                                     n_0_flow=pd.Series(index=self.data.lines.index, data=0),
                                     n_1_flow=pd.Series(index=self.data.lines.index, data=0))


    def create_mean_geoplot_data(self, market_result):
        """Creates geoplot data struct from results supplied as market_result.

        Based on :meth:`~geoplot_data_struct` and  :meth:`~create_geoplot_data` this method fills 
        the data struct with data and results from the market result specified which is an 
        instance of :class:`~pomato.data.Results`. This data struct is intended for the 
        static geoplot, which visualizes the results in average flows, injections, generation and 
        prices. 

        Parameters
        ----------
        market_result : :class:`~pomato.data.Results`
            Market result which gets subsumed into the predefined data struct which is used in the 
            geoplot functionality.
        """
        data_struct = self.create_geoplot_data(market_result)

        data_struct.inj = data_struct.inj.groupby("n").mean().reindex(market_result.grid.nodes.index).INJ
        data_struct.n_0_flow = data_struct.n_0_flow.abs().mean(axis=1)
        data_struct.n_1_flow = data_struct.n_1_flow.abs().mean(axis=1)
        data_struct.dc_flow = data_struct.dc_flow.pivot(index="dc", columns="t", values="F_DC") \
                                .abs().mean(axis=1).reindex(market_result.data.dclines.index).fillna(0)
        
        data_struct.prices = data_struct.prices[["n", "marginal"]].groupby("n").mean()

        return data_struct

    def create_geoplot_data(self, market_result):
        """Creates geoplot data struct from results supplied as market_result.

        Based on :meth:`~geoplot_data_struct`this method fills the data struct with data and results
        from the market result specified which is an instance of :class:`~pomato.data.Results`.
        This data struct is intended for the dynamic geoplot, which visualizes all timesteps of the 
        result.

        Parameters
        ----------
        market_result : :class:`~pomato.data.Results`
            Market result which gets subsumed into the predefined data struct which is used in the 
            geoplot functionality.
        """
        data_struct = self.geoplot_data_struct()
        data_struct.lines = market_result.data.lines
        data_struct.nodes = market_result.data.nodes
        data_struct.dclines = market_result.data.dclines
        data_struct.inj = market_result.INJ
        data_struct.dc_flow = market_result.F_DC
                                
        data_struct.gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "node"]],
                                   market_result.G, left_index=True, right_on="p")
        data_struct.demand = self.process_demand_data(market_result)
        data_struct.n_0_flow = market_result.n_0_flow()
        data_struct.n_1_flow = market_result.absolute_max_n_1_flow()
        data_struct.prices = market_result.price()

        return data_struct

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
        geoplot_data_struct = self.create_geoplot_data(market_result)
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

        geoplot_data_struct.demand.to_csv(str(data_path.joinpath('demand.csv')),
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

    def process_generation_data(self, market_result):
        """Prepare Generation Data"""
        # Save relevant variables from market result
        gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "node"]],
                        market_result.G, left_index=True, right_on="p")
        return gen

    def process_demand_data(self, market_result):
        """Process total nodal demand composed of load data and market results of storage/heatpump usage."""

        map_pn = market_result.data.plants.node.reset_index()
        map_pn.columns = ['p', 'n']
        demand = market_result.data.demand_el.copy()
        demand.rename(columns={"node": "n", "timestep": "t", "demand_el": "d_el"}, inplace=True)
        if not market_result.D_ph.empty:
            demand_ph = pd.merge(market_result.D_ph, map_pn[["p", "n"]], 
                                 how="left", on="p").groupby(["n", "t"], as_index=False).sum()
            demand = pd.merge(demand, demand_ph[["D_ph", "n", "t"]], how="outer", on=["n", "t"])
        else:
            demand["D_ph"] = 0
        if not market_result.D_es.empty:
            demand_es = pd.merge(market_result.D_es, map_pn[["p", "n"]], 
                                 how="left", on="p").groupby(["n", "t"], as_index=False).sum()
            demand = pd.merge(demand, demand_es[["D_es", "n", "t"]], how="outer", on=["n", "t"])
        else:
            demand["D_es"] = 0
        demand.fillna(value=0, inplace=True)
        demand["d_total"] = demand.d_el + demand.D_ph + demand.D_es
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
