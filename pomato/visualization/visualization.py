""""""

import logging
import types
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pomato
import pomato.tools as tools
from plotly.offline import plot
from plotly.subplots import make_subplots


class Visualization():
    """
    The visualisation module of Pomato bundles processing and plotting methods to visualize pomato results.

    Its instantiated with access to DataManagement and in extention to market results instantiated
    into in it. The plotting functionality is implemented with the plotting libraries Bokeh and Plotly. 

    The first is used to create a interactive geo plot in :class:`~pomato.visualization.GeoPlot` 
    the latter to create interactive visualization of capacity, generation and storage usage. 
   
    """

    def __init__(self, wdir, data):
        # Impoort Logger
        self.logger = logging.getLogger('Log.visualization.Visualization')
        self.wdir = wdir
        self.data = data
        self.package_dir = Path(pomato.__path__[0])
    
    def create_generation_plot(self, market_result):
        """Create Generation plot.
        """
        gen = market_result.generation()
        if "tech" in market_result.data.plants.columns:
            gen = pd.merge(gen, market_result.data.plants.tech, left_on="p", right_index=True)
        else:
            gen["tech"] = gen.plant_type
        gen = gen[["fuel", "tech", "t", "G"]].groupby(["fuel", "tech", "t"]).sum().reset_index()
        gen_colors = self.color_map(gen)
        gen = pd.merge(gen, gen_colors, on=["fuel", "tech"])
        gen.loc[:, "G"] *= 1/1000
        fig = px.area(gen, x="t", y="G", color="name", 
                    color_discrete_map=gen_colors[["color", "name"]].set_index("name").color.to_dict())

        fig.layout.xaxis.title="Time"
        fig.layout.yaxis.title="Generation/Load [GW]"

        inf = market_result.infeasibility(drop_zero=False)
        inf["infeasibility"] = inf.pos - inf.neg
        inf = inf[["t", "infeasibility"]].groupby("t").sum()/1000

        d = pd.merge(market_result.demand(), market_result.data.net_export, left_on=["t", "n"], right_on=["timestep", "node"])
        d = d[["t", "demand_el", "D_ph", "D_es", "net_export"]].groupby("t").sum()/1000
        d.loc[:, "demand_el"] -= (d.net_export)
        d = d.loc[market_result.model_horizon, :]

        fig.add_trace(go.Scatter(x=d.index, y=d.demand_el - inf.infeasibility, line=dict(color="#000000"), name="demand")) 
        fig.add_trace(go.Scatter(x=d.index, y=d.demand_el - inf.infeasibility + d.D_es + d.D_ph, fill='tonexty', mode= 'none', fillcolor="#BFBDE5", name="storage charging"))

        plot(fig)

    def create_installed_capacity_plot(self, market_result):
        """Create plot visualizing installed capacity per market area."""
        plants = market_result.data.plants
        plants["zone"] = market_result.data.nodes.loc[plants.node, "zone"].values
        if not "tech" in plants.columns:
            plants["tech"] = plants.plant_type

        plants = (plants[["tech", "fuel", "zone", "g_max"]].groupby(["zone", "tech", "fuel"]).sum()/1000).reset_index()
        plant_colors = self.color_map(plants)
        plants = pd.merge(plants, plant_colors, on=["fuel", "tech"])
        fig = px.bar(plants, x="zone", y="g_max", color="name", 
                    color_discrete_map=plant_colors[["color", "name"]].set_index("name").color.to_dict())

        fig.layout.yaxis.title="Installed Capacity [GW]"
        fig.layout.xaxis.title="Zone/Country"
        plot(fig)

    def create_storage_plot(self, market_result):
        """Storage plot."""
        es_gen = market_result.storage_generation()
        
        if es_gen.empty:
            self.logger.warning("No storage results to plot.")
            return None

        es_gen = es_gen[["t", "G", "D_es", "L_es"]].groupby(["t"]).sum().reset_index()
        fig = px.line(pd.melt(es_gen, id_vars=["t"], value_vars=["G", "D_es"]), x="t", y="value", color='variable')
        fig2 = px.line(pd.melt(es_gen, id_vars=["t"], value_vars=["L_es"]), x="t", y="value", color='variable')
        fig2.update_traces(yaxis="y2")
        # Create figure with secondary y-axis
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        subfig.add_traces(fig.data + fig2.data)
        subfig.layout.xaxis.title="Time"
        subfig.layout.yaxis.title="Storage Charging (D_es)/Dischargin (G) [MW]"
        subfig.layout.yaxis2.title="Storage Level [MWh]"
        subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
        plot(subfig)

    def result_data_struct(self):
        """Data struct, as a standart template for result processing.
        
        Returns
        -------
        result_data, types.SimpleNamespace
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
    
    def create_result_data(self, market_result):
        """Creates result data struct from results supplied as market_result.

        Based on :meth:`~result_data_struct`this method fills the data struct with data and results
        from the market result specified which is an instance of :class:`~pomato.data.Results`.
        This data struct is intended for the generation of visualizations of result in e.g. the
        dynamic geoplot.

        Parameters
        ----------
        market_result : :class:`~pomato.data.Results`
            Market result which gets subsumed into the predefined data struct.
        """
        data_struct = self.result_data_struct()
        data_struct.lines = market_result.data.lines
        data_struct.nodes = market_result.data.nodes
        data_struct.dclines = market_result.data.dclines
        data_struct.inj = market_result.INJ
        data_struct.dc_flow = market_result.F_DC
        data_struct.gen = market_result.generation()
        data_struct.demand = market_result.demand()
        data_struct.n_0_flow = market_result.n_0_flow()
        data_struct.n_1_flow = market_result.absolute_max_n_1_flow(sensitivity=0.1)
        data_struct.prices = market_result.price()

        return data_struct

    def create_averaged_result_data(self, market_result):
        """Creates averaged result data struct from results supplied as market_result.

        Based on :meth:`~result_data_struct` and  :meth:`~create_result_data` this method fills 
        the data struct with data and results from the market result specified which is an 
        instance of :class:`~pomato.data.Results`. All results are averaged in useful ways. This 
        data struct is intended for the static geoplot, which visualizes the results in 
        average flows, injections, generation and prices. 

        Parameters
        ----------
        market_result : :class:`~pomato.data.Results`
            Market result which gets subsumed into the predefined data struct.
        """

        data_struct = self.create_result_data(market_result)

        data_struct.inj = data_struct.inj.groupby("n").mean().reindex(market_result.grid.nodes.index).INJ
        data_struct.n_0_flow = data_struct.n_0_flow.abs().mean(axis=1)
        data_struct.n_1_flow = data_struct.n_1_flow.abs().mean(axis=1)
        data_struct.dc_flow = data_struct.dc_flow.pivot(index="dc", columns="t", values="F_DC") \
                                .abs().mean(axis=1).reindex(market_result.data.dclines.index).fillna(0)
        
        data_struct.prices = data_struct.prices[["n", "marginal"]].groupby("n").mean()

        return data_struct

    def color_map(self, gen):
        """Fuel colors for generation/capacity plots."""

        basic_fuel_color_map = {
            "coal": ["#8C7F77", "#73645D", "#5E4F48"],
            "lignite": ["#BC8873", "#A67662", "#895B4B"],
            "gas": ["#FA5827", "#E44214", "#C62200"],
            "uran": ["#C44B50", "#7B011A", "#9C242E"],
            "oil": ["#565752", "#282924"],
            "waste": ["#000000", "#141510"],
            "wind": ["#6699AA", "#00303E", "#326775"],
            "sun": ["#FFEB3C", "#F4E01D", "#EBD601"],
            "hydro": ["#6881E5", "#5472D4", "#2E56B5"],
            "water": ["#6881E5", "#5472D4", "#2E56B5"],
            "geothermal": ["#FDCF94", "#5472D4", "#2E56B5"],
            "biomass": ["#AEE570", "#96CD58", "#7DB343"],
            "other": ["#87959E", "#CCDAE3", "#B6C9D0", "#235E58", "#013531"],
        }

        color_df = gen[["fuel", "tech"]].groupby(["fuel", "tech"]).sum()
        color_df["color"] = ""
        for fuel in basic_fuel_color_map:
            condition = color_df.index.get_level_values("fuel").str.lower().str.contains(fuel)
            number_of_fuels = sum(condition)
            color_df.loc[condition, "color"] = basic_fuel_color_map[fuel][:number_of_fuels]
        
        fill = "#EAF9FE"
        number_to_fill = len(basic_fuel_color_map["other"]) - sum(color_df.color == "") 
        if number_to_fill > 0:
            basic_fuel_color_map["other"].extend([fill for x in range(0, number_to_fill)])
        color_df.loc[color_df.color == "", "color"] = basic_fuel_color_map[fuel][:sum(color_df.color == "")]

        color_df = color_df.reset_index()
        color_df["name"] = ""
        for fuel in color_df.fuel.unique():
            condition = color_df.fuel == fuel
            if sum(condition) > 1:
                color_df.loc[condition, "name"] = color_df[condition].fuel + " " + color_df[condition].tech
            else:
                color_df.loc[condition, "name"]  = color_df[condition].fuel
        color_df["name"] = color_df["name"].apply(tools.remove_duplicate_words_string)
        return color_df






