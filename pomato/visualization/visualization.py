"""Visualization Module of POMATO"""
# pylint: disable-msg=E1101

import logging
import types
from pathlib import Path

import matplotlib
import io
from PIL import Image

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pomato
import pomato.tools as tools
from matplotlib import cm
from plotly.offline import plot
from plotly.subplots import make_subplots
from pomato.visualization.geoplot_functions import line_coordinates, line_colors, add_prices_layer
from pomato.visualization.dashboard import Dashboard

def color_map(gen):
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

    color_df = gen[["fuel", "technology"]].groupby(["fuel", "technology"]).sum()
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
            color_df.loc[condition, "name"] = color_df[condition].fuel + " " + color_df[condition].technology
        else:
            color_df.loc[condition, "name"]  = color_df[condition].fuel
    color_df["name"] = color_df["name"].apply(tools.remove_duplicate_words_string)
    return color_df

class Visualization():
    """
    The visualization module of Pomato bundles processing and plotting methods to visualize pomato results.

    Its instantiated with access to DataManagement and in extention to market results instantiated
    into in it. The plotting functionality is implemented with the plotting library Plotly.    
    """
    def __init__(self, wdir, data):
        # Impoort Logger
        self.logger = logging.getLogger('Log.visualization.Visualization')
        self.wdir = wdir
        self.data = data

    def create_geo_plot(self, market_result, show_redispatch=False, show_prices=False, 
                        timestep=None, line_color_option=0, show_plot=True, filepath=None):
        """Creates Geoplot of market result.

        The geoplot is a interactive plotly figure showing lineloading, redispatch and prices 
        depending on optional arguments. 

        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement`
            Market result which is plotted. 
        show_redispatch : bool, optional
            Include redispatch, this requires the *market_result* argument to be the redispatch 
            result and the corresponding  market results to be instantiated. Defaults to False. 
        show_prices : bool, optional
            Include a visual representation of the locational marginal prices, by default False.
        timestep : int, string, optional
            Timestep for which the load, redispatch or price is plotted. Can be timestep index or
            timestep string identifier. If None, average values are presented. Defaults to None.
        line_color_option : int, optional
            Lines are colored based on N-0 flows (flow_option = 0), N-1 flows (flow_option = 1) and 
            voltage levels (flow_option = 2), by default 0.
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """    

        dclines = market_result.data.dclines.copy()
        lines = market_result.data.lines.copy()
        nodes = market_result.data.nodes.copy()
        if isinstance(timestep, int):
            timestep = market_result.model_horizon[timestep]
        if show_redispatch: 
            gen = market_result.redispatch()  
            if isinstance(timestep, str) and isinstance(gen, pd.DataFrame):
                gen = gen.loc[gen.t == timestep, ["node", "delta", "delta_abs"]].groupby("node").sum()
                gen = pd.merge(nodes["zone"], gen, how="left", left_index=True, right_index=True).fillna(0)
            elif isinstance(gen, pd.DataFrame):
                gen = gen.loc[:, ["node", "delta", "delta_abs"]].groupby("node").sum()
                gen = pd.merge(nodes["zone"], gen, how="left", left_index=True, right_index=True).fillna(0)
            else:
                show_redispatch = False

        if isinstance(timestep, str):
            result_data = market_result.create_result_data()
            n_0_flows = result_data.n_0_flow[timestep]
            dcline_flow = result_data.dc_flow.loc[result_data.dc_flow.t == timestep].set_index("dc").F_DC.reindex(dclines.index)
            prices = result_data.prices[result_data.prices.t == timestep].groupby("n").mean()
            n_1_flows = result_data.n_1_flow[timestep]
        else:
            result_data = market_result.create_averaged_result_data()
            n_0_flows = result_data.n_0_flow
            dcline_flow = result_data.dc_flow
            prices = result_data.prices
            n_1_flows = result_data.n_1_flow

        fig = go.Figure()
        dcline_coords = np.array(result_data.dcline_coordinates)
        lons, lats = [], []
        customdata = None
        for dcline in dclines.index:
            i = dclines.index.get_loc(dcline)
            lats.extend(dcline_coords[0][i, :])
            lats.append(None)
            lons.extend(dcline_coords[1][i, :])
            lons.append(None)
            data = [dcline, dclines.loc[dcline, "capacity"], dcline_flow.loc[dcline]]
            if isinstance(customdata, np.ndarray):
                customdata = np.vstack([customdata,
                                        [data for x in range(0, len(dcline_coords[1][i]))],
                                        [None for x in range(0, len(data))]])
            else:
                customdata = np.vstack([[data for x in range(0, len(dcline_coords[1][i]))],
                                        [None for x in range(0, len(data))]])

        hovertemplate_dclines = "<br>".join(["Line: %{customdata[0]}", 
                                             "Capcity: %{customdata[1]:.2f} MW",
                                             "Flow %{customdata[2]:.2f} MW"]) + "<extra></extra>"
        fig.add_trace(
            go.Scattermapbox(
                lon = lons,
                lat = lats,
                mode = 'lines',
                line = dict(width = 2, color="#1F77B4"),
                opacity = 0.4,
                customdata=customdata,
                hovertemplate=hovertemplate_dclines
            )
        )

        line_coords = np.array(result_data.line_coordinates)
        lines["colors"], lines["alpha"] = line_colors(lines, n_0_flows, 
                                                      n_1_flows, option=line_color_option)
        hovertemplate_lines = "<br>".join(["Line: %{customdata[0]}", 
                                           "Capcity: %{customdata[1]:.2f} MW",
                                           "N-0 Flow %{customdata[2]:.2f} MW",
                                           "N-1 Flow %{customdata[3]:.2f} MW"]) + "<extra></extra>"
        # Add Lines for each color
        for c in lines.colors.unique():
            tmp_lines = lines[lines.colors == c]
            # tmp_lines_idx = [lines.index.get_loc(line) for line in tmp_lines.index]
            alpha = lines.alpha.unique()[0]
            lons, lats = [], []
            customdata = None
            for line in tmp_lines.index:
                i = lines.index.get_loc(line)
                lats.extend(line_coords[0][i, :])
                lats.append(None)
                lons.extend(line_coords[1][i, :])
                lons.append(None)
                data = [line, lines.loc[line, "capacity"], n_0_flows.loc[line], n_1_flows.loc[line]]
                if isinstance(customdata, np.ndarray):
                    customdata = np.vstack([customdata,
                                            [data for x in range(0, len(line_coords[1][i]))],
                                            [None for x in range(0, len(data))]])
                else:
                    customdata = np.vstack([[data for x in range(0, len(line_coords[1][i]))],
                                            [None for x in range(0, len(data))]])

            fig.add_trace(
                go.Scattermapbox(
                    lon = lons,
                    lat = lats,
                    mode = 'lines',
                    line = dict(width = 2, color=c),
                    opacity = alpha,
                    customdata=customdata,
                    hovertemplate=hovertemplate_lines
                    )
                )
            
        if show_redispatch:
            gen.loc[gen.delta_abs == 0, "delta_abs"] = 1e-3
            sizeref = max(2*max(gen['delta_abs'])/12**2, 1)
            for condition, color in zip([gen.delta < 0, gen.delta == 0, gen.delta > 0], ["red", "#3283FE", "green"]):
                markers = go.scattermapbox.Marker(
                    size = gen.loc[condition, 'delta_abs'],
                    sizeref = sizeref,
                    sizemin = 3,
                    color = color,
                    # line = {"color": 'rgb(40,40,40)'},
                    # line_width=0.5,
                    autocolorscale=True
                    )
                customdata = gen.loc[condition, ["zone", "delta_abs"]].reset_index()
                fig.add_trace(go.Scattermapbox(
                            lon = nodes.loc[condition, 'lon'],
                            lat = nodes.loc[condition, 'lat'],
                            marker = markers,
                            customdata=customdata,
                            hovertemplate=
                            "<br>".join([
                                "Node: %{customdata[0]}",
                                "Zone: %{customdata[1]}",
                                "Redispatch: %{customdata[2]:.2f} MW",
                            ]) + "<extra></extra>"
                        ))
        else:
            fig.add_trace(go.Scattermapbox(
                        lon = nodes.lon,
                        lat = nodes.lat,
                        mode = 'markers',
                        marker = go.scattermapbox.Marker(
                            color = "#3283FE",
                            opacity=0.4
                            ), 
                        customdata=nodes[["zone"]].reset_index(),
                        hovertemplate=
                        "<br>".join([
                            "Node: %{customdata[0]}",
                            "Zone: %{customdata[1]}",
                        ]) + "<extra></extra>"
                    ))

        if show_prices:
            prices_layer, coordinates, hight_width = add_prices_layer(nodes, prices)
            price_fig = go.Figure(
                data=go.Contour(z=prices_layer, showscale=False, 
                                colorscale="Viridis", ncontours=10))
            price_fig.update_layout(
                width=2e3, height=2e3*hight_width, 
                xaxis = {'visible': False},
                yaxis = {'visible': False},
                margin={"r":0,"t":0,"l":0,"b":0})
            
            img_pil = Image.open(io.BytesIO(price_fig.to_image()))
            price_layer =  {   
                    "sourcetype": "image",
                    "source": img_pil,
                    "coordinates": coordinates,
                    "opacity": 0.2,
                }
            # Price Colorbar
            price_colorbar = go.Scatter(x=[None],y=[None],
                                        mode='markers',
                                        marker=dict(
                                            colorscale="Viridis", 
                                            showscale=True,
                                            cmin=prices_layer.min(),
                                            cmax=prices_layer.max(),
                                            colorbar=dict(thickness=5)
                                        ), hoverinfo='none')
            fig.add_trace(price_colorbar)
        else:
            price_layer = {}
            lines_colorbar = go.Scatter(x=[None],y=[None],
                                        mode='markers',
                                        marker=dict(
                                            colorscale="RdYlGn", 
                                            reversescale=True,
                                            showscale=True,
                                            cmin=0,
                                            cmax=1,
                                            colorbar=dict(thickness=5)
                                        ), hoverinfo='none')
            fig.add_trace(lines_colorbar)

        center = {
            'lon': round((max(nodes.lon) + min(nodes.lon)) / 2, 6),
            'lat': round((max(nodes.lat) + min(nodes.lat)) / 2, 6)
            }
        fig.update_layout(    
            showlegend = False,
            margin={"r":0,"t":0,"l":0,"b":0},
            mapbox= {"style": "carto-positron",
                    "layers": [price_layer],
                    "zoom": 4,
                    "center": center},
            xaxis = {'visible': False},
            yaxis = {'visible': False})

        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_generation_plot(self, market_result, nodes=None, show_plot=True, filepath=None):
        """Create interactive generation plot.

        The generation plot shows generation by fuel/type/technology over the model horizon for the 
        specified market result and optionally subset of nodes. The figure is generated using the 
        plotly package and can be returned or saved depending on the input arguments.
        
        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement`
            Market result which is plotted. 
        nodes: list, optional,
            Show only generation at the supplied subset of nodes. By default None, which shows all
            generation. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """

        gen = market_result.generation()
        demand = market_result.demand()
        inf = market_result.infeasibility(drop_zero=False)
        net_export = market_result.data.net_export

        if isinstance(nodes, list):
            gen = gen[gen.node.isin(nodes)]
            demand = demand[demand.n.isin(nodes)]
            inf = inf[inf.n.isin(nodes)]
            net_export = net_export[net_export.node.isin(nodes)]

        if gen.empty:
            return go.Figure()

        gen = gen[["fuel", "technology", "t", "G"]].groupby(["fuel", "technology", "t"]).sum().reset_index()
        gen_colors = color_map(gen)
        gen = pd.merge(gen, gen_colors, on=["fuel", "technology"])
        gen.loc[:, "G"] *= 1/1000
        fig = px.area(gen, x="t", y="G", color="name", 
                      color_discrete_map=gen_colors[["color", "name"]].set_index("name").color.to_dict())

        fig.layout.xaxis.title="Time"
        fig.layout.yaxis.title="Generation/Load [GW]"
        
        inf["infeasibility"] = inf.pos - inf.neg
        inf = inf[["t", "infeasibility"]].groupby("t").sum()/1000

        d = pd.merge(demand, net_export, left_on=["t", "n"], right_on=["timestep", "node"])
        d = d[["t", "demand_el", "D_ph", "D_es", "net_export"]].groupby("t").sum()/1000
        d.loc[:, "demand_el"] -= (d.net_export)
        d = d.loc[market_result.model_horizon, :]

        fig.add_trace(go.Scatter(x=d.index, y=d.demand_el - inf.infeasibility, line=dict(color="#000000"), name="demand")) 
        fig.add_trace(go.Scatter(x=d.index, y=d.demand_el - inf.infeasibility + d.D_es + d.D_ph, fill='tonexty', mode= 'none', fillcolor="#BFBDE5", name="storage charging"))
            
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_generation_overview(self, market_result, show_plot=True, filepath=None):
        """Create pie chart, showing generation by type/fuel in relation to the total generation.

        The overview shows generation by fuel/type/technology as a pie chart, illustrating the share
        of each type. The figure is generated using the 
        plotly package and can be returned or saved depending on the input arguments.
        
        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement`
            Market result which is plotted. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """
        gen = market_result.generation()
        flh = market_result.full_load_hours()
            
        gen = gen[["fuel", "technology", "G", "g_max"]].groupby(["fuel", "technology"]).sum().reset_index()
        flh = flh.groupby(["fuel", "technology"]).mean().reset_index()
        gen = pd.merge(gen, flh, on=["fuel", "technology"])
        gen.loc[:, "G"] *= 1/1000
        gen.loc[:, "flh"] *= 100
        gen.loc[:, "utilization"] *= 100

        gen_colors = color_map(gen)
        gen = pd.merge(gen, gen_colors, on=["fuel", "technology"])
        gen.rename(columns={"G": "Generation"}, inplace=True)
        fig = px.pie(gen, values='Generation', color='name', custom_data=gen[["name", "flh", "utilization"]],
                    color_discrete_map=gen_colors[["color", "name"]].set_index("name").color.to_dict())
        fig.update_traces(hovertemplate='<b>%{customdata[0][0]}</b> \
                                         <br>Generation = %{value:.2f} GWh \
                                         <br>Full Load Hours=%{customdata[0][1]:.2f} [% of capacity]\
                                         <br>Utilization=%{customdata[0][2]:.2f} [% of available capacity]\
                                         <extra></extra>')
       
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_lineflow_plot(self, market_result, lines, show_plot=True, filepath=None):
        """Create line flow plot for selected lines.        

        The lineflow plot contains three different plots, showing the N-0, N-1 and a lineflow 
        duration plot for the specified lines. The figure is generated using the 
        plotly package and can be returned or saved depending on the input arguments.
        
        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement`
            Market result which is plotted. 
        lines : list, 
            List of line elements to be included in the plot.
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """
        n_0 = market_result.n_0_flow()
        n_1 = market_result.absolute_max_n_1_flow()
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=("N-0 Flow in [MW]", "N-1 Flow in [MW]", 
                                            "Power Flow Duration [MW]"),
                            shared_xaxes=True, vertical_spacing=0.1)

        for i,line in enumerate(lines):
            color_n_0 = matplotlib.colors.rgb2hex(matplotlib.cm.Paired(i))
            color_n_1 = matplotlib.colors.rgb2hex(matplotlib.cm.Accent(i))
            
            fig.append_trace(go.Scatter(
                x=n_0.loc[line, :].index,
                y=n_0.loc[line, :].values,
                legendgroup=line,
                name=line, line=dict(color=color_n_0)
            ), row=1, col=1)
            
            fig.append_trace(go.Scatter(
                x=n_1.loc[line, :].index,
                y=n_1.loc[line, :].values,
                legendgroup=line,
                name=line, line=dict(color=color_n_1)
            ), row=2, col=1)
            
            fig.append_trace(go.Scatter(
                x=market_result.model_horizon,
                y=n_0.loc[line, :].abs().sort_values(ascending=False),
                legendgroup=line, showlegend=False,
                line=dict(color=color_n_0),
                name=line,
            ), row=3, col=1)
            
            fig.append_trace(go.Scatter(
                x=market_result.model_horizon,
                y=n_1.loc[line, :].abs().sort_values(ascending=False),
                legendgroup=line, showlegend=False,
                line=dict(color=color_n_1),
                name=line,
            ), row=3, col=1)

        # fig.update_xaxes(visible=False, row=1, col=1)
        # fig.update_xaxes(visible=False, row=2, col=1)
        # fig.update_yaxes(title="Power Flow Duration [MW]", row=3, col=1)

        fig.update_layout(autosize=True)
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_installed_capacity_plot(self, market_result, show_plot=True, filepath=None):
        """Create plot visualizing installed capacity per market area.

        The installed capacity plot visualizes the installed capacity as stacked bar charts. The 
        figure is generated using the plotly package and can be returned or saved depending on the input arguments.
        
        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement`
            Market result which is plotted. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """
        plants = market_result.data.plants
        plants["zone"] = market_result.data.nodes.loc[plants.node, "zone"].values
        if not "technology" in plants.columns:
            plants["technology"] = plants.plant_type

        plants = (plants[["technology", "fuel", "zone", "g_max"]].groupby(["zone", "technology", "fuel"]).sum()/1000).reset_index()
        plant_colors = color_map(plants)
        plants = pd.merge(plants, plant_colors, on=["fuel", "technology"])
        fig = px.bar(plants, x="zone", y="g_max", color="name", 
                     color_discrete_map=plant_colors[["color", "name"]].set_index("name").color.to_dict())
        fig.layout.yaxis.title = "Installed Capacity [GW]"
        fig.layout.xaxis.title = "Zone/Country"
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_storage_plot(self, market_result, storages=None, show_plot=True, filepath=None):
        """Create plot storage utilization.

        The storage plot visualizes the usage of electricity storages over the model horizon. The
        values are summed over all storage units, or units can be specified by the optional input 
        argument. The figure is generated using the plotly package and can be returned or 
        saved depending on the input arguments.
        
        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement`
            Market result which is plotted. 
        storages : list, optional
            Only show selected storage units. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """

        es_gen = market_result.storage_generation()
        if es_gen.empty:
            self.logger.warning("No storage results to plot.")
            return None
        
        if isinstance(storages, list):
            es_gen = es_gen[es_gen.p.isin(storages)]

        es_gen = es_gen[["t", "G", "D_es", "L_es"]].groupby(["t"]).sum().reset_index()

        fig = px.line(pd.melt(es_gen, id_vars=["t"], value_vars=["G", "D_es"]), x="t", y="value", color='variable')
        fig2 = px.line(pd.melt(es_gen, id_vars=["t"], value_vars=["L_es"]), x="t", y="value", color='variable')
        fig2.update_traces(yaxis="y2")
        # Create figure with secondary y-axis
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        subfig.add_traces(fig.data + fig2.data)
        subfig.layout.xaxis.title = "Time"
        subfig.layout.yaxis.title = "Storage Charging (D_es)/Discharging (G) [MW]"
        subfig.layout.yaxis2.title = "Storage Level [MWh]"
        subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
        
        if filepath:
            subfig.write_html(str(filepath))
        if show_plot:
            plot(subfig)
        else:
            return subfig









