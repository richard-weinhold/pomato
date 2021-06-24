"""Visualization Module of POMATO"""

import io
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pomato
import pomato.tools as tools
from matplotlib import cm
from numpy.lib.arraysetops import isin
from PIL import Image
from plotly.offline import plot
from plotly.subplots import make_subplots
from pomato.visualization.geoplot_functions import (_create_geo_json,
                                                    add_prices_layer,
                                                    line_colors,
                                                    line_coordinates)

BASIC_FUEL_COLOR_MAP = {
        "uran": ["#C44B50", "#7B011A", "#9C242E"],
        "lignite": ["#BC8873", "#A67662", "#895B4B"],
        "coal": ["#8C7F77", "#73645D", "#5E4F48"],
        "geothermal": ["#FDCF94", "#5472D4", "#2E56B5"],
        "hydro": ["#6881E5", "#5472D4", "#2E56B5", "#0E47A1", "#00125E"],
        "water": ["#6881E5", "#5472D4", "#2E56B5", "#0E47A1", "#00125E"],
        "gas": ["#FA5827", "#E44214", "#C62200"],
        "biomass": ["#AEE570", "#96CD58", "#7DB343"],
        "oil": ["#565752", "#282924"],
        "waste": ["#000000", "#141510"],
        "other": ["#87959E", "#CCDAE3", "#B6C9D0", "#235E58", "#013531"],
        "wind": ["#6699AA", "#00303E", "#326775"],
        "sun": ["#FFEB3C", "#F4E01D", "#EBD601"],
    }

def color_map(gen):
    """Fuel colors for generation/capacity plots."""
    color_df = gen[["fuel", "technology"]].groupby(["fuel", "technology"], observed=True).sum()
    for fuel in BASIC_FUEL_COLOR_MAP:
        condition = color_df.index.get_level_values("fuel").str.lower().str.contains(fuel)
        number_of_fuels = sum(condition)
        color_df.loc[condition, "color"] = BASIC_FUEL_COLOR_MAP[fuel][:number_of_fuels]
    
    fill = "#EAF9FE"
    other_fuel_colors = BASIC_FUEL_COLOR_MAP["other"]
    number_to_fill =  sum(color_df.color.isna()) - len(BASIC_FUEL_COLOR_MAP["other"])
    if number_to_fill > 0: # if more colors are needed than existing on other add filler
        other_fuel_colors.extend([fill for x in range(0, number_to_fill)])
    color_df.loc[color_df.color.isna(), "color"] = other_fuel_colors[:sum(color_df.color.isna())]

    color_df = color_df.reset_index()
    color_df["name"] = ""
    for fuel in color_df.fuel.unique():
        condition = color_df.fuel == fuel
        if sum(condition) > 1:
            color_df.loc[condition, "name"] = color_df[condition].fuel.astype(str) + " " + color_df[condition].technology.astype(str)
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
        self.logger = logging.getLogger('Log.pomato.visualization.Visualization')
        self.wdir = wdir

    def create_geo_plot(self, market_result, show_redispatch=False, show_prices=False, show_infeasibility=False,
                        show_curtailment=False, timestep=None, threshold=0, highlight_nodes=None, 
                        highlight_lines=None,  line_color_option=0, show_plot=True, filepath=None):
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
                gen = pd.merge(nodes["zone"], gen, how="left", left_index=True, right_index=True)
                nodes = pd.merge(nodes, gen[["delta", "delta_abs"]].fillna(0), left_index=True, right_index=True)
            elif isinstance(gen, pd.DataFrame):
                gen = gen.loc[:, ["node", "delta", "delta_abs"]].groupby("node").sum()
                gen = pd.merge(nodes["zone"], gen, how="left", left_index=True, right_index=True)
                nodes = pd.merge(nodes, gen[["delta", "delta_abs"]].fillna(0), left_index=True, right_index=True)
            else:
                show_redispatch = False
        
        if show_curtailment: 
            curtailment = market_result.curtailment()
            if isinstance(timestep, str):
                curtailment = curtailment.loc[curtailment.t == timestep, ["node", "CURT"]].groupby("node").sum()
                curtailment = pd.merge(nodes["zone"], curtailment, how="left", left_index=True, right_index=True)
                nodes = pd.merge(nodes, curtailment["CURT"].fillna(0), left_index=True, right_index=True)
            else:
                curtailment = curtailment[["node", "CURT"]].groupby("node").sum()
                curtailment = pd.merge(nodes["zone"], curtailment, how="left", left_index=True, right_index=True)
                nodes = pd.merge(nodes, curtailment["CURT"].fillna(0), left_index=True, right_index=True)
        
        if show_infeasibility: 
            infeasibility = market_result.infeasibility()
            if isinstance(timestep, str):
                infeasibility = infeasibility.loc[infeasibility.t == timestep, ["n", "pos", "neg"]].groupby("n").sum()
                infeasibility = pd.merge(nodes["zone"], infeasibility, how="left", left_index=True, right_index=True)
                nodes = pd.merge(nodes, infeasibility[["pos", "neg"]].fillna(0), left_index=True, right_index=True)
            else:
                infeasibility = infeasibility[["n", "pos", "neg"]].groupby("n").sum()
                infeasibility = pd.merge(nodes["zone"], infeasibility, how="left", left_index=True, right_index=True)
                nodes = pd.merge(nodes, infeasibility[["pos", "neg"]].fillna(0), left_index=True, right_index=True)

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
        # DC Lines
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
        lines["colors"], lines["alpha"] = line_colors(lines, n_0_flows, n_1_flows, 
                                                      option=line_color_option,
                                                      threshold=threshold,
                                                      highlight_lines=highlight_lines)

        hovertemplate_lines = "<br>".join(["Line: %{customdata[0]}", 
                                           "Capcity: %{customdata[1]:.2f} MW",
                                           "N-0 Flow %{customdata[2]:.2f} MW",
                                           "N-1 Flow %{customdata[3]:.2f} MW"]) + "<extra></extra>"
        # Add Lines for each color
        for color, alpha in (lines[["colors", "alpha"]].apply(tuple, axis=1).unique()):
            tmp_lines = lines[(lines.colors == color)&(lines.alpha == alpha)]
            # tmp_lines_idx = [lines.index.get_loc(line) for line in tmp_lines.index]
            alpha = alpha
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
                    line = dict(width = 2, color=color),
                    opacity = alpha,
                    customdata=customdata,
                    hovertemplate=hovertemplate_lines
                    )
                )
            
        # Highlight Nodes
        affected_nodes = []
        if isinstance(highlight_nodes, list):
            condition = nodes.index.isin(highlight_nodes)
            affected_nodes = highlight_nodes
            fig.add_trace(go.Scattermapbox(
                lon = nodes.loc[condition, 'lon'],
                lat = nodes.loc[condition, 'lat'],
                mode = 'markers',
                marker = go.scattermapbox.Marker(
                    color = "#EF7FFF", # Pink 
                    opacity=1,
                    size=30),
                customdata=nodes.loc[condition, ["zone"]].reset_index(),
                hovertemplate=
                "<br>".join([
                    "Node: %{customdata[0]}",
                    "Zone: %{customdata[1]}",
                ]) + "<extra></extra>"
            ))

        if show_redispatch and any(nodes.delta_abs > 0):
            affected_nodes = nodes.index[nodes.delta_abs > 0]
            sizeref = max(2*max(nodes['delta_abs'])/12**2, 1)
            for condition, color in zip([nodes["delta"] < 0, nodes["delta"] > 0], ["red", "green"]):
                markers = go.scattermapbox.Marker(
                    size = nodes.loc[condition, 'delta_abs'],
                    sizeref = sizeref,
                    sizemin = 3,
                    color = color,
                    opacity=0.8,
                    # line = {"color": 'rgb(40,40,40)'},
                    # line_width=0.5,
                    autocolorscale=True
                    )
                customdata = nodes.loc[condition, ["zone", "delta_abs"]].reset_index()
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

        elif show_curtailment and any(nodes.CURT > 0):
            condition = (nodes.CURT > 0)
            affected_nodes = nodes.index[condition]
            sizeref = max(2*max(nodes.loc[condition, 'CURT'])/12**2, 1)
            fig.add_trace(go.Scattermapbox(
                lon = nodes.loc[condition, 'lon'],
                lat = nodes.loc[condition, 'lat'],
                mode = 'markers',
                marker = go.scattermapbox.Marker(
                    color = "#8A31BD", # Purple
                    opacity=0.8,
                    sizeref = sizeref,
                    sizemin = 3,
                    size=nodes.loc[condition, "CURT"]),
                customdata=nodes.loc[condition, ["zone", "CURT"]].reset_index(),
                hovertemplate=
                "<br>".join([
                    "Node: %{customdata[0]}",
                    "Zone: %{customdata[1]}",
                    "Curtailment: %{customdata[2]:.2f} MW",
                ]) + "<extra></extra>"
            ))

        elif show_infeasibility and any(nodes[["pos", "neg"]] > 0):
            affected_nodes = nodes.index[(nodes[["pos", "neg"]] > 0).any(axis=1)]
            sizeref = max(4*nodes[["pos", "neg"]].max().max()/12**2, 1)
            for col, color in zip(["pos", "neg"], ["#4575B4", "#F46D43"]):
                condition = nodes[col] > 0
                fig.add_trace(go.Scattermapbox(
                    lon = nodes.loc[condition, 'lon'],
                    lat = nodes.loc[condition, 'lat'],
                    mode = 'markers',
                    marker = go.scattermapbox.Marker(
                        color = color,
                        opacity=0.8,
                        sizeref = sizeref,
                        sizemin = 3,
                        size=nodes.loc[condition, col]),
                    customdata=nodes.loc[condition, ["zone", "pos", "neg"]].reset_index(),
                    hovertemplate=
                    "<br>".join([
                        "Node: %{customdata[0]}",
                        "Zone: %{customdata[1]}",
                        "Infeasibilities: + %{customdata[2]:.2f}, - %{customdata[3]:.2f} MW",
                    ]) + "<extra></extra>"
                ))
        
        # Plot the remaining nodes
        fig.add_trace(go.Scattermapbox(
            lon = nodes[~nodes.index.isin(affected_nodes)].lon,
            lat = nodes[~nodes.index.isin(affected_nodes)].lat,
            mode = 'markers',
            marker = go.scattermapbox.Marker(
                color = "#3283FE", # Blue
                opacity=0.8
                ), 
            customdata=nodes[~nodes.index.isin(affected_nodes)][["zone"]].reset_index(),
            hovertemplate=
            "<br>".join([
                "Node: %{customdata[0]}",
                "Zone: %{customdata[1]}",
            ]) + "<extra></extra>"
        ))

        if show_prices:
            compress = True
            colorscale = "RdBu_r"
            contours = 12
            prices_layer, coordinates, hight_width = add_prices_layer(nodes, prices, compress)
            price_fig = go.Figure(
                data=go.Contour(z=prices_layer, showscale=False, 
                                colorscale=colorscale, ncontours=contours))
            price_fig.update_layout(
                width=2e3, height=2e3*hight_width, 
                xaxis = {'visible': False},
                yaxis = {'visible': False},
                margin={"r":0,"t":0,"l":0,"b":0}
                )
            
            img_pil = Image.open(io.BytesIO(price_fig.to_image()))
            price_layer =  {   
                    "below": 'traces',
                    "sourcetype": "image",
                    "source": img_pil,
                    "coordinates": coordinates,
                    "opacity": .5,
                }
            # Price Colorbar
            price_colorbar = go.Scatter(x=[None],y=[None],
                                        mode='markers',
                                        marker=dict(
                                            colorscale=colorscale, 
                                            showscale=True,
                                            cmin=prices_layer.min().round(2),
                                            cmax=prices_layer.max().round(2),
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

        map_layer = {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
            "source": [
                "https://stamen-tiles.a.ssl.fastly.net/toner-background/{z}/{x}/{y}.png"
            ]
        }

        fig.update_layout(    
            showlegend = False,
            margin={"r":0,"t":0,"l":0,"b":0},
            mapbox= {
                "style": "white-bg",
                "layers": [map_layer, price_layer],
                "zoom": 3,
                "center": center},
            xaxis = {'visible': False},
            yaxis = {'visible': False})

        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_zonal_geoplot(self, market_result, timestep=None, highlight_lines=None, show_plot=True, filepath=None):
        
        fig = self.create_geo_plot(market_result, timestep=timestep, 
                                   highlight_lines=highlight_lines, show_plot=False)
        fig.update_traces(marker_showscale=False)

        commercial_exchange = market_result.EX.copy()
        net_position = market_result.net_position()
        geojson = _create_geo_json(self.data.zones, self.data.nodes)
        if isinstance(timestep, int):
            timestep = market_result.model_horizon[timestep]
        if isinstance(timestep, str):
            commercial_exchange = commercial_exchange[commercial_exchange.t == timestep].groupby(["z", "zz"], observed=True).mean().reset_index()
            net_position = net_position.loc[timestep]
        else:
            net_position = net_position.mean(axis=0)
            commercial_exchange = commercial_exchange.groupby(["z", "zz"], observed=True).mean().reset_index()

        custom_data = []
        for zone in self.data.zones.index:
            data = [zone, net_position.loc[zone]]
            tmp_ex_to = commercial_exchange.loc[(commercial_exchange.z == zone)&(commercial_exchange.zz != zone)]
            tmp_ex_from = commercial_exchange.loc[(commercial_exchange.zz == zone)&(commercial_exchange.z != zone)]
            tmp = []
            for i in tmp_ex_to[tmp_ex_to.EX > 0].index:
                tmp.append(tmp_ex_to.loc[i, "zz"] + ": " + str(tmp_ex_to.loc[i, "EX"].round()))
            data.append("TO: " + " | ".join(tmp))
            tmp = []
            for i in tmp_ex_from[tmp_ex_from.EX > 0].index:
                tmp.append(tmp_ex_from.loc[i, "z"] + ": " + str(tmp_ex_from.loc[i, "EX"].round()))
            data.append("FROM: " + " | ".join(tmp))
            custom_data.append(data)

        hovertemplate = ("<b>Zone: %{customdata[0]}</b>" +
                        "<br>Net Position: %{customdata[1]:.2f}" + 
                        "<br>Commercial Exchange: " + 
                        "<br>%{customdata[2]}" +
                        "<br>%{customdata[3]}" +
                        "<extra></extra>")

        fig.add_trace(go.Choroplethmapbox(
                geojson=geojson, 
                colorscale="deep",
                locations=self.data.zones.index, # Spatial coordinates
                z = net_position.values,
                customdata=custom_data,
                hovertemplate=hovertemplate,
                marker=dict(
                    line=dict(width=3),
                    opacity=0.45),
                colorbar=dict(thickness=5)
                ))

        # re-sort layers to bring the zones to the bottom.
        fig.data = tuple(fig.data[i] for i in ([-1] + [i for i in range(0, len(fig.data)-1)]))
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
        market_result : :class:`~pomato.data.Result`
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

        gen = gen[["fuel", "technology", "t", "G", "g_max"]].groupby(["fuel", "technology", "t"], observed=True).sum().reset_index()
        gen_colors = color_map(gen)
        gen = pd.merge(gen, gen_colors, on=["fuel", "technology"])
        gen.loc[:, "G"] *= 1/1000

        gen["utilization"] = gen.G / gen.g_max
        sort_fuel_name = gen[["name", "utilization"]].groupby("name").mean().sort_values(by="utilization", ascending=False)
        gen = gen.sort_values(by="t", key=market_result._sort_timesteps)
        fig = px.area(gen, x="t", y="G", color="name", 
                      color_discrete_map=gen_colors[["color", "name"]].set_index("name").color.to_dict(),
                      category_orders={"name": list(sort_fuel_name.index)})

        fig.layout.xaxis.title="Time"
        fig.layout.yaxis.title="Generation/Load [GW]"

        inf["infeasibility"] = inf.pos - inf.neg
        inf = inf[["t", "infeasibility"]].groupby("t").sum()/1000
        inf = inf.loc[market_result.model_horizon, :]
        
        demand = pd.merge(demand, net_export, left_on=["t", "n"], 
                          right_on=["timestep", "node"], how="left")
        demand["net_export"].fillna(0, inplace=True)

        demand = demand[["t", "demand_el", "D_ph", "D_es", "net_export"]].groupby("t").sum()/1000
        demand.loc[:, "demand_el"] -= (demand.net_export)
        demand = demand.loc[market_result.model_horizon, :]
        
        fig.add_trace(
            go.Scatter(x=demand.index, y=demand.demand_el - inf.infeasibility, 
                       line=dict(color="#000000"), name="demand")) 
        fig.add_trace(
            go.Scatter(x=demand.index, y=demand.demand_el - inf.infeasibility + demand.D_es + demand.D_ph, 
                       fill='tonexty', mode= 'none', fillcolor="#BFBDE5", name="storage charging"))
            
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_available_intermittent_capacity_plot(self, data, zones=None, show_plot=True, filepath=None):
        """Visualize available intermittet generation capacities over time.

        Parameters
        ----------
        data : :class:`~pomato.data.DataManagement`
            Pomato data object. 
        zones: list, optional,
            Subset of zones to include in the plot. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """        
        ava = self.data.availability.copy()
        plants = self.data.plants.copy()

        plants["zone"] = self.data.nodes.loc[self.data.plants.node, "zone"].values
        if isinstance(zones, list):
            plants = plants[plants.zone.isin(zones)]
      
        if not "technology" in plants.columns:
            plants["technology"] = plants.plant_type
        ava= pd.merge(ava, plants[["g_max", "fuel", "technology", "zone"]], left_on="plant", right_index=True, how="left")
        ava["Available Capacity [GW]"] = ava.availability * ava.g_max / 1000
        cols = ["timestep", "fuel", "technology", "zone", "Available Capacity [GW]"]
        ava = ava[cols].groupby(cols[:-1], observed=True).sum().reset_index()
        capacity_colors = color_map(ava)
        ava = pd.merge(ava, capacity_colors, on=["fuel", "technology"])

        # Sort by variance
        sort_names = ava.groupby("name", observed=True).var().sort_values(by="Available Capacity [GW]", ascending=True).index
        fig = px.area(ava, x="timestep", y="Available Capacity [GW]", color="name", 
                    line_group="zone", 
                    color_discrete_map=capacity_colors[["color", "name"]].set_index("name").color.to_dict(),
                    category_orders={"name": list(sort_names)})

        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig
    
    def create_generation_pie(self, market_result, show_plot=True, filepath=None):
        """Create pie chart, showing generation by type/fuel in relation to the total generation.

        The resulting figure generation by fuel/type/technology as a pie chart, illustrating the
        share of each type. The figure is generated using the plotly package and can be returned or
        saved depending on the input arguments.

        Parameters
        ----------
        market_result : :class:`~pomato.data.DataManagement` Market result which is plotted.
            show_plot : bool, optional Shows plot after generation. If false, returns plotly figure
            instead. By default True. filepath : pathlib.Path, str, optional If filepath is
            supplied, saves figure as .html, by default None
        """
        gen = market_result.generation()
        flh = market_result.full_load_hours()
            
        gen = gen[["fuel", "technology", "G", "g_max"]].groupby(["fuel", "technology"], observed=True).sum().reset_index()
        flh = flh.groupby(["fuel", "technology"], observed=True).mean().reset_index()
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

        fig.update_layout(autosize=True, template="none")
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_installed_capacity_plot(self, data=None, zones=None,
                                       show_plot=True, filepath=None):
        """Create plot visualizing installed capacity per market area.

        The installed capacity plot visualizes the installed capacity as stacked bar charts. The 
        figure is generated using the plotly package and can be returned or saved depending on the input arguments.
        
        Parameters
        ----------
        data : :class:`~pomato.data.DataManagement` or :class:`~pomato.data.Results`, optional
            Data to plot, defaults to class attribute data. 
        zones : list, optional 
            Subset of zones to include in the capacity plot. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """
        if not data:
            data = self.data
        if isinstance(data, pomato.data.Results):
            data = data.data
        
        if not isinstance(data, pomato.data.DataManagement):
            raise TypeError("Please supply a Result oder DataManagement instance.")
        
        plants = data.plants.copy()
        plants["zone"] = data.nodes.loc[plants.node, "zone"].values

        if not "technology" in plants.columns:
            plants["technology"] = plants.plant_type
        if isinstance(zones, list):
            plants = plants[plants.zone.isin(zones)]

        plants = (plants[["technology", "fuel", "zone", "g_max"]].groupby(["zone", "technology", "fuel"], observed=True).sum()/1000).reset_index()
        plant_colors = color_map(plants)
        plants = pd.merge(plants, plant_colors, on=["fuel", "technology"])
        fig = px.bar(plants, x="zone", y="g_max", color="name", 
                     color_discrete_map=plant_colors[["color", "name"]].set_index("name").color.to_dict())
        fig.update_xaxes(categoryorder='array', categoryarray=data.zones.index)
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

    def create_fb_domain_plot(self, fb_domain, show_plot=True, filepath=None):
        """Create FlowBased Domain plot. 

        This is a copy of the previous :meth:`~pomato.visualization.FBDomain.create_fbmc_figure`
        using plotly instead of matplotlib. This allows for proper integration in the Dashboard
        functionality including interaction with the geo plot. 

        Input argument remains an instance of :class:`~pomato.visualization.FBDomain` which can be
        created by utilizing :meth:`~pomato.visualization.FBDomainPlots` module. 
        """
        # fb_domain = domain_plot
        fig = go.Figure()
        scale = 2
        n0_lines_x, n0_lines_y = [], []
        n1_lines_x, n1_lines_y = [], []
        frm_x, frm_y = [], []
        hover_data_n0, hover_data_n1, hover_data_frm = [], [], []

        hover_points = len(fb_domain.domain_equations[0][1])
        tmp = fb_domain.domain_data.reset_index(drop=True)
        for i in tmp[tmp.co == "basecase"].index:
            n0_lines_x.extend(fb_domain.domain_equations[i][0])
            n0_lines_y.extend(fb_domain.domain_equations[i][1])
            n0_lines_x.append(None)
            n0_lines_y.append(None)
            data = [tmp.loc[i, "cb"], tmp.loc[i, "co"], tmp.loc[i, "ram"]]
            hover_data_n0.append(np.vstack([[data for n in range(0, hover_points)], [None, None, None]]))
        
        for i in tmp[(tmp.co == "FRM")&(tmp.in_domain)].index:
            frm_x.extend(fb_domain.domain_equations[i][0])
            frm_y.extend(fb_domain.domain_equations[i][1])
            frm_x.append(None)
            frm_y.append(None)
            data = [tmp.loc[i, "cb"], tmp.loc[i, "co"], tmp.loc[i, "ram"]]
            hover_data_frm.append(np.vstack([[data for n in range(0, hover_points)], [None, None, None]]))
        
        for i in tmp[(tmp.co != "basecase")&(tmp.co != "FRM")].index:
            n1_lines_x.extend(fb_domain.domain_equations[i][0])
            n1_lines_y.extend(fb_domain.domain_equations[i][1])
            n1_lines_x.append(None)
            n1_lines_y.append(None)
            data = [tmp.loc[i, "cb"], tmp.loc[i, "co"], tmp.loc[i, "ram"]]
            hover_data_n1.append(np.vstack([[data for n in range(0, hover_points)], [None, None, None]]))
        
        hovertemplate = "<br>".join(["cb: %{customdata[0]}", 
                                     "co: %{customdata[1]}", 
                                     "ram: %{customdata[2]:.2f}"]) + "<extra></extra>"
        if len(hover_data_n0) > 0:
            fig.add_trace(
                go.Scatter(x=n0_lines_x, y=n0_lines_y, name='N-0 Constraints',
                        line = dict(width = 1.5, color="dimgrey"),
                        customdata=np.vstack(hover_data_n0),
                        hovertemplate=hovertemplate
                        )
                )

        if len(hover_data_frm) > 0:
            fig.add_trace(
                go.Scatter(x=frm_x, y=frm_y, name='FRM',
                        line = dict(dash='dash', width = 1.5, color="royalblue"),
                        customdata=np.vstack(hover_data_frm),
                        hovertemplate=hovertemplate
                        )
                )
            
        fig.add_trace(
            go.Scatter(x=n1_lines_x, y=n1_lines_y, name='N-1 Constraints',
                    line = dict(width = 1, color="lightgrey"),
                    opacity=0.6,
                    customdata=np.vstack(hover_data_n1),
                    hovertemplate=hovertemplate

                        )
            )
        fig.add_trace(
                go.Scatter(x=fb_domain.feasible_region_vertices[:, 0], 
                            y=fb_domain.feasible_region_vertices[:, 1],
                            line = dict(width = 1, color="red"),
                            opacity=1, name=f"FB Domain<br>Volume: {int(fb_domain.volume)}",
                            mode='lines', hoverinfo="none"
                        )
                )
        fig.update_layout(xaxis_range=[fb_domain.x_min, fb_domain.x_max],
                          yaxis_range=[fb_domain.y_min, fb_domain.y_max],
                          template='simple_white',
                          hovermode="closest")

        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_cost_overview(self, market_results, show_plot=True, filepath=None):
        """Create objective value overview of multiple model results. 

        Parameters
        ----------
        market_results : list of :class:`~pomato.data.DataManagement`
            Market results which is plotted. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """

        if not all([isinstance(r, pomato.data.Results) or isinstance(r, pomato.data.results.Results) for r in  market_results]):
            raise TypeError("Submit list of market results")
    
        cost_data = pd.DataFrame()
        columns = ["COST_H", "COST_EX", "COST_G", "COST_CURT", "COST_REDISPATCH", "COST_INFEASIBILITY_EL"]
        colors = ["#D6616B", "#969696", "#A1D99B", "#9E9AC8", "#E7969C", "#08519C"]
        for result in market_results:
            objective_values = result.result_attributes["objective"]
            name = result.result_attributes["title"] # .replace("_redispatch", "").replace("/FBMC", "").replace("_20", "")
            data = [(name, c, objective_values[c]/1e6) for c in columns]
            tmp = pd.DataFrame(data=data, columns=["result", "costs", "value"])
            tmp["color"] = colors
            tmp["total_costs"] = tmp.value.sum()
            tmp["percentage_of_total_costs"] = tmp.value / tmp.value.sum()*100
            if cost_data.empty:
                cost_data = tmp
            else:
                cost_data = pd.concat([cost_data, tmp])

        hovertemplate = '<b>%{customdata[0]}</b> \
                        <br>Costs = %{customdata[1]:.2f} \
                        <br>Total costs = %{customdata[2]:.2f} \
                        <br>%{customdata[3]:.0f}% of total costs\
                        <extra></extra>'
        data = []                 
        for cost in cost_data.costs.unique():
            data.append(go.Bar(
                name=cost, 
                x=cost_data.loc[cost_data.costs == cost, "result"],
                y=cost_data.loc[cost_data.costs == cost, "value"],
                marker_color=cost_data.loc[cost_data.costs == cost, "color"],
                marker_line_color='black',
                hovertemplate=hovertemplate,
                customdata=cost_data.loc[cost_data.costs == cost, ["costs", "value", "total_costs", "percentage_of_total_costs"]],
                width=0.9,
                )
            )
        layout = go.Layout(
            template="simple_white",
            barmode='stack',
            yaxis=dict(title='Costs [mio $]'),
            )
        fig = go.Figure(data=data, layout=layout)
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_generation_overview(self, market_results, show_plot=True, filepath=None):
        """Create generation overview of multiple model results. 

        Parameters
        ----------
        market_results : list of :class:`~pomato.data.Results`
            Market results which is plotted. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """
        if not all([isinstance(r, pomato.data.Results) or isinstance(r, pomato.data.results.Results) for r in  market_results]):
            raise TypeError("Submit list of market results")

        gen = pd.DataFrame()
        for result in market_results: 
            if (result.result_attributes["is_redispatch_result"] and 
                result.result_attributes["corresponding_market_result_name"]):
                tmp = result.redispatch()
                tmp = tmp.rename(columns={"G_redispatch": "G"})
            else:
                tmp = result.generation()
                tmp["delta"] = 0
                tmp["delta_abs"] = 0
            tmp["result"] = result.result_attributes["title"]
            if gen.empty:
                gen = tmp
            else:
                gen = pd.concat([gen, tmp], ignore_index=True)
            
        names = color_map(gen)
        gen.loc[:, ["G", "delta", "delta_abs"]] /= 1000
        gen["positive_redispatch"] = gen.loc[gen.delta > 0, "delta"]
        gen["negative_redispatch"] = gen.loc[gen.delta < 0, "delta"]
        gen = gen.groupby(["fuel", "technology", "result"]).sum().reset_index()

        gen = pd.merge(gen, names, on=["fuel", "technology"])
        gen["percentage_gen"] = 0
        gen["total_redispatch"] = 0
        gen["precentage_redispatch"] = 0
        for r in gen.result.unique():
            gen.loc[gen.result == r, "percentage_gen"] = gen.loc[gen.result == r, "G"]/gen.loc[gen.result == r, "G"].sum()*100
            gen.loc[gen.result == r, "total_redispatch"] = gen.loc[gen.result == r, "delta_abs"].sum()
            gen.loc[gen.result == r, "precentage_redispatch"] = gen.loc[gen.result == r, "delta_abs"]/gen.loc[gen.result == r, "delta_abs"].sum()*100

        hovertemplate_gen = '<b>%{customdata[0]}</b> \
                            <br>Generation = %{customdata[1]:.2f} GWh \
                            <br>%{customdata[2]:.0f}% of total generation \
                            <extra></extra>'
        hovertemplate_red = '<b>%{customdata[0]}</b> \
                            <br>Redispatch = %{customdata[1]:.2f} GWh \
                            <br>Total redispatch = %{customdata[2]:.2f} GWh \
                            <br>%{customdata[3]:.0f}% of total redispatch \
                            <extra></extra>'
        data = []                 
        for name in gen.name.unique():
            data.append(go.Bar(
                name=name, 
                legendgroup=name,
                x=[gen.loc[gen.name == name, "result"], ['Generation']*len(market_results)],
                y=gen.loc[gen.name == name, "G"],
                marker_color=gen.loc[gen.name == name, "color"],
                marker_line_color='black',
                hovertemplate=hovertemplate_gen,
                customdata=gen.loc[gen.name == name, ["name", "G", "percentage_gen"]],
                width=0.8,
                yaxis='y1'))
            for redispatch in ["positive_redispatch", "negative_redispatch"]:
                data.append(go.Bar(
                    name=name, 
                    legendgroup=name,
                    showlegend=False,
                    x=[gen.loc[gen.name == name, "result"], ['Redispatch']*len(market_results)],
                    y=gen.loc[gen.name == name, redispatch],
                    marker_color=gen.loc[gen.name == name, "color"],
                    hovertemplate=hovertemplate_red,
                    customdata=gen.loc[gen.name == name, ["name", redispatch, "total_redispatch", "precentage_redispatch"]],
                    width=0.8,
                    opacity=0.7,
                    yaxis='y2')
                    )
        layout = go.Layout(
            template="simple_white",
            legend=dict(x=1.1),
            barmode='relative',
            yaxis=dict(title='Generation [GWh]'),
            yaxis2=dict(title='Redispatch [GWh]', 
                        overlaying='y',
                        side='right'),
            )
        fig = go.Figure(data=data, layout=layout)
        result_tiles = [r.result_attributes["title"] for r in market_results]
        fig.update_xaxes(categoryorder='array', categoryarray=result_tiles)

        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig

    def create_merit_order(self, data=None, timestep=None, show_plot=True, filepath=None):
        """Create merit order of the input data. 

        Parameters
        ----------
        data : :class:`~pomato.data.DataManagement` or :class:`~pomato.data.Results`, optional
            Data to plot, defaults to class attribute data. 
        show_plot : bool, optional
            Shows plot after generation. If false, returns plotly figure instead. By default True.
        filepath : pathlib.Path, str, optional
            If filepath is supplied, saves figure as .html, by default None
        """
        if not data:
            plants = self.data.plants.copy()
        elif isinstance(data, pomato.data.DataManagement):
            plants = data.plants.copy()
        elif isinstance(data, pomato.data.Results):
            plants = data.data.plants.copy()
        else:
            raise TypeError("Data input argument not correct type. ")

        if not "technology" in plants.columns:
            plants["technology"] = plants.plant_type
        if "name" in plants.columns:
            plants = plants.drop("name", axis=1)

        if timestep:
            ava = self.data.availability[self.data.availability.timestep == timestep].copy()
            ava = ava.drop("timestep", axis=1).set_index("plant")
            plants.loc[ava.index, "g_max"] *= ava.availability

        color = color_map(plants)
        plants = pd.merge(plants, color, on=["technology", "fuel"])
        plants = plants[["color", "name", "g_max", "mc_el"]].groupby(["color", "name", "mc_el"], observed=True).sum().reset_index()
        plants = plants.sort_values("mc_el").reset_index()
        plants["x"] = plants['g_max'].cumsum() - plants['g_max']*0.5
        hovertemplate = (
            '<b>%{customdata[0]}</b> \
            <br>Cost = %{customdata[1]:.2f} $ per MWh \
            <br>Installed Capacity = %{customdata[2]:.2f} MW \
            <extra></extra>')

        data = []
        for name in color.name:
            data.append(go.Bar(
                name=name,
                x=plants.loc[plants.name == name, "x"],
                y=plants.loc[plants.name == name, "mc_el"],
                width=plants.loc[plants.name == name, "g_max"],
                marker_color=plants.loc[plants.name == name, "color"],
                legendgroup=name,
                hovertemplate=hovertemplate,
                customdata=plants.loc[plants.name == name, ["name", "mc_el", "g_max"]]
            ))
            
        layout = go.Layout(
            template="simple_white",
            bargap=0,
            legend=dict(x=1.1),
            yaxis=dict(title='Costs [$ per MWh]'),
            xaxis=dict(title='Installed Capacity [MW]'),

        )
        fig = go.Figure(data=data, layout=layout)
        if filepath:
            fig.write_html(str(filepath))
        if show_plot:
            plot(fig)
        else:
            return fig
