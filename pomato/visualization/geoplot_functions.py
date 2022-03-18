"""Collection of functions used in the generation of the GeoPlot."""

import io
from math import atan2

import geojson
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.spatial
import shapely
import shapely.geometry
import shapely.ops
from PIL import Image


def merc(lat, lon):
    """convert lat lon to x,y"""
    r_major = 6378137.000
    coord_x = r_major * np.radians(lon)
    scale = coord_x/lon
    coord_y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    return(coord_x, coord_y)


def _create_geo_json(zones, nodes, boundaries=None):
    node_coordinates = nodes.loc[:, ["lon", "lat"]].values
    if not boundaries:
        lon_min, lat_min = node_coordinates.min(axis=0)*.99
        lon_max, lat_max= node_coordinates.max(axis=0)*1.01
        # Four corners counter clockwise, first element == last element
        boundaries = [[lon_min, lat_min], [lon_min, lat_max], 
                        [lon_max, lat_max], [lon_max, lat_min], 
                        [lon_min, lat_min]]

    points = np.vstack([node_coordinates, np.array(boundaries)])
    vor = scipy.spatial.Voronoi(points) #pylint: disable=no-member
    boundaries = shapely.geometry.Polygon(boundaries)
    features = []
    for zone in zones.index:
        node_polygons = []
        for node in nodes[nodes.zone == zone].index:
            node_index = nodes.index.get_loc(node)
            vor_regions = vor.regions[vor.point_region[node_index]]
            if -1 in vor_regions:
                vor_regions.remove(-1)
            vertices = vor.vertices[vor.regions[vor.point_region[node_index]]]
            center_x, center_y = (max(vertices[:, 0]) + min(vertices[:, 0])) / 2, (max(vertices[:, 1]) + min(vertices[:, 1])) / 2
            vertices = [list(x) for x in vertices]
            vertices.sort(key=lambda c:atan2(c[0] - center_x, c[1] - center_y))
            vertices.append(vertices[0])
            node_polygons.append(shapely.geometry.Polygon(vertices).intersection(boundaries))
        zone_polygon = shapely.ops.unary_union(node_polygons)    
        features.append(geojson.Feature(geometry=zone_polygon.__geo_interface__, id=zone))  
    return geojson.FeatureCollection(features)

def _build_raster(nodes, plot_width, plot_hight, alpha=4):
    """Build Raster for prices layer"""

    raster = np.zeros((plot_width, plot_hight))
    known_points_coords = [[b.x, b.y] for i,b in nodes.iterrows()]
    raster[nodes.x.values, nodes.y.values] = nodes.marginal.values

    raster_coords = np.array([[x,y] for x in range(plot_width) for y in range(plot_hight)])
    distance_matrix = scipy.spatial.distance.cdist(raster_coords, known_points_coords)
    condition = np.all(distance_matrix > 0, axis=1)
    distance_matrix = distance_matrix[condition]
    tmp = np.divide(1.0, np.power(distance_matrix, alpha))
    raster_values = tmp.dot(nodes.marginal.values)/tmp.sum(axis=1)
    
    for i, (x, y) in enumerate(raster_coords[condition]):
        raster[x, y] = raster_values[i]
    return raster

def add_prices_layer(nodes, prices, compress=True):
    """Adds prices layer to Geoplot"""

    if isinstance(compress, bool):
        quantile = .01
        prices.loc[prices.marginal > prices.marginal.quantile(1 - quantile), "marginal"] = prices.marginal.quantile(1 - quantile)
        prices.loc[prices.marginal < prices.marginal.quantile(quantile), "marginal"] = prices.marginal.quantile(quantile)
    elif isinstance(compress, tuple):
        prices.loc[prices.marginal < compress[0], "marginal"] = compress[0]
        prices.loc[prices.marginal > compress[1], "marginal"] = compress[1]


    nodes = pd.merge(nodes[["lat", "lon"]], prices, left_index=True, right_index=True)
    lat, lon = list(nodes.lat.values), list(nodes.lon.values)
    lat.extend([min(lat)*0.99, max(lat)*1.01])
    lon.extend([min(lon)*0.95, max(lon)*1.05])
    # Calculate plot dimensions
    xx, yy = merc(np.array(lat), np.array(lon))
    ratio = (max(xx) - min(xx))/(max(yy) - min(yy))
    # size = (max(yy) - min(yy))/5e4
    size = 300
    # prices Plot Coordinates (0,0) (plot_width, plot_hight)
    x = ((xx - min(xx))/max(xx - min(xx))*ratio*size).astype(int)
    y = ((yy - min(yy))/max(yy - min(yy))*size).astype(int)
    nodes["x"], nodes["y"] = x[:-2], y[:-2]
    plot_width, plot_hight = x.max() + 1, y.max() + 1
    prices_layer = _build_raster(nodes, plot_width, plot_hight, alpha=4)
    
    lon_min, lon_max = min(lon), max(lon)
    lat_min, lat_max = min(lat), max(lat)
    corners = [[lon_max, lat_min], [lon_max, lat_max], 
               [lon_min, lat_max], [lon_min, lat_min]]

    return prices_layer, corners, plot_hight/plot_width


def line_colors(line_data, flow_type="n_0_flow", threshold=0, loading_range=(0,100)):
    """Line colors in 10 shades of RedYellowGreen palette"""
    ## 0: N-0 Flows, 1: N-1 Flows 2: Line voltage levels
    # timesteps = 't'+ "{0:0>4}".format(int(slider.value))
    stepsize = round((loading_range[1] - loading_range[0])/10, 3)
    steps = [loading_range[0] + i*stepsize for i in range(0, 10)]
    RdYlGn = ('#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', 
              '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026') 
    flows = line_data[flow_type].to_frame()
    flows.columns = ["flow"]
    flows["alpha"] = 0.8
    condition_threshold = abs(flows.flow.values)/line_data.capacity < threshold/100
    flows.loc[condition_threshold, "alpha"] = 0.1
    flows["color"] = RdYlGn[0]
    for idx, loading in enumerate(steps):
        condition = abs(flows.flow.values)/line_data.capacity > loading/100
        flows.loc[condition, "color"] = RdYlGn[idx]
    color = list(flows.color.values)
    line_alpha = list(flows.alpha.values)


    return color, line_alpha

def line_coordinates(lines, nodes):
    """Create line coordinates for the geo plot.

    Each line contains 5 coordinates. Start, middle and end point. And 2 coordinates to visualize
    multilines where they branch out across a circle around start and end nodes.  

    Parameters
    ----------
    lines : pd.DataFrame
        Line data, with columns node_i, node_j indicating connections.
    nodes : pd.DataFrame
        Node data, with columns lat, lon for coordinates.

    Returns
    -------
    line_x, line_y, lists
        Lists for x,y coordinates each line a list of 5 coordinates, start, start radius, middle, 
        end radius end.  
    """    
    # Add Columns to lines with total systems and an index
    # so each line from a system has an relative index e.g. 1/3 or 4/5
    tmp = lines[["node_i", "node_j"]].copy()
    tmp["systems"] = 1
    tmp = tmp.groupby(["node_i", "node_j"], observed=True).sum()
    tmp = tmp.reset_index()
    lines["systems"] = 1
    lines["no"] = 1
    for node_i, node_j, systems in zip(tmp.node_i, tmp.node_j, tmp.systems):
        condition = (lines.node_i == node_i)&(lines.node_j == node_j)
        lines.loc[condition, "systems"] = systems
        # np.array bc of bug when assigning a 2-elm list
        lines.loc[condition, "no"] = np.array([nr for nr in range(0, systems)])

    line_x, line_y = [], []
    for l in lines.index:
        xi, yi = (nodes.lat[lines.node_i[l]], nodes.lon[lines.node_i[l]])
        xj, yj = (nodes.lat[lines.node_j[l]], nodes.lon[lines.node_j[l]])
        mx = xj - xi
        my = yj - yi
        # multiple lines are spread across a circle with radius d around each node
        # starting from PI/4 in equal steps (in angle) to -PI/4 from reference point
        # reference point is the intersection of the circle and the line to the other node
        # the 20 and pi/5 are purely visual
        d = 0.05*np.power((np.sqrt(np.power(mx, 2) + np.power(my, 2))), 1/2)
        if lines.systems[l] == 1:
            idx = 0
        else:
            idx = lines.no[l]/(lines.systems[l] - 1) - 0.5
        if mx == 0:
            alpha = np.pi/4*idx + np.pi/2
            alpha2 = 3/2*np.pi - np.pi/4*idx
        elif mx > 0: # bottom left -> top right | top left -> bottom right
            alpha = np.arctan(my/mx) + np.pi/4*idx
            alpha2 = np.arctan(my/mx) + np.pi - np.pi/4*idx
        elif mx < 0: # bottom right -> top left | top right -> bottom right
            alpha2 = np.arctan(my/mx) + np.pi/4*idx
            alpha = np.arctan(my/mx) + np.pi - np.pi/4*idx

        # lx contains start point, point on circle for multiple lines on start point,
        # a point 1/2 of the way for the hover menus to stick to
        # point on circle for multiple lines on end point, end point
        line_x.append([xi, xi + np.cos(alpha)*d,
                   0.5*(xi + np.cos(alpha)*d + xj + np.cos(alpha2)*d),
                   xj + np.cos(alpha2)*d, xj])
        line_y.append([yi, yi + np.sin(alpha)*d,
                   0.5*(yi + np.sin(alpha)*d + yj + np.sin(alpha2)*d),
                   yj + np.sin(alpha2)*d, yj])
    return line_x, line_y

def line_voltage_colors(lines): 
    #{380: 'red', 400: 'red', 220: 'green', 232: 'green', 165: 'grey', 150: 'grey', 132: 'black'}
    if not "voltage" in lines.columns:
        lines[["voltage"]] = lines[["type"]].copy()
    tmp = lines[["voltage"]].copy()
    tmp["voltage"] = lines.loc[:, "voltage"].apply(pd.to_numeric, errors='coerce')
    tmp["color"] = ""
    for line in tmp.index:
        if tmp.loc[line, "voltage"] > 500:
            tmp.loc[line, "color"] = "blue"
        elif tmp.loc[line, "voltage"] > 300:
            tmp.loc[line, "color"] = "red"
        elif tmp.loc[line, "voltage"] > 200:
            tmp.loc[line, "color"] = "green" 
        elif tmp.loc[line, "voltage"] > 100:
            tmp.loc[line, "color"] = "black" 
        elif tmp.loc[line, "voltage"] <= 100:
            tmp.loc[line, "color"] = "grey" 
        else:
            tmp.loc[line, "color"] = "purple"

    return list(tmp.color)


def create_redispatch_trace(nodes, reference_size, plotly_function):
    nodes.loc[:, ["delta_abs", "delta_pos", "delta_neg"]] /= 1000

    if isinstance(reference_size, (int, float)):
        sizeref = 2*reference_size/20**2
    else:
        sizeref = 2*max(nodes['delta_abs'])/20**2

    trace = []
    for pos_neg, condition, color in zip(["delta_neg", "delta_pos"], [nodes["delta_neg"] < 0, nodes["delta_pos"] > 0], ["red", "green"]):
        markers = plotly_function["marker"](
            size = nodes.loc[condition, pos_neg].abs(),
            sizeref = sizeref,
            sizemin = 1,
            sizemode='area',
            color = color,
            opacity=0.7,
            # line = {"color": 'rgb(40,40,40)'},
            # line_width=0 if ,
            autocolorscale=True
            )
        # Custom Data One for both pos/neg redispatch
        customdata = nodes.loc[condition, ["zone", "delta_pos", "delta_neg"]].reset_index()
        trace.append(plotly_function["geo"](
            lon = nodes.loc[condition, 'lon'],
            lat = nodes.loc[condition, 'lat'],
            marker = markers,
            customdata=customdata,
            hovertemplate=
            "<br>".join([
                "Node: %{customdata[0]}",
                "Zone: %{customdata[1]}",
                "Positive Redispatch: %{customdata[2]:.2f} GWh",
                "Negative Redispatch: %{customdata[3]:.2f} GWh",
            ]) + "<extra></extra>"
        ))
    return trace

def create_curtailment_trace(nodes, plotly_function):
    condition = (nodes.CURT > 0)
    sizeref = max(2*max(nodes.loc[condition, 'CURT'])/12**2, 1)
    trace = plotly_function["geo"](
        lon = nodes.loc[condition, 'lon'],
        lat = nodes.loc[condition, 'lat'],
        mode = 'markers',
        marker = plotly_function["marker"](
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
            "Curtailment: %{customdata[2]:.2f} GW",
        ]) + "<extra></extra>"
    )
    return trace
    
def create_infeasibilities_trace(nodes, plotly_function):
    sizeref = max(4*nodes[["pos", "neg"]].max().max()/12**2, 1)
    trace = []
    for col, color in zip(["pos", "neg"], ["#4575B4", "#F46D43"]):
        condition = nodes[col] > 0
        trace.append(plotly_function["geo"](
            lon = nodes.loc[condition, 'lon'],
            lat = nodes.loc[condition, 'lat'],
            mode = 'markers',
            marker = plotly_function["marker"](
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
                "Infeasibilities: + %{customdata[2]:.2f}, - %{customdata[3]:.2f} GWh",
            ]) + "<extra></extra>"
        ))
    return trace

def create_price_layer(nodes, prices):
    colorscale = "RdBu_r"
    contours = 15
    # price_high = 60
    # price_low = 50
    price_compress = True
    prices_layer, coordinates, hight_width = add_prices_layer(nodes, prices, price_compress)
    price_fig = go.Figure(
        data=go.Contour(z=prices_layer, 
                        showscale=False, 
                        colorscale=colorscale, 
                        ncontours=contours,
                        # contours=dict(start=price_low,end=price_high,size=30/contours)
                        )
    )
    price_fig.update_layout(
        width=5e3, height=5e3*hight_width, 
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
            "opacity": .25,
        }
    # Price Colorbar
    price_colorbar = go.Scatter(
        x=[None],y=[None], 
        mode='markers',
        hoverinfo='none',
        marker=dict(
            colorscale=colorscale, 
            showscale=True,
            cmin=prices_layer.min().round(2),
            cmax=prices_layer.max().round(2),
            colorbar=dict(thickness=5)) 
    )

    return price_layer, price_colorbar

def create_custom_data_lines(lines, dispay_data, line_coords, subset=None):
    if not isinstance(subset, list):
        subset = lines.index
    lons, lats = [], []
    customdata = np.empty((0, len(dispay_data.columns) + 1))
    for line in subset:
        i = lines.index.get_loc(line)
        lats.extend(line_coords[0, i])
        lats.append(None)
        lons.extend(line_coords[1, i])
        lons.append(None)
        customdata = np.vstack(
            [customdata, 
            np.tile([line] + list(dispay_data.loc[line, :]), (len(line_coords[0, i]),1)),
            np.repeat(None, len(dispay_data.columns) + 1)
        ])
    return lons, lats, customdata
