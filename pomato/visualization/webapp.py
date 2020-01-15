import os.path
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

from bokeh import palettes
from bokeh.io import show
from bokeh.layouts import column, row, widgetbox
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Select, \
                         LassoSelectTool, Circle, MultiLine, \
                         Label, TapTool
from bokeh.models.widgets import RadioButtonGroup, Slider
# pylint: disable-msg=C0103

def merc(lat, lon):
    """convert lat lon to x,y"""
    r_major = 6378137.000
    coord_x = r_major * np.radians(lon)
    scale = coord_x/lon
    coord_y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    return(coord_x, coord_y)


def update_line_colors(lines, lineflow, range_start=0, range_end=100):
    """Line colors in 10 shades of RedYellowGreeen palette"""
    ## 0: N-0 Flows, 1: N-1 Flows 2: Line voltage levels
    stepsize = round((range_end - range_start)/10, 3)
    steps = [range_start + i*stepsize for i in range(0, 10)]
    lineflow = lineflow.to_frame()
    lineflow.columns = ["flow"]
    lineflow["alpha"] = 0.4
    lineflow["color"] = palettes.RdYlGn[10][0]
    for idx, loading in enumerate(steps):
        cond = abs(lineflow.flow.values)/lines.maxflow > loading/100
        lineflow.loc[cond, "color"] = palettes.RdYlGn[10][idx]
    color = list(lineflow.color.values)
    line_alpha = list(lineflow.alpha.values)

    return color, line_alpha

def prepare_line_plot(lines, nodes):
    """Creates Line Data Source Object"""
    # Add Columns to lines with total systems and an index
    # so each line from a system has an relative index e.g. 1/3 or 4/5
    tmp = lines[["node_i", "node_j"]].copy()
    tmp.loc[:, "systems"] = 1
    tmp = tmp.groupby(["node_i", "node_j"]).sum()
    tmp = tmp.reset_index()
    lines.loc[:, "systems"] = 1
    lines.loc[:, "no"] = 1
    for node_i, node_j, systems in zip(tmp.node_i, tmp.node_j, tmp.systems):
        condition = (lines.node_i == node_i)&(lines.node_j == node_j)
        lines.loc[condition, "systems"] = systems
        # np.array bc of bug when assining a 2-elm list
        lines.loc[condition, "no"] = np.array([nr for nr in range(0, systems)])

    lx, ly = [], []
    for l in lines.index:
        if lines.systems[l] == 1:
            xi, yi = merc(nodes.lat[lines.node_i[l]], nodes.lon[lines.node_i[l]])
            xj, yj = merc(nodes.lat[lines.node_j[l]], nodes.lon[lines.node_j[l]])
            lx.append([xi, (xi + xj)*0.5, xj])
            ly.append([yi, (yi + yj)*0.5, yj])
        else:
            xi, yi = merc(nodes.lat[lines.node_i[l]], nodes.lon[lines.node_i[l]])
            xj, yj = merc(nodes.lat[lines.node_j[l]], nodes.lon[lines.node_j[l]])
            mx = xj - xi
            my = yj - yi
            # multiple lines are spread across a circle with radius d around each node
            # starting from PI/4 in equal steps ( in angle) to -PI/4 from referece poin
            # reference point is the intersection of the circle and the line to the other node
            # the 20 and pi/5 are purely visual
            d = 36*np.power((np.sqrt(np.power(mx, 2) + np.power(my, 2))), 1/3)
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

            # lx contains start point, point on circle for mult lines on start point,
            # a point 1/2 of the way for the hover menue to stick to
            # point on circle for mult lines on end point, end point
            lx.append([xi, xi + np.cos(alpha)*d,
                       0.5*(xi + np.cos(alpha)*d + xj + np.cos(alpha2)*d),
                       xj + np.cos(alpha2)*d, xj])
            ly.append([yi, yi + np.sin(alpha)*d,
                       0.5*(yi + np.sin(alpha)*d + yj + np.sin(alpha2)*d),
                       yj + np.sin(alpha2)*d, yj])

    return lx, ly

def get_tilesource():
  from bokeh.tile_providers import get_provider, Vendors
  from bokeh.models import WMTSTileSource
  # STAMEN_LITE = WMTSTileSource(
  #     url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png',
  #     # url='http://tile.stamen.com/terrain-lines/{Z}/{X}/{Y}.png',
  #     # url='https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}@2x.png',
  #     attribution=(
  #         'Probably cc'
  #     ))
  tile_source = get_provider("STAMEN_TONER_BACKGROUND")
  return tile_source

def create_static_plot(lines, nodes, dclines, inj, flow_n_0, flow_n_1, flow_dc):

    coords_x, coords_y, lat, lon = [], [], [], []
    for i in nodes.index:
        coord_x, coord_y = merc(nodes.lat[i], nodes.lon[i])
        coords_x.append(coord_x)
        coords_y.append(coord_y)
        lat.append(nodes.lat[i])
        lon.append(nodes.lon[i])

    nodes_dict = {"x": coords_x,
                  "y": coords_y,
                  "lat": lat,
                  "lon": lon,
                  "node": list(nodes.index),
                  "name": list(nodes.name),
                  "zone": list(nodes.zone),
                  "inj": list(inj)}

    lx, ly = prepare_line_plot(lines, nodes)


    color, line_alpha = update_line_colors(lines, flow_n_1)

    line_dict = {"lx": lx, "ly": ly,
             "line": list(lines.index),
             "max_flow": list(lines.maxflow),
             "flow": list(flow_n_0.values),
             "n_1_flow": list(flow_n_1.values),
             "node_i": list(lines.node_i),
             "node_j": list(lines.node_j),
             "contingency": list(lines.contingency),
             "color": color,
             "line_alpha": line_alpha}

    if flow_dc.empty:
        flow_dc = []

    lx_dc, ly_dc = [], []
    for l in dclines.index:
        xi, yi = merc(nodes.lat[dclines.node_i[l]], nodes.lon[dclines.node_i[l]])
        xj, yj = merc(nodes.lat[dclines.node_j[l]], nodes.lon[dclines.node_j[l]])
        lx_dc.append([xi, (xi+xj)/2, xj])
        ly_dc.append([yi, (yi+yj)/2, yj])



    dcline_dict = {"lx": lx_dc,
                   "ly": ly_dc,
                   "line": list(dclines.index),
                   "flow": list(flow_dc.values),
                   "max_flow": list(dclines.maxflow),
                   "node_i": list(dclines.node_i),
                   "node_j": list(dclines.node_j)}

    # Main plot
    ## Dimensions of map (=de)
    lat_max, lat_min = nodes.lat.max(), nodes.lat.min()
    lon_max, lon_min = nodes.lon.max() + 0.2, nodes.lon.min() - 0.2
    x_range, y_range = [], []
    for coord in [[lat_min, lon_min], [lat_max, lon_max]]:
        x, y = merc(coord[0], coord[1])
        x_range.append(x)
        y_range.append(y)

    hover_line = [("Line", "@line"),
                  ("Node_i", "@node_i"),
                  ("Node_j", "@node_j"),
                  ("Capacity", "@max_flow"),
                  ("Flow", "@flow"),
                  ("N-1 Flow", "@n_1_flow"),
                  ("Contingency", "@contingency")]

    hover_dcline = [("Line", "@line"),
                    ("Node_i", "@node_i"),
                    ("Node_j", "@node_j"),
                    ("Capacity", "@max_flow"),
                    ("Flow", "@flow")]

    hover_node = [("Node", "@node"),
                  ("Name", "@name"),
                  ("Zone", "@zone"),
                  ("(lat,lon)", "(@lat, @lon)"),
                  ("Inj", "@inj")]

    fig = figure(tools="pan,wheel_zoom", active_scroll="wheel_zoom",
                 x_range=(x_range), y_range=(y_range), plot_height=700, plot_width=800)

    STAMEN_LITE = get_tilesource()
    fig.add_tile(STAMEN_LITE)
    fig.axis.visible = False

    # lines and dc lines as line plots
    l = fig.multi_line("lx", "ly", color="color", hover_line_color='color',
                       line_width=3, line_alpha="line_alpha",
                       hover_line_alpha=1.0, source=line_dict)

    dc = fig.multi_line("lx", "ly", color="blue", hover_line_color='blue',
                        line_width=3, line_alpha=0.6, hover_line_alpha=1.0,
                        line_dash="dashed", source=dcline_dict)

    # hover tool which picks up information from lines and dc lines
    fig.add_tools(HoverTool(renderers=[l], show_arrow=False,
                            line_policy='next', tooltips=hover_line))

    fig.add_tools(HoverTool(renderers=[dc], show_arrow=False,
                            line_policy='next', tooltips=hover_dcline))

    ## nodes plot
    color = palettes.Category20b[4]
    # color sample for various states normal-hover-selected-notselected
    n = fig.circle("x", "y", size=5, fill_alpha=0.8, 
                    fill_color=color[3], source=nodes_dict)
    fig.add_tools(HoverTool(renderers=[n], tooltips=hover_node))

    show(fig)
    return fig