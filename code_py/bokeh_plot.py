import os.path
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json

from bokeh import palettes
from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Select, WMTSTileSource, LassoSelectTool, Circle, Label
from bokeh.models.widgets import RadioButtonGroup, Slider

pd.options.mode.chained_assignment = None  # default='warn'
wdir = Path(sys.argv[1])
# wdir = Path("C:/Users/riw/tubCloud/Uni/Market_Tool/pomato/data_temp/bokeh_files")


STAMEN_LITE = WMTSTileSource(
    url='http://tile.stamen.com/toner-lite/{Z}/{X}/{Y}.png',
    attribution=(
        'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
        'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'
        'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
        'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
    ))

def init_market_data(market_db):
    """load data from market_result/market_db forder"""
    data_dir = wdir.joinpath("market_result").joinpath(market_db)
    nodes = pd.read_csv(data_dir.joinpath("nodes.csv"), index_col=0)
    dclines = pd.read_csv(data_dir.joinpath("dclines.csv"), index_col=0)
    fuel = pd.read_csv(data_dir.joinpath("fuel.csv"), index_col=0)
    g_by_fuel = pd.read_csv(data_dir.joinpath("g_by_fuel.csv"), index_col=0)
    demand = pd.read_csv(data_dir.joinpath("demand.csv"), index_col=0)
    inj = pd.read_csv(data_dir.joinpath("inj.csv"), index_col=0)
    f_dc = pd.read_csv(data_dir.joinpath("f_dc.csv"), index_col=0)
    t_dict = json.load(open(data_dir.joinpath("t.json")))

    return nodes, dclines, fuel, g_by_fuel, demand, inj, f_dc, t_dict["t_first"], t_dict["t_last"]

def init_grid_data(market_db):
    """load data from market_result/market_db forder"""
    data_dir = wdir.joinpath("market_result").joinpath(market_db)
    lines = pd.read_csv(data_dir.joinpath("lines.csv"), index_col=0)
    n_1_flows = pd.read_csv(data_dir.joinpath("n_1_flows.csv"), index_col=0)
    n_0_flows = pd.read_csv(data_dir.joinpath("n_0_flows.csv"), index_col=0)

    return lines, n_0_flows, n_1_flows

def update_market_data(attr, old, new):
    """update market data"""
    label.text = "Loading Market Data..."

    source_ind.data["color"] = ["red"]
    _, _, _, g_by_fuel, demand, inj, f_dc, t_first, t_last = init_market_data(select_market_db.value)

    source_g_fuel.data = ColumnDataSource.from_df(g_by_fuel)
    source_d.data = ColumnDataSource.from_df(demand)
    source_inj.data = ColumnDataSource.from_df(inj)
    source_dc.data = ColumnDataSource.from_df(f_dc)
    slider.start = t_first
    slider.end = t_last
    slider.value = t_first
    label.text = "Data is Loaded!"
    source_ind.data["color"] = ["green"]

def update_grid_data(attr, old, new):

    label.text = "Loading Grid Data..."
    source_ind.data["color"] = ["red"]
    lines, n_0_flows, n_1_flows = init_grid_data(select_market_db.value)

    source_n_0_flows.data = ColumnDataSource.from_df(n_0_flows)
    source_n_1_flows.data = ColumnDataSource.from_df(n_1_flows)
    source_lines_data.data = ColumnDataSource.from_df(lines)

    line_dict = create_line_source(lines)
    source_lines.data = line_dict
    label.text = "Data is Loaded!"
    source_ind.data["color"] = ["green"]

def merc(lat, lon):
    """convert lat lon to x,y"""
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
    return(x, y)

def available_files_by_ext(folder, ext):
    """ check folder for files with ext"""
    list_files = []
    for file in folder.rglob(f"*.{ext}"):
        list_files.append(file.name)
    return(list_files)

def update_line_colors(option=0):
    """Line colors in 10 shades of RedYellowGreeen palette"""
    ## 0: N-0 Flows, 1: N-1 Flows 2: Line voltage levels
    t = 't'+ "{0:0>4}".format(int(slider.value))
    lines = source_lines_data.to_df()
    if option == 1:
        n_1_flows = source_n_1_flows.to_df()
        color = []
        line_alpha = []
        for line in lines.index:
            color.append(palettes.RdYlGn11[min(int(abs(n_1_flows[t][line])/(lines.maxflow[line])*10),10)])
            line_alpha.append(0.3)
    elif option == 0:
        n_0_flows = source_n_0_flows.to_df()
        color = []
        line_alpha = []
        for line in lines.index:
            color.append(palettes.RdYlGn11[min(int(abs(n_0_flows[t][line])/(lines.maxflow[line])*10),10)])
            line_alpha.append(0.3)

    elif option == 2:
        color = []
        line_alpha = []
        type_color = {380: 'red', 220: 'green', 150: 'black'}
        if "cnec" in lines.columns:
            for l in lines.index:
                if lines.cnec[l]:
                    color.append("blue")
                    line_alpha.append(1.0)
                elif lines.cb[l]:
                    color.append("purple")
                    line_alpha.append(1.0)
                else:
                    color.append(type_color[lines.type[l]])
                    line_alpha.append(0.1)
        else:
            for l in lines.index:
                color.append(type_color[lines.type[l]])
                line_alpha.append(0.6)
    return color, line_alpha

def injections(t):
    """update net injection depending of t"""
    df_inj = source_inj.to_df()
    t = 't'+ "{0:0>4}".format(int(t))
    return list(df_inj[(df_inj.t==t)].INJ.values)

def dclines_flow(t):
    t = 't'+ "{0:0>4}".format(int(t))
    df_dc = source_dc.to_df()
    return list(df_dc.F_DC[df_dc.t == t])

def n_0_flow(t):
    t = 't'+ "{0:0>4}".format(int(t))
    n_0_flows = source_n_0_flows.to_df()
    return list(n_0_flows[t])

def n_1_flow(t):
    t = 't'+ "{0:0>4}".format(int(t))
    n_1_flows = source_n_1_flows.to_df()
    return list(n_1_flows[t])

### Bokeh Core Function which are associated with WIdgets
def update_node_injections(attrname, old, new):
    source_nodes.data["inj"] = injections(slider.value)

def update_line_loadings(attrname, old, new):
    source_lines.data["color"], source_lines.data["line_alpha"] = update_line_colors(option=flow_type_botton.active)
    source_lines.data["flow"] = n_0_flow(slider.value)
    source_lines.data["n_1_flow"] = n_1_flow(slider.value)
    source_dclines.data["flow"] = dclines_flow(slider.value)

def update_stacked_bars(attrname, old, new):
    nodes = []
    for i in n.data_source.selected['1d']['indices']:
        nodes.append(source_nodes.data["node"][i])

    if nodes != []:
        gen_dict, y_max = gen_fuel(int(slider.value),nodes)
        list_fuels = list(gen_dict.keys())
        list_fuels.remove("type")
        x,y,leg,c = [],[],[],[]
        for i,f in enumerate(list_fuels):
            if sum(gen_dict[f]) != 0:
                leg.append(f)
                c.append(palettes.viridis(len(list_fuels))[i])
                x.append("gen")
                y.append(-1)

        source_stacked_bars.data = gen_dict
        source_legend.data= dict(legend=leg,color=c,x=x,y=y)
        fig_bar.y_range.end = np.ceil(y_max/10000)*10000

## Create/Init source objects
def gen_fuel(t,nodes):
    """Generate source for Generation/Demand in Bar-Plot"""
    ## Plot consists of all gen fuels and demand
    ## - first column only gen - second column only demand
    gen_fuel = source_g_fuel.to_df()
    demand = source_d.to_df()
    demand = demand.drop("index", axis=1)
    gen_fuel = gen_fuel.drop("index", axis=1)

    t = 't'+ "{0:0>4}".format(int(t))
    tmp = gen_fuel[(gen_fuel.t==t)&(gen_fuel.node.isin(nodes))].groupby("fuel").sum()
    gen_max = tmp.values.sum()
    tmp["D"] = 0
    fuel_dict = {}
    for i in fuel.fuel:
        if i in tmp.index:
            custom_list = tmp.loc[i].tolist()
            fuel_dict[i] = custom_list
        else:
            fuel_dict[i] = [0,0]
    fuel_dict["type"] = ["gen", "dem"]
    dem_max = demand.d_total[(demand.n.isin(nodes))&(demand.t == t)].sum()

    fuel_dict["dem"] = [0, dem_max]
    return fuel_dict, max(dem_max, gen_max)

def create_line_source(lines):
    # Add Columns to lines with total systems and an index
    # so each line from a system has an relative index e.g. 1/3 or 4/5
    tmp = lines[["node_i", "node_j"]]
    tmp["tmp"] = 1
    tmp = tmp.groupby(["node_i", "node_j"]).sum()
    tmp = tmp.reset_index()
    lines["systems"] = 1
    lines["no"] = 1
    for i,j,s in zip(tmp.node_i, tmp.node_j, tmp.tmp):
        lines.systems[(lines.node_i == i)&(lines.node_j == j)] = s
        # np.array bc of bug when assining a 2-elm list
        lines.no[(lines.node_i == i)&(lines.node_j == j)] = np.array([x for x in range(0,s)])

    lx,ly = [],[]
    for l in lines.index:
        if lines.systems[l] == 1:
            xi,yi = merc(nodes.lat[lines.node_i[l]], nodes.lon[lines.node_i[l]])
            xj,yj = merc(nodes.lat[lines.node_j[l]], nodes.lon[lines.node_j[l]])
            lx.append([xi,(xi+xj)*0.5,xj])
            ly.append([yi,(yi+yj)*0.5,yj])
        else:
            xi,yi = merc(nodes.lat[lines.node_i[l]], nodes.lon[lines.node_i[l]])
            xj,yj = merc(nodes.lat[lines.node_j[l]], nodes.lon[lines.node_j[l]])
            mx = xj-xi
            my = yj-yi
            # multiple lines are spread across a circle with radius d around each node
            # starting from PI/4 in equal steps ( in angle) to -PI/4 from referece poin
            # reference point is the intersection of the circle and the line to the other node
            # the 20 and pi/5 are purely visual
            d = 24*np.power((np.sqrt(np.power(mx,2) + np.power(my,2))),1/3)
            idx = (lines.no[l])/(lines.systems[l]-1) - 0.5
            if mx == 0:
                alpha = np.pi/4*idx + np.pi/2
                alpha2 = 3/2*np.pi - np.pi/4*idx
            elif mx>0: # bottom left -> top right | top left -> bottom right
                alpha = np.arctan(my/mx) + np.pi/4*idx
                alpha2 = np.arctan(my/mx) + np.pi - np.pi/4*idx
            elif mx<0: # bottom right -> top left | top right -> bottom right
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

    color, line_alpha = update_line_colors()

    line_dict = {"lx": lx, "ly": ly,
                 "line": list(lines.index),
                 "max_flow": list(lines.maxflow),
                 "flow": n_0_flow(slider.value),
                 "n_1_flow": n_1_flow(slider.value),
                 "node_i": list(lines.node_i),
                 "node_j": list(lines.node_j),
                 "contingency": list(lines.contingency),
                 "color": color,
                 "line_alpha": line_alpha}

    return line_dict


# Init Data and Widgets
option_market_db = [path for path in os.listdir(wdir.joinpath("market_result")) if "." not in path]
select_market_db = Select(title="Model Run:", value=option_market_db[0], options=option_market_db)

nodes, dclines, fuel, g_by_fuel, demand, inj, f_dc, t_first, t_last = init_market_data(select_market_db.value)
lines, n_0_flows, n_1_flows = init_grid_data(select_market_db.value)

#Market Data
source_g_fuel = ColumnDataSource(g_by_fuel)
source_d = ColumnDataSource(demand)
source_dc = ColumnDataSource(f_dc)
source_inj = ColumnDataSource(inj)

#Grid Data
source_lines_data = ColumnDataSource(lines)
source_n_0_flows = ColumnDataSource(n_0_flows)
source_n_1_flows = ColumnDataSource(n_1_flows)

## Widgets and function calls
slider = Slider(start=t_first, end=t_last, value=t_first, step=1, title="Timestep",
                callback_throttle=250, callback_policy="throttle") # Throttle doesnt work as of now

flow_type_botton = RadioButtonGroup(name="Choose Flow Case:", width=300,
        labels=["N-0", "N-1", "Voltage Level"], active=0)
flow_type_botton.on_change('active', update_line_loadings)

##Update things when slider or selects are changed
slider.on_change('value', update_line_loadings)
slider.on_change('value', update_stacked_bars)
slider.on_change('value', update_node_injections)

select_market_db.on_change('value', update_market_data)
select_market_db.on_change('value', update_stacked_bars)
select_market_db.on_change('value', update_line_loadings)
select_market_db.on_change('value', update_node_injections)
select_market_db.on_change('value', update_grid_data)

## Init plot things from either data or function
line_dict = create_line_source(lines)
source_lines = ColumnDataSource(line_dict)

x,y,lat,lon = [],[],[],[]
for i in nodes.index:
    cx,cy = merc(nodes.lat[i], nodes.lon[i])
    x.append(cx)
    y.append(cy)
    lat.append(nodes.lat[i])
    lon.append(nodes.lon[i])

lx_dc,ly_dc = [],[]
for l in dclines.index:
    xi,yi = merc(nodes.lat[dclines.node_i[l]], nodes.lon[dclines.node_i[l]])
    xj,yj = merc(nodes.lat[dclines.node_j[l]], nodes.lon[dclines.node_j[l]])
    lx_dc.append([xi,(xi+xj)/2,xj])
    ly_dc.append([yi,(yi+yj)/2,yj])

## Init source objects for nodes dclines
source_nodes = ColumnDataSource(data=dict(
                x=x,
                y=y,
                lat=lat,
                lon=lon,
                node=list(nodes.index),
                name=list(nodes.name),
                zone=list(nodes.zone),
                inj=injections(slider.value)))

source_dclines = ColumnDataSource(data=dict(
                lx=lx_dc,
                ly=ly_dc,
                line=list(dclines.index),
                flow=dclines_flow(slider.value),
                node_i = list(dclines.node_i),
                node_j = list(dclines.node_j)))

hover_line =[#("index", "$index"),
            ("Line", "@line"),
            # ("Name", "@name"),
            ("Node_i", "@node_i"),
            ("Node_j", "@node_j"),
            ("Capacity", "@max_flow"),
            ("Flow", "@flow"),
            ("N-1 Flow", "@n_1_flow"),
            ("Contingency", "@contingency")]

hover_dcline =[#("index", "$index"),
            ("Line", "@line"),
            # ("Name", "@name"),
            ("Node_i", "@node_i"),
            ("Node_j", "@node_j"),
            ("Capacity", "@max_flow"),
            ("Flow", "@flow"),
            ("N-1 Flow", "@n_1_flow"),
            ("Contingency", "@contingency")]

hover_node = [#("index", "$index"),
                ("Node", "@node"),
                ("Name", "@name"),
                ("Zone", "@zone"),
                ("(lat,lon)", "(@lat, @lon)"),
                ("Inj", "@inj")]

### Initial Node that is selected
gen_dict, _ = gen_fuel(202, list(nodes.index.values))
list_fuels = list(gen_dict.keys())
list_fuels.remove("type")

color = palettes.viridis(len(list_fuels))

x,y,leg,c = [],[],[],[]
for i,f in enumerate(list_fuels):
    if sum(gen_dict[f]) != 0:
        leg.append(f)
        c.append(palettes.viridis(len(list_fuels))[i])
        x.append("gen")
        y.append(-1)

source_legend = ColumnDataSource(data=dict(legend=leg,color=c,x=x,y=y))
source_stacked_bars = ColumnDataSource(data=gen_dict)

## bar Chart with generation by fuel
fig_bar = figure(x_range = ["gen", "dem"], y_range= (0,2000), title="Nodal Generation",
                 toolbar_location=None, tools="", plot_height=1000, plot_width=300)
fig_bar.vbar_stack(stackers=list_fuels, x='type', width=.9, color=color,
             source=source_stacked_bars)
## Dummy plot to create the legend for the barplot # Needed bc vbar_stack does
## not support to subset the data presented in the legen ( e.g. will always display all fuels)
fig_bar.circle("x","y", alpha=1, color="color", legend="legend", source=source_legend)

# Main plot
## Dimensions of map (=de)
de =[[57, 4.4], [43, 16.4]]
x_range, y_range = [],[]
for c in de:
    x,y = merc(c[0],c[1])
    x_range.append(x)
    y_range.append(y)

fig = figure(tools = "pan,wheel_zoom,lasso_select", active_scroll="wheel_zoom",
             x_range=(x_range), y_range=(y_range), plot_width=1200, plot_height=1000)
fig.axis.visible = False
fig.select(LassoSelectTool).select_every_mousemove = False

# lines and dc lines as line plots
l = fig.multi_line("lx","ly", color="color", hover_line_color='color', line_width=4, line_alpha="line_alpha", hover_line_alpha=1.0, source=source_lines)
dc = fig.multi_line("lx","ly", color="blue", hover_line_color='blue', line_width=4, line_alpha=0.6, hover_line_alpha=1.0, line_dash="dashed", source=source_dclines)
# hover tool which picks up information from lines and dc lines
fig.add_tools(HoverTool(renderers=[l,dc], show_arrow=False, line_policy='next', tooltips=hover_line))

## nodes plot
# color sample for various states normal-hover-selected-notselected
color = palettes.Category20b[4]
n = fig.circle("x", "y", size=4,fill_alpha=0.8, fill_color=color[3], source=source_nodes)
fig.add_tools(HoverTool(renderers=[n], tooltips=hover_node))
# define color when (not-) selected
selected_node = Circle(fill_alpha=0.8, fill_color=color[0], line_color=None)
nonselected_node = Circle(fill_alpha=0.6, fill_color=color[3], line_color=None)
# and trigger it correctly
n.selection_glyph = selected_node
n.nonselection_glyph = nonselected_node
n.data_source.on_change('selected', update_stacked_bars)

#inidcator for data-load
source_ind = ColumnDataSource(data=dict(x=[1],y=[1],color=["green"]))
fig_indicator = figure(x_range=(0,8), y_range=(0,2), toolbar_location=None,
                       tools="", plot_height=100, plot_width=350)
label = Label(x=1.8, y=.8, text="Data is Loaded!", text_alpha=1)
fig_indicator.add_layout(label)

fig_indicator.axis.visible = False
fig_indicator.grid.visible = False
fig_indicator.outline_line_alpha = 0
fig_indicator.circle("x", "y", size=25, fill_alpha=0.6, fill_color="color", line_color=None, source=source_ind)

# add the geographical map
fig.add_tile(STAMEN_LITE)
# set up layout = row(row|column)
# UI sclaing for everthing except the widget column with the inducator plot
widgets =  column(widgetbox(slider, flow_type_botton, select_market_db, width=300), fig_indicator)
main_map = row(children=[fig, fig_bar])
layout = row(children=[main_map, widgets], sizing_mode="scale_height")

curdoc().add_root(layout)

