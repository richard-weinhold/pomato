"""Input Processing of POMATO, which processes the raw ipunt data into something usefule.

This collection of functions can be used to modify the input data, but the main idea is to have 
properly formatted input data which does not require additional processing.
"""

import numpy as np
import pandas as pd

def default_net_position(data):
    """Add default net position."""
    data.net_position = pd.DataFrame(index=data.demand_el.timestep.unique(), 
                                            columns=data.zones.index, 
                                            data=data.options["data"]["default_net_position"]).stack().reset_index()

    data.net_position.columns = [col for col in data.model_structure["net_position"].keys() if col != "index"]

def process_inflows(data):
    """Process inflows to (hydro-) storages.

    If no raw data create an all zero timeseries for all electric storage (plant_type es)
    power plants
    """
    if data.inflows.empty:
        inflows_columns = [col for col in data.model_structure["inflows"].keys() if col != "index"]
        data.inflows = pd.DataFrame(columns=inflows_columns)
        data.inflows["timestep"] = data.demand_el.timestep.unique()

    tmp = data.inflows.pivot(index="timestep", columns="plant", values="inflow").fillna(0)
    condition = data.plants.plant_type.isin(data.options["optimization"]["plant_types"]["es"])
    for es_plant in data.plants.index[condition]:
        if es_plant not in tmp.columns:
            tmp[es_plant] = 0
    data.inflows = pd.melt(tmp.reset_index(), id_vars=["timestep"], value_name="inflow").dropna()


def unique_mc(data):
    """Make marginal costs unique.

    This is done by adding a small increment multiplied by the number if plants with the
    same mc. This makes the solver find a unique solition (at leat in regards to generation
    scheduel) and is sopposed to have positive effect on solvetime.
    """
    for marginal_cost in data.plants.mc_el:
        condition_mc = data.plants.mc_el == marginal_cost
        data.plants.loc[condition_mc, "mc"] = \
        data.plants.mc_el[condition_mc] + \
        [int(x)*1E-4 for x in range(0, len(data.plants.mc_el[condition_mc]))]

def line_susceptance(data):
    """Calculate line susceptance for lines that have none set.

    This is not maintained as the current grid data set includes this parameter. However, this
    Was done with the simple formula b = length/type ~ where type is voltage level. While this
    is technically wrong, it works with linear load flow, as it only relies on the
    conceptual "conductance"/"resistance" of each circuit/line in relation to others.
    """
    if ("x per km" in data.lines.columns)&("voltage" in data.nodes.columns):
        data.lines['x'] = data.lines['x per km'] * data.lines["length"] * 1e-3
        data.lines.loc[data.lines.technology == "transformer", 'x'] = 0.01
        base_mva = 100
        base_kv = data.nodes.loc[data.lines.node_i, "voltage"].values
        # base_kv = 110
        v_base = base_kv * 1e3
        s_base = base_mva * 1e6
        z_base = np.power(v_base,2)/s_base
        data.lines['x'] = np.divide(data.lines['x'], z_base)
        data.lines['b'] = np.divide(1, data.lines['x'])

