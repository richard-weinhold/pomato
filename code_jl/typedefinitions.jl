# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v0.6.2 (required)
# ----------------------------------
#
# This file:
# Julia mutable struct definitions for efficient data handling
# -------------------------------------------------------------------------------------------------

mutable struct Grid
    index::Any
    ptdf::Array{Float64}
    ram::Float64
    reference_flow::Dict
    function Grid(index, ptdf, ram)
        z = new()
        z.index = index
        z.ptdf = ptdf
        z.ram = ram
        return z
    end
end

mutable struct Zone
    index::Any
    demand::Dict
    # ntc_to::Dict
    nodes::Any
    net_position::Dict
    net_export::Dict
    plants::Array
    function Zone(index, demand, nodes, plants)
        z = new()
        z.index = index
        z.demand = demand
        z.nodes = nodes
        z.plants = plants
        return z
    end
end

mutable struct Heatarea
    index::Any
    name::String
    demand::Dict
    plants::Array
    function Heatarea(index, demand, plants)
        ha = new()
        ha.index = index
        ha.demand = demand
        ha.plants = plants
        return ha
    end
end

mutable struct Node
    index::Any
    zone::Zone
    slack::Bool
    name::String
    demand::Dict
    net_export::Dict
    plants::Array
    function Node(index, zone, slack, name, plants)
        n = new()
        n.index = index
        n.zone = zone
        n.slack = slack
        n.name = name
        n.plants = plants
        return n
    end
end

mutable struct Plant
    index::Any
    efficiency::Float64
    g_max::Float64
    h_max::Float64
    tech::Any
    heatarea::Heatarea
    mc::Float64
    node::Node
    function Plant(index, efficiency, g_max, h_max, tech, mc)
        p = new()
        p.index = index
        p.efficiency = efficiency
        p.g_max = g_max
        p.h_max = h_max
        p.tech = tech
        p.mc = mc
        return p
    end
end

mutable struct DC_Line
    index::Any
    node_i::Node
    node_j::Node
    maxflow::Float64
    function DC_Line(index, node_i, node_j, maxflow)
        l = new()
        l.index = index
        l.node_i = node_i
        l.node_j = node_j
        l.maxflow = maxflow
        return l
    end
end

mutable struct Availability
    plant::Plant
    value::Dict
    function Availability(plant, value)
        a = new()
        a.plant = plant
        a.value = value
        return a
    end
end
