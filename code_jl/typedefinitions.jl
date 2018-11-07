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
# Julia type definitions for efficient data handling
# -------------------------------------------------------------------------------------------------

type Zone
    index::Any
    demand::Dict
    # ntc_to::Dict
    nodes::Any
    function Zone(index, demand, nodes)
        z = new()
        z.index = index
        z.demand = demand
        z.nodes = nodes
        return z
    end
end

type Heatarea
    index::Any
    name::String
    demand::Dict
    function Heatarea(index, demand)
        z = new()
        z.index = index
        z.demand = demand
        return z
    end
end

type Node
    index::Any
    zone::Zone   
    slack::Bool  
    name::String
    demand::Dict
    function Node(index, zone, slack, name)
        n = new()
        n.index = index
        n.zone = zone
        n.slack = slack
        n.name = name
        return n
    end
end

type Plant
    index::Any             
    efficiency::Float64 
    g_max::Float64   
    h_max::Float64   
    tech::Any
    heatarea::Heatarea
    mc::Float64     
    node::Node       
    has_avail::Bool
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

type DC_Line
    index::Any
    node_i::Node
    node_j::Node
    capacity::Float64
    function DC_Line(index, node_i, node_j, capacity)
        l = new()
        l.index = index
        l.node_i = node_i
        l.node_j = node_j
        l.capacity = capacity
        return l 
    end
end

type Availability
    plant::Plant
    value::Dict
    function Availability(plant, value)
        a = new()
        a.plant = plant
        a.value = value
        return a
    end
end
