# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v1.1.0 (required)
# ----------------------------------
#
# This file:
# Julia mutable struct definitions for efficient data handling
# -------------------------------------------------------------------------------------------------

mutable struct Grid
    # Attributes
    index::Any
    ptdf::Vector{Float64}
    ram::Float64
    # Optional Attributes
    reference_flow::Dict
    function Grid(index::Any, 
                  ptdf::Vector{Float64}, 
                  ram::Float64)
        z = new()
        z.index = index
        z.ptdf = ptdf
        z.ram = ram
        return z
    end
end

mutable struct Zone
    # Attributes
    index::Any
    demand::Dict
    nodes::Array
    plants::Array
    # Optional Attributes
    net_position::Dict
    net_export::Dict
    ntc::Dict
    function Zone(index::Any, 
                  demand::Dict, 
                  nodes::Array, 
                  plants::Array)
        z = new()
        z.index = index
        z.demand = demand
        z.nodes = nodes
        z.plants = plants
        return z
    end
end

mutable struct Heatarea
    # Attributes
    index::Any
    demand::Dict
    plants::Array
    function Heatarea(index::Any, 
                      demand::Dict, 
                      plants::Array)
        ha = new()
        ha.index = index
        ha.demand = demand
        ha.plants = plants
        return ha
    end
end

mutable struct Node
    # Attributes
    index::Any
    zone::Any
    slack::Bool
    demand::Dict
    plants::Array
    # Optional Attributes
    net_export::Dict
    slack_zone::Array
    function Node(index::Any, 
                  zone::Any, 
                  demand::Dict, 
                  slack::Bool, 
                  plants::Array)
        n = new()
        n.index = index
        n.zone = zone
        n.demand = demand
        n.slack = slack
        n.plants = plants
        return n
    end
end

mutable struct Plant
    # Attributes
    index::Any
    node::Any
    mc::Float64
    g_max::Float64
    h_max::Float64
    eta::Float64
    tech::Any
    # Optional Attributes
    heatarea::Heatarea
    availability::Dict
    function Plant(index::Any, 
                   node::Any,
                   mc::Float64,
                   eta::Float64, 
                   g_max::Float64, 
                   h_max::Float64, 
                   tech::Any)
        p = new()
        p.index = index
        p.node = node
        p.mc = mc
        p.eta = eta
        p.g_max = g_max
        p.h_max = h_max
        p.tech = tech
        return p
    end
end

mutable struct DC_Line
    # Attributes
    index::Any
    node_i::Any
    node_j::Any
    maxflow::Float64
    function DC_Line(index::Any, 
                     node_i::Any, 
                     node_j::Any, 
                     maxflow::Float64)
        l = new()
        l.index = index
        l.node_i = node_i
        l.node_j = node_j
        l.maxflow = maxflow
        return l
    end
end
