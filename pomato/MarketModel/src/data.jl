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
    index::Int
    name::String
    ptdf::Vector{Float64}
    ram::Float64
    # Optional Attributes
    timestep::String
    reference_flow::Dict
    zone::Union{String, Nothing}
    function Grid(index::Int,
                  name::String,
                  ptdf::Vector{Float64},
                  ram::Float64)
        z = new()
        z.index = index
        z.name = name
        z.ptdf = ptdf
        z.ram = ram
        return z
    end
end

mutable struct Zone
    # Attributes
    index::Int
    name::Any
    demand::Array
    nodes::Array
    plants::Array
    res_plants::Array
    # Optional Attributes
    net_position::Array
    net_export::Array
    ntc::Array
    function Zone(index::Int,
                  name::Any,
                  demand::Array,
                  nodes::Array,
                  plants::Array,
                  res_plants::Array)
        z = new()
        z.index = index
        z.name = name
        z.demand = demand
        z.nodes = nodes
        z.plants = plants
        z.res_plants = res_plants
        return z
    end
end

mutable struct Heatarea
    # Attributes
    index::Int
    name::Any
    demand::Array
    plants::Array
    res_plants::Array
    function Heatarea(index::Int,
                      name::Any,
                      demand::Array,
                      plants::Array,
                      res_plants::Array)
        ha = new()
        ha.index = index
        ha.name = name
        ha.demand = demand
        ha.plants = plants
        ha.res_plants = res_plants
        return ha
    end
end

mutable struct Node
    # Attributes
    index::Int
    name::Any
    zone::Int
    slack::Bool
    demand::Array
    plants::Array
    res_plants::Array
    # Optional Attributes
    net_export::Array
    slack_zone::Array
    function Node(index::Int,
                  name::Any,
                  zone::Int,
                  demand::Array,
                  slack::Bool,
                  plants::Array,
                  res_plants::Array)
        n = new()
        n.index = index
        n.name = name
        n.zone = zone
        n.demand = demand
        n.slack = slack
        n.plants = plants
        n.res_plants = res_plants
        return n
    end
end

mutable struct Renewables
        index::Int
        name::Any
        g_max::Float64
        h_max::Float64
        mc_el::Float64
        mc_heat::Float64
        mu::Array
        mu_heat::Array
        sigma::Array
        sigma_heat::Array
        node::Int
        plant_type::Any
        function Renewables(index::Int, name::Any,
                            g_max::Float64, h_max::Float64,
                            mc_el::Float64, mc_heat::Float64,
                            availability::Array, node::Int, plant_type::Any)
            res = new()
            res.index = index
            res.g_max = g_max
            res.h_max = h_max
            res.mc_el = mc_el
            res.mc_heat = mc_heat
            res.name = name
            factor_sigma =  0.3
            res.mu = availability * g_max
            res.mu_heat = availability * h_max
            res.sigma = (availability * factor_sigma)*g_max
            res.sigma_heat = (availability * factor_sigma)*h_max
            res.node = node
            res.plant_type = plant_type
            return res
        end
    end


mutable struct Plant
    # Attributes
    index::Int
    name::Any
    node::Int
    mc_el::Float64
    mc_heat::Float64
    g_max::Float64
    h_max::Float64
    eta::Float64
    plant_type::Any
    # Optional Attributes
    storage_capacity::Float64
    inflow::Array
    function Plant(index::Int,
                   name::Any,
                   node::Int,
                   mc_el::Float64,
                   mc_heat::Float64,
                   eta::Float64,
                   g_max::Float64,
                   h_max::Float64,
                   plant_type::Any)
        p = new()
        p.index = index
        p.name = name
        p.node = node
        p.mc_el = mc_el
        p.mc_heat = mc_heat
        p.eta = eta
        p.g_max = g_max
        p.h_max = h_max
        p.plant_type = plant_type
        return p
    end
end

mutable struct DC_Line
    # Attributes
    index::Int
    name::Any
    node_i::Int
    node_j::Int
    maxflow::Float64
    function DC_Line(index::Int,
                     name::Any,
                     node_i::Int,
                     node_j::Int,
                     maxflow::Float64)
        l = new()
        l.index = index
        l.name = name
        l.node_i = node_i
        l.node_j = node_j
        l.maxflow = maxflow
        return l
    end
end

mutable struct Timestep
    index::Int
    name::String
    function Timestep(index::Int, name::String)
        t = new()
        t.index = index
        t.name = name
        return t
    end
end

mutable struct Data
    # Attributes
    nodes::Vector{Node}
    zones::Vector{Zone}
    heatareas::Vector{Heatarea}
    plants::Vector{Plant}
    renewables::Vector{Renewables}
    grid::Vector{Grid}
    dc_lines::Vector{DC_Line}
    t::Vector{Timestep}
    folders::Dict{String, String}

    function Data(nodes::Vector{Node}, zones::Vector{Zone},
                  heatareas::Vector{Heatarea}, plants::Vector{Plant},
                  renewables::Vector{Renewables}, grid::Vector{Grid},
                  dc_lines::Vector{DC_Line}, t::Vector{Timestep})
          d = new()
          d.nodes = nodes
          d.zones = zones
          d.heatareas = heatareas
          d.plants = plants
          d.renewables = renewables
          d.grid = grid
          d.dc_lines = dc_lines
          d.t = t
          return d
      end
end
