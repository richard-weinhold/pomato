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
# Define Data-Subsets based on general properties of the data
# -------------------------------------------------------------------------------------------------


# Power-to-heat plants
PH_TECH = ["heatpump", "elheat"]
# Plants that use time-series for capacity of p
TS_TECH = ["wind onshore", "wind offshore", "solar", "solarheat", "iheat"]
# El storgae technologies 
EL_STORAGE_TECH = ["psp", "reservoir"]
#Conventional Plants
function get_co(plants::Dict)
    set = []
    for p in keys(plants)
        if (plants[p].g_max > 0) & (plants[p].tech != "dem")
            push!(set, plants[p].index)
        end
    end
    return set
end

# Heat plants
function get_he(plants::Dict)
    set = []
    for p in keys(plants)
        if plants[p].h_max > 0 
            push!(set, plants[p].index)
        end
    end
    return set
end

# CHP plants
function get_chp(plants::Dict)
    set = []
    for p in keys(plants)
         if (plants[p].g_max > 0) & (plants[p].h_max > 0)
            push!(set, plants[p].index)
        end
    end
    return set
end

#Electricity Storage
function get_es(plants::Dict)
    set = []
    for p in keys(plants)
         if in(plants[p].tech, EL_STORAGE_TECH) & (plants[p].g_max > 0)
            push!(set, plants[p].index)
        end
    end
    return set
end

#Heat Storage
function get_hs(plants::Dict)
    set = []
    for p in keys(plants)
         if (plants[p].tech == "storage") & (plants[p].h_max > 0)
            push!(set, plants[p].index)
        end
    end
    return set
end

# Power to heat
function get_ph(plants::Dict)
    set = []
    for p in keys(plants)
         if in(plants[p].tech, PH_TECH)
            push!(set, plants[p].index)
        end
    end
    return set
end

# Demand Units
function get_d(plants::Dict)
    set = []
    for p in keys(plants)
         if plants[p].tech == "dem"
            push!(set, plants[p].index)
        end
    end
    return set
end

# Plants that use time-series for their capacity (e.g. RES)
function get_ts(plants::Dict)
    set = []
    for p in keys(plants)
         if in(plants[p].tech, TS_TECH)
            push!(set, plants[p].index)
        end
    end
    return set
end

# Slack nodes
function get_slack(nodes::OrderedDict)
    set = []
    for n in keys(nodes)
        if nodes[n].slack
            push!(set, nodes[n].index)
        end
    end
    return set
end

# DC Lines
function get_dclines(dclines::Dict)
    # Only DC lines are included as lines in the model
    # Rest of the network is represented by PTDF
    return collect(keys(dclines))
end
