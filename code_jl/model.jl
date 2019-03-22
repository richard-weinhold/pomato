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
# Central optimization implemented in JuMP
# -------------------------------------------------------------------------------------------------

function build_and_run_model(model_horizon::OrderedDict, 
                             options::Dict,
                             plants::Dict{String, Plant}, 
                             plants_in_ha::Dict{String, Dict{String, Plant}}, 
                             plants_at_node::Dict{String, Dict{String, Plant}},
                             plants_in_zone::Dict{String, Dict{String, Plant}}, 

                             availabilities::Dict{String, Availability},
                             nodes::OrderedDict{String, Node}, 
                             zones::OrderedDict{String, Zone}, 
                             heatareas::Dict{String, Heatarea},
                             nodes_in_zone::Dict{String, Dict{String, Node}},
                             
                             dc_lines::Dict{String, DC_Line}, 
                             grid::Dict{String, Grid},
                             ntc::Dict{Tuple, Number}, 
                             slack_zones::Dict
                             )

# Check for feasible model_type
model_type = options["type"]
possible_types = ["base", "dispatch", "ntc", "nodal", "cbco_nodal", "cbco_zonal", "d2cf"]
if !in(model_type, possible_types)
    println("Error: Model_Type $(model_type) unkown. Calculate Base Model")
    model_type = "base"
end
# Base Constraints are part of all models
# So far dispatch is identical to base, therefore no extra condition around the constraint
println("Starting $(model_type) Model")

# Timesteps
# needs to be numeric in order to perform operations such as t-1
t_set = collect(keys(model_horizon))
t_start = collect(keys(model_horizon))[1]
t_end = collect(keys(model_horizon))[end]

# Create plant subset lists
p_set = collect(keys(plants))
co_set = get_co(plants)
he_set = get_he(plants)
chp_set = get_chp(plants)
es_set = get_es(plants)
hs_set = get_hs(plants)
ph_set = get_ph(plants)
d_set = get_d(plants)
ts_set = get_ts(plants)

# Create area, node and line sets
slack_set = get_slack(nodes)
ha_set = collect(keys(heatareas))
z_set = collect(keys(zones))
n_set = collect(keys(nodes))
dc_set = collect(keys(dc_lines))
cb_set = collect(keys(grid))

# Create incidence matrix for dc-lines
dc_incidence = Dict()
for dc in dc_set
    tmp = Dict()
    for n in n_set
        tmp[n] = 0
    end
    dc_incidence[dc] = tmp
    dc_incidence[dc][dc_lines[dc].node_i.index] =  1
    dc_incidence[dc][dc_lines[dc].node_j.index] = -1
end

# Setup model
# disp = Model(solver=ClpSolver(SolveType=5))
# disp = Model(solver=GurobiSolver(Presolve=2, PreDual=2, Threads=8))
# disp = Model(solver=GurobiSolver(Method=0,Threads=1))
disp = Model(solver=GurobiSolver(Method=2))
# Variables
@variable(disp, G[t_set, p_set] >= 0) # El. power generation per plant p
@variable(disp, H[t_set, p_set] >= 0) # Heat generation per plant p
@variable(disp, D_es[t_set, es_set] >= 0) # El. demand of storage plants
@variable(disp, D_hs[t_set, hs_set] >= 0) # El. demand of heat storage
@variable(disp, D_ph[t_set, ph_set] >= 0) # El. demand of power to heat
@variable(disp, D_d[t_set, d_set] >= 0) # Electricity Demand by Demand Units
@variable(disp, L_es[t_set, es_set] >= 0) # Level of electricity storage
@variable(disp, L_hs[t_set, hs_set] >= 0) # Level of heat storage
@variable(disp, EX[t_set, z_set, z_set] >= 0) # Commercial Exchanges between zones (row from, col to)
@variable(disp, INJ[t_set, n_set]) # Net Injection at Node n
@variable(disp, F_DC[t_set, dc_set]) # Flow in DC Line dc

if options["infeas_heat"]
    # Relaxing at high costs to avoid infeasibility in heat EB
    @variable(disp, 0 <= INFEAS_H_NEG[t_set, ha_set] <= options["infeasibility_bound"]) 
    @variable(disp, 0 <= INFEAS_H_POS[t_set, ha_set] <= options["infeasibility_bound"])
else
    @variable(disp, INFEAS_H_NEG[t_set, ha_set] == 0)
    @variable(disp, INFEAS_H_POS[t_set, ha_set] == 0)
end

if options["infeas_el_nodal"]
    @variable(disp, 0 <= INFEAS_EL_N_NEG[t_set, n_set] <= options["infeasibility_bound"])
    @variable(disp, 0 <= INFEAS_EL_N_POS[t_set, n_set] <= options["infeasibility_bound"])
else
    @variable(disp, INFEAS_EL_N_NEG[t_set, n_set] == 0)
    @variable(disp, INFEAS_EL_N_POS[t_set, n_set] == 0)
end
if options["infeas_el_zonal"]
    @variable(disp, 0 <= INFEAS_EL_Z_NEG[t_set, z_set] <= options["infeasibility_bound"])
    @variable(disp, 0 <= INFEAS_EL_Z_POS[t_set, z_set] <= options["infeasibility_bound"])
else
    @variable(disp, INFEAS_EL_Z_NEG[t_set, z_set] == 0)
    @variable(disp, INFEAS_EL_Z_POS[t_set, z_set] == 0)
end

if options["infeas_lines"]
    @variable(disp, 0 <= INFEAS_LINES[t_set, cb_set] <= options["infeasibility_bound"])
else
    @variable(disp, INFEAS_LINES[t_set, cb_set] == 0)
end

if options["infeas_lines_ref"]
    @variable(disp, 0 <= INFEAS_REF_FLOW[t_set, cb_set] <= options["infeasibility_bound"])
else
    @variable(disp, INFEAS_REF_FLOW[t_set, cb_set] == 0)
end

## Make Constraint references to get the duals after solve
@constraintref EB_nodal[t_set, n_set]
JuMP.registercon(disp, :EB_nodal, EB_nodal)
@constraintref EB_zonal[t_set, z_set]
JuMP.registercon(disp, :EB_zonal, EB_zonal)

@objective(disp, Min, sum(sum(G[t, p]*plants[p].mc for p in p_set) for t in t_set)
                      + sum(sum(H[t, p]*plants[p].mc for p in p_set) for t in t_set)
                      + (sum(INFEAS_EL_Z_POS) + sum(INFEAS_EL_Z_NEG))*1e4
                      + (sum(INFEAS_EL_N_POS) + sum(INFEAS_EL_N_NEG))*1e2
                      + (sum(INFEAS_H_NEG) + sum(INFEAS_H_POS))*1e3
                      + (sum(INFEAS_LINES)*1e3)
                      + (sum(INFEAS_REF_FLOW)*1e1))

println("Building Constraints")
# Constraints for all times
for t in t_set
    # Applies to: Dispatch
    # Base Constraint
    for p in setdiff(p_set, union(he_set, ts_set, d_set))
        @constraint(disp, G[t, p] <= plants[p].g_max)
    end

    # All plants without he, ts and demand units
    for p in setdiff(he_set, union(chp_set, ts_set))
        @constraint(disp, H[t, p] <= plants[p].h_max)
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in chp_set
        @constraint(disp, G[t, p] >= ((plants[p].g_max*0.85) / plants[p].h_max) * H[t, p])
        @constraint(disp, G[t, p] <= plants[p].g_max * (1-(0.15 * H[t,p] / plants[p].h_max)))
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in intersect(ts_set, co_set)
        # WARNING this formulation requires the availabilty df to be correctly sorted by times
        @constraint(disp, G[t, p] <= plants[p].g_max * availabilities[p].value[model_horizon[t]])
        @constraint(disp, G[t, p] >= plants[p].g_max * availabilities[p].value[model_horizon[t]] * 0.8)
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in intersect(ts_set, he_set)
        @constraint(disp, H[t, p] <= plants[p].h_max * availabilities[p].value[model_horizon[t]])
        @constraint(disp, H[t, p] >= plants[p].h_max * availabilities[p].value[model_horizon[t]] * 0.8)
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in d_set
        @constraint(disp, D_d[t, p] <= plants[p].g_max * availabilities[p].value[model_horizon[t]])
        @constraint(disp, D_d[t, p] >=  plants[p].g_max * availabilities[p].value[model_horizon[t]] * 0.8)
    end
    for p in ph_set
        @constraint(disp, D_ph[t, p] == H[t, p] * plants[p].efficiency)
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in es_set
        #
        @constraint(disp, L_es[t, p]  == (t>t_start ? L_es[t-1, p] : plants[p].g_max*2)
                                        - G[t, p]
                                        + plants[p].efficiency*D_es[t, p])
        @constraint(disp, L_es[t, p] <= plants[p].g_max*8)
        @constraint(disp, D_es[t, p] <= plants[p].g_max)
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in hs_set
        @constraint(disp, L_hs[t, p] ==  (t>t_start ? plants[p].efficiency*L_hs[t-1, p] : plants[p].h_max*2)
                                        - H[t, p]
                                        + D_hs[t, p])
        @constraint(disp, L_hs[t, p] <= plants[p].h_max*4)
        @constraint(disp, D_hs[t, p] <= plants[p].h_max)
    end


    # Applies to: Dispatch
    # Base Constraint
    for ha in ha_set
        # WARNING this formulation requires the demand df to be correctly sorted by times
        p_in_ha_set = collect(keys(plants_in_ha[ha]))
        @constraint(disp, heatareas[ha].demand[model_horizon[t]] ==
                              sum(H[t, p] for p in p_in_ha_set)
                            - sum(D_hs[t, p] for p in intersect(p_in_ha_set, hs_set))
                            + INFEAS_H_POS[t, ha] - INFEAS_H_NEG[t, ha])
    end

    # Zonal Energy Balance
    # Applies to: Dispatch
    # Base Constraint
    for z in z_set
        p_in_z_set = collect(keys(plants_in_zone[z]))
        EB_zonal[t,z] = @constraint(disp, zones[z].demand[model_horizon[t]] + zones[z].net_export[model_horizon[t]] ==
                                  sum(G[t, p] for p in intersect(p_in_z_set, co_set))
                                - sum(D_ph[t, p] for p in intersect(p_in_z_set, ph_set))
                                - sum(D_es[t, p] for p in intersect(p_in_z_set, es_set))
                                - sum(D_d[t, p] for p in intersect(p_in_z_set, d_set))
                                - sum(EX[t, z, zz] for zz in z_set)
                                + sum(EX[t, zz, z] for zz in z_set)
                                + sum(INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n] for n in collect(keys(nodes_in_zone[z])))
                                # + INFEAS_EL_Z_POS[t, z] - INFEAS_EL_Z_NEG[t, z]
                                )
    end

    # Nodal Energy Balance
    # Applies to: Dispatch
    # Base Constraint
    for n in n_set
        p_in_n_set = collect(keys(plants_at_node[n]))
        EB_nodal[t,n] = @constraint(disp, nodes[n].demand[model_horizon[t]] + nodes[n].net_export[model_horizon[t]] ==
                                  sum(G[t, p] for p in intersect(p_in_n_set, co_set))
                                - sum(D_ph[t, p] for p in intersect(p_in_n_set, ph_set))
                                - sum(D_es[t, p] for p in intersect(p_in_n_set, es_set))
                                - sum(D_d[t, p] for p in intersect(p_in_n_set, d_set))
                                - sum((F_DC[t, dc]*dc_incidence[dc][n]) for dc in dc_set)
                                - INJ[t, n]
                                + INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n]
                                )
    end

    # DC Lines Constraints
    # Applies to: Dispatch
    # Base Constraint
    for dc in dc_set
        @constraint(disp, F_DC[t, dc] <= dc_lines[dc].maxflow)
        @constraint(disp, F_DC[t, dc] >= -dc_lines[dc].maxflow)
    end

    # NTC Constraints
    # Applies to: ntc
    if in(model_type, ["ntc"])
        for z in z_set
            for zz in z_set
                @constraint(disp, EX[t, z, zz] <=  ntc[(z, zz)])
            end
        end
    end

     # Slack Constraint
    # Applies to: ntc, nodal, cbco_nodal, cbco_zonal
    if in(model_type, ["ntc", "nodal", "cbco_nodal", "cbco_zonal", "d2cf"])
        for slack in slack_set
            # INJ have to be balanced within a slack_zone
            @constraint(disp, 0 == sum(INJ[t, n] for n in slack_zones[slack]))
        end
    end

    # Cbco Constraints
    # Applies to: cbco_nodal, nodal
    if in(model_type, ["cbco_nodal", "nodal"])
        for cb in cb_set
            @constraint(disp, sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set)) <= grid[cb].ram ) #+ INFEAS_LINES[t, cb])
            @constraint(disp, sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set)) >= -(grid[cb].ram)) #+ INFEAS_LINES[t, cb]))
        end
    end

    # Applies to: cbco_zonal
    if in(model_type, ["cbco_zonal"])
        for cb in cb_set
            @constraint(disp, sum((sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))
                                    *grid[cb].ptdf[i] for (i, z) in enumerate(z_set)) <= grid[cb].ram + INFEAS_LINES[t, cb])
            @constraint(disp, sum((sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))
                                    *grid[cb].ptdf[i] for (i, z) in enumerate(z_set)) >= -(grid[cb].ram + INFEAS_LINES[t, cb]))
        end
    end

    # Applies to d2cf model:
    # set net_position net_position:
    if in(model_type, ["d2cf"])
        for z in ["DE", "NL", "BE", "FR"]
            ### net_position when positive -> export
            @constraint(disp, sum(EX[t, z, zz] - EX[t, zz, z] for zz in z_set) <=
                              zones[z].net_position[model_horizon[t]] + 0.2*abs(zones[z].net_position[model_horizon[t]])
                              )
            @constraint(disp, sum(EX[t, z, zz] - EX[t, zz, z] for zz in z_set) >=
                              zones[z].net_position[model_horizon[t]] - 0.2*abs(zones[z].net_position[model_horizon[t]])
                              )
        end

        # set reference flows in CNECS
        for cb in cb_set
            if grid[cb].reference_flow[model_horizon[t]] != 0
                @constraint(disp, sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set))
                                  <= grid[cb].reference_flow[model_horizon[t]] + INFEAS_REF_FLOW[t, cb])
                @constraint(disp, sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set))
                                  >= grid[cb].reference_flow[model_horizon[t]]  - INFEAS_REF_FLOW[t, cb])
            end
            # Upper/Lower Bound on CBs
            @constraint(disp, sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set))
                              <= grid[cb].ram) # + INFEAS_LINES[t, cb])
            @constraint(disp, sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set))
                              >= -grid[cb].ram) # - INFEAS_LINES[t, cb])
        end
    end
end

for p in es_set
    @constraint(disp, L_es[t_end, p] >= 2*plants[p].g_max)
end

println("Solving...")
t_start = time_ns()
println()
# println("Number of Linear Constraints in the Model: ", MathProgBase.numlinconstr(disp))
solve(disp)
println("Objective: $(getobjectivevalue(disp))")
t_elapsed = time_ns() - t_start
println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")

result_folder = WDIR*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM")
if !isdir(result_folder)
    println("Creating Results Folder: ", result_folder)
    mkdir(result_folder)
end

results_parameter = [[:G, [:t, :p], false],
					 [:H, [:t, :p], false],
					 [:D_es, [:t, :p], false],
					 [:D_hs, [:t, :p], false],
					 [:D_ph, [:t, :p], false],
					 [:D_d, [:t, :p], false],
					 [:L_es, [:t, :p], false],
					 [:L_hs, [:t, :p], false],
					 [:EX, [:t, :z, :zz], false],
					 [:INJ, [:t, :n], false],
					 [:F_DC, [:t, :dc], false],
					 [:INFEAS_H_NEG, [:t, :ha], false],
					 [:INFEAS_H_POS, [:t, :ha], false],
					 [:INFEAS_EL_N_NEG, [:t, :n], false],
					 [:INFEAS_EL_N_POS, [:t, :n], false],
					 [:INFEAS_EL_Z_NEG, [:t, :z], false],
					 [:INFEAS_EL_Z_POS, [:t, :z], false],
					 [:INFEAS_LINES, [:t, :cb], false],
					 [:INFEAS_REF_FLOW, [:t, :cb], false],
					 [:EB_nodal, [:t, :n], true],
					 [:EB_zonal, [:t, :z], true],
					]

for par in results_parameter
	jump_to_df(disp, par[1], par[2], par[3], result_folder)
end

# Misc Results or Data
misc_result = Dict()
misc_result["Objective Value"] = getobjectivevalue(disp)
# write the file with the stringdata variable information
open(result_folder*"/misc_result.json", "w") do f
        write(f, JSON.json(misc_result))
end
end
