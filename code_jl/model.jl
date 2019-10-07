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

function build_and_run_model(WDIR::String,
                             model_horizon::OrderedDict,
                             options::Dict,
                             plant_types::Dict,

                             plants::Dict{String, Plant},
                             nodes::OrderedDict{String, Node},
                             zones::OrderedDict{String, Zone},
                             heatareas::Dict{String, Heatarea},

                             grid::Dict{String, Grid},
                             dc_lines::Dict{String, DC_Line}
                             )


result_folder = WDIR*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM")
if !isdir(result_folder)
    println("Creating Results Folder")
    mkdir(result_folder)
end
### Set Options
model_type = options["type"]
curtailment_electricity = options["parameters"]["curtailment"]["electricity"]
curtailment_heat = options["parameters"]["curtailment"]["heat"]
chp_efficiency = options["parameters"]["chp_efficiency"]
storage_start = 0.65

# Check for feasible model_type
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
co_set = get_co(plants, plant_types)
he_set = get_he(plants)
chp_set = get_chp(plants)
es_set = get_es(plants, plant_types)
hs_set = get_hs(plants, plant_types)
ph_set = get_ph(plants, plant_types)
d_set = get_d(plants, plant_types)
ts_set = get_ts(plants, plant_types)

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
    dc_incidence[dc][dc_lines[dc].node_i] =  1
    dc_incidence[dc][dc_lines[dc].node_j] = -1
end

# Setup model
# disp = Model(solver=ClpSolver(SolveType=5))
# disp = Model(solver=GurobiSolver(Presolve=2, PreDual=2, Threads=8))
# disp = Model(solver=GurobiSolver(Method=0,Threads=1))
disp = Model(with_optimizer(Gurobi.Optimizer, Method=1, LogFile=result_folder*"/log.txt"))
# disp = Model(with_optimizer(GLPK.Optimizer))

# Variables
@variable(disp, G[t_set, p_set] >= 0) # El. power generation per plant p
@variable(disp, H[t_set, p_set] >= 0) # Heat generation per plant p
@variable(disp, D_es[t_set, es_set] >= 0) # El. demand of storage plants
@variable(disp, D_hs[t_set, hs_set] >= 0) # El. demand of heat storage
@variable(disp, D_ph[t_set, ph_set] >= 0) # El. demand of power to heat
@variable(disp, L_es[t_set, es_set] >= 0) # Level of electricity storage
@variable(disp, L_hs[t_set, hs_set] >= 0) # Level of heat storage
@variable(disp, EX[t_set, z_set, z_set] >= 0) # Commercial Exchanges between zones (row from, col to)
@variable(disp, INJ[t_set, n_set]) # Net Injection at Node n
@variable(disp, F_DC[t_set, dc_set]) # Flow in DC Line dc

@variable(disp, COST_G)
@variable(disp, COST_H)
@variable(disp, COST_EX >= 0)
@variable(disp, COST_INEAS_EL >= 0)
@variable(disp, COST_INEAS_H >= 0)
@variable(disp, COST_INEAS_LINES >= 0)

if options["infeasibility"]["heat"]
    # Relaxing at high costs to avoid infeasibility in heat EB
    @variable(disp, 0 <= INFEAS_H_NEG[t_set, ha_set] <= options["infeasibility"]["bound"])
    @variable(disp, 0 <= INFEAS_H_POS[t_set, ha_set] <= options["infeasibility"]["bound"])
else
    @variable(disp, INFEAS_H_NEG[t_set, ha_set] == 0)
    @variable(disp, INFEAS_H_POS[t_set, ha_set] == 0)
end

if options["infeasibility"]["electricity"]
    @variable(disp, 0 <= INFEAS_EL_N_NEG[t_set, n_set] <= options["infeasibility"]["bound"])
    @variable(disp, 0 <= INFEAS_EL_N_POS[t_set, n_set] <= options["infeasibility"]["bound"])

else
    @variable(disp, INFEAS_EL_N_NEG[t_set, n_set] == 0)
    @variable(disp, INFEAS_EL_N_POS[t_set, n_set] == 0)
end

if options["infeasibility"]["lines"]
    @variable(disp, 0 <= INFEAS_LINES[t_set, cb_set] <= options["infeasibility"]["bound"])
else
    @variable(disp, INFEAS_LINES[t_set, cb_set] == 0)
end


@objective(disp, Min, COST_G + COST_H + COST_EX + COST_INEAS_EL + COST_INEAS_H + COST_INEAS_LINES)

@constraint(disp, COST_G == sum(sum(G[t, p]*plants[p].mc_el for p in p_set) for t in t_set))
@constraint(disp, COST_H == sum(sum(H[t, p]*plants[p].mc_heat for p in p_set) for t in t_set))
@constraint(disp, COST_EX == sum(EX)*1e-1)
@constraint(disp, COST_INEAS_EL == (sum(INFEAS_EL_N_POS) + sum(INFEAS_EL_N_NEG))*1e2)
@constraint(disp, COST_INEAS_H == (sum(INFEAS_H_NEG) + sum(INFEAS_H_POS))*1e3)
@constraint(disp, COST_INEAS_LINES == sum(INFEAS_LINES)*1e3)

println("Building Constraints")
# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=setdiff(p_set, union(he_set, ts_set))],
    G[t, p] <= plants[p].g_max)

# All plants without he, ts and demand units
@constraint(disp, [t=t_set, p=setdiff(he_set, union(chp_set, ts_set))],
    H[t, p] <= plants[p].h_max)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=chp_set],
    G[t, p] >= ((plants[p].g_max*(1 - chp_efficiency)) / plants[p].h_max) * H[t, p])

@constraint(disp, [t=t_set, p=chp_set],
    G[t, p] <= plants[p].g_max * (1-(chp_efficiency * H[t,p] / plants[p].h_max)))

# Applies to: Dispatch
# Base Constraint
# WARNING this formulation requires the availabilty df to be correctly sorted by times
@constraint(disp, [t=t_set, p=intersect(ts_set, co_set)],
    G[t, p] <= plants[p].g_max * plants[p].availability[model_horizon[t]])
@constraint(disp, [t=t_set, p=intersect(ts_set, co_set)],
    G[t, p] >= plants[p].g_max * plants[p].availability[model_horizon[t]] * curtailment_electricity)

# Applies to: Dispatch
# Base Constraint

@constraint(disp, [t=t_set, p=intersect(ts_set, he_set)],
    H[t, p] <= plants[p].h_max * plants[p].availability[model_horizon[t]])
@constraint(disp, [t=t_set, p=intersect(ts_set, he_set)],
    H[t, p] >= plants[p].h_max * plants[p].availability[model_horizon[t]] * curtailment_heat)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=ph_set],
    D_ph[t, p] == H[t, p] / plants[p].eta)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=es_set],
    L_es[t, p]  == (t>t_start ? L_es[t-1, p] : storage_start*plants[p].storage_capacity)
                   + plants[p].inflow[model_horizon[t]]
                   - G[t, p]
                   + plants[p].eta*D_es[t, p])

@constraint(disp, [t=t_set, p=es_set],
    L_es[t, p] <= plants[p].storage_capacity)
@constraint(disp, [t=t_set, p=es_set],
    D_es[t, p] <= plants[p].g_max*0)

@constraint(disp, [p=es_set],
    L_es[t_end, p] >= storage_start*plants[p].storage_capacity)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=hs_set],
    L_hs[t, p] ==  (t>t_start ? plants[p].eta*L_hs[t-1, p] : storage_start*plants[p].storage_capacity)
                   - H[t, p]
                   + D_hs[t, p])

@constraint(disp, [t=t_set, p=hs_set],
    L_hs[t, p] <= plants[p].storage_capacity)

@constraint(disp, [t=t_set, p=hs_set],
    D_hs[t, p] <= plants[p].h_max)

# @constraint(disp, [p=hs_set],
#     L_hs[t_end, p] >= storage_start*plants[p].storage_capacity)

# Applies to: Dispatch
# Base Constraint
# WARNING this formulation requires the demand df to be correctly sorted by times
@constraint(disp, [t=t_set, ha=ha_set],
    heatareas[ha].demand[model_horizon[t]] ==
      sum(H[t, p] for p in heatareas[ha].plants)
    - sum(D_hs[t, p] for p in intersect(heatareas[ha].plants, hs_set))
    + INFEAS_H_POS[t, ha] - INFEAS_H_NEG[t, ha])

# Zonal Energy Balance
# Applies to: Dispatch
# Base Constraint
@constraint(disp, EB_zonal[t=t_set, z=z_set],
    zones[z].demand[model_horizon[t]] ==
    + zones[z].net_export[model_horizon[t]]
    + sum(G[t, p] for p in intersect(zones[z].plants, co_set))
    - sum(D_ph[t, p] for p in intersect(zones[z].plants, ph_set))
    - sum(D_es[t, p] for p in intersect(zones[z].plants, es_set))
    - sum(EX[t, z, zz] for zz in z_set)
    + sum(EX[t, zz, z] for zz in z_set)
    + sum(INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n] for n in zones[z].nodes)
    )

# Nodal Energy Balance
# Applies to: Dispatch
# Base Constraint
@constraint(disp, EB_nodal[t=t_set, n=n_set],
    nodes[n].demand[model_horizon[t]] ==
    + nodes[n].net_export[model_horizon[t]]
    + sum(G[t, p] for p in intersect(nodes[n].plants, co_set))
    - sum(D_ph[t, p] for p in intersect(nodes[n].plants, ph_set))
    - sum(D_es[t, p] for p in intersect(nodes[n].plants, es_set))
    - sum((F_DC[t, dc]*dc_incidence[dc][n]) for dc in dc_set)
    - INJ[t, n]
    + INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n]
    )

# DC Lines Constraints
# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, dc=dc_set],
    F_DC[t, dc] <= dc_lines[dc].maxflow)
@constraint(disp, [t=t_set, dc=dc_set],
    F_DC[t, dc] >= -dc_lines[dc].maxflow)

# NTC Constraints
# Applies to: ntc
if in(model_type, ["ntc", "zonal", "cbco_zonal"])
    @constraint(disp, [t=t_set, z=z_set, zz=z_set],
        EX[t, z, zz] <=  zones[z].ntc[zz])
end

# Slack Constraint
# Applies to: ntc, nodal, cbco_nodal, cbco_zonal
# "ntc"
if in(model_type, ["ntc", "nodal", "cbco_nodal", "cbco_zonal", "d2cf"])
    # INJ have to be balanced within a slack_zone
    @constraint(disp, [t=t_set, slack=slack_set],
        0 == sum(INJ[t, n] for n in nodes[slack].slack_zone))
end

# Cbco Constraints
# Applies to: cbco_nodal, nodal
if in(model_type, ["cbco_nodal", "nodal"])
    @constraint(disp, [t=t_set, cb=cb_set], grid[cb].ptdf' * INJ[t, :] <= grid[cb].ram + INFEAS_LINES[t, cb])
    @constraint(disp, [t=t_set, cb=cb_set], grid[cb].ptdf' * INJ[t, :] >= -grid[cb].ram - INFEAS_LINES[t, cb])
end

# Applies to: cbco_zonal
if in(model_type, ["cbco_zonal"])
    @constraint(disp, [t=t_set, cb=cb_set],
        sum((sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))*grid[cb].ptdf[i] for (i, z) in enumerate(z_set))
        <= grid[cb].ram + INFEAS_LINES[t, cb])
    @constraint(disp, [t=t_set, cb=cb_set],
        sum((sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))*grid[cb].ptdf[i] for (i, z) in enumerate(z_set))
        >= -(grid[cb].ram - INFEAS_LINES[t, cb]))
end


# Applies to nodal model for basecase calculation:
if in(model_type, ["cbco_nodal_s"])
    for t in t_set
        for z in ["DE", "NL", "BE", "FR"]
            ### net_position when positive -> export
            @constraint(disp, sum(EX[t, z, zz] - EX[t, zz, z] for zz in z_set) <=
                              zones[z].net_position[model_horizon[t]] + 0.2*abs(zones[z].net_position[model_horizon[t]])
                              )
            @constraint(disp, sum(EX[t, z, zz] - EX[t, zz, z] for zz in z_set) >=
                              zones[z].net_position[model_horizon[t]] - 0.2*abs(zones[z].net_position[model_horizon[t]])
                              )
        end
    end
end

println("Solving...")
t_start = time_ns()
@time JuMP.optimize!(disp)
println("Objective: $(JuMP.objective_value(disp))")
t_elapsed = time_ns() - t_start
println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")

results_parameter = [[:G, [:t, :p], false],
                     [:H, [:t, :p], false],
                     [:D_es, [:t, :p], false],
                     [:D_hs, [:t, :p], false],
                     [:D_ph, [:t, :p], false],
                     [:L_es, [:t, :p], false],
                     [:L_hs, [:t, :p], false],
                     [:EX, [:t, :z, :zz], false],
                     [:INJ, [:t, :n], false],
                     [:F_DC, [:t, :dc], false],
                     [:INFEAS_H_NEG, [:t, :ha], false],
                     [:INFEAS_H_POS, [:t, :ha], false],
                     [:INFEAS_EL_N_NEG, [:t, :n], false],
                     [:INFEAS_EL_N_POS, [:t, :n], false],
                     [:INFEAS_LINES, [:t, :cb], false],
                     [:EB_nodal, [:t, :n], true],
                     [:EB_zonal, [:t, :z], true],
					          ]

println("Saving results to results folder: ", result_folder)
for par in results_parameter
	jump_to_df(disp, par[1], par[2], par[3], model_horizon, result_folder)
end

# Misc Results or Data
misc_result = Dict()
misc_result["Objective Value"] = JuMP.objective_value(disp)
misc_result["COST_G"] = JuMP.value(COST_G)
misc_result["COST_H"] = JuMP.value(COST_H)
misc_result["COST_EX"] = JuMP.value(COST_EX)
misc_result["COST_INEAS_EL"] = JuMP.value(COST_INEAS_EL)
misc_result["COST_INEAS_H"] = JuMP.value(COST_INEAS_H)
misc_result["COST_INEAS_LINES"] = JuMP.value(COST_INEAS_LINES)
misc_result["Solve Status"] = JuMP.termination_status(disp)

# write the file with the stringdata variable information
open(result_folder*"/misc_result.json", "w") do f
        write(f, JSON.json(misc_result))
end

# End build_and_run_model function
return disp
end
