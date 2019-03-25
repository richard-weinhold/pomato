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
                             nodes::OrderedDict{String, Node}, 
                             zones::OrderedDict{String, Zone}, 
                             heatareas::Dict{String, Heatarea},
                             
                             grid::Dict{String, Grid},
                             dc_lines::Dict{String, DC_Line}, 
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
    dc_incidence[dc][dc_lines[dc].node_i] =  1
    dc_incidence[dc][dc_lines[dc].node_j] = -1
end

# Setup model
# disp = Model(solver=ClpSolver(SolveType=5))
# disp = Model(solver=GurobiSolver(Presolve=2, PreDual=2, Threads=8))
# disp = Model(solver=GurobiSolver(Method=0,Threads=1))
disp = Model(with_optimizer(Gurobi.Optimizer)) #, Presolve=0)) #, Method=2))

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
# @constraintref EB_nodal[t_set, n_set]
# JuMP.registercon(disp, :EB_nodal, EB_nodal)
# @constraintref EB_zonal[t_set, z_set]
# JuMP.registercon(disp, :EB_zonal, EB_zonal)

@objective(disp, Min, sum(sum(G[t, p]*plants[p].mc for p in p_set) for t in t_set)
                      + sum(sum(H[t, p]*plants[p].mc for p in p_set) for t in t_set)
                      + (sum(INFEAS_EL_Z_POS) + sum(INFEAS_EL_Z_NEG))*1e4
                      + (sum(INFEAS_EL_N_POS) + sum(INFEAS_EL_N_NEG))*1e2
                      + (sum(INFEAS_H_NEG) + sum(INFEAS_H_POS))*1e3
                      + (sum(INFEAS_LINES)*1e3)
                      + (sum(INFEAS_REF_FLOW)*1e1))

println("Building Constraints")
# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=setdiff(p_set, union(he_set, ts_set, d_set))],
    G[t, p] <= plants[p].g_max)

# All plants without he, ts and demand units
@constraint(disp, [t=t_set, p=setdiff(he_set, union(chp_set, ts_set))],
    H[t, p] <= plants[p].h_max)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=chp_set],
    G[t, p] >= ((plants[p].g_max*0.85) / plants[p].h_max) * H[t, p])
@constraint(disp, [t=t_set, p=chp_set],
    G[t, p] <= plants[p].g_max * (1-(0.15 * H[t,p] / plants[p].h_max)))

# Applies to: Dispatch
# Base Constraint
# WARNING this formulation requires the availabilty df to be correctly sorted by times
@constraint(disp, [t=t_set, p=intersect(ts_set, co_set)],
    G[t, p] <= plants[p].g_max * plants[p].availability[model_horizon[t]])
@constraint(disp, [t=t_set, p=intersect(ts_set, co_set)],
    G[t, p] >= plants[p].g_max * plants[p].availability[model_horizon[t]] * 0.8)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=intersect(ts_set, he_set)],
    H[t, p] <= plants[p].h_max * plants[p].availability[model_horizon[t]])
@constraint(disp, [t=t_set, p=intersect(ts_set, he_set)],
    H[t, p] >= plants[p].h_max * plants[p].availability[model_horizon[t]] * 0.8)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=d_set],
    D_d[t, p] <= plants[p].g_max * plants[p].availability[model_horizon[t]])
@constraint(disp, [t=t_set, p=d_set],
    D_d[t, p] >=  plants[p].g_max * plants[p].availability[model_horizon[t]] * 0.8)
# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=ph_set],
    D_ph[t, p] == H[t, p] * plants[p].eta)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=es_set],
    L_es[t, p]  == (t>t_start ? L_es[t-1, p] : plants[p].g_max*2)
                   - G[t, p]
                   + plants[p].eta*D_es[t, p])

@constraint(disp, [t=t_set, p=es_set], 
    L_es[t, p] <= plants[p].g_max*8)
@constraint(disp, [t=t_set, p=es_set],
    D_es[t, p] <= plants[p].g_max)
@constraint(disp, [p=es_set],
    L_es[t_end, p] >= 2*plants[p].g_max)

# Applies to: Dispatch
# Base Constraint
@constraint(disp, [t=t_set, p=hs_set],
    L_hs[t, p] ==  (t>t_start ? plants[p].eta*L_hs[t-1, p] : plants[p].h_max*2)
                   - H[t, p]
                   + D_hs[t, p])
@constraint(disp, [t=t_set, p=hs_set],
    L_hs[t, p] <= plants[p].h_max*4)
@constraint(disp, [t=t_set, p=hs_set],
    D_hs[t, p] <= plants[p].h_max)

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
    zones[z].demand[model_horizon[t]] + zones[z].net_export[model_horizon[t]] ==
      sum(G[t, p] for p in intersect(zones[z].plants, co_set))
    - sum(D_ph[t, p] for p in intersect(zones[z].plants, ph_set))
    - sum(D_es[t, p] for p in intersect(zones[z].plants, es_set))
    - sum(D_d[t, p] for p in intersect(zones[z].plants, d_set))
    - sum(EX[t, z, zz] for zz in z_set)
    + sum(EX[t, zz, z] for zz in z_set)
    + sum(INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n] for n in zones[z].nodes)
    # + INFEAS_EL_Z_POS[t, z] - INFEAS_EL_Z_NEG[t, z]
    )

# Nodal Energy Balance
# Applies to: Dispatch
# Base Constraint
@constraint(disp, EB_nodal[t=t_set, n=n_set], 
    nodes[n].demand[model_horizon[t]] + nodes[n].net_export[model_horizon[t]] ==
      sum(G[t, p] for p in intersect(nodes[n].plants, co_set))
    - sum(D_ph[t, p] for p in intersect(nodes[n].plants, ph_set))
    - sum(D_es[t, p] for p in intersect(nodes[n].plants, es_set))
    - sum(D_d[t, p] for p in intersect(nodes[n].plants, d_set))
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
if in(model_type, ["ntc", "cbco_nodal"])
    @constraint(disp, [t=t_set, z=z_set, zz=z_set],
        EX[t, z, zz] <=  zones[z].ntc[zz])
end

# Slack Constraint
# Applies to: ntc, nodal, cbco_nodal, cbco_zonal
if in(model_type, ["ntc", "nodal", "cbco_nodal", "cbco_zonal", "d2cf"])
    # INJ have to be balanced within a slack_zone
    @constraint(disp, [t=t_set, slack=slack_set],
        0 == sum(INJ[t, n] for n in nodes[slack].slack_zone))
end

# Cbco Constraints
# Applies to: cbco_nodal, nodal
if in(model_type, ["cbco_nodal", "nodal"])
    @constraint(disp, [t=t_set, cb=cb_set],
        sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set)) <= grid[cb].ram ) #+ INFEAS_LINES[t, cb])
    @constraint(disp, [t=t_set, cb=cb_set],
        sum(INJ[t, n]*grid[cb].ptdf[i] for (i, n) in enumerate(n_set)) >= -(grid[cb].ram)) #+ INFEAS_LINES[t, cb]))
end

# Applies to: cbco_zonal
if in(model_type, ["cbco_zonal"])
    @constraint(disp, [t=t_set, cb=cb_set],
        sum((sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))*grid[cb].ptdf[i] for (i, z) in enumerate(z_set)) 
        <= grid[cb].ram + INFEAS_LINES[t, cb])
    @constraint(disp, [t=t_set, cb=cb_set],
        sum((sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))*grid[cb].ptdf[i] for (i, z) in enumerate(z_set)) 
        >= -(grid[cb].ram + INFEAS_LINES[t, cb]))
end

# Applies to d2cf model:
# set net_position net_position:
if in(model_type, ["d2cf"])
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

println("Solving...")
t_start = time_ns()
JuMP.optimize!(disp)
println("Objective: $(JuMP.objective_value(disp))")
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
misc_result["Objective Value"] = JuMP.objective_value(disp)
# write the file with the stringdata variable information
open(result_folder*"/misc_result.json", "w") do f
        write(f, JSON.json(misc_result))
end

# End build_and_run_model function
end


