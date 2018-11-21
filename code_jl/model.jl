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

function build_and_run_model(model_horizon, opt_setup, 
                             plants, plants_in_ha, plants_at_node, plants_in_zone, availabilities,
                             nodes, zones, slack_zones, heatareas, nodes_in_zone,
                             ntc, dc_lines, cbco)

# Check for feasible model_type
model_type = opt_setup["opt"]
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
t_set = keys(model_horizon) 
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
cb_set = collect(keys(cbco))

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
disp = Model(solver=GurobiSolver())
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

if opt_setup["infeas_heat"]
    @variable(disp, INFEAS_H_NEG[t_set, ha_set] >= 0) # Relaxing at high costs to avoid infeasibility in heat EB
    @variable(disp, INFEAS_H_POS[t_set, ha_set] >= 0) # - " -
else
    @variable(disp, INFEAS_H_NEG[t_set, ha_set] == 0)
    @variable(disp, INFEAS_H_POS[t_set, ha_set] == 0)
end

if opt_setup["infeas_el_nodal"]
    @variable(disp, INFEAS_EL_N_NEG[t_set, n_set] >= 0)
    @variable(disp, INFEAS_EL_N_POS[t_set, n_set] >= 0)
else
    @variable(disp, INFEAS_EL_N_NEG[t_set, n_set] == 0)
    @variable(disp, INFEAS_EL_N_POS[t_set, n_set] == 0)
end
if opt_setup["infeas_el_zonal"]
    @variable(disp, INFEAS_EL_Z_NEG[t_set, z_set] >= 0)
    @variable(disp, INFEAS_EL_Z_POS[t_set, z_set] >= 0)
else
    @variable(disp, INFEAS_EL_Z_NEG[t_set, z_set] == 0)
    @variable(disp, INFEAS_EL_Z_POS[t_set, z_set] == 0)
end

if opt_setup["infeas_lines"]
    @variable(disp, INFEAS_LINES[t_set, cb_set] >= 0)
    @variable(disp, INFEAS_REF_FLOW[t_set, cb_set] >= 0)
else
    @variable(disp, INFEAS_LINES[t_set, cb_set] == 0)
    @variable(disp, INFEAS_REF_FLOW[t_set, cb_set] == 0)
end

## Dicts to store EB constraints for duals
EB_nodal_dict = Dict()
EB_zonal_dict = Dict()

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
        @constraint(disp, G[t, p] >= plants[p].g_max * availabilities[p].value[model_horizon[t]] * 0)
    end

    # Applies to: Dispatch
    # Base Constraint
    for p in intersect(ts_set, he_set)
        @constraint(disp, H[t, p] <= plants[p].h_max * availabilities[p].value[model_horizon[t]])
        @constraint(disp, H[t, p] >= plants[p].h_max * availabilities[p].value[model_horizon[t]] * 0.2)
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
       
        EB_zonal = @constraint(disp, zones[z].demand[model_horizon[t]] + zones[z].net_export[model_horizon[t]] == 
                              sum(G[t, p] for p in intersect(p_in_z_set, co_set))
                            - sum(D_ph[t, p] for p in intersect(p_in_z_set, ph_set))
                            - sum(D_es[t, p] for p in intersect(p_in_z_set, es_set))
                            - sum(D_d[t, p] for p in intersect(p_in_z_set, d_set))
                            - sum(EX[t, z, zz] for zz in z_set)
                            + sum(EX[t, zz, z] for zz in z_set)
                            + sum(INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n] for n in collect(keys(nodes_in_zone[z])))
                            # + INFEAS_EL_Z_POS[t, z] - INFEAS_EL_Z_NEG[t, z]
                            )
        EB_zonal_dict[length(EB_zonal_dict) + 1] = Dict("t" => model_horizon[t], "z" => z, "EB_zonal" => EB_zonal)
    end

    # Nodal Energy Balance
    # Applies to: Dispatch
    # Base Constraint
    for n in n_set
        p_in_n_set = collect(keys(plants_at_node[n]))
        EB_nodal = @constraint(disp, nodes[n].demand[model_horizon[t]] + nodes[n].net_export[model_horizon[t]] == 
                              sum(G[t, p] for p in intersect(p_in_n_set, co_set))
                            - sum(D_ph[t, p] for p in intersect(p_in_n_set, ph_set))
                            - sum(D_es[t, p] for p in intersect(p_in_n_set, es_set))
                            - sum(D_d[t, p] for p in intersect(p_in_n_set, d_set))
                            - sum((F_DC[t, dc]*dc_incidence[dc][n]) for dc in dc_set)
                            - INJ[t, n]
                            + INFEAS_EL_N_POS[t, n] - INFEAS_EL_N_NEG[t, n]
                            )
        EB_nodal_dict[length(EB_nodal_dict) + 1] = Dict("t" => model_horizon[t], 
                                                        "n" => n, 
                                                        "EB_nodal" => EB_nodal)
    end

    # DC Lines Constraints
    # Applies to: Dispatch
    # Base Constraint
    for dc in dc_set
        @constraint(disp, F_DC[t, dc] <= dc_lines[dc].capacity)
        @constraint(disp, F_DC[t, dc] >= -dc_lines[dc].capacity)
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
            @constraint(disp, sum(INJ[t, n]*cbco[cb]["ptdf"][i] for (i, n) in enumerate(n_set)) <= cbco[cb]["ram"] + INFEAS_LINES[t, cb])
        end
    end

    # Applies to: cbco_zonal
    if in(model_type, ["cbco_zonal"])
        for cb in cb_set
            @constraint(disp, sum( (sum(EX[t, zz, z] for zz in z_set) - sum(EX[t, z, zz] for zz in z_set))
                                    *cbco[cb]["ptdf"][i] for (i, z) in enumerate(z_set)) <= cbco[cb]["ram"] + INFEAS_LINES[t, cb])
        end
    end

    # Applies to d2cf model:
    # set net_position net_position:
    if in(model_type, ["d2cf"])
        for z in ["DE", "NL", "BE", "FR", "LU"]
            ### net_position when positive -> export
            @constraint(disp, sum(EX[t, z, zz] - EX[t, zz, z] for zz in z_set) <= 
                              zones[z].net_position[model_horizon[t]] + 0.1*abs(zones[z].net_position[model_horizon[t]])
                              )
            @constraint(disp, sum(EX[t, z, zz] - EX[t, zz, z] for zz in z_set) >= 
                              zones[z].net_position[model_horizon[t]] - 0.1*abs(zones[z].net_position[model_horizon[t]])
                              )
        end
        # set reference flows in CNECS
        for cb in cb_set
            if ("f_ref" in keys(cbco[cb])) && (cbco[cb]["f_ref"][model_horizon[t]] != 0)
                @constraint(disp, sum(INJ[t, n]*cbco[cb]["ptdf"][i] for (i, n) in enumerate(n_set)) 
                                  <= cbco[cb]["f_ref"][model_horizon[t]] + INFEAS_REF_FLOW[t, cb])
                @constraint(disp, sum(INJ[t, n]*cbco[cb]["ptdf"][i] for (i, n) in enumerate(n_set)) 
                                  >= cbco[cb]["f_ref"][model_horizon[t]]  - INFEAS_REF_FLOW[t, cb])
            end
            # Upper/Lower Bound on CBs
            @constraint(disp, sum(INJ[t, n]*cbco[cb]["ptdf"][i] for (i, n) in enumerate(n_set)) 
                              <= cbco[cb]["f_max"]*1.0 + INFEAS_LINES[t, cb])
            @constraint(disp, sum(INJ[t, n]*cbco[cb]["ptdf"][i] for (i, n) in enumerate(n_set)) 
                              >= -cbco[cb]["f_max"]*1.0 - INFEAS_LINES[t, cb])
        end
    end
end

for p in es_set
    @constraint(disp, L_es[t_end, p] >= 2*plants[p].g_max)
end

println("Solving...")
tic()
println("Number of Linear Constraints in the Model: ", MathProgBase.numlinconstr(disp))
solve(disp)
println("Objective: $(getobjectivevalue(disp))")
println("Solvetime: $(toq()) seconds")

result_folder = WDIR*"\\data_temp\\julia_files\\results\\"*Dates.format(now(), "dmm_HHMM")
if !isdir(result_folder)
    mkdir(result_folder)
end

# Prepare Output
G_dict = Dict()
H_dict = Dict()
d_el_dict = Dict()
D_es_dict = Dict()
L_es_dict = Dict()
D_hs_dict = Dict()
L_hs_dict = Dict()
D_ph_dict = Dict()
D_d_dict = Dict()
INFEAS_H_dict = Dict()
INFEAS_EL_Z_dict = Dict()
INFEAS_EL_N_dict = Dict()
INFEAS_LINES_dict = Dict()
INFEAS_REF_FLOW_dict = Dict()
INJ_dict = Dict()
EX_dict = Dict()
F_DC_dict = Dict()

for t in t_set  
    t_ind =model_horizon[t]
    for p in p_set
        G_dict[length(G_dict) + 1] = Dict("t" => t_ind, "p" => p, "G" => getvalue(G[t, p]))
        H_dict[length(H_dict) + 1] = Dict("t" => t_ind, "p" => p, "H" => getvalue(H[t, p]))
        if p in es_set
            D_es_dict[length(D_es_dict) + 1] = Dict("t" => t_ind, "es" => p, "D_es" => getvalue(D_es[t, p]))
            L_es_dict[length(L_es_dict) + 1] = Dict("t" => t_ind, "es" => p, "L_es" => getvalue(L_es[t, p]))
        end
        if p in hs_set
            D_hs_dict[length(D_hs_dict) + 1] = Dict("t" => t_ind, "hs" => p, "D_hs" => getvalue(D_hs[t, p]))
            L_hs_dict[length(L_hs_dict) + 1] = Dict("t" => t_ind, "hs" => p, "L_hs" => getvalue(L_hs[t, p]))
        end
        if p in ph_set
            D_ph_dict[length(D_ph_dict) + 1] = Dict("t" => t_ind, "ph" => p, "D_ph" => getvalue(D_ph[t, p]))
        end
        if p in d_set
            D_d_dict[length(D_d_dict) + 1] = Dict("t" => t_ind, "d" => p, "D_d" => getvalue(D_d[t, p]))
        end
    end
    for ha in ha_set
        INFEAS_H_dict[length(INFEAS_H_dict) + 1] = Dict("t" => t_ind, "ha" => ha, 
                                                        "INFEAS_H_NEG" => getvalue(INFEAS_H_NEG[t, ha]), 
                                                        "INFEAS_H_POS" => getvalue(INFEAS_H_POS[t, ha]))
    end
    for z in z_set
        INFEAS_EL_Z_dict[length(INFEAS_EL_Z_dict) + 1] = Dict("t" => t_ind, "z" => z, 
                                                          "INFEAS_EL_Z_NEG" => getvalue(INFEAS_EL_Z_NEG[t, z]), 
                                                          "INFEAS_EL_Z_POS" => getvalue(INFEAS_EL_Z_POS[t, z]))
    end
    for n in n_set
        INFEAS_EL_N_dict[length(INFEAS_EL_N_dict) + 1] = Dict("t" => t_ind, "n" => n, 
                                                          "INFEAS_EL_N_NEG" => getvalue(INFEAS_EL_N_NEG[t, n]), 
                                                          "INFEAS_EL_N_POS" => getvalue(INFEAS_EL_N_POS[t, n]))

        INJ_dict[length(INJ_dict) + 1] = Dict("t" => t_ind, "n" => n, "INJ" => getvalue(INJ[t, n]))
        d_el_dict[length(d_el_dict) + 1] = Dict("t" => t_ind, "n" => n, "d_el" => nodes[n].demand[model_horizon[t]])

    end
    for z_from in z_set
        for z_to in z_set
            EX_dict[length(EX_dict) + 1] = Dict("t" => t_ind, "z" => z_from, "zz" => z_to, "EX" => getvalue(EX[t, z_from, z_to]))
        end
    end
    for dc in dc_set
        F_DC_dict[length(F_DC_dict) + 1] = Dict("t" => t_ind, "dc" => dc, "F_DC" => getvalue(F_DC[t, dc]))
    end
    for cb in cb_set
        INFEAS_LINES_dict[length(INFEAS_LINES_dict) + 1] = Dict("t" => t_ind, "cb" => cb, "INFEAS_LINES" => getvalue(INFEAS_LINES[t, cb]))
        INFEAS_REF_FLOW_dict[length(INFEAS_REF_FLOW_dict) + 1] = Dict("t" => t_ind, "cb" => cb, "INFEAS_REF_FLOW" => getvalue(INFEAS_REF_FLOW[t, cb]))
    end

end
for i in keys(EB_nodal_dict)
   EB_nodal_dict[i]["EB_nodal"] = getdual(EB_nodal_dict[i]["EB_nodal"])
end
for i in keys(EB_zonal_dict)
   EB_zonal_dict[i]["EB_zonal"] = getdual(EB_zonal_dict[i]["EB_zonal"])
end

# Save results
open(result_folder*"\\G.json", "w") do f
        write(f, JSON.json(G_dict))
end
open(result_folder*"\\H.json", "w") do f
        write(f, JSON.json(H_dict))
end
open(result_folder*"\\D_es.json", "w") do f
        write(f, JSON.json(D_es_dict))
end
open(result_folder*"\\d_el.json", "w") do f
        write(f, JSON.json(d_el_dict))
end
open(result_folder*"\\L_es.json", "w") do f
        write(f, JSON.json(L_es_dict))
end
open(result_folder*"\\D_hs.json", "w") do f
        write(f, JSON.json(D_hs_dict))
end
open(result_folder*"\\L_hs.json", "w") do f
        write(f, JSON.json(L_hs_dict))
end
open(result_folder*"\\D_ph.json", "w") do f
        write(f, JSON.json(D_ph_dict))
end
open(result_folder*"\\D_d.json", "w") do f
        write(f, JSON.json(D_d_dict))
end
open(result_folder*"\\INFEAS_H.json", "w") do f
        write(f, JSON.json(INFEAS_H_dict))
end
open(result_folder*"\\INFEAS_EL_Z.json", "w") do f
        write(f, JSON.json(INFEAS_EL_Z_dict))
end
open(result_folder*"\\INFEAS_EL_N.json", "w") do f
        write(f, JSON.json(INFEAS_EL_N_dict))
end
open(result_folder*"\\INJ.json", "w") do f
        write(f, JSON.json(INJ_dict))
end
open(result_folder*"\\EX.json", "w") do f
        write(f, JSON.json(EX_dict))
end
open(result_folder*"\\F_DC.json", "w") do f
        write(f, JSON.json(F_DC_dict))
end
open(result_folder*"\\INFEAS_LINES.json", "w") do f
        write(f, JSON.json(INFEAS_LINES_dict))
end
open(result_folder*"\\INFEAS_REF_FLOW.json", "w") do f
        write(f, JSON.json(INFEAS_REF_FLOW_dict))
end
open(result_folder*"\\EB_nodal.json", "w") do f
        write(f, JSON.json(EB_nodal_dict))
end
open(result_folder*"\\EB_zonal.json", "w") do f
        write(f, JSON.json(EB_zonal_dict))
end

# Misc Results or Data
misc_result = Dict()
misc_result["Objective Value"] = getobjectivevalue(disp)
# write the file with the stringdata variable information
open(result_folder*"\\misc_result.json", "w") do f
        write(f, JSON.json(misc_result))
end

end
