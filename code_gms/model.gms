

$if not set wdir                         $abort "No wdir specified"

$if set ddir                             $set data_folder %wdir%data_temp\gms_files\data\%ddir%
$if set rdir                             $set result_folder %rdir%

$if not set ddir                         $set data_folder %wdir%\data_temp\gms_files\data
$if not set rdir                         $set result_folder %wdir%\data_temp\gms_files\results\default

$if not dexist %result_folder%           $call mkdir %result_folder%

$if not set infeasibility_electricity    $set infeasibility_electricity true
$if not set infeasibility_heat           $set infeasibility_heat true
$if not set infeasibility_lines          $set infeasibility_lines true
$if not set infeasibility_bound          $set infeasibility_bound 1000

$if not set chp_efficiency               $set chp_efficiency 0.15
$if not set curtailment_electricity      $set curtailment_electricity 0.8
$if not set curtailment_heat             $set curtailment_heat 0
$if not set stor_start                   $set stor_start 0.65

$if not set model_type                   $set model_type dispatch

$if %model_type% == cbco_zonal           $set data_type zonal
$if %model_type% == zonal                $set data_type zonal
$if %model_type% == cbco_nodal           $set data_type nodal
$if %model_type% == nodal                $set data_type nodal

$offlisting
$offsymxref offsymlist

option
    limrow = 0
    limcol = 0
    solprint = off
    sysout = off
    reslim = 10000
    solver = gurobi
;

$onecho >gurobi.opt
threads = 4
method = 2
names = 0
$offecho


Sets
t        Hour
p        Plants
tech     Technology Set
co(p)    Conventional Plants of p
he(p)    Heat-Only Plants of p
chp(p)   CHP Plants of p
es(p)    Electricity Storage Plants of p
ph(p)    Power to Heat of p
hs(p)    Heat Storage Plants of p
ts(p)    Generation Plants using Time-Series for Capacity of p
n        Nodes
z        Zones Electric
ha       Heat Area
dc       DC Lines
cb       Critical Branch

map_nz(n,z)      Node-Zone El Mapping
map_pha(p,ha)    Plant-Heat Area Mapping
map_pn(p,n)      Plant-Node Mapping
slack(n)         Slacknode
map_ns(n,slack)  Node to Slack mapping
;

Alias(n,nn);
Alias(z,zz);

Parameter
***Generation related parameters
         mc_el(p)                Marginal Costs for 1 MWh_el of Plant p
         mc_heat(p)              Marginal Costs for 1 MWh_th of Plant p
         eta(p)                  Efficiency for Plants [MW_el per MW_th] or Storage [loss per period %]
         g_max(p)                Maximum Generation of Plant p
         h_max(p)                Maximum Heat-Generation of Plant h(p)
         ava(t,p)                Availability of Time-Series Dependant Generation
         es_cap(p)               Electricity Storage Capacity
         hs_cap(p)               Heat Storage Capacity

***Demand related parameters
         d_el(t,n)               Electricity demand at node n
         d_h(t,ha)               Heat demand at node n
***Grid related parameters
         ntc(z,zz)               NTC between two zones
         ptdf(cb,*)              Critical Branch Critical Outage - for Zone or Node to Line Sensitivities
         ram(cb)                 Remaining Available Margin on CB
         dc_max(dc)              Maximum transmission capacity of DC Line
         inc_dc(dc,n)            Incedence Matrix for DC Connections
         net_position(t,z)
         net_export(t,n)
         inflows(t, p)          Inflows into storage es in timestep t in [MWh]

;
*###############################################################################
*                                 UPLOAD
*###############################################################################

$call gams %wdir%\code_gms\dataimport --data_type=%data_type% --data_folder=%data_folder% suppress=1

$gdxin %data_folder%\dataset.gdx
$load t p n z ha dc
$load chp co he ph es ts hs
$load cb slack map_pn map_nz map_pha map_ns
$load mc_el mc_heat g_max h_max ava inflows eta hs_cap es_cap d_el d_h
$load ntc ptdf ram dc_max inc_dc net_export net_position
$gdxin
;


set t_end(t);
t_end(t) = No;
t_end(t)$(ord(t) eq card(t)) = Yes;

*$stop
*****SET UP SETS
Variables
COST            Total System Cost
COST_G
COST_H
INJ(t,n)        Net Injection at Node n
F_DC(t,dc)      Flow in DC Line dc
;

Positive Variables
D_hs(t,hs)      Heat Demand by Heat Storage
D_es(t,es)      Electricity Demand by Electricity Storage
D_ph(t,ph)      Electricity Demand by Power to Heat Plants
G(t,p)          Generation per Plant p
H(t,p)          Heat generation of Plant h(p)
L_es(t,es)      Electricity Storage Level at t
L_hs(t,hs)      Heat Storage Level at t
EX(t,z,zz)      Commercial Flow From z to zz

INFEAS_EL_N_POS(t,n)    Relaxing the EB at high costs to avoid infeas
INFEAS_EL_N_NEG(t,n)
INFEAS_H_POS(t,ha)
INFEAS_H_NEG(t,ha)
INFEAS_LINES(t,cb)   Infeasibility Variable for Lines


COST_EX
COST_INEAS_EL
COST_INEAS_H
COST_INEAS_LINES
;


scalar
curtailment_electricity /%curtailment_electricity%/
curtailment_heat /%curtailment_heat%/
chp_efficiency /%chp_efficiency%/
;


$ifthen %infeasibility_electricity% == True
         INFEAS_EL_N_POS.up(t,n) = %infeasibility_bound%;
         INFEAS_EL_N_NEG.up(t,n) = %infeasibility_bound%;
$else
         INFEAS_EL_N_POS.fx(t,n) = 0;
         INFEAS_EL_N_NEG.fx(t,n) = 0;
$endif

$ifthen %infeasibility_heat% == True
         INFEAS_H_POS.up(t,ha) = %infeasibility_bound%;
         INFEAS_H_NEG.up(t,ha) = %infeasibility_bound%;
$else
         INFEAS_H_POS.fx(t,ha) = 0;
         INFEAS_H_NEG.fx(t,ha) = 0;
$endif

$ifthen %infeasibility_lines% == True
         INFEAS_LINES.up(t,cb) = %infeasibility_bound%;
$else
         INFEAS_LINES.fx(t,cb) = 0;
$endif

Equations
Obj              Objective Function - Total System Costs
DEF_COST_G
DEF_COST_H
DEF_COST_EX
DEF_COST_INEAS_EL
DEF_COST_INEAS_H
DEF_COST_INEAS_LINES


Gen_Max_El       Max Generation for non-CHP Conventional Plans
Gen_Max_H        Maximum Generation for non-CHP Heat Plants

Gen_Max_CHP1     Generation Constraint CHP Plants for Heat and Electricity - 1
Gen_Max_CHP2     Generation Constraint CHP Plants for Heat and Electricity - 2
Gen_Max_RES      Generation Constraint for el RES - Max Available Capacity
Gen_Min_RES      Generation Constraint for el RES - Max Available Courtailment - 15%
Gen_Max_RES_h    Generation Constraint for h RES - Max Available Capacity
Gen_Min_RES_h    Generation Constraint for h RES - Max Available Courtailment - 15%
Gen_PH           Generation Constraint Heat Pumps - Heat Generation induces Electricity Demand

STOR_EL          Electricity Storage Balance
STOR_EL_Cap      Electricity Storage level Capacity
STOR_EL_Max      Maximum Discharge
STOR_EL_End      Electricity Storage End Level
STOR_H           Heat Storage Balance
STOR_H_Cap       Heat Storage level Capacity
STOR_H_Max       Maximum Discharge
STOR_H_End       Heat Storage End Level

EB_Heat          Energy Balance Heat - Demand = H - Demand_HS (Heat Storage)
EB_Nodal         Energy Balance Electricity - Demand = G - Demand_ES - Demand_HP - NET Injection (per Node)
EB_Zonal         Energy Balance Electricity - Demand = G - Demand_ES - Demand_HP - NET Injection (per Zone)

CON_DC_up        Upper Bound in DC Line Flow
CON_DC_lo        Lower Bound in DC Line Flow
CON_Slack        Slack Definition (for Nodal Disptach)

CON_NTC
CON_CBCO_nodal_p   Flow on Critical Branchens < RAM based on Nodal Injection - CBCO provide Node-Line Sensitivity
CON_CBCO_nodal_n   Flow on Critical Branchens < RAM based on Nodal Injection - CBCO provide Node-Line Sensitivity
CON_CBCO_zonal_p   Flow on Critical Branchens < RAM based on Zonal Injection - CBCO provide Node-Line Sensitivity
CON_CBCO_zonal_n   Flow on Critical Branchens < RAM based on Zonal Injection - CBCO provide Node-Line Sensitivity

CON_NEX_u
CON_NEX_l
;

Obj..                                            COST =e= COST_G + COST_H + COST_EX
                                                          + COST_INEAS_EL + COST_INEAS_H + COST_INEAS_LINES
;

DEF_Cost_G..                                           COST_G =e= sum((t,p), G(t,p)* mc_el(p));
DEF_COST_H..                                           COST_H =e= sum((t,p), H(t,p) * mc_heat(p));
DEF_COST_INEAS_EL..                                    COST_INEAS_EL =e=  sum((t,n), (INFEAS_EL_N_POS(t,n) + INFEAS_EL_N_NEG(t,n))*1E4);
DEF_COST_INEAS_H..                                     COST_INEAS_H =e=  sum((t,ha), (INFEAS_H_POS(t,ha) + INFEAS_H_NEG(t,ha))*1E4);
DEF_COST_INEAS_LINES..                                 COST_INEAS_LINES =e= sum((t,cb), INFEAS_LINES(t,cb)*1E4);
DEF_COST_EX..                                          COST_EX =E= sum((t,z,zz), EX(t,z,zz)*1);

Gen_Max_El(t,p)$(not (he(p) or ts(p)))..                 G(t,p) =l= g_max(p)
;
Gen_Max_H(t,p)$(he(p) and not (chp(p) or ts(p)))..       H(t,p) =l= h_max(p)
;
Gen_Max_CHP1(t,chp)..                            G(t,chp) =g= (g_max(chp)*(1-chp_efficiency))/ h_max(chp) * H(t,chp)
;
Gen_Max_CHP2(t,chp)..                            G(t,chp) =l= g_max(chp)*(1-(chp_efficiency*H(t,chp)/h_max(chp)))
;
Gen_Max_RES(t,p)$(ts(p) and co(p))..             G(t,p)  =l= g_max(p) * ava(t,p)
;
Gen_Min_RES(t,p)$(ts(p) and co(p))..             G(t,p)  =g= g_max(p) * ava(t,p) * curtailment_electricity
;
Gen_Max_RES_h(t,ts)$(h_max(ts))..                H(t,ts)  =l= h_max(ts) * ava(t,ts)
;
Gen_Min_RES_h(t,ts)$(h_max(ts))..                H(t,ts)  =g= h_max(ts) * ava(t,ts) * curtailment_heat
;
Gen_PH(t,ph)..                                   D_ph(t,ph) =E= H(t,ph) / eta(ph)
;

STOR_EL(t,es)..                                  L_es(t,es) =e= L_es(t-1,es)$(ord(t)>1)
                                                        - G(t,es)
                                                        + eta(es)*D_es(t,es)
                                                        + inflows(t, es)
                                                        + %stor_start%*es_cap(es)$(ord(t) eq 1)
;

STOR_EL_Cap(t,es)..                      L_es(t,es) =l= es_cap(es)
;
STOR_EL_Max(t,es)..                      D_es(t,es) =l= 0
*g_max(es)
;
STOR_EL_End(t_end,es)..                  L_es(t_end,es) =g= %stor_start%*es_cap(es)
;

STOR_H(t,hs)..                           L_hs(t,hs) =e= eta(hs)*L_hs(t,hs-1)$(ord(t)>1)
                                                        - H(t,hs)
                                                        + D_hs(t,hs)
                                                        + 0.65*hs_cap(hs)$(ord(t) eq 1)
;
STOR_H_Cap(t,hs)..                       L_hs(t,hs) =l= hs_cap(hs)
;
STOR_H_Max(t,hs)..                       D_hs(t,hs) =l= h_max(hs)
;
STOR_H_End(t_end,hs)..                   L_hs(t_end,hs) =g= hs_cap(hs)/2
;

EB_Heat(t,ha)..                  d_h(t,ha) =e= sum(map_pha(he,ha), H(t,he))
                                               - sum(map_pha(hs,ha), D_hs(t,hs))
                                               + INFEAS_H_POS(t,ha) - INFEAS_H_NEG(t,ha)
;

EB_Zonal(t,z)..                  sum(map_nz(n,z), d_el(t,n) - net_export(t,n)) =e=      sum(map_nz(n,z), sum(map_pn(co,n), G(t,co)))
                                                                                      - sum(map_nz(n,z), sum(map_pn(ph,n), D_ph(t,ph)))
                                                                                      - sum(map_nz(n,z), sum(map_pn(es,n), D_es(t,es)))
                                                                                      - sum(zz, EX(t,z,zz))
                                                                                      + sum(zz, EX(t,zz,z))
                                                                                      + sum(map_nz(n,z), INFEAS_EL_N_POS(t,n) - INFEAS_EL_N_NEG(t,n))
;

EB_Nodal(t,n)..                  d_el(t,n) - net_export(t,n) =e=   sum(map_pn(co,n), G(t,co))
                                                                 - sum(map_pn(ph,n), D_ph(t,ph))
                                                                 - sum(map_pn(es,n), D_es(t,es))
                                                                 - sum(dc, F_DC(t,dc)*inc_dc(dc,n))
                                                                 - INJ(t,n)
                                                                 + (INFEAS_EL_N_POS(t,n) - INFEAS_EL_N_NEG(t,n))
;

CON_DC_up(t,dc)..                F_DC(t,dc) =l= dc_max(dc)
;
CON_DC_lo(t,dc)..                F_DC(t,dc) =g= -dc_max(dc)
;
CON_Slack(t,slack)..             sum(map_ns(n,slack), INJ(t,n)) =E= 0
;
CON_NTC(t,z,zz)..                EX(t,z,zz) =L= ntc(z,zz)
;
CON_CBCO_nodal_p(t,cb)..         sum(n, INJ(t,n)*ptdf(cb,n)) =l= ram(cb) + INFEAS_LINES(t,cb)
;
CON_CBCO_nodal_n(t,cb)..         sum(n, INJ(t,n)*ptdf(cb,n)) =g= -ram(cb) - INFEAS_LINES(t,cb)
;
CON_CBCO_zonal_p(t,cb)..         sum(z, sum(zz, EX(t,zz,z) - EX(t,z,zz))*ptdf(cb,z)) =l= ram(cb) + INFEAS_LINES(t,cb)
;
CON_CBCO_zonal_n(t,cb)..         sum(z, sum(zz, EX(t,zz,z) - EX(t,z,zz))*ptdf(cb,z)) =g= -ram(cb) - INFEAS_LINES(t,cb)
;

*set cwe(z) /DE, NL, BE, FR/;
*CON_NEX_u(t, cwe)..              sum(zz, EX(t,cwe,zz) - EX(t,zz,cwe)) =L= net_position(t, cwe) + 0.2*abs(net_position(t, cwe));
*CON_NEX_l(t, cwe)..              sum(zz, EX(t,cwe,zz) - EX(t,zz,cwe)) =g= net_position(t, cwe) - 0.2*abs(net_position(t, cwe));


Model Base_Model
/Obj
DEF_Cost_G
DEF_COST_H
DEF_COST_EX
DEF_COST_INEAS_EL
DEF_COST_INEAS_H
DEF_COST_INEAS_LINES

Gen_Max_El
Gen_Max_H
Gen_Max_CHP1
Gen_Max_CHP2
Gen_Max_RES
Gen_Min_RES
Gen_Max_RES_h
Gen_Min_RES_h
Gen_PH
STOR_EL
STOR_EL_Cap
STOR_EL_Max
STOR_EL_End
STOR_H
STOR_H_Cap
STOR_H_Max
*STOR_H_End
CON_DC_up
CON_DC_lo

EB_Heat
EB_Nodal
EB_Zonal
/;


Model model_dispatch
/
Base_Model
/;
Model model_ntc
/
Base_Model
CON_NTC
/;
Model model_cbco_nodal
/Base_Model
*CON_NTC
CON_CBCO_nodal_p
CON_CBCO_nodal_n
CON_Slack
/;
Model model_nodal
/Base_Model
*CON_NTC
CON_CBCO_nodal_p
CON_CBCO_nodal_n
CON_Slack
/;
Model model_cbco_zonal
/Base_Model
CON_NTC
CON_CBCO_zonal_p
CON_CBCO_zonal_n
CON_Slack
/;

*$stop
model_%model_type%.optfile = 1;
Solve model_%model_type% min COST using lp;

$include %wdir%\code_gms\result_export

