

*$if not set wdir                         $set wdir C:\Users\B036017\Documents\ramses_model\pomato
$if not set wdir                         $set wdir C:\Users\riw\tubCloud\Uni\Market_Tool\pomato

$if set ddir                             $set data_folder %wdir%\data_temp\gms_files\data\%ddir%
$if set rdir                             $set result_folder %rdir%

$if not set ddir                         $set data_folder %wdir%\data_temp\gms_files\data
$if not set rdir                         $set result_folder %wdir%\data_temp\gms_files\results\default

$if not set threads                      $set threads 4
$if not set dataset                      $set dataset dataset.gdx

$if not dexist %result_folder%           $call mkdir %result_folder%

$if not set infeasibility_electricity    $set infeasibility_electricity true
$if not set infeasibility_heat           $set infeasibility_heat true
$if not set infeasibility_lines          $set infeasibility_lines false
$if not set infeasibility_bound          $set infeasibility_bound 10000000

$if not set curtailment_electricity      $set curtailment_electricity 0.8
$if not set curtailment_heat             $set curtailment_heat 0
$if not set stor_start                   $set stor_start 0.65

$if not set model_type                   $set model_type dispatch

$if %model_type% == cbco_zonal           $set data_type zonal
$if %model_type% == zonal                $set data_type zonal
$if %model_type% == ntc                  $set data_type zonal
$if %model_type% == cbco_nodal           $set data_type nodal
$if %model_type% == nodal                $set data_type nodal

$offlisting
$offsymxref offsymlist

option
    limrow = 0
    limcol = 0
    solprint = off
    sysout = off
    reslim = 1000000
    solver = gurobi
;

$onecho >gurobi.opt
threads = %threads%
method = 1
names = 0
$offecho

Sets
t        Hour
p        Plants
tech     Technology Set
co(p)    Conventional Plants of p (alle el producerende)

he(p)    DH Plants of p    (alle varme producerende)
ho(p)    DH Plants of p (Heat Only)
cd(p)    Condensing plants
chp(p)   CHP Plants of p

es(p)    Electricity Storage Plants of p
ph(p)    Power to Heat of p
hs(p)    Heat Storage Plants of p
ts(p)    Generation Plants using Time-Series for Capacity of p
bp(p)    CHP Plants strictly backpressure
et(p)    CHP Plants with extration (can produce above the backpressure line)
bb(p)    CHP Plants with bypass (can produce below the backpressure line)
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
         cm(p)                   El per Heat at backpressure of CHP-plant
         cv(p)                   CHP-efficience of CHP-plant
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
         net_export(t,n)         Export from not-modeled regions
         inflows(t, p)           Inflows into storage es in timestep t in [MWh]

;
*###############################################################################
*                                 UPLOAD
*###############################################################################

$call gams %wdir%\pomato\ENS_pomato_gms\dataimport --data_folder=%data_folder% suppress=1

$gdxin %data_folder%\%dataset%
$load t p n z ha dc
$load chp co he ho cd ph es ts hs bp et bb
$load cb slack map_pn map_nz map_pha map_ns
$load mc_el mc_heat g_max h_max cm cv ava inflows eta hs_cap es_cap d_el d_h
$load ntc ptdf ram dc_max inc_dc net_export
$gdxin
;

*$stop
set t_end(t);
t_end(t) = No;
t_end(t)$(ord(t) eq card(t)) = Yes;

*$stop
*****SET UP SETS
Variables
COST            Total System Cost
COST_G          Cost for electricity generation
COST_H          Cost for heat generation
INJ(t,n)        Net Injection at Node n

;

Positive Variables
D_hs(t,hs)      Heat Demand by Heat Storage
D_es(t,es)      Electricity Demand by Electricity Storage
D_ph(t,ph)      Electricity Demand by Power to Heat Plants
G(t,p)          Generation per Plant p
H(t,p)          Heat generation of Plant h(p)
H_b(t,bb)        Heat generation with bypass of CHP-Plant h_b(chp)
L_es(t,es)      Electricity Storage Level at t
L_hs(t,hs)      Heat Storage Level at t
EX(t,z,zz)      Commercial Flow From z to zz
F_DC_pos(t,dc)      Flow in DC Line dc
F_DC_neg(t,dc)      Flow in DC Line dc

INFEAS_EL_N_POS(t,n)     Relaxing the EB-Electricity at high costs to avoid infeas
INFEAS_EL_N_NEG(t,n)     Relaxing the EB-Electricity at high costs to avoid infeas
INFEAS_H_POS(t,ha)       Relaxing the EB-Heat at high costs to avoid infeas
INFEAS_H_NEG(t,ha)       Relaxing the EB-Heat at high costs to avoid infeas
INFEAS_LINES(t,cb)       Infeasibility Variable for Lines (not advised)

COST_DC
COST_EX
COST_INEAS_EL
COST_INEAS_H
COST_INEAS_LINES
;

scalar
curtailment_electricity /%curtailment_electricity%/
curtailment_heat /%curtailment_heat%/
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
DEF_COST_DC
DEF_COST_INEAS_EL
DEF_COST_INEAS_H
DEF_COST_INEAS_LINES


Gen_Max_El       Max Generation for non-CHP Conventional Plants
Gen_Max_H        Maximum Generation for non-CHP Heat Plants
Gen_Max_H_BB     Maximum bypass heatproduction for Bypass Plants

Gen_Max_ET       Generation Constraint ET Plants for Heat and Electricity (extration)
Gen_Max_BP       Generation Constraint BP Plants for Heat and Electricity (back pressure)
Gen_Min_BP       Generation Constraint BP Plants for Heat and Electricity (back pressure)
Gen_Max_H_b      Generation Constraint BB Plants for Heat and Electricity (bypass)
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
;



Obj..                                                  COST =e= COST_G + COST_H + COST_EX
                                                                + COST_INEAS_EL + COST_INEAS_H + COST_INEAS_LINES + COST_DC
;

DEF_COST_G..                                           COST_G =e= sum((t,co), G(t,co)* mc_el(co));
DEF_COST_H..                                           COST_H =e= sum((t,he), H(t,he) * mc_heat(he))
                                                                 + sum((t,bb)$(cv(bb) gt 0),H_b(t,bb)*mc_heat(bb) * (cm(bb)/cv(bb)) )
                                                                 + sum((t,bb)$(cv(bb) eq 0),H_b(t,bb)*mc_heat(bb));

DEF_COST_INEAS_EL..                                    COST_INEAS_EL =e=  sum((t,n), (INFEAS_EL_N_POS(t,n) + INFEAS_EL_N_NEG(t,n))*1E4);
DEF_COST_INEAS_H..                                     COST_INEAS_H =e=  sum((t,ha), (INFEAS_H_POS(t,ha) + INFEAS_H_NEG(t,ha))*1E4);
DEF_COST_INEAS_LINES..                                 COST_INEAS_LINES =e= sum((t,cb), INFEAS_LINES(t,cb)*1E4);
DEF_COST_EX..                                          COST_EX =E= sum((t,z,zz), EX(t,z,zz)*0.1);
DEF_COST_DC..                                          COST_DC =E= sum((dc, t), (F_DC_pos(t,dc) + F_DC_neg(t,dc))*0.1);

Gen_Max_El(t,p)$(not (he(p) or ts(p)))..               G(t,p) =l= g_max(p)
;
Gen_Max_H(t,he)..                                      H(t,he) =l= h_max(he)
;
Gen_Max_ET(t,et)..                                     G(t,et) =l= g_max(et)-(cv(et)*H(t,et))
;
Gen_Min_BP(t,p)$(chp(p) and not et(p))..         G(t,p) =g= cm(p) * H(t,p)
;
Gen_Max_BP(t,bp)..                               G(t,bp) =l= cm(bp) * H(t,bp)
;
Gen_Max_H_b(t,bb)..                              H_b(t,bb) =l= h_max(bb) - H(t,bb)
;
Gen_Max_RES(t,ts)$(g_max(ts))..                  G(t,ts)  =l= g_max(ts) * ava(t,ts)
;
Gen_Min_RES(t,ts)$(g_max(ts))..                  G(t,ts)  =g= g_max(ts) * ava(t,ts) * curtailment_electricity
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

STOR_H(t,hs)..                           L_hs(t,hs) =e= L_hs(t-1,hs)$(ord(t)>1)
                                                        - H(t,hs)
                                                        + eta(hs)*D_hs(t,hs)
                                                        + 0*hs_cap(hs)$(ord(t) eq 1)
;
STOR_H_Cap(t,hs)..                       L_hs(t,hs) =l= hs_cap(hs)
;
STOR_H_Max(t,hs)..                       D_hs(t,hs) =l= h_max(hs)
;
STOR_H_End(t_end,hs)..                   L_hs(t_end,hs) =g= 0
*hs_cap(hs)* 0.65
;

EB_Heat(t,ha)..                  d_h(t,ha) =e= sum(map_pha(he,ha), H(t,he))
                                               + sum(map_pha(bb,ha), H_b(t,bb))
                                               - sum(map_pha(hs,ha), D_hs(t,hs))
                                               + INFEAS_H_POS(t,ha) - INFEAS_H_NEG(t,ha)
;
*
EB_Zonal(t,z)..                  sum(map_nz(n,z), d_el(t,n) + net_export(t,n)) =e=      sum(map_nz(n,z), sum(map_pn(co,n), G(t,co)))
                                                                                      - sum(map_nz(n,z), sum(map_pn(ph,n), D_ph(t,ph)))
                                                                                      - sum(map_nz(n,z), sum(map_pn(es,n), D_es(t,es)))
                                                                                      - sum(zz, EX(t,z,zz))
                                                                                      + sum(zz, EX(t,zz,z))
                                                                                      + sum(map_nz(n,z), INFEAS_EL_N_POS(t,n) - INFEAS_EL_N_NEG(t,n))
;

EB_Nodal(t,n)..                  d_el(t,n) + net_export(t,n) =e=   sum(map_pn(co,n), G(t,co))
                                                                 - sum(map_pn(ph,n), D_ph(t,ph))
                                                                 - sum(map_pn(es,n), D_es(t,es))
                                                                 - sum(dc, (F_DC_pos(t,dc) - F_DC_neg(t,dc))*inc_dc(dc,n))
                                                                 - INJ(t,n)
                                                                 + (INFEAS_EL_N_POS(t,n) - INFEAS_EL_N_NEG(t,n))
;

CON_DC_up(t,dc)..                (F_DC_pos(t,dc) - F_DC_neg(t,dc)) =l= dc_max(dc)
;
CON_DC_lo(t,dc)..                (F_DC_pos(t,dc) - F_DC_neg(t,dc)) =g= -dc_max(dc)
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


Model Base_Model
/Obj
DEF_COST_G
DEF_COST_H
DEF_COST_EX
DEF_COST_DC
DEF_COST_INEAS_EL
DEF_COST_INEAS_H
DEF_COST_INEAS_LINES

Gen_Max_El
Gen_Max_H
Gen_Max_ET
Gen_Max_BP
Gen_Min_BP
Gen_Max_H_b

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
CON_Slack
/;
Model model_cbco_nodal
/Base_Model
CON_NTC
CON_CBCO_nodal_p
CON_CBCO_nodal_n
CON_Slack
/;
Model model_nodal
/Base_Model
CON_NTC
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

$include %wdir%\pomato\ENS_pomato_gms\result_export

