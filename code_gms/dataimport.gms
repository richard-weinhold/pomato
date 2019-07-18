
$if not set data_folder          $set data_folder data
$if not set data_type            $set data_type nodal

Sets
t        Hour
p        Plants
tech     Technology Set
es_tech(tech)    es tech
hs_tech(tech)    hs tech
ph_tech(tech)    ph tech
ts_tech(tech)    ts tech
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
map_nz(n,z)              Node-Zone El Mapping
map_pha(p,ha)            Plant-Heat Area Mapping
map_pn(p,n)              Plant-Node Mapping
map_ptech(p,tech)
slack(n)                 Slacknode
map_ns(n,slack)          Node to Slack mapping
map_dcn(dc,n,n)
;


Alias(n,nn);
Alias(z,zz);

Parameter
***Data Load Paramters
         zone_data
         heatarea_data
         node_data
         slack_data
         plant_data
         cbco_data
         dclines_data
         plant_types
         plant_mapping
         demand_el_data
         demand_h_data
         ava_data
         net_position_data
         net_export_data
         ntc_data

***Generation related parameters
         mc(p)                   Marginal Costs of Technology s
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
         ntc(z,zz)
;
scalar test;

*###############################################################################
*                                 UPLOAD
*###############################################################################
$onUNDF
$call csv2gdx %data_folder%\zones.csv output=%data_folder%\zones.gdx id=zone_data Index=(1) Values=(1) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\zones.gdx
$load zone_data
$load z=dim1
;

$call csv2gdx %data_folder%\nodes.csv output=%data_folder%\nodes.gdx id=node_data Index=(1,3) Values=(4) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\nodes.gdx
$load n=dim1
$load map_nz=node_data
$load node_data
$gdxin
;


$call csv2gdx %data_folder%\slack_zones.csv output=%data_folder%\slack_zones.gdx id=slack_data Index=(1) Values=(2..LastCol) UseHeader=Y StoreZero=Y valueDim=Y
$gdxin %data_folder%\slack_zones.gdx
$load slack_data
$load slack=dim2
$gdxin
;
map_ns(n,slack)$(slack_data(n, slack) eq 1) = Yes;

$call csv2gdx %data_folder%\heatareas.csv output=%data_folder%\heatareas.gdx id=heatarea_data Index=(1) UseHeader=Y StoreZero=Y valueDim=1
$gdxin %data_folder%\heatareas.gdx
$load ha=dim1
$gdxin
;

$call csv2gdx %data_folder%\plants.csv output=%data_folder%\plants.gdx id=plant_data Index=(1) Values=(2,5,6,7) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\plants.gdx
$load plant_data
$load p=dim1
$gdxin
;


$call csv2gdx %data_folder%\plant_types.csv output=%data_folder%\plant_types.gdx id=plant_types Index=(1) Values=(2,3,4,5) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\plant_types.gdx
$load plant_types
$load tech=dim1
$gdxin
;

$call csv2gdx %data_folder%\plants.csv output=%data_folder%\map_pn.gdx id=map_pn Index=(1,4) Values=(5) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\map_pn.gdx
$load map_pn=map_pn
$gdxin
;

$call csv2gdx %data_folder%\plants.csv output=%data_folder%\map_ptech.gdx id=map_ptech Index=(1,3) Values=(5) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\map_ptech.gdx
$load map_ptech=map_ptech
$gdxin
;

$call csv2gdx %data_folder%\plants.csv output=%data_folder%\map_pha.gdx id=map_pha Index=(1,8) Values=(5) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\map_pha.gdx
$load map_pha=map_pha
$gdxin
;

es_tech(tech)$(plant_types(tech, "es") eq 1) = Yes;
hs_tech(tech)$(plant_types(tech, "hs") eq 1) = Yes;
ph_tech(tech)$(plant_types(tech, "ph") eq 1) = Yes;
ts_tech(tech)$(plant_types(tech, "ts") eq 1) = Yes;

es(p) = No;
hs(p) = No;
ts(p) = No;
ph(p) = No;

es(p)$sum(es_tech, map_ptech(p,es_tech)) = Yes ;
hs(p)$sum(hs_tech, map_ptech(p,hs_tech)) = Yes ;
ts(p)$sum(ts_tech, map_ptech(p,ts_tech)) = Yes ;
ph(p)$sum(ph_tech, map_ptech(p,ph_tech)) = Yes ;

mc(p) = plant_data(p,"mc");
eta(p) = plant_data(p, "eta");
g_max(p) = plant_data(p, "g_max");
h_max(p) = plant_data(p, "h_max");
es_cap(es) = plant_data(es, "g_max")*8;
hs_cap(hs) = plant_data(hs, "h_max")*8;

chp(p) = No;
chp(p)$((g_max(p) > 0) and (h_max(p) > 0)) = Yes;

co(p) = No;
co(p)$(g_max(p) > 0) = Yes;

he(p) = No;
he(p)$(h_max(p) > 0) = Yes;

$call csv2gdx %data_folder%\demand_el.csv output=%data_folder%\demand_el.gdx id=demand_el_data Index=(1) Values=(2..LastCol) UseHeader=Y
$gdxin %data_folder%\demand_el.gdx
$load t=dim1
$load demand_el_data
;

d_el(t,n) = demand_el_data(t,n)


$call csv2gdx %data_folder%\demand_h.csv output=%data_folder%\demand_h.gdx id=demand_h_data Index=(1) Values=(2..LastCol) UseHeader=Y storeZero=Y
$gdxin %data_folder%\demand_h.gdx
;

if (card(ha)>0,
         execute_load "%data_folder%\demand_h.gdx", d_h=demand_h_data;
else
         d_h(t,ha) = 0;
);

$call csv2gdx %data_folder%\availability.csv output=%data_folder%\availability.gdx id=ava_data Index=(1) Values=(2..LastCol) UseHeader=Y
$gdxin %data_folder%\availability.gdx
*$load ava_data
;
if (card(ts)>0,
         execute_load "%data_folder%\availability.gdx", ava=demand_el_data;
else
         ava(t,p) = 0;
);

$call csv2gdx %data_folder%\cbco.csv output=%data_folder%\cbco.gdx id=cbco_data Index=(1) Values=(2..LastCol) UseHeader=Y
$gdxin %data_folder%\cbco.gdx
$load cbco_data
$load cb=dim1
;

$if %data_type% == zonal ptdf(cb, z) = cbco_data(cb, z);
$if %data_type% == nodal ptdf(cb, n) = cbco_data(cb, n);

ram(cb) = cbco_data(cb, "ram");

$call csv2gdx %data_folder%\dclines.csv output=%data_folder%\dclines.gdx id=dclines_data Index=(1) Values=(LastCol) UseHeader=Y
$gdxin %data_folder%\dclines.gdx
$load dclines_data
$load dc=dim1
;
dc_max(dc) = dclines_data(dc);

$call csv2gdx %data_folder%\dclines.csv output=%data_folder%\map_dcn.gdx id=map_dcn Index=(1,2,3) Values=(LastCol) UseHeader=Y
$gdxin %data_folder%\map_dcn.gdx
$load map_dcn
;

inc_dc(dc,n) = 0;
inc_dc(dc,n)$sum(nn, map_dcn(dc,n,nn)) = 1;
inc_dc(dc,n)$sum(nn, map_dcn(dc,nn,n)) = -1;

set test_nex;
$call csv2gdx %data_folder%\net_position.csv output=%data_folder%\net_position.gdx id=net_position_data Index=(1) Values=(3..LastCol) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\net_position.gdx
$load test_nex=*
;
if (card(test_nex)>0,
         execute_load "%data_folder%\net_position.gdx", net_position=net_position_data;
else
         net_position(t,z) = 0;
);

$call csv2gdx %data_folder%\net_export.csv output=%data_folder%\net_export.gdx id=net_export Index=(1) Values=(3..LastCol) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\net_export.gdx
$load net_export
;

set test_ntc;
$call csv2gdx %data_folder%\ntc.csv output=%data_folder%\ntc.gdx id=ntc_data Index=(1,2) Values=(3) UseHeader=Y StoreZero=Y
$gdxin %data_folder%\ntc.gdx
$load test_ntc=*
;
if (card(test_ntc)>0,
         execute_load "%data_folder%\net_position.gdx", ntc=ntc_data;
else
         ntc(z,zz) = 0;
);

execute_unload "%data_folder%\dataset.gdx",
t p n z ha dc
chp co he ph es ts hs
cb slack map_pn map_nz map_pha map_ns
mc g_max h_max ava eta hs_cap es_cap d_el d_h
ntc ptdf ram dc_max inc_dc net_export net_position
;




