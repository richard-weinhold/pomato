
$if not set result_folder  $set result_folder C:\Users\riw\tubCloud\Uni\Market_Tool\pomato\data_temp\gms_files\results\demand_2030_hi


Variables
COST
COST_G
COST_H
COST_EX
COST_INEAS_EL
COST_INEAS_H
COST_INEAS_LINES
;
Equation EB_Nodal, EB_Zonal

$gdxin "%result_folder%\result.gdx"
$load EB_Zonal EB_Nodal COST COST_G COST_H COST_EX COST_INEAS_EL COST_INEAS_H COST_INEAS_LINES
$gdxin
;
execute_unload "%result_folder%/tmp_result.gdx";
execute 'gdxdump %result_folder%/tmp_result.gdx output=%result_folder%/EB_nodal.csv symb=EB_nodal format=csv CSVAllFields EpsOut=0 header="t,n,Level,EB_nodal,Lower,Upper,Scale"'
execute 'gdxdump %result_folder%/tmp_result.gdx output=%result_folder%/EB_zonal.csv symb=EB_zonal format=csv CSVAllFields EpsOut=0 header="t,z,Level,EB_zonal,Lower,Upper,Scale"'


$if not set json_file  $set json_file %result_folder%\misc_results.json
File results / %json_file% /;
put results;
put '{"Objective Value":', COST.L,
    ', "COST_G":', COST_G.L,
    ', "COST_H":', COST_H.L,
    ', "COST_EX":', COST_EX.L,
    ', "COST_INEAS_EL":', COST_INEAS_EL.L,
    ', "COST_INEAS_H":', COST_INEAS_H.L,
    ', "COST_INEAS_LINES":', COST_INEAS_LINES.L,
    ', "Solve Status":', 1,
    '}'/;
putclose;







