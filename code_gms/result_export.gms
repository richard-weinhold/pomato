
execute_unload "%result_folder%\result.gdx";
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\G.csv symb=G format=csv header="t,p,G"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\H.csv symb=H format=csv header="t,p,H"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\D_es.csv symb=D_es format=csv header="t,p,D_es"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\D_hs.csv symb=D_hs format=csv header="t,p,D_hs"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\D_ph.csv symb=D_ph format=csv header="t,p,D_ph"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\L_es.csv symb=L_es format=csv header="t,p,L_es"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\L_hs.csv symb=L_hs format=csv header="t,p,L_hs"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\EX.csv symb=EX format=csv header="t,z,zz,EX"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\INJ.csv symb=INJ format=csv header="t,n,INJ"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\G.csv symb=G format=csv header="t,p,G"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\F_DC.csv symb=F_DC format=csv header="t,dc,F_DC"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\INFEAS_H_NEG.csv symb=INFEAS_H_NEG format=csv header="t,ha,INFEAS_H_NEG"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\INFEAS_H_POS.csv symb=INFEAS_H_POS format=csv header="t,ha,INFEAS_H_POS"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\INFEAS_EL_N_NEG.csv symb=INFEAS_EL_N_NEG format=csv header="t,n,INFEAS_EL_N_NEG"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\INFEAS_EL_N_POS.csv symb=INFEAS_EL_N_POS format=csv header="t,n,INFEAS_EL_N_POS"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\INFEAS_LINES.csv symb=INFEAS_LINES format=csv header="t,cb,INFEAS_LINES"'

execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\EB_nodal.csv symb=EB_nodal format=csv CSVAllFields EpsOut=0 header="t,n,Level,EB_nodal,Lower,Upper,Scale"'
execute 'gdxdump %result_folder%\result.gdx output=%result_folder%\EB_zonal.csv symb=EB_zonal format=csv CSVAllFields EpsOut=0 header="t,z,Level,EB_zonal,Lower,Upper,Scale"'

File results / %result_folder%\misc_result.json /;
put results;
put '{"Objective Value":', COST.L,
    ', "COST_G":', COST_G.L,
    ', "COST_H":', COST_H.L,
    ', "COST_EX":', COST_EX.L,
    ', "COST_INEAS_EL":', COST_INEAS_EL.L,
    ', "COST_INEAS_H":', COST_INEAS_H.L,
    ', "COST_INEAS_LINES":', COST_INEAS_LINES.L,
    ', "Solve Status":', model_%model_type%.modelstat,
    '}'/;
putclose;


