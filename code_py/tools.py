import pandas as pd
import json
from pathlib import Path

def create_folder_structure(base_path, logger=None):
	folder_structure = {
		"code_jl": {},
		"code_py": {},
		"data_input": {},
		"data_output": {},
		"data_temp": {
				"bokeh_files": {
						"market_result": {}
						},
				"julia_files": {
						"data": {},
						"results": {},
						"cbco_data": {},
						},
				"python_files": {
						"tmp_tables": {},
						},
					},
	  "logs": {},
	  "profiles": {},
	}

	if logger:
		logger.info("Creating Folder Structure")
	try:
		folder = folder_structure
		while folder:
			subfolder_dict = {}
			for subfolder in folder:
				if not Path.is_dir(base_path.joinpath(subfolder)):
					if logger:
						logger.info(f"creating folder {subfolder}")
					Path.mkdir(base_path.joinpath(subfolder))
				if folder[subfolder]:
					for subsubfolder in folder[subfolder]:
						subfolder_dict[subfolder + "/" + subsubfolder] = folder[subfolder][subsubfolder]
			folder = subfolder_dict.copy()
	except:
		if logger:
			logger.error("Cound not create folder structure!")
		else:
			print("Cound not create folder structure!")

def find_xy_limits(list_plots):
	try:
		x_max = 0
		x_min = 0
		y_max = 0
		y_min = 0
		for plots in list_plots:
			tmpx = plots[0].reshape(len(plots[0]),)
			tmpy = plots[1].reshape(len(plots[1]),)
			x_coords = tmpx.tolist()
			y_coords = tmpy.tolist()
			x_coords.extend([x_max,x_min])
			y_coords.extend([y_max,y_min])
			x_max = max(x_coords)
			x_min = min(x_coords)
			y_max = max(y_coords)
			y_min = min(y_coords)

		return {"x_max": x_max, "x_min": x_min,
				"y_max": y_max, "y_min": y_min}
	except:
		print('error:find_xy_limits')

def split_length_in_ranges(step_size, length):
	ranges = []
	if step_size > length:
		ranges.append(range(0, length))
	else:
		ranges = []
		step_size = int(step_size)
		for i in range(0, int(length/step_size)):
			ranges.append(range(i*step_size, (i+1)*step_size))
		ranges.append(range((i+1)*step_size, length))
	return ranges


def from_named_range(wb, name):
	position_obj = wb.name_map[name]
	address = position_obj[0].__dict__['formula_text'].split('!')
	sheet = wb.sheet_by_name(address[0])
	rng = address[1][1:].replace(":", "").split('$')
	dic = {}
	header = sheet.row_values(int(rng[1])-1, start_colx=a2i(rng[0]), end_colx=a2i(rng[2]))
	for idx,r in enumerate(range(int(rng[1]), int(rng[3]))):
		tmp = sheet.row_values(r, start_colx=a2i(rng[0]), end_colx=a2i(rng[2]))
		dic[idx] = {header[i]: tmp[i] for i in range(0,len(tmp))}
		df = pd.DataFrame.from_dict(dic, orient='index')
	return(df)


def a2i(alph): # alph_to_index_from_excelrange
	import numpy as np
	# A = 0 .... Z = 25, AA=26
	n = len(alph)
	p = 0
	for i in range(0, n):
		p += (ord(alph[i]) - ord('A') + 1) * np.power(26,(n-1-i))
	return(int(p-1))

def array2latex(array):
	for i in array:
		row = []
		for j in i:
			row.append(str(round(j,3)))
		print(' & '.join(row) + ' \\\ ')


def delete_empty_subfolders(folder):
    """Delete all empty subfolders to input folder"""
    non_empty_subfolders = {f.parent for f in folder.rglob("*") if f.is_file()}
    for subfolder in folder.iterdir():
        if subfolder not in non_empty_subfolders:
            subfolder.rmdir()

def default_options():

	json_str = """{
					"optimization": {
						"type": "dispatch",
						"model_horizon": [0,1],
						"infeas_heat": true,
						"infeas_el_nodal": true,
						"infeas_el_zonal": true,
						"infeas_lines": false,
						"infeas_lines_ref": false
					},
					"grid": {
						"type": "dispatch",
						"capacity_multiplier": 1,
						"reference_flows": false,
						"precalc_filename": "",
						"senstitivity": 5e-2,
						"cbco_option": ""
					},
					"data": {
						"unique_mc": true,
						"round_demand": true,
						"default_efficiency": 0.5,
						"default_mc": 200,
						"co2_price": 20,
						"all_lines_cb": false,
						"d2cf_data": false
					}
				}"""

	return json.loads(json_str)

