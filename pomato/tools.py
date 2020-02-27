import pandas as pd
import json
from pathlib import Path
import pexpect
from pexpect import popen_spawn

class FileAdapter():
    def __init__(self, logger):
        self.logger = logger
    def write(self, data):
        # NOTE: data can be a partial line, multiple lines
        data = data.strip()  # ignore leading/trailing whitespace
        if data: # non-blank
            # pass
            self.logger.info(data.decode(errors="ignore"))
    def flush(self):
        pass  # leave it to logging to flush properly

class InteractiveJuliaProcess():
    def __init__(self, wdir, logger, julia_model):

        self.julia_process = popen_spawn.PopenSpawn('julia --project=project_files/pomato',
                                                    cwd=wdir, timeout=100000,
                                                    logfile=FileAdapter(logger))
        self.logger = logger
        self.solved = False

        if julia_model == "market_model":
            self.julia_process.sendline('using MarketModel')
            self.julia_process.expect(["Initialized", "ERROR"])
        elif julia_model == "cbco":
            self.julia_process.sendline('using RedundancyRemoval')
            self.julia_process.expect(["Initialized", "ERROR"])
        else:
            self.logger.error("Model Options not available!")

    def run(self, command):
        self.julia_process.sendline(command)
        self.julia_process.expect(["Everything Done!", "ERROR"])

        if self.julia_process.after != "ERROR":
            self.solved = True

    def terminate(self):
        self.julia_process.close()


def newest_file_folder(folder, keyword="", number_of_elm=1):
    """Return newest (n) folders/files from a folder.

    Plattform sensitive function to return the last file generated in a specified folder.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to look for files in.
    keyword : string, optional
        A supplied string reduces the number of possibilities and makes it more robust.
    number_of_files : int, optional
        Specify the number of files or folders returned.
    """
    df = pd.DataFrame()
    df["elm"] = [i for i in folder.iterdir() if keyword in i.name]

    try:  # This should work in Windows
        df["time"] = [i.lstat().st_ctime for i in folder.iterdir() if keyword in i.name]
    except AttributeError:  # This should work in OSX
        df["time"] = [i.lstat().st_birthtime for i in folder.iterdir() if keyword in i.name]
    except AttributeError:  # Fallback option (linux, but the first try should work there as well)
        df["time"] = [i.lstat().st_mtime for i in folder.iterdir() if keyword in i.name]

    if number_of_elm > 1:
        return list(df.nlargest(2, "time").elm)
    else:
        return df.elm[df.time.idxmax()]


def create_folder_structure(base_path, logger=None):
    """Create folder structure to run POMATO.

    Since the repository does not conatin the empty folders for
    temporary data, this function checks whether these exist and
    creates them if nessesary. This is only valid if the process is
    run from the pomato root folder, here checked by looking if the
    pomato package folder exists.



    Parameters
    ----------
    base_path : pathlib.Path
        Pomato root folder.
    logger : logger, optional
        If a logger is supplied the status messages will be logged there.

    Raises
    ------
    RuntimeError
        If the base folder appearently does not coincide with the pomato base folder
        an exception is raised.
    """
    folder_structure = {
        "pomato": {},
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
                "gms_files": {
                        "data": {},
                        "results": {},
                        },
                    },
        "logs": {},
        "profiles": {},
    }

    if logger:
        logger.info("Creating Folder Structure")

    if not base_path.joinpath("pomato").is_dir():
        if logger:
            logger.error("Process is not run from pomato root folder!")
        else:
            print("Process is not run from pomato root folder!")
        raise RuntimeError
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


def _delete_empty_subfolders(folder):
    """Delete all empty subfolders to input folder"""
    non_empty_subfolders = {f.parent for f in folder.rglob("*") if f.is_file()}
    for subfolder in folder.iterdir():
        if subfolder not in non_empty_subfolders:
            subfolder.rmdir()

def options():
    """Returns the default options of POMATO.


    """
    options_dict = {"optimization": {}, "grid": {}, "data": {}}
    options_dict["optimization"]: {
        "type": "cbco_nodal",
        "solver": "glpk",
        "gams": False,
        "model_horizon": [0, 2],
        "heat_model": False,
        "split_timeseries": True,
        "redispatch": {
            "include": True,
            "cost": 1},
        "curtailment": {
            "include": False,
            "cost": 1E2},
        "parameters": {
            "storage_start": 0.65},
        "infeasibility": {
            "heat": {
                "include": False,
                "cost": 1E3,
                "bound": 20},
            "electricity": {
                "include": True,
                "cost": 1E3,
                "bound": 20},
            "lines": {
                "include": False,
                "cost": 1E3,
                "bound": 20}},
        "plant_types": {
            "es": ["hydro_res", "hydro_psp"],
            "hs": [],
            "ts": ["wind", "solar"],
            "ph": [],}
        }
    options_dict["grid"] = {
            "cbco_option": "full",
            "precalc_filename": "",
            "senstitivity": 5e-2,
            "capacity_multiplier": 1,
            "preprocess": True,
            "gsk": "gmax",
            }

    options_dict["data"] = {
        "data_type": "ieee",
        "stacked": [],
        "process": [],
        "process_input": False,
        "unique_mc": False,
        "round_demand": True,
        "default_efficiency": 0.5,
        "default_mc": 200,
        "co2_price": 20,
        }


    return json.loads(json_str)

def gams_modelstat_dict(modelstat):
    """ Returns GAMS ModelStat String for int input"""
    gams_status_dict = {1:   "Optimal",
                        2:   "Locally Optimal",
                        3:   "Unbounded",
                        4:   "Infeasible",
                        5:   "Locally Infeasible",
                        6:   "Intermediate Infeasible",
                        7:   "Intermediate Nonoptimal",
                        8:   "Integer Solution",
                        9:   "Intermediate Non-Integer",
                        10:  "Integer Infeasible",
                        11:  "Licensing Problems - No Solution",
                        12:  "Error Unknown",
                        13:  "Error No Solution",
                        14:  "No Solution Returned",
                        15:  "Solved Unique",
                        16:  "Solved",
                        17:  "Solved Singular",
                        18:  "Unbounded - No Solution",
                        19:  "Infeasible - No Solution"}

    if modelstat in gams_status_dict.keys():
        return gams_status_dict[modelstat]
    else:
        return f"Unknown GAMS ModelStat {modelstat}"