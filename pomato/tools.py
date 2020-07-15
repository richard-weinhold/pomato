"""Collection of tools potentially used by multiple components of POMATO.

This collection is characterized by a certain degree of generality and they cannot be 
attributed to a specified component of pomato.
"""

import json
import operator
import subprocess
import threading
import shutil, os 
import time
from functools import reduce
from pathlib import Path

import pandas as pd
import pomato._installation.manage_julia_env as julia_management

class JuliaDaemon():
    """Class to communicate with a julia daemon process.
    
    The RedundancyRemoval and MarketModel processes are written in Julia. 
    This class's purpose is to communicate with a daemon process in julia
    that runs these processes on demand and allows to excecute them multiple times
    without restarting the julia process, which would require lengthy precompile everytime 
    instead of one lengthy precompile. 
    
    This is implemented by two daemon processes, one in python the other in julia, or more specifically
    a threaded julia daemon is initialized in python.
    This allows start-up of the julia process while other pomato related processes are done. 
    The communication is done through a json file in the data_temp/julia_files folder.

    
    Attributes
    ----------
    daemon_file : pathlib.Path
        Path to json file for process management.
    julia_daemon : Thread with julia subprocess.
        Threaded subprocess of the julia daemon process.
    julia_module : str
        Defines whether RedundancyRemoval or MarketModel is initialized/used.
    julia_daemon_path : pathlib.Path
        Description
    solved : bool
        Indicator if julia process has successfully concluded.
    wdir : pathlib.Path
        Workingdirectory, should  be pomato root directory.
   
    """
    def __init__(self, logger, wdir, package_dir, julia_module):
        if not julia_module in ["market_model", "redundancy_removal"]:
            raise TypeError("The JuliaDaemon has to be initialized with market_model or redundancy_removal")

        self.julia_module = julia_module

        self.logger = logger
        self.wdir = wdir
        self.package_dir = package_dir
        self.daemon_file = wdir.joinpath(f"data_temp/julia_files/daemon_{julia_module}.json")
        self.julia_daemon_path = package_dir.joinpath("julia_daemon.jl")
        self.write_daemon_file(self.default_daemon_file())
        # Start Julia daemon in a thread
        self.julia_daemon = threading.Thread(target=self.start_julia_daemon, args=())
        self.julia_daemon.start()
        self.solved = False

    @property
    def is_alive(self):
        return self.julia_daemon.is_alive()
        
    def start_julia_daemon(self):
        """Stat julia daemon"""
        args = ["julia", "--project=" + str(self.package_dir.joinpath("_installation/pomato")),
                str(self.julia_daemon_path), self.julia_module]
        with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, cwd=str(self.wdir)) as programm:
            for line in programm.stdout:
                if not any(w in line.decode(errors="ignore") for w in ["Academic license"]):
                    self.logger.info(line.decode("UTF-8", errors="ignore").replace("[ Info:", "").strip())

    def join(self):
        """Exit the julia daemon and join python threads"""
        if self.julia_daemon.is_alive():
            file = self.read_daemon_file()
            file["break"] = True
            self.write_daemon_file(file)
            self.julia_daemon.join()

    def default_daemon_file(self):
        """Return default daemon file
        
        Returns
        -------
        file : dict
            daemon file as dictionary. 
        """
        file = {"processing": False,
                "run": False,
                "ready": False,
                "type": self.julia_module,
                "file_suffix": "py",
                "redispatch": False,
                "chance_constrained": False,
                "multi_threaded": True,
                "data_dir": "/data/",
                "break": False}
        return file

    def disable_multi_threading(self):
        """Disable multithreading."""
        file = self.read_daemon_file()
        file["multi_threaded"] = False
        self.write_daemon_file(file)

    def write_daemon_file(self, file):
        """Write (updated) file to disk"""
        while True:
            try:
                with open(self.daemon_file, 'w') as config:
                    json.dump(file, config, indent=2)
                    break
            except:
                self.logger.warning("Failed to write to daemon file.")
                time.sleep(1)

    def read_daemon_file(self):
        """Read daemon file from disk"""
        while True:
            try:
                with open(self.daemon_file, 'r') as jsonfile:
                    file = json.load(jsonfile)
                return file
            except:
                self.logger.warning("Failed to read from daemon file.")
                time.sleep(1)

    def halt_while_processing(self):
        """Halt python main thread, while julia is processing.
        Sometimes its better for the user to wait until julia is done.
        """
        progress_indicator = 1
        counter = 0
        while True:
            time.sleep(0.1)
            file = self.read_daemon_file()
            if not file["processing"]:
                self.logger.info("Programm Done")
                break
            else:
                # self.logger.info("Waiting for processing to complete")
                counter += 1
                if counter > 100:
                    if progress_indicator < 0:
                        dots = "\\"
                    else:
                        dots = "/"
                    print("\r" + dots + "Waiting for processing to complete" + dots, end="")
                    progress_indicator *= -1
                    counter = 0


    def halt_until_ready(self):
        """Halt python main thread until julia is initialized.

        Julias startup time can be fairly long, therefore its started immediately 
        when pomato is initialized.

        However, when a julia process is started it has to be ready. This method 
        halts the main thread until julia is ready. 
        """
        progress_indicator = 1
        counter = 0
        while True:
            time.sleep(0.1)
            file = self.read_daemon_file()
            if file["ready"]:
                self.logger.info("Process ready!")
                break
            else:
                counter += 1
                if counter > 50:
                    if progress_indicator < 0:
                        dots = "\\"
                    else:
                        dots = "/"
                    print("\r" + dots + "Waiting until Julia is ready" + dots, end="")
                    progress_indicator *= -1
                    counter = 0

    def run(self, args=None):
        """Run julia process.
        Writes commands to daemon file, initiating start of a process. Halts while it is active
        and set attribute solved as true.
        
        Parameters
        ----------
        args : dict, optional
            Dictionary with pairs of values to change in the daemon file.       
        """
        self.solved = False
        self.halt_until_ready()
        file = self.read_daemon_file()
        file["run"] = True
        file["processing"] = True
        if args:
            for k,v in args.items():
                file[k] = v
        self.write_daemon_file(file)
        time.sleep(0.1)
        self.halt_while_processing()
        self.solved = True

def newest_file_folder(folder, keyword="", number_of_elm=1):
    """Return newest (n) folders/files from a folder.

    Platform sensitive function to return the last file generated in a specified folder.

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
    
    if df["elm"].empty:
        raise FileNotFoundError("No results in folder.")

    try:  # This should work in Windows
        df["time"] = [i.lstat().st_ctime for i in folder.iterdir() if keyword in i.name]
    except AttributeError:  # This should work in OSX
        df["time"] = [i.lstat().st_birthtime for i in folder.iterdir() if keyword in i.name]
    except AttributeError:  # Fallback option (linux, but the first try should work there as well)
        df["time"] = [i.lstat().st_mtime for i in folder.iterdir() if keyword in i.name]

    if number_of_elm > 1:
        return list(df.nlargest(number_of_elm, "time").elm)
    else:
        return df.elm[df.time.idxmax()]

def create_folder_structure(base_path, logger=None):
    """Create folder structure to run POMATO.

    Since the repository does not contain the empty folders for
    temporary data, this function checks whether these exist and
    creates them if necessary. This is only valid if the process is
    run from the pomato root folder, here checked by looking if the
    pomato package folder exists.

    Parameters
    ----------
    base_path : pathlib.Path
        Pomato root folder.
    logger : logger, optional
        If a logger is supplied the status messages will be logged there.

    """
    folder_structure = {
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
            logger.error("Could not create folder structure!")
        else:
            print("Cound not create folder structure!")

def find_xy_limits(list_plots):
    """Find max/min of a list of coordinates, i.e. a canvas where all points are included."""
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
    """Split a range 1:N in a list of ranges with specified length."""
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

def _delete_empty_subfolders(folder):
    """Delete all empty subfolders to input folder"""
    non_empty_subfolders = {f.parent for f in folder.rglob("*") if f.is_file()}
    for subfolder in folder.iterdir():
        if subfolder not in non_empty_subfolders:
            subfolder.rmdir()

def default_options():
    """Returns the default options of POMATO.

    """
    options_dict = {"optimization": {}, "grid": {}, "data": {}}
    options_dict["optimization"] = {
        "type": "cbco_nodal",
        "model_horizon": [0, 2],
        "heat_model": False,
        "constrain_nex": False,
        "timeseries": {
            "split": True,
            "market_horizon": 1000,
            "redispatch_horizon": 24},
        "redispatch": {
            "include": False,
            "zonal_redispatch": True,
            "zones": [],
            "cost": 1},
        "curtailment": {
            "include": False,
            "cost": 1E2},
        "chance_constrained": {
            "include": False,
            "fixed_alpha": True,
            "cc_res_mw": 50,
            "alpha_plants_mw": 200},
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
            "es": [],
            "hs": [],
            "ts": [],
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

    return options_dict

def add_default_options(option_dict):
    """Takes the loaded option dict and adds missing values from default options.
    
    Uses function that are "taken" from https://stackoverflow.com/a/14692747
    Parameters
    ----------
    option_dict : option_dict
        Optionfile loaded from disk.
    """
    options = default_options()
    for i in _dict_generator(option_dict):
        try: 
            _setInDict(options, i, _getFromDict(option_dict, i))
        except:
            raise ValueError(".".join(i) + " is not a valid option")
    return options

def _dict_generator(indict, pre=None):
    """Flatten Option Dict.

    Source: https://stackoverflow.com/a/12507546
    """
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in _dict_generator(value, pre + [key]):
                    yield d
            else:
                yield pre + [key]
    else:
        yield pre + [indict]

def _getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def _setInDict(dataDict, mapList, value):
    _getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def copytree(src, dst, symlinks=False, ignore=None):
    # https://stackoverflow.com/a/12514470
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)