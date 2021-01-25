import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import xlrd

def _mpc_data_pu_to_real(lines,  base_kv, base_mva):
    """Convert pu to actual units for the mpc case."""
    v_base = base_kv * 1e3
    s_base = base_mva * 1e6
    z_base = np.power(v_base,2)/s_base
    lines['r_pu'] = np.divide(lines['r'], z_base)
    lines['x_pu'] = np.divide(lines['x'], z_base)
    return lines

def _mpc_data_structure():
    data_structure = [
        ["nodes", "index", "any", False],
        ["nodes", "zone", "zones.index", False],
        ["nodes", "slack", "bool", False],
        ["nodes", "lat", "float64", False],
        ["nodes", "lon", "float64", False],
        ["nodes", "name", "str", False],
        ["nodes", "voltage", "float64", False],
        ["zones", "index", "any", False],
        ["plants", "index", "any", False],
        ["plants", "node", "nodes.index", False],
        ["plants", "mc_el", "float64", False],
        ["plants", "g_max", "float64", False],
        ["plants", "plant_type", "any", False],
        ["plants", "fuel", "any", False],
        ["lines", "index", "any", False],
        ["lines", "node_i", "nodes.index", False],
        ["lines", "node_j", "nodes.index", False],
        ["lines", "b", "float64", False],
        ["lines", "capacity", "float64", False],
        ["lines", "contingency", "bool", False],
        ["lines", "technology", "float64", True],
        ["lines", "type", "float64", True],
        ["lines", "voltage", "any", True],
        ["demand_el", "index", "any", False],
        ["demand_el", "timestep", "any", False],
        ["demand_el", "node", "nodes.index", False],
        ["demand_el", "demand_el", "float", False]]

    return pd.DataFrame(data_structure, columns=["data", "attributes", "type", "optional"]).set_index("data")

class DataWorker(object):
    """Data Worker Module reads data from disk.

    This module's purpose is to hide all the file system specific functions
    and allow for rather seemless data import.
    An instance of the DataManagement class acts as the carrier for the
    read data and is used as an attribute of DataWorker.

    The based on *file_path* the module tries to import xlsx or matpower (.m
    or .mat) cases. Reading the excel file is done by going through the
    pre-defined data tables in *data.data_attributes* and attaching them
    to the DataManagement instance for processing and validation. Matpower
    cases are imported manually and

    Attributes
    ----------
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class with processed input data.

    Parameters
    ----------
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class with processed input data.
    file_path : pathlib.Path
        Filepath to input data.

    """

    def __init__(self, data, file_path):
        self.logger = logging.getLogger('log.pomato.data.DataWorker')
        self.data = data

        if ".xls" in str(file_path):
            self.logger.info("Loading data from Excel file")
            self.read_xls(file_path)
            self.data.data_structure.set_index("data", inplace=True)

        elif ".zip" in str(file_path):
            self.logger.info("Loading data from zipped data archive")
            self.read_csv_zipped(file_path)
            self.data.data_structure.set_index("data", inplace=True)
        elif ".mat" in str(file_path):
            self.logger.info("Loading data from matpower .mat case-file")
            self.process_matpower_case(file_path, ".mat")
        elif ".m" in str(file_path):
            self.logger.info("Loading data from matpower .m case-file")
            self.process_matpower_case(file_path, ".m")
        elif file_path.is_dir():
            self.logger.info("Loading data from folder")
            self.read_csv_folder(file_path)
            self.data.data_structure.set_index("data", inplace=True)
        else:             
            self.logger.error("Filepath: %s", str(file_path))
            self.logger.error("Data Type not supported, only .xls(x), .zip, .mat or folder")
            raise TypeError

    def stack_data(self, data, columns):
        """Stacks data that comes in a wide format.

        As xls files have a limit when it comes to table length, some data will
        be stored wide instead (e.g. demand or availability). To properly use
        this data in the model the data will be stacked based on the
        configuration in the options file (options["data"]["stacked"]).

        """
        data = data.stack().reset_index()
        data.columns = columns
        data = data.sort_values("timestep")
        return data.infer_objects()

    def read_xls(self, xls_filepath):
        """Read excel file at specified filepath.

        Parameters
        ----------
        xls_filepath : pathlib.Path
            Filepath to input excel file.

        """
        if xls_filepath.suffix == ".xls":
            engine = "xldr"
        else:
            engine = "openpyxl"
        
        xls = pd.ExcelFile(xls_filepath, engine=engine)
        self.data.data_structure = xls.parse("data_structure", engine=engine)
        self.data.data_attributes.update({d: False for d in self.data.data_structure.data.unique()})
        for data in self.data.data_attributes:
            try:
                raw_data = xls.parse(data, engine=engine, index_col=0).infer_objects()
                self._set_data_attribute(data, raw_data)
            except xlrd.XLRDError as error_msg:
                self.data.missing_data.append(data)
                self.logger.debug(error_msg)
            except KeyError as error_msg:
                self.data.missing_data.append(data)
                self.logger.debug(error_msg)

    def read_csv_folder(self, folder):
        """Read csv files from specified folder.

        Parameters
        ----------
        folder : pathlib.Path
            Path to folder containing input data .csv files.

        """
        self.data.data_structure = pd.read_csv(folder.joinpath('data_structure.csv'))
        self.data.data_attributes.update({d: False for d in self.data.data_structure.data.unique()})
        for data in self.data.data_attributes:
            try:
                csv_file = folder.joinpath(data + ".csv")
                raw_data = pd.read_csv(csv_file, index_col=0).infer_objects()
                self._set_data_attribute(data, raw_data)
            except KeyError as error_msg:
                self.data.missing_data.append(data)
                self.logger.warning(error_msg)
            except FileNotFoundError as error_msg:
                self.data.missing_data.append(data)
                self.logger.debug(error_msg)

    def _set_data_attribute(self, data_name, data): 
        """Sets the read in data as attribute of DataManagement.

        Parameters
        ----------
        data_name : str,
            name of the data that is read and set as attribute e.g. nodes, lines
        data : pd.DataFrame,
            Read in data, as 
        """        
        cols = self.data.data_structure.loc[self.data.data_structure.data == data_name]["attributes"][1:]
        condition_cols = any([col in data.columns for col in cols])
        if not condition_cols and len(cols) > 1 and not data.empty:
            setattr(self.data, data_name, self.stack_data(data, cols))
        else:
            setattr(self.data, data_name, data)
        self.data.data_attributes[data_name] = True

    def read_csv_zipped(self, zip_filepath):
        """Read csv files zipped into archive at specified filepath.

        Parameters
        ----------
        zip_filepath : pathlib.Path
            Filepath to .zip file.

        """
        from zipfile import ZipFile
        with ZipFile(zip_filepath) as zip_archive:
            with zip_archive.open('data_structure.csv', 'r') as csv_file:
                self.data.data_structure = pd.read_csv(csv_file)
            self.data.data_attributes.update({d: False for d in self.data.data_structure.data.unique()})
            for data in self.data.data_attributes:
                try:
                    with zip_archive.open(data + '.csv', 'r') as csv_file:
                        raw_data = pd.read_csv(csv_file, index_col=0).infer_objects()
                        self._set_data_attribute(data, raw_data)
                except KeyError as error_msg:
                    self.data.missing_data.append(data)
                    self.logger.debug(error_msg)
        

    def read_mat_file(self, mat_filepath):
        """Read mat file at specified filepath.

        Reading a .mat file which is intended to be used with matlab
        as a way to exchange data, requires the package *sio.loadmat*.

        Parameters
        ----------
        mat_filepath : pathlib.Path
            Filepath to input matpower .mat casefile file.
        Returns
        -------
        caseinfo : str
            Matpower case information, i.e. case name.
        busname : list like
            List of bus names. These are sometimes provided with the casedata, returns empty if not.
        baseMVA :
            BaseMVA for the system.
        bus_df : pd.DataFrame
            DataFrame containing the bus/nodes data.
        gen_df : pd.DataFrame
            DataFrame containing the generator/plants data.
        branch_df  pd.DataFrame
            DataFrame containing the branch/lines data.
        gencost_df : pd.DataFrame
            DataFrame containing the generation costs.
        """

        self.logger.info("Reading MatPower Casefile")

        MPCOLNAMES = {'bus_keys': 
                        np.array(['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area',
                                  'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']),
                     'gen_keys': 
                        np.array(["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", 
                                  "status", "Pmax", "Pmin"]),
                     'branch_keys':
                        np.array(['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB',
                                  'rateC', 'ratio', 'angle', 'status', 'angmin', 'angmax']),
                     'gencost_keys': np.array(['model', 'startup', 'shutdown', 'n'])}

        case_raw = sio.loadmat(mat_filepath)
        mpc = case_raw['mpcn']
        bus = mpc['bus'][0,0]
        gen = mpc['gen'][0,0]
        baseMVA = mpc['baseMVA'][0,0][0][0]
        branch = mpc['branch'][0,0]
        gencost = mpc['gencost'][0,0]

        try:
            busname = mpc['bus_name'][0,0]
        except:
            busname = np.array([])
        docstring = mpc['docstring'][0,0]
        n = int(gencost[0,3])

        for i in range(n):
            MPCOLNAMES['gencost_keys'] = np.append(MPCOLNAMES['gencost_keys'], 'c{}'.format(n-i-1))
        bus_df = pd.DataFrame(bus, columns=MPCOLNAMES['bus_keys'])
        gen_df = pd.DataFrame(gen, columns=MPCOLNAMES['gen_keys'])
        branch_df = pd.DataFrame(branch, columns=MPCOLNAMES['branch_keys'])
        gencost_df = pd.DataFrame(gencost, columns=MPCOLNAMES['gencost_keys'])
        caseinfo = docstring[0]

        return caseinfo, busname, baseMVA, bus_df, gen_df, branch_df, gencost_df

    def read_m_file(self, m_filepath):
        """Read .m file at specified filepath.

        Reading a .m file with the providing the same return as
        method read_mat_file.

        Returns the necessary data in DataFrames to be processes in
        :meth:`~process_matpower_case`.

        Parameters
        ----------
        m_filepath : pathlib.Path
            Filepath to input matpower .m casefile file.

        Returns
        -------
        caseinfo : str
            Matpower case information, i.e. case name.
        busname : list like
            List of bus names. These are sometimes provided with the casedata, returns empty if not.
        baseMVA :
            BaseMVA for the system.
        bus_df : pd.DataFrame
            DataFrame containing the bus/nodes data.
        gen_df : pd.DataFrame
            DataFrame containing the generator/plants data.
        branch_df  pd.DataFrame
            DataFrame containing the branch/lines data.
        gencost_df : pd.DataFrame
            DataFrame containing the generation costs.
        """
        with open(m_filepath) as mfile:
            raw_text = mfile.read()

        raw_text = raw_text.splitlines()
        is_table = False
        tables, table = [], []
        for line in raw_text:
            if "function mpc" in line:
                caseinfo = line.split()[-1]
            if "mpc.baseMVA" in line:
                baseMVA = float(line.lstrip("mpc.baseMVA = ").rstrip(";"))
            if "%%" in line:
                table = []
                is_table = True
            if "];" in line:
                is_table = False
                tables.append(table)
            if is_table:
                table.append(line)

        df_dict = {}
        for table in tables:
            name = table[0].lstrip("%% ")
            columns = table[1].lstrip("%\t").split()
            data = [row.split(";")[0].split() for row in table[3:]]
            df = pd.DataFrame(data=data, columns=columns, dtype=float)
            df_dict[name] = df

        busname = np.array([])
        bus_df = df_dict["bus data"]
        gen_df = df_dict["generator data"]
        gencost_df = df_dict["generator cost data"]
        gencost_columns = ["model", "startup", "shutdown", "n"] + \
                          ["c" + str(x) for x in range(int(gencost_df.n.values[0]-1), -1, -1)]
        gencost_df.columns = gencost_columns

        branch_df = df_dict["branch data"]

        return caseinfo, busname, baseMVA, bus_df, gen_df, branch_df, gencost_df

    def process_matpower_case(self, casefile, m_type):
        """Process Matpower Case.

        Based on the read data, processing the matpower case data into somthing
        compatible with pomato. This is a fairly manual process, but it should work
        with most .mat/.m as the structure is the same (most of the time).

        This methods populates the DataManagement instance with the corresponding data.
        Additionally it can add coordinates if a correspnonding file is provided.

        Parameters
        ----------
        casefile : pathlib.Path
            Filepath to input matpower .m casefile file.
        m_type : str
            Filetype, .mat or .m, set by parent method.
        """
        if m_type == ".mat":
            caseinfo, busname, baseMVA, bus_df, gen_df, branch_df, gencost_df = self.read_mat_file(casefile)
        elif m_type == ".m":
            caseinfo, busname, baseMVA, bus_df, gen_df, branch_df, gencost_df = self.read_m_file(casefile)
        else:
            self.logger.error(f"Error when reading {m_type} file")

        mpc_buses = {'idx': bus_df['bus_i'],
                     'zone': bus_df['zone'],
                     'Pd': bus_df['Pd'],
                     'Qd': bus_df['Qd'],
                     'baseKV': bus_df['baseKV']
                    }

        # find and set slack bus
        if 3.0 in bus_df['type']:
            slackbus_idx = bus_df['type'][bus_df['type'] == 3.0].index[0]
            slackbus = bus_df['bus_i'][slackbus_idx]
            self.logger.info("Slackbus read as {:.0f}".format(slackbus))
        else:
            slackbus_idx = 0
            slackbus = bus_df['bus_i'][0]
            self.logger.info("Slackbus set to default {}".format(slackbus))

        slack = np.zeros(len(bus_df['bus_i']))
        slack[slackbus_idx] = 1
        mpc_buses['slack'] = slack
        mpc_buses['slack'] = mpc_buses['slack'].astype(bool)

        # add verbose names if available
        if busname.any():
            b_name = []
            for b in busname:
                b_name.append(b[0][0])
            b_name = np.array(b_name)
            mpc_buses['name'] = b_name

        line_idx = ['l{}'.format(i) for i in range(0,len(branch_df.index))]
        mpc_lines = {
                'idx': line_idx,
                'node_i': branch_df['fbus'],
                'node_j': branch_df['tbus'],
                'capacity': branch_df['rateA'],
                'b_other': branch_df['b'],
                'r': branch_df['r'],
                'x': branch_df['x']
                }
        mpc_lines = _mpc_data_pu_to_real(mpc_lines, mpc_buses['baseKV'][0], baseMVA)

        contingency = np.ones(len(mpc_lines['idx']))
        mpc_lines['contingency'] = contingency.astype(bool)

        ng = len(gen_df.index)
        gen_idx = ['g{}'.format(i) for i in range(ng)]
        mpc_generators = {
                    'idx': gen_idx,
                    'g_max': gen_df['Pmax'],
                    # 'g_max_Q': gen_df['Qmax'],
                    'node': gen_df['bus'],
                    # 'apf': gen_df['apf'],
                    'mc_el': gencost_df['c1'][list(range(0,ng))],
                    # 'mc_Q': np.zeros(ng)
                    }

        self.data_source = caseinfo
        self.data.lines = pd.DataFrame(mpc_lines).set_index('idx')
        self.data.nodes = pd.DataFrame(mpc_buses).set_index('idx')
        self.data.plants = pd.DataFrame(mpc_generators).set_index('idx')
        self.data.plants = self.data.plants[self.data.plants.g_max > 0]
        ### Make ieee case ready for the market model
        self.data.nodes["name"] = ["n" + str(int(idx)) for idx in self.data.nodes.index]
        self.data.nodes.rename(columns={"baseKV": "voltage"}, inplace=True)
        self.data.nodes.set_index("name", drop=False, inplace=True)
        self.data.nodes.zone = ["z" + str(int(idx)) for idx in self.data.nodes.zone]

        self.data.lines.node_i = ["n" + str(int(idx)) for idx in self.data.lines.node_i]
        self.data.lines.node_j = ["n" + str(int(idx)) for idx in self.data.lines.node_j]
        self.data.lines["voltage"] = self.data.nodes.loc[self.data.lines.node_i, "voltage"].values
        self.data.lines["technology"] = "ac"
        condition_tranformer = (self.data.nodes.loc[self.data.lines.node_i, "voltage"].values 
                                != self.data.nodes.loc[self.data.lines.node_j, "voltage"].values)
        
        self.data.lines.loc[condition_tranformer, "technology"] = "transformer"

        self.data.zones = pd.DataFrame(index=set(self.data.nodes.zone.values))
        self.data.plants.node = ["n" + str(int(idx)) for idx in self.data.plants.node]
        
        tmp_demand = pd.DataFrame(index=["t0001", "t0002"], data=self.data.nodes.Pd.to_dict())
        tmp_demand = tmp_demand.stack().reset_index()
        tmp_demand.columns = ["timestep", "node", "demand_el"]
        self.data.demand_el = tmp_demand 
        self.data.plants = self.data.plants[["g_max", "mc_el", "node"]]
        
        condition = [i%2==0 for i in range(0, len(self.data.plants))]
        self.data.plants.loc[condition, "plant_type"] = "type 1"
        self.data.plants.loc[~np.array(condition), "plant_type"] = "type 2"
        self.data.plants.loc[condition, "fuel"] = "fuel 1"
        self.data.plants.loc[~np.array(condition), "fuel"] = "fuel 2"

        
        self.data.net_export = tmp_demand.copy()
        self.data.net_export.columns = ["timestep", "node", "net_export"]
        self.data.net_export.loc[:, "net_export"] = 0

        # add coordinates from CSV with X and Y columns for the grid coordinates
        if Path(str(casefile).split(".")[0] + "_coordinates.csv").is_file():
            xy = pd.read_csv(str(casefile).split(".")[0] + "_coordinates.csv", sep=",", index_col=0)
            self.data.nodes[["lat", "lon"]] = xy
        else:
            self.data.nodes.loc[:, "lat"] = 0
            self.data.nodes.loc[:, "lon"] = 0

        self.data.data_structure = _mpc_data_structure()
        # mark read data as true in datamanagement attributes
        self.data.data_attributes["lines"] = True
        self.data.data_attributes["nodes"] = True
        self.data.data_attributes["plants"] = True
        self.data.data_attributes["zones"] = True
        self.data.data_attributes["demand_el"] = True
