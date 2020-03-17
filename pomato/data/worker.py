import sys
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
import logging
import pyproj


def _mpc_data_pu_to_real(lines,  base_kv, base_mva):
    """Convert pu to actual units for the mpc case."""
    v_base = base_kv * 1e3
    s_base = base_mva * 1e6
    z_base = np.power(v_base,2)/s_base
    lines['r'] = np.multiply(lines['r'], z_base)
    lines['x'] = np.multiply(lines['x'], z_base)
    lines['b_other'] = np.divide(lines['b_other'], z_base)
    lines['b'] = np.divide(1, lines['x'])
    return lines

class DataWorker(object):
    """Data Woker Module reads data from disk.

    This module's pupose is to hide all the file system specific functions
    and allow for rather seemless data import.
    An instance of the DataManagement class acts as the carrier for the
    read data and is used as an attribute of DataWorker.

    The based on *file_path* the module tries to import xlsx or matpower (.m
    or .mat) cases. Reading the excel file is done by going through the
    pre-defined data tables in *data.data_attributes* and attaching them
    to the DataManagement instance for processisng and validation. Matpower
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
        self.logger = logging.getLogger('Log.MarketModel.DataManagement.DataWorker')
        self.data = data

        if ".xls" in str(file_path):
            self.logger.info("Loading data from Excel file")
            self.read_xls(file_path)

        elif ".mat" in str(file_path):
            self.logger.info("Loading data from matpower .mat case-file")
            self.process_matpower_case(file_path, ".mat")
        elif ".m" in str(file_path):

            self.logger.info("Loading data from matpower .m case-file")
            self.process_matpower_case(file_path, ".m")
        else:
            self.logger.warning("Data Type not supported, only .xls(x) or .mat")

    def read_xls(self, xls_filepath):
        """Read excel file at speciefied filepath.

        Parameters
        ----------
        xls_filepath : pathlib.Path
            Filepath to input excel file.

        """
        xls = pd.ExcelFile(xls_filepath)
        for data in self.data.data_attributes:
            try:
                setattr(self.data, data, xls.parse(data, index_col=0).infer_objects())
                self.data.data_attributes[data] = True
            except:
                self.logger.warning(f"{data} not in excel file")

    def read_mat_file(self, mat_filepath):
        """Read mat file at speciefied filepath.

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

        MPCOLNAMES = {'bus_keys': np.array(['bus_i', 'type', 'Pd',
                                            'Qd', 'Gs', 'Bs', 'area',
                                            'Vm', 'Va', 'baseKV',
                                            'zone', 'Vmax', 'Vmin']),

                      'gen_keys': np.array(['bus', 'Pg', 'Qg', 'Qmax',
                                            'Qmin', 'Vg', 'mBase',
                                            'status', 'Pmax', 'Pmin',
                                            'Pc1', 'Pc2', 'Qc1min',
                                            'Qc1max', 'Qc2min', 'Qc2max',
                                            'ramp_agc', 'ramp_10',
                                            'ramp_30', 'ramp_q', 'apf']),

                        'branch_keys': np.array(['fbus', 'tbus', 'r', 'x',
                                                 'b', 'rateA', 'rateB',
                                                 'rateC', 'ratio', 'angle',
                                                 'status', 'angmin', 'angmax']),

                        'gencost_keys': np.array(['model', 'startup',
                                                  'shutdown', 'n'])}

        case_raw = sio.loadmat(mat_filepath)
        mpc = case_raw['mpc']
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
        """Read .m file at speciefied filepath.

        Reading a .m file with the providing the same return as
        method read_mat_file.

        Returns the nessesary data in DataFrames to be processes in
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
        compadible with pomato. This is a fairly manual process, but it should work
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
        mpc_buses['net_injection'] = np.zeros(len(mpc_buses['idx']))

        # add verbose names if available
        if busname.any():
            b_name = []
            for b in busname:
                b_name.append(b[0][0])
            b_name = np.array(b_name)
            mpc_buses['name'] = b_name

        lineidx = ['l{}'.format(i) for i in range(0,len(branch_df.index))]
        mpc_lines = {
                'idx': lineidx,
                'node_i': branch_df['fbus'],
                'node_j': branch_df['tbus'],
                'maxflow': branch_df['rateA'],
                'b_other': branch_df['b'],
                'r': branch_df['r'],
                'x': branch_df['x']
                }
        mpc_lines = _mpc_data_pu_to_real(mpc_lines, mpc_buses['baseKV'][0], baseMVA)

        contingency = np.ones(len(mpc_lines['idx']))
        mpc_lines['contingency'] = contingency.astype(bool)

        ng = len(gen_df.index)
        genidx = ['g{}'.format(i) for i in range(ng)]
        mpc_generators = {
                    'idx': genidx,
                    'g_max': gen_df['Pmax'],
                    # 'g_max_Q': gen_df['Qmax'],
                    'node': gen_df['bus'],
                    # 'apf': gen_df['apf'],
                    'mc_el': gencost_df['c1'][list(range(0,ng))],
                    # 'mc_Q': np.zeros(ng)
                    }

        # if len(gencost_df.index) == 2*ng:
        #     mpc_generators['mc_Q'] = gencost_df['c1'][list(range(ng,2*ng))].tolist

        self.data.lines = pd.DataFrame(mpc_lines).set_index('idx')
        self.data.nodes = pd.DataFrame(mpc_buses).set_index('idx')
        self.data.plants = pd.DataFrame(mpc_generators).set_index('idx')
        self.data.plants = self.data.plants[self.data.plants.g_max > 0]
        ### Make ieee case ready for the market model
        self.data.nodes["name"] = ["n" + str(int(idx)) for idx in self.data.nodes.index]
        self.data.nodes.set_index("name", drop=False, inplace=True)
        self.data.nodes.zone = ["z" + str(int(idx)) for idx in self.data.nodes.zone]

        self.data.lines.node_i = ["n" + str(int(idx)) for idx in self.data.lines.node_i]
        self.data.lines.node_j = ["n" + str(int(idx)) for idx in self.data.lines.node_j]
        self.data.zones = pd.DataFrame(index=set(self.data.nodes.zone.values))
        self.data.plants.node = ["n" + str(int(idx)) for idx in self.data.plants.node]
        self.data.demand_el = pd.DataFrame(index=["t0001", "t0002"], data=self.data.nodes.Pd.to_dict())
        self.data.demand_el = self.data.demand_el.stack().reset_index()
        self.data.demand_el.columns = ["timestep", "node", "demand_el"]
        self.data.plants = self.data.plants[["g_max", "mc_el", "node"]]
        self.data.plants.loc[:, "plant_type"] = "default"

        self.data.net_export = self.data.demand_el.copy()
        self.data.net_export.columns = ["timestep", "node", "net_export"]
        self.data.net_export.loc[:, "net_export"] = 0

        # add coordinates from CSV with X and Y columns for the grid coordinates
        if Path(str(casefile).split(".")[0] + "_coordinates.csv").is_file():
            xy = pd.read_csv(str(casefile).split(".")[0] + "_coordinates.csv",
                             sep=";", index_col=0)
            lat0, lon0 = 25, 82.5
            projection = pyproj.Proj(f"+proj=stere +lat_0={str(lat0)} +lon_0={str(lon0)} \
                                         +k=1 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs")

            coord = pd.DataFrame(columns=["lon", "lat"], index=self.data.nodes.index,
                                data = [projection(x*4000,y*4000, inverse=True) for x,y in zip(xy.X, xy.Y)])
            coord = coord[["lat", "lon"]]
            self.data.nodes[["lat", "lon"]] = coord
        else:
            self.data.nodes.loc[:, "lat"] = 0
            self.data.nodes.loc[:, "lon"] = 0

        # mark read data as true in datamanagement attributes
        self.data.data_attributes["lines"] = True
        self.data.data_attributes["nodes"] = True
        self.data.data_attributes["plants"] = True
        self.data.data_attributes["zones"] = True
        self.data.data_attributes["demand_el"] = True

