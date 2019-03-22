
import logging
import subprocess
import json
import datetime as dt
import numpy as np
import pandas as pd
import tables

from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

from pathlib import Path
import tools

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

class CBCOModule(object):
    """ Class to do all calculations in connection with cbco calculation"""
    def __init__(self, wdir, grid, data, option):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.CBCOModule')
        self.logger.info("Initializing the CBCOModule....")

        self.wdir = wdir
        self.jdir = wdir.joinpath("data_temp/julia_files")
        tools.create_folder_structure(self.wdir, self.logger)

        self.grid = grid
        self.data = data
        self.options = option

        # Attributes
        self.grid_representation = {}
        self.cbco_info = None
        self.cbco_index = None
        self.A, self.b = None, None
        self.A_base, self.b_base = None, None

        self.logger.info("CBCOModule Initialized!")

    def create_grid_representation(self):

        # determining what grid represenation is wanted
        # options are: dispacth (default), ntc, nodal, cbco_nodal,
        # cbco_zonal (tbd), d2cf
        optimization_option = self.options["optimization"]

        # Data Structure of grid_representation
        self.grid_representation["option"] = optimization_option["type"]
        self.grid_representation["mult_slacks"] = self.grid.mult_slack
        self.grid_representation["slack_zones"] = self.grid.slack_zones()
        self.grid_representation["cbco"] = pd.DataFrame()

        if optimization_option["type"] == "ntc":
            self.process_ntc()
        elif optimization_option["type"] == "nodal":
            self.process_nodal()
        elif optimization_option["type"] == "cbco_nodal":
            self.process_cbco_nodal()
        elif optimization_option["type"] == "d2cf":
           self.process_d2cf()
        else:
            self.logger.info("No grid represenation needed for dispatch model")

    def process_ntc(self):
        """process grid information for NTC representation in market model"""
        self.grid_representation["ntc"] = self.data.ntc

    def process_nodal(self):
        """process grid information for nodal N-0 representation in market model"""
        ptdf_df = pd.DataFrame(index=self.grid.lines.index,
                       columns=self.grid.nodes.index,
                       data=np.round(self.grid.ptdf, decimals=4))
        ptdf_df["ram"] = self.grid.lines.maxflow*self.options["grid"]["capacity_multiplier"]

        self.grid_representation["cbco"] = ptdf_df


    def process_cbco_nodal(self):
        """process grid information for cbco nodal representation in market model"""
        grid_option = self.options["grid"]
        self.A, self.b, self.cbco_info = self.create_Ab(grid_option["senstitivity"], grid_option["preprocess"])

        if grid_option["precalc_filename"]:
            try:
                filename = grid_option["precalc_filename"]
                self.logger.info(f"Using cbco indices from pre-calc: {filename}")
                precalc_cbco = pd.read_csv(self.jdir.joinpath(f"cbco_data/{filename}.csv"),
                                           delimiter=',')
                self.cbco_index = list(precalc_cbco.constraints.values)
                self.logger.info("Number of CBCOs from pre-calc: " + str(len(self.cbco_index)))
            except FileNotFoundError:
                self.logger.warning("FileNotFound: No Precalc available")
                self.logger.warning("Running nomal CBCO Algorithm - ConvexHull only")
        else:
            # 3 valid args supported for cbco_option:
            # clarkson, clarkson_base, convex_hull, full_cbco (default)
            if grid_option["cbco_option"] == "full_cbco":
                self.cbco_index = [i for i in range(0, len(self.b))]

            elif grid_option["cbco_option"] == "convex_hull":
                self.cbco_index = self.reduce_Ab_convex_hull()

            elif grid_option["cbco_option"] == "clarkson_base":
                # self.cbco_index = self.reduce_Ab_convex_hull()
                self.A_base, self.b_base = self.base_constraints()
                self.cbco_index = self.clarkson_algorithm()

            elif grid_option["cbco_option"] == "clarkson":
                # self.cbco_index = self.reduce_Ab_convex_hull()
                self.A_base, self.b_base = np.array([]), np.array([])
                self.cbco_index = self.clarkson_algorithm()

            elif grid_option["cbco_option"] == "save":
                self.A_base, self.b_base = np.array([]), np.array([])
                self.write_Ab(self.jdir.joinpath("cbco_data"), "py")
                self.cbco_index = [i for i in range(0, len(self.b))]

            elif grid_option["cbco_option"] == "save_base":
                self.A_base, self.b_base = self.base_constraints()
                self.write_Ab(self.jdir.joinpath("cbco_data"), "py")
                self.cbco_index = [i for i in range(0, len(self.b))]
            else:
                raise

        self.grid_representation["cbco"] = self.return_cbco()
        self.grid_representation["cbco"].ram *= grid_option["capacity_multiplier"]

    def process_d2cf(self):
        """process grid information for d2cf representation in market model"""
        grid_option = self.options["grid"]
        self.A, self.b, self.cbco_info = self.create_Ab(grid_option["senstitivity"])
        self.cbco_index = [i for i in range(0, len(self.b))]
        ptdf = self.return_cbco()

        cbs = self.grid.lines.index[self.grid.lines.cb]
        if grid_option["cbco_option"] == "co as cb":
            cos = ptdf.co[ptdf.cb.isin(cbs)].unique()
            ptdf = ptdf[(ptdf.cb.isin(cbs)|ptdf.cb.isin(cos))&(ptdf.co == "basecase")]
        elif grid_option["cbco_option"] == "cbco":
            ptdf = ptdf[ptdf.cb.isin(cbs)]
        else:
            ptdf = ptdf[ptdf.cb.isin(cbs)&(ptdf.co == "basecase")]

        ptdf.ram *= grid_option["capacity_multiplier"]
        self.grid_representation["cbco"] = ptdf

        if grid_option["reference_flows"]:
            self.grid_representation["reference_flows"] = self.data.reference_flows
            self.grid_representation["reference_flows"].columns = [x + "_basecase" for x in self.data.reference_flows.columns]
        else:
            self.grid_representation["reference_flows"] = pd.DataFrame(index=self.data.demand_el.index)

        for line in ptdf.index:
            if not line in self.grid_representation["reference_flows"].columns:
                self.grid_representation["reference_flows"][line] = 0

    def create_Ab(self, lodf_sensitivity=0, preprocess=True):
        """
        Create all relevant N-1 ptdfs in the for of Ax<b (ptdf x < ram):
        For each line as CB add basecase (N-0)
        and COs based on the senstitivity in LODF (default = 5%)
        return ptdf, corresponding ram and df with the relevant info
        """
        A = [self.grid.ptdf]
        label_lines = list(self.grid.lines.index)
        label_outages = ["basecase" for i in range(0, len(self.grid.lines.index))]

        for idx, line in enumerate(self.grid.lines.index[self.grid.lines.contingency]):
            outages = list(self.grid.lodf_filter(line, lodf_sensitivity))
            label_lines.extend([line for i in range(0, len(outages))])
            label_outages.extend(outages)

        # estimate size of array = nr_elements * bits per element (float64) / (8 * 1e6) MB
        estimate_size = len(label_lines)*(len(self.grid.nodes.index) + 1)*64/(8*1e6)
        self.logger.info(f"Estimated size in RAM for A is: {estimate_size} MB")
        if estimate_size > 5000:
            raise

        for idx, line in enumerate(self.grid.lines.index[self.grid.lines.contingency]):
            outages = list(self.grid.lodf_filter(line, lodf_sensitivity))
            tmp_ptdf = np.vstack([self.grid.create_n_1_ptdf_cbco(line, o) for o in outages])
            A.append(tmp_ptdf)

        A = np.concatenate(A).reshape(len(label_lines), len(list(self.grid.nodes.index)))
        b = self.grid.lines.maxflow[label_lines].values.reshape(len(label_lines), 1)

        # Processing: Rounding, remove duplicates and 0...0 rows
        if preprocess:
            A = np.round(A, decimals=6)
            self.logger.info("Preprocessing Ab...")
            _, idx = np.unique(np.hstack((A,b)), axis=0, return_index=True)
            idx = np.sort(idx)
            A = A[idx]
            b = b[idx]
            label_lines = [label_lines[x] for x in idx]
            label_outages = [label_outages[x] for x in idx]

        df_info = pd.DataFrame(columns=list(self.grid.nodes.index), data=A)
        df_info["cb"] = label_lines
        df_info["co"] = label_outages
        df_info["ram"] = b
        df_info = df_info[["cb", "co", "ram"] + list(list(self.grid.nodes.index))]
        return A, b, df_info

    def write_Ab(self, folder, suffix):

        if isinstance(self.A, np.ndarray) and isinstance(self.b, np.ndarray):
            self.logger.info("Saving A, b...")
            np.savetxt(folder.joinpath(f"A_{suffix}.csv"),
                       np.asarray(self.A), delimiter=",")

            np.savetxt(folder.joinpath(f"b_{suffix}.csv"),
                       np.asarray(self.b), delimiter=",")

        if isinstance(self.cbco_index, list):
            self.logger.info(f"Saving I...")
            np.savetxt(folder.joinpath(f"I_{suffix}.csv"),
                       np.array(self.cbco_index).astype(int),
                       fmt='%i', delimiter=",")

        if isinstance(self.A_base, np.ndarray) and isinstance(self.b_base, np.ndarray):
            self.logger.info(f"Saving A_base, b_base...")
            np.savetxt(folder.joinpath(f"A_base_{suffix}.csv"),
                       np.asarray(self.A_base), delimiter=",")

            np.savetxt(folder.joinpath(f"b_base_{suffix}.csv"),
                       np.asarray(self.b_base), delimiter=",")

        self.logger.info(f"Saved everything to folder \n {str(folder)}")

    def base_constraints(self):
        """ Create Base Constraints for Clarkson algorithm"""
        infeas_upperbound = self.options["optimization"]["infeasibility_bound"]
        base_constraints = []
        base_rhs = []

        for node in self.data.nodes.index:
            condition_storage = (self.data.plants.node==node)& \
                                (self.data.plants.tech.isin(["psp", "reservoir"]))
            condition_el_heat = (self.data.plants.node==node)& \
                                (self.data.plants.tech.isin(["heatpump", "elheat"]))

            max_dc_inj = self.data.dclines.maxflow[(self.data.dclines.node_i == node)| \
                                                   (self.data.dclines.node_j == node)].sum()

            upper = max(self.data.plants.g_max[self.data.plants.node == node].sum() \
                        - self.data.demand_el[node].min() \
                        + max_dc_inj \
                        + infeas_upperbound, 0)

            lower = max(self.data.demand_el[node].max() \
                        + self.data.plants.g_max[condition_storage].sum() \
                        + self.data.plants.g_max[condition_el_heat].sum() \
                        + max_dc_inj \
                        + infeas_upperbound, 0)

            row = np.zeros(len(self.data.nodes.index))
            row[self.data.nodes.index.get_loc(node)] = 1
            base_constraints.extend([row, -row])
            base_rhs.extend([max(upper, lower), max(upper, lower)])

        # base_constraints.append(np.ones(len(self.data.nodes.index)))
        # base_rhs.append(0)

        A_base = np.vstack(base_constraints)
        b_base = np.array(base_rhs).reshape(len(base_rhs), 1)

        return A_base, b_base


    def clarkson_algorithm(self):
        ## save A,b to csv
        self.write_Ab(self.jdir.joinpath("cbco_data"), "py")

        args = ["julia", "--project=project_files/cbco",
                str(self.wdir.joinpath("code_jl/cbco_model.jl")),
                "py", str(self.wdir)]

        t_start = dt.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        # with open(self.wdir.joinpath("logs").joinpath('cbco_reduction.log'), 'w') as log:
        # shell=false needed for mac (and for Unix in general I guess)
        with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
            for line in programm.stdout:
                self.logger.info(line.decode().strip())

        t_end = dt.datetime.now()
        self.logger.info("End-Time: " + t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: " + str((t_end-t_start).total_seconds()) + " sec")

        if programm.returncode == 0:
            df = pd.DataFrame()
            df["files"] = [i for i in self.jdir.joinpath("cbco_data").iterdir()]
            df["time"] = [i.lstat().st_mtime for i in self.jdir.joinpath("cbco_data").iterdir()]
            file = df.files[df.time.idxmax()]
            self.logger.info(f"cbco list save for later use to: \n{file.stem}.csv")
            cbco = pd.read_csv(file, delimiter=',').constraints.values

            return list(cbco)
        else:
            self.logger.critical("Error in Julia code")

    def return_range_of_Ab(self, r):
        """return range of A and b"""
        A, b = self.A[r], self.b[r]
        return A, b.reshape(len(b), 1)

    def reduce_Ab_convex_hull(self):
        """
        Given an system Ax = b, where A is a list of ptdf and b the corresponding ram
        Reduce will find the set of ptdf equations which constrain the solution domain
        (which are based on the N-1 ptdfs)
        """
        try:
            ranges = split_length_in_ranges(5e4, len(self.b))
            self.logger.info(f"Splitting A in {len(ranges)} segments")
            vertices = []
            for r in ranges:
                A, b = self.return_range_of_Ab(r)
                D = A/b
                model = PCA(n_components=8).fit(D)
                D_t = model.transform(D)
                k = ConvexHull(D_t, qhull_options="Qx")
                vertices.extend(k.vertices + r[0])
                self.logger.info("BeepBeepBoopBoop")

            return vertices #np.array(cbco_rows)

        except:
            self.logger.exception('error:reduce_ptdf')

    def return_cbco(self):
        """returns cbco dataframe with A and b"""
        return_df = self.cbco_info.iloc[self.cbco_index].copy()
        return_df.loc[:, "index"] = return_df.cb + "_" + return_df.co
        return_df = return_df.set_index("index")
        return return_df

