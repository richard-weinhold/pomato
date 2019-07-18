"""
CBCO Module
Creates Grid Representation for the Merket Model

Options: Full, Reduced, None etc


"""
import logging
import subprocess
import datetime as dt
import itertools
import numpy as np
import pandas as pd


from scipy import spatial
from sklearn.decomposition import PCA


import tools

def split_length_in_ranges(step_size, length):
    """
    [1,..,R] in [1,...,r1], [r1+1,...,r2], etc

    """
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

class CBCOModule():
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
        """
        Creates Grid Representation:
        based on set option:
        options are: dispacth (default), ntc, nodal, zonal, cbco_nodal, cbco_zonal, d2cf
        """
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
        elif optimization_option["type"] == "zonal":
            self.process_zonal()
        elif optimization_option["type"] == "cbco_nodal":
            self.process_cbco_nodal()
        elif optimization_option["type"] == "cbco_zonal":
            self.process_cbco_zonal()
        elif optimization_option["type"] == "d2cf":
            self.process_d2cf()
        else:
            self.logger.info("No grid represenation needed for dispatch model")

    def process_ntc(self):
        """process grid information for NTC representation in market model"""

        tmp = []
        for from_zone, to_zone in itertools.combinations(set(self.data.nodes.zone), 2):
            lines = []

            from_nodes = self.data.nodes.index[self.data.nodes.zone == from_zone]
            to_nodes = self.data.nodes.index[self.data.nodes.zone == to_zone]

            condition_i_from = self.data.lines.node_i.isin(from_nodes)
            condition_j_to = self.data.lines.node_j.isin(to_nodes)

            condition_i_to = self.data.lines.node_i.isin(to_nodes)
            condition_j_from = self.data.lines.node_j.isin(from_nodes)

            lines += list(self.data.lines.index[condition_i_from&condition_j_to])
            lines += list(self.data.lines.index[condition_i_to&condition_j_from])

            dclines = []
            condition_i_from = self.data.dclines.node_i.isin(from_nodes)
            condition_j_to = self.data.dclines.node_j.isin(to_nodes)

            condition_i_to = self.data.dclines.node_i.isin(to_nodes)
            condition_j_from = self.data.dclines.node_j.isin(from_nodes)

            dclines += list(self.data.dclines.index[condition_i_from&condition_j_to])
            dclines += list(self.data.dclines.index[condition_i_to&condition_j_from])


            if lines or dclines:
                tmp.append([from_zone, to_zone, 1e5])
                tmp.append([to_zone, from_zone, 1e5])
            else:
                tmp.append([from_zone, to_zone, 0])
                tmp.append([to_zone, from_zone, 0])

        self.grid_representation["ntc"] = pd.DataFrame(tmp, columns=["zone_i", "zone_j", "ntc"])


    def process_nodal(self):
        """process grid information for nodal N-0 representation in market model"""
        ptdf_df = pd.DataFrame(index=self.grid.lines.index,
                               columns=self.grid.nodes.index,
                               data=np.round(self.grid.ptdf, decimals=4))

        ptdf_df["ram"] = self.grid.lines.maxflow*self.options["grid"]["capacity_multiplier"]

        self.grid_representation["cbco"] = ptdf_df

    def process_zonal(self):
        """process grid information for zonal N-0 representation in market model"""
        gsk = self.options["grid"]["gsk"]
        grid_option = self.options["grid"]

        if grid_option["cbco_option"] == "clarkson":            
            A = self.grid.ptdf
            label_lines = list(self.grid.lines.index)
            label_outages = ["basecase" for i in range(0, len(self.grid.lines.index))]
            A = np.dot(A, self.create_gsk(gsk))
            b = self.grid.lines.maxflow[label_lines].values.reshape(len(label_lines), 1)
            columns = list(self.data.zones.index)
            df_info = pd.DataFrame(columns=columns, data=A)
            df_info["cb"] = label_lines
            df_info["co"] = label_outages
            df_info["ram"] = b
            df_info = df_info[["cb", "co", "ram"] + list(columns)]
            self.A, self.b, self.cbco_info = A, b, df_info
            self.A_base, self.b_base = np.array([]), np.array([])
            self.cbco_index = self.clarkson_algorithm()
            self.grid_representation["cbco"] = self.return_cbco()


        else:
            ptdf = np.dot(self.grid.ptdf, self.create_gsk(gsk))
            ptdf_df = pd.DataFrame(index=self.grid.lines.index,
                                   columns=self.data.zones.index,
                                   data=np.round(ptdf, decimals=4))

            ptdf_df["ram"] = self.grid.lines.maxflow*self.options["grid"]["capacity_multiplier"]

            self.grid_representation["cbco"] = ptdf_df


    def process_cbco_zonal(self):
        """Creating a Reduced Zonal N-1 Representation, with Convex Hull Algorithm"""
        grid_option = self.options["grid"]

        self.A, self.b, self.cbco_info = self.create_Ab(grid_option["senstitivity"],
                                                        preprocess=False,
                                                        gsk=grid_option["gsk"])

        self.cbco_index = self.reduce_Ab_convex_hull()

        self.grid_representation["cbco"] = self.return_cbco()
        self.grid_representation["cbco"].ram *= grid_option["capacity_multiplier"]
        self.process_ntc()


    def process_cbco_nodal(self):
        """process grid information for nodal N-1 representation in market model"""
        grid_option = self.options["grid"]
        self.A, self.b, self.cbco_info = self.create_Ab(grid_option["senstitivity"],
                                                        grid_option["preprocess"])

        if grid_option["precalc_filename"]:
            try:
                filename = grid_option["precalc_filename"]
                self.logger.info("Using cbco indices from pre-calc: %s", filename)
                precalc_cbco = pd.read_csv(self.jdir.joinpath(f"cbco_data/{filename}.csv"),
                                           delimiter=',')
                self.cbco_index = list(precalc_cbco.constraints.values)
                self.logger.info("Number of CBCOs from pre-calc: %s", str(len(self.cbco_index)))
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
                self.logger.warning("No valid cbco_option set!")

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
            columns = [line + "_basecase" for line in self.data.reference_flows.columns]
            self.grid_representation["reference_flows"].columns = columns

        else:
            df = pd.DataFrame(index=self.data.demand_el.index)
            self.grid_representation["reference_flows"] = df

        for line in ptdf.index:
            if not line in self.grid_representation["reference_flows"].columns:
                self.grid_representation["reference_flows"][line] = 0

    def create_Ab(self, lodf_sensitivity=0, preprocess=True, gsk=None):
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

        # estimate size of array = nr_elements * bytes per element
        # (float64 + sep = 8 + 1) / (1024**2) MB
        estimate_size = len(label_lines)*len(self.grid.nodes.index)*(8 + 1)/(1024*1024)

        self.logger.info("Estimated size in RAM for A is: %d MB", estimate_size)
        if estimate_size > 5000:
            raise Exception('Matrix A too large!')

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
            _, idx = np.unique(np.hstack((A, b)), axis=0, return_index=True)
            idx = np.sort(idx)
            A = A[idx]
            b = b[idx]
            label_lines = [label_lines[x] for x in idx]
            label_outages = [label_outages[x] for x in idx]

        if gsk:
            A = np.dot(A, self.create_gsk(gsk))
            columns = list(self.data.zones.index)
        else:
            columns = list(self.grid.nodes.index)

        df_info = pd.DataFrame(columns=columns, data=A)
        df_info["cb"] = label_lines
        df_info["co"] = label_outages
        df_info["ram"] = b
        df_info = df_info[["cb", "co", "ram"] + list(columns)]
        return A, b, df_info

    def write_Ab(self, folder, suffix):

        if isinstance(self.A, np.ndarray) and isinstance(self.b, np.ndarray):
            self.logger.info("Saving A, b...")
            np.savetxt(folder.joinpath(f"A_{suffix}.csv"),
                       np.asarray(self.A), delimiter=",")

            np.savetxt(folder.joinpath(f"b_{suffix}.csv"),
                       np.asarray(self.b), delimiter=",")

        if isinstance(self.cbco_index, list):
            self.logger.info("Saving I...")
            np.savetxt(folder.joinpath(f"I_{suffix}.csv"),
                       np.array(self.cbco_index).astype(int),
                       fmt='%i', delimiter=",")

        if isinstance(self.A_base, np.ndarray) and isinstance(self.b_base, np.ndarray):
            self.logger.info("Saving A_base, b_base...")
            np.savetxt(folder.joinpath(f"A_base_{suffix}.csv"),
                       np.asarray(self.A_base), delimiter=",")

            np.savetxt(folder.joinpath(f"b_base_{suffix}.csv"),
                       np.asarray(self.b_base), delimiter=",")

        self.logger.info("Saved everything to folder: \n %s", str(folder))

    def base_constraints(self):
        """ Create Base Constraints for Clarkson algorithm"""
        infeas_upperbound = self.options["optimization"]["infeasibility_bound"]
        base_constraints = []
        base_rhs = []

        for node in self.data.nodes.index:
            condition_storage = (self.data.plants.node == node) & \
                                (self.data.plants.tech.isin(["psp", "reservoir"]))
            condition_el_heat = (self.data.plants.node == node) & \
                                (self.data.plants.tech.isin(["heatpump", "elheat"]))

            max_dc_inj = self.data.dclines.maxflow[(self.data.dclines.node_i == node) | \
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

        args = ["julia", "--project=project_files/pomato",
                str(self.wdir.joinpath("code_jl/cbco_model.jl")),
                "py", str(self.wdir)]

        t_start = dt.datetime.now()
        self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))
        with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
            for line in programm.stdout:
                self.logger.info(line.decode().strip())

        t_end = dt.datetime.now()
        self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")

        if programm.returncode == 0:
            df = pd.DataFrame()
            df["files"] = [i for i in self.jdir.joinpath("cbco_data").iterdir()]
            df["time"] = [i.lstat().st_mtime for i in self.jdir.joinpath("cbco_data").iterdir()]
            file = df.files[df.time.idxmax()]
            self.logger.info("cbco list save for later use to: \n%s", file.stem + ".csv")
            cbco = pd.read_csv(file, delimiter=',').constraints.values
            return_value = list(cbco)
        else:
            self.logger.critical("Error in Julia code")
            return_value = None

        return return_value

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

        ranges = split_length_in_ranges(5e4, len(self.b))
        self.logger.info("Splitting A in %d segments", len(ranges))
        vertices = []
        for r in ranges:
            A, b = self.return_range_of_Ab(r)
            D = A[:, A.any(axis=0)]/b
            # D = A/b

            if np.size(D, 1) > 8:
                model = PCA(n_components=8).fit(D)
                D = model.transform(D)

            k = spatial.qhull.ConvexHull(D, qhull_options="QJ")
            # k = ConvexHull(D, qhull_options="Qx")
            vertices.extend(k.vertices + r[0])
            self.logger.info("BeepBeepBoopBoop")
        return vertices #np.array(cbco_rows)

    def return_cbco(self):
        """returns cbco dataframe with A and b"""
        return_df = self.cbco_info.iloc[self.cbco_index].copy()
        return_df.loc[:, "index"] = return_df.cb + "_" + return_df.co
        return_df = return_df.set_index("index")
        return return_df

    def create_gsk(self, option="flat"):
        """returns GSK, either flat or gmax"""

        gsk = pd.DataFrame(index=self.data.nodes.index)
        conv_fuel = ['uran', 'lignite', 'hard coal', 'gas', 'oil', 'hydro', 'waste']
        condition = self.data.plants.fuel.isin(conv_fuel)&(self.data.plants.tech != "psp")
        gmax_per_node = self.data.plants.loc[condition, ["g_max", "node"]].groupby("node").sum()

        for zone in self.data.zones.index:
            nodes_in_zone = self.data.nodes.index[self.data.nodes.zone == zone]
            gsk[zone] = 0
            gmax_in_zone = gmax_per_node[gmax_per_node.index.isin(nodes_in_zone)]
            if option == "gmax":
                if not gmax_in_zone.empty:
                    gsk_value = gmax_in_zone.g_max/gmax_in_zone.values.sum()
                    gsk.loc[gsk.index.isin(gmax_in_zone.index), zone] = gsk_value

            elif option == "flat":
                # if not gmax_in_zone.empty:
                gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)

        return gsk.values
