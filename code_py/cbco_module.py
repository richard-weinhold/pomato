
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
#import gams_cbco_reduction as cbco_reduction
# from pomato.resources import JULIA_PATH

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
    def __init__(self, wdir, grid_object):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.CBCOModule')
        self.logger.info("Initializing the CBCOModule....")

        self.wdir = wdir
        self.jdir = wdir.joinpath("data_temp/julia_files")
        self.nodes = grid_object.nodes
        self.lines = grid_object.lines
        self.grid = grid_object
        ### Prepare Matrix A and vector b
        ## potentially as file
        # self.Ab_file_path = self.create_Ab()
        self.A, self.b, self.cbco_info = self.create_Ab()
        tools.create_folder_structure(self.wdir, self.logger)
        self.cbco_index = []

        self.logger.info("CBCOModule Initialized!")

    def create_Ab(self, lodf_sensitivity=5e-2, preprocess=True):
    # def create_Ab(self, lodf_sensitivity=0, preprocess=False):
        """
        Create all relevant N-1 ptdfs in the for of Ax<b (ptdf x < ram):
        For each line as CB add basecase (N-0) 
        and COs based on the senstitivity in LODF (default = 5%)
        return ptdf, corresponding ram and df with the relevant info
        """
        A = [self.grid.ptdf]
        label_lines = list(self.grid.lines.index)
        label_outages = ["basecase" for i in range(0, len(self.grid.lines.index))]

        for idx, line in enumerate( self.lines.index[self.lines.contingency]):
            outages = list(self.grid.lodf_filter(line, lodf_sensitivity))
            label_lines.extend([line for i in range(0, len(outages))])
            label_outages.extend(outages)

        # estimate size of array = nr_elements * bits per element (float32) / (8 * 1e6) MB
        estimate_size = len(label_lines)*len(self.nodes.index)*32/(8*1e6)
        self.logger.info(f"Estimated size in RAM for A is: {estimate_size} MB")
        if estimate_size > 3000:
            raise

        for idx, line in enumerate( self.lines.index[self.lines.contingency]):
            outages = list(self.grid.lodf_filter(line, lodf_sensitivity))
            tmp_ptdf = np.vstack([self.grid.create_n_1_ptdf_cbco(line,o) for o in outages])
            A.append(tmp_ptdf)

        A = np.concatenate(A).reshape(len(label_lines), len(list(self.nodes.index)))
        b = self.lines.maxflow[label_lines].values.reshape(len(label_lines), 1)

        # Processing: Rounding, remove duplicates and 0...0 rows
        if preprocess:
            self.logger.info("Preprocessing Ab...")
            A = np.round(A, decimals=6)
            _, idx = np.unique(np.hstack((A,b)), axis=0, return_index=True)
            idx = np.sort(idx)
            A = A[idx]
            b = b[idx]
            label_lines = [label_lines[x] for x in idx]
            label_outages = [label_outages[x] for x in idx]
        
        df_info = pd.DataFrame(columns=list(self.nodes.index), data=A)
        df_info["cb"] = label_lines
        df_info["co"] = label_outages
        df_info["ram"] = b
        df_info = df_info[["cb", "co", "ram"] + list(list(self.nodes.index))]
        
        return A, b, df_info

    def main(self, use_precalc=None, only_convex_hull=True):

        self.logger.info("CBCO Main Function...")
        if use_precalc:
            try:
                self.logger.info(f"Using cbco indices from pre-calc {use_precalc}")
                precalc_cbco = pd.read_csv(self.jdir.joinpath(f"cbco_data/{use_precalc}.csv"), delimiter=',')
                self.cbco_index = list(precalc_cbco.constraints.values)
                self.logger.info("Number of CBCOs from pre-calc: " + str(len(self.cbco_index)))
            except FileNotFoundError:
                self.logger.warning("FileNotFound: No Precalc available")
                self.logger.warning("Running nomal CBCO Algorithm - ConvexHull only")
                use_precalc = False
                only_convex_hull=True
        if not use_precalc:
            self.cbco_algorithm(only_convex_hull)

    def add_to_cbco_index(self, add_cbco):
        """adds the indecies of the manulally added cbco to cbco_index"""
        # make sure its np.array
        if not isinstance(add_cbco, list):
            add_cbco = list(add_cbco)
        self.cbco_index = list(set(self.cbco_index + add_cbco))

    def julia_cbco_interface(self, A, b, cbco_index):
        ## save A,b to csv
        ## save cbco_index for starting set A', and b' as csv
        np.savetxt(self.jdir.joinpath("cbco_data").joinpath("A_py.csv"), np.asarray(A), delimiter=",")
        np.savetxt(self.jdir.joinpath("cbco_data").joinpath("b_py.csv"), np.asarray(b), delimiter=",")
        ## fmt='%i' is needed to save as integer
        np.savetxt(self.jdir.joinpath("cbco_data").joinpath("I_py.csv"), np.array(cbco_index).astype(int),
                   fmt='%i', delimiter=",")

        args = ["julia", "--project=project_files/cbco", str(self.wdir.joinpath("code_jl/cbco_model.jl")), "py"]
        t_start = dt.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        # with open(self.wdir.joinpath("logs").joinpath('cbco_reduction.log'), 'w') as log:
            # shell=false needed for mac (and for Unix in general I guess)
        with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
            for line in programm.stdout:
                # log.write(line.decode())
                self.logger.info(line.decode().split(":")[1].strip())

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

    def cbco_algorithm(self, only_convex_hull=True):
        """
        Creating Ax = b Based on the list of N-1 ptdfs and ram
        Reduce it by:
        1) using the convex hull method to get a subset A' from A, where A'x<=b'
           is a non rendundant system of inequalities
        2) Using the julia algorithm to check all linear inequalities of A
           against A' and add those which are non-redundant but missed bc of
           the low dim of the convexhull problem
        """
        try:
            self.add_to_cbco_index(self.reduce_Ab_convex_hull())
            self.logger.info("Number of CBCOs from ConvexHull Method: " + str(len(self.cbco_index)))

            if not only_convex_hull:
                self.logger.info("Running Julia CBCO-Algorithm...")
                cbco_index = self.julia_cbco_interface(self.A, self.b, self.cbco_index)
                self.add_to_cbco_index(cbco_index)
                self.logger.info("Number of CBCOs after Julia CBCO-Algorithm: " + str(len(self.cbco_index)))

        except:
            self.logger.exception('e:cbco')

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
        return_df = self.cbco_info.iloc[self.cbco_index]
        return_df["index"] = return_df.cb + "_" + return_df.co
        return_df = return_df.set_index("index")

        return return_df


