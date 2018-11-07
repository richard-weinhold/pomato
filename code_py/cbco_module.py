
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
import shutil
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
        self.jdir = wdir.joinpath("julia-files")
        self.nodes = grid_object.nodes
        self.lines = grid_object.lines

        self.Ab_file_path = self.create_Ab_file(grid_object)
        # self.Ab_file_path = "Ab_expandeble_compressed.hdf5"
        # self.A = np.array(A)
        # self.b = np.array(b).reshape(len(b), 1)


        self.create_folders(wdir)
        self.cbco_index = np.array([], dtype=np.int32)

        self.logger.info("CBCOModule Initialized!")

    # def __getstate__(self):
    #     """
    #     Method to remove logger attribute from __dict__
    #     needed when pickeled
    #     """
    #     d = dict(self.__dict__)
    #     del d["logger"]
    #     return d

    # def __setstate__(self, d):
    #     """
    #     Method updates self with modified __dict__ without logger
    #     needed when pickeled
    #     """
    #     self.__dict__.update(d) # I *think* this is a safe way to do it


    def create_Ab_file(self, grid_object):
        """ Storing Matrix A and Vector b to disk to save memory"""
        # set up for Ab file
        # dtype = np.array(grid_object.ptdf, dtype=np.float16).dtype
        dtype = np.dtype("Float16")
        shape = grid_object.ptdf.shape[-1]
        expectedrows = len(self.lines)*(len(self.lines)+1)*2

        ## Init PyTables
        hdf5_path = self.wdir.joinpath("temp_data/Ab_expandeble_compressed.hdf5")
        hdf5_file = tables.open_file(str(hdf5_path), mode='w')
        filters = tables.Filters(complevel=1, complib='zlib')
        A_storage = hdf5_file.create_earray(hdf5_file.root, 'A',
                                              tables.Atom.from_dtype(dtype),
                                              shape=(0, shape),
                                              filters=filters,
                                              expectedrows=expectedrows)

        b_storage = hdf5_file.create_earray(hdf5_file.root, 'b',
                                                  tables.Atom.from_dtype(dtype),
                                                  shape=(0,),
                                                  filters=filters,
                                                  expectedrows=expectedrows)

        ## N-0 Matrix
        ptdf = grid_object.ptdf
        ram_array = grid_object.update_ram(grid_object.ptdf, option="array")
        A_storage.append(np.vstack([grid_object.ptdf, -grid_object.ptdf]))
        b_storage.append(np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0))

        for idx, line in enumerate(grid_object.lines.index):
            ptdf = grid_object.create_n_1_ptdf_outage(idx)
            # ram_array = grid_object.update_ram(ptdf, option="array")
            A_storage.append(np.vstack([ptdf, -ptdf]))
            b_storage.append(np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0))

        hdf5_file.close()
        ## To be clear
        Ab_file_path = str(hdf5_path)
        return Ab_file_path

    def retrieve_Ab(self, selection):
        """getting the rows/values from A,b, input either array/list with indices, range or index"""
        Ab_file = tables.open_file(self.Ab_file_path, mode='r')
        A = Ab_file.root.A[selection, :]
        b = Ab_file.root.b[selection]
        Ab_file.close()

        if isinstance(b, np.ndarray):
            b = b.reshape(len(b), 1)
        return A, b

    def main(self, use_precalc=None, only_convex_hull=True):

        self.logger.info("CBCO Main Function...")
        if use_precalc:
            try:
                self.logger.info(f"Using cbco indices from pre-calc {use_precalc}")
                precalc_cbco = np.genfromtxt(self.jdir.joinpath(f"cbco_data/{use_precalc}.csv"), delimiter=',')
                self.cbco_index = np.array(precalc_cbco, dtype=np.int32)
                self.logger.info("Number of CBCOs from pre-calc: " + str(len(self.cbco_index)))
            except FileNotFoundError:
                self.logger.warning("FileNotFound: No Precalc available")
                self.logger.warning("Running nomal CBCO Algorithm - ConvexHull only")
                use_precalc = False
                only_convex_hull=True
        if not use_precalc:
            self.cbco_algorithm(only_convex_hull)

    def return_cbco(self):
        """ return the cbco infor for the market market"""
        self.logger.info("Creating the CBCO information to be used in the MarketModel")
        info = {}
        for n in self.cbco_index:
            info[n] = self.create_cbcomodule_return(n)
        cbco = {}
        for i in self.cbco_index: # range(0, len(b)): #
            A, b = self.retrieve_Ab(i)
            cbco['cbco'+ "{0:0>4}".format(i+1)] = {'ptdf': list(np.array(A, dtype=np.float)),'ram': int(b)}

        self.logger.info(f"Returning a total of {len(self.cbco_index)} CBCOs")
        ## save cbco_index to wdir/julia/cbco_data
        np.savetxt(self.wdir.joinpath("julia-files/cbco_data/cbco_" + dt.datetime.now().strftime("%d%m_%H%M") + ".csv"),
                   self.cbco_index, delimiter=",")
        self.logger.info("CBCOs stores as indices in cbco_data/cbco.csv for re-use")
        return(info, cbco)

    def add_to_cbco_index(self, add_cbco):
        """adds the indecies of the manulally added cbco to cbco_index"""
        # make sure its np.array
        if not isinstance(add_cbco, np.ndarray):
            add_cbco = np.array(add_cbco, dtype=np.int32)
        self.cbco_index = np.union1d(self.cbco_index, add_cbco)

    def create_folders(self, wdir):
        """ create folders for julia cbco_analysis"""
        if not wdir.joinpath("julia-files").is_dir():
            wdir.joinpath("julia-files").mkdir()
        if not wdir.joinpath("julia-files/cbco_data").is_dir():
            wdir.joinpath("julia-files/cbco_data").mkdir()

    def julia_cbco_interface(self, A, b, cbco_index):
        ## save A,b to csv
        ## save cbco_index for starting set A', and b' as csv
        np.savetxt(self.jdir.joinpath("cbco_data").joinpath("A.csv"), np.asarray(A), delimiter=",")
        np.savetxt(self.jdir.joinpath("cbco_data").joinpath("b.csv"), np.asarray(b), delimiter=",")
        ## fmt='%i' is needed to save as integer
        np.savetxt(self.jdir.joinpath("cbco_data").joinpath("cbco_index.csv"), cbco_index.astype(int),
                   fmt='%i', delimiter=",")

        args = ["julia", str(self.jdir.joinpath("cbco.jl")), str(self.jdir)]
        t_start = dt.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        with open(self.wdir.joinpath("logs").joinpath('cbco_reduction.log'), 'w') as log:
            # shell=false needed for mac (and for Unix in general I guess)
            with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
                for line in programm.stdout:
                    log.write(line.decode())
                    self.logger.info(line.decode().strip())

        if programm.returncode == 0:
            tmp_cbco = np.genfromtxt(self.jdir.joinpath("/cbco_data/cbco.csv"), delimiter=',')
#            tmp_cbco = self.add_negative_constraints(tmp_cbco)
            return np.array(tmp_cbco, dtype=np.int32)
        else:
            self.logger.critical("Error in Julia code")

    def cbco_algorithm(self, only_convex_hull):
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

                A, b = self.return_positive_cbcos()
                cbco_index = self.cbco_index_full_to_positive_Ab(self.cbco_index)
                self.logger.info("Number of positive CBCOs for Julia CBCO-Algorithm: " + str(len(cbco_index)))

                cbco_index = self.julia_cbco_interface(A, b, cbco_index)
                self.cbco_index = self.cbco_index_positive_to_full_Ab(cbco_index)
                self.logger.info("Number of CBCOs after Julia CBCO-Algorithm: " + str(len(self.cbco_index)))

        except:
            self.logger.exception('e:cbco')

    def reduce_Ab_convex_hull(self):
        """
        Given an system Ax = b, where A is a list of ptdf and b the corresponding ram
        Reduce will find the set of ptdf equations which constrain the solution domain
        (which are based on the N-1 ptdfs)
        """
        try:
            ranges = split_length_in_ranges(5e4, len(self.lines)*(len(self.lines)+1)*2)
            self.logger.info(f"Splitting A in {len(ranges)} segments")
            vertices = np.array([], dtype=np.int32)
            for r in ranges:
                A, b = self.retrieve_Ab(r)
                D = A/b

#                %%timeit
#                model = PCA(n_components=8, svd_solver="arpack", random_state=1).fit(D)
#                D_t = model.transform(D)
#                k = ConvexHull(D_t, qhull_options="QJ")
#                tmp_vertices = list(k.vertices)
#                print(len(k.vertices))

                model = PCA(n_components=8).fit(D)
                D_t = model.transform(D)
                k = ConvexHull(D_t, qhull_options="QJ")
#                tmp_vertices += list(k.vertices)
#                print(len(tmp_vertices))
#                print(len(list(set(tmp_vertices))))
#                vertices = np.append(vertices, np.array(list(set(tmp_vertices))) + r[0])
                vertices = np.append(vertices, k.vertices + r[0])
                self.logger.info("BeepBeepBoopBoop")

            return vertices #np.array(cbco_rows)
        except:
            self.logger.exception('error:reduce_ptdf')

    def cbco_index_full_to_positive_Ab(self, array_cbco_index):
        """Get the corresponding indecies for the postive Ab Matrix from the
        cbco indecies for the full Ab matrix
        """
        ## Only pos constraints
        idx_pos = []
        for i in array_cbco_index:
#            idx_pos.append(int(i/lines)%2==0)
            idx_pos.append(int(i/len(self.lines))%2==0)
        idx_pos = np.array(idx_pos)

        ## Map to the only-positive indices
        tmp = []
        for i in array_cbco_index[idx_pos]:
#            tmp.append(i - ((lines)*(int((i-1)/(lines)))))
             tmp.append(i - (len(self.lines)*(int((i-1)/len(self.lines)))))

        return np.array(tmp)

    def cbco_index_positive_to_full_Ab(self, array_cbco_index):
        """Get the corresponding indecies for the full Ab Matrix from the
        cbco indecies Ab matrix for only positve constraints
        """
        ## Map to full matrix indices
        idx = []
        for i in array_cbco_index:
            idx.append(i + (len(self.lines)*(int((i-1)/len(self.lines)))))
#            idx.append(i + (len(self.lines)*(int(i/len(self.lines)) - 1)))
        idx = np.array(idx)
        ## add corresponding negative constraints
        tmp = []
        for i in idx:
            tmp.append(i)
            tmp.append(i + (len(self.lines)))
        return np.array(tmp)

    def return_positive_cbcos(self):
        """return A', b', where they are the posotive cbcos from A,b
            optional: return an array of cbcos, that belong to a positive constraint"""
        idx_pos = []
        for i in range(0, len(self.b)):
            idx_pos.append(int(i/len(self.lines))%2==0)
        idx_pos = np.array(idx_pos)
        return self.A[idx_pos], self.b[idx_pos]

    def return_index_from_cbco(self, additional_cbco):
        """Creates a list of cbco_indecies for list of [cb,co], both pos/neg constraints"""
        cbco_index = []
        for [line, out] in additional_cbco:
            cbco_index.append((self.lines.index.get_loc(out) + 1)*2*len(self.lines) + \
                               self.lines.index.get_loc(line))
            cbco_index.append((self.lines.index.get_loc(out) + 1)*2*len(self.lines) + \
                               len(self.lines) + self.lines.index.get_loc(line))

        return cbco_index

    def create_cbcomodule_return(self, index):
        """translates cbco index from reduce method to cb-co from lines data"""
        # Ordering: L+1 N-1 ptdfs, [N-0, N-1_l1, N-1_l2 .... N-1_lL]
        # Each ptdf contains L*2 equations ptdf_1, ... , ptdf_N, ram
        # 0-L constraints have positive ram, L+1 - 2*L Negative ram
        pos_or_neg = {0: 'pos', 1: 'neg'}
        if index/(len(self.lines)*2) < 1: # N-0 ptdf
            info = {'Line': self.lines.index[int(index%(len(self.lines)))],
                    '+/-':  pos_or_neg[int(index/len(self.lines))%2],
                    'Outage': 'N-0'}
        else: # N-1 ptdfs
            info = {'Line': self.lines.index[int(index%(len(self.lines)))],
                    '+/-':  pos_or_neg[int(index/len(self.lines))%2],
                    'Outage': self.lines.index[int(index/(len(self.lines)*2))-1]}
        return info

    #### OLD CODE
    def reduce_ptdf_gams(self, A, b):
        """ Equiv to reduce_ptdf but using gams algorithm"""
        from pathlib import Path
        self.logger.critical("DEPRICATED")
        #add slack to cbco calc sum INJ = 0 for all slack zones
        slack_zones_idx = self.slack_zones_index()
        slack_cbco = []
        for nodes_idx in slack_zones_idx:
            add = np.zeros(len(self.nodes))
            add[nodes_idx] = 1
            add = add.reshape(1, len(self.nodes))
            A = np.concatenate((A, add), axis=0)
            b = np.append(b, [0.00001])
            b = b.reshape(len(b), 1)
            slack_cbco.append(len(b)-1)
        cbco = cbco_reduction.LPReduction(Path.cwd(), A, b)
        cbco.algorithm()
        cbco_rows = cbco.cbco_rows
        for index in slack_cbco:
            if index in cbco_rows:
                cbco_rows.remove(index)
        return np.array(cbco_rows)