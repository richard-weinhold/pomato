import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd
import tables

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import tools as tools


class FBMCDomain(object):
    """Class to store all relevant information of an FBMC Plot"""
    def __init__(self, plot_information, plot_equations, hull_information, xmax, xmin, ymax, ymin):

        self.gsk_strat = plot_information["gsk_strat"]
        self.timestep = plot_information["timestep"]
        self.domain_x = plot_information["domain_x"]
        self.domain_y = plot_information["domain_y"]

        self.title = self.timestep + "_" + self.gsk_strat

        self.plot_equations = plot_equations
        self.hull_information = hull_information

        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        
        # set-up: dont show the graphs when created
        plt.ioff()

    def plot_fbmc_domain(self, folder):
        """Plot the domain"""
        hull_plot_x = self.hull_information["hull_plot_x"]
        hull_plot_y = self.hull_information["hull_plot_y"]
        hull_coord_x = self.hull_information["hull_coord_x"]
        hull_coord_y = self.hull_information["hull_coord_y"]
        fig = plt.figure()
        ax = plt.subplot()
        scale = 1.05
        ax.set_xlim(self.xmin*scale, self.xmax*scale)
        ax.set_ylim(self.ymin*scale, self.ymax*scale)

        title = 'FBMC Domain between: ' + "-".join(self.domain_x) + ' and ' + "-".join(self.domain_y) \
                + '\n Number of CBCOs: ' + str(len(hull_plot_x)-1) \
                + "\n GSK Strategy: " + self.gsk_strat + " - Timestep: " + self.timestep

        for elem in self.plot_equations:
            ax.plot(elem[0], elem[1], c='lightgrey', ls='-')

        ax.plot(hull_plot_x, hull_plot_y, 'r--', linewidth=2)
        ax.set_title(title)
        ax.scatter(hull_coord_x, hull_coord_y)
        fig.savefig(str(folder.joinpath(f"FBMC_{self.title}.png")))
        fig.clf()
        

class FBMCModule(object):
    """ Class to do all calculations in connection with cbco calculation"""
    def __init__(self, wdir, grid_object, injections, frm_fav):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.FBMCModule')
        self.logger.info("Initializing the FBMCModule....")

        self.wdir = wdir
        self.fbmcdir = wdir.joinpath("fbmc_files")

        self.grid_object = grid_object
        self.nodes = grid_object.nodes
        self.lines = grid_object.lines
        self.injections = injections
        self.frm_fav = frm_fav

        self.gsk_strat = "flat"
        self.timestep = "t0001"

        self.fbmc_plots = {}

        self.create_folders(wdir)
        # self.zonal_Ab_file_path = self.create_zonal_Ab_file(grid_object)

        self.A = None
        self.b = None
        # set-up: dont show the graphs when created
        plt.ioff()
        plt.close("all")

        self.logger.info("FBMCModule  Initialized!")

    def create_folders(self, wdir):
        """ create folders for julia cbco_analysis"""
        if not wdir.joinpath("fbmc_files").is_dir():
            wdir.joinpath("fbmc_files").mkdir()
        if not wdir.joinpath("temp_data").is_dir():
            wdir.joinpath("temp_data").mkdir()

    def update_plot_setup(self, timestep, gsk_strat):
        self.logger.info("Setting Net Injection and Updating Ab Matrix")
        self.grid_object.nodes.net_injection = self.injections.INJ[self.injections.t == timestep].values
        self.gsk_strat, self.timestep = gsk_strat, timestep
        ## recalculating Ab
        self.create_zonal_Ab()

    def set_xy_limits_forall_plots(self):

        xmin = min([self.fbmc_plots[plot].xmin for plot in self.fbmc_plots])
        xmax = max([self.fbmc_plots[plot].xmax for plot in self.fbmc_plots])
        ymin = min([self.fbmc_plots[plot].ymin for plot in self.fbmc_plots])
        ymax = max([self.fbmc_plots[plot].ymax for plot in self.fbmc_plots])

        for plots in self.fbmc_plots:
            self.logger.info(f"Resetting x and y limits for {self.fbmc_plots[plots].title}")
            self.fbmc_plots[plots].xmin = xmin
            self.fbmc_plots[plots].xmax = xmax
            self.fbmc_plots[plots].ymin = ymin
            self.fbmc_plots[plots].ymax = ymax

    def load_gsk(self):
#        self = fbmc
#        self.gsk_strat = "g_max_G / flat"
#        self.timestep = "t0001"
        self.logger.info(f"Creating GSKs with GSK Strategy: {self.gsk_strat} for timestep {self.timestep}")

        gsk_raw = pd.read_csv(self.wdir.joinpath("data/gsk_3.csv"), sep=";")
#        gsk_raw["zone"] = self.nodes.zone[gsk_raw.node].values
        gsk_raw = gsk_raw[["node",
                           "zone",
                            self.gsk_strat]][(gsk_raw.timestamp == self.timestep)]

        gsk_raw = gsk_raw.set_index("node")

        gsk = self.nodes.zone.to_frame()
        for zone in list(self.nodes.zone.unique()):
            # zone = "FR"
            if gsk_raw[gsk_raw.zone == zone].empty:
                gsk[zone] = [0]*len(self.nodes)
                gsk[zone][gsk.zone == zone] = 1/len(gsk[zone][gsk.zone == zone])
            else:
#                gsk[zone] = [0]*len(self.nodes)
                tmp = pd.merge(gsk, gsk_raw[gsk_raw.zone == zone],
                               how="left", left_index=True,
                               right_index=True).fillna(0)

                gsk[zone] = tmp[self.gsk_strat].values
                gsk[zone] *= 1/gsk[zone].sum()
        gsk = gsk.drop(["zone"], axis=1)
#        t = gsk.values
        return gsk.values

    def update_gsk(self, option="dict"):
            """
            update generation shift keys based on value in nodes
            option allow to return an array instead of dict for zonal ptdf calc
            """
            try:
                nodes_in_zone = {}
                for zone in self.nodes.zone.unique():
                    nodes_in_zone[zone] = self.nodes.index[self.nodes.zone == zone]
                gsk_dict = {}
                for zone in list(self.nodes.zone.unique()):
                    gsk_zone = {}
                    sum_weight = 0
                    for node in nodes_in_zone[zone]:
                        sum_weight += self.nodes.gsk[node]
                    for node in nodes_in_zone[zone]:
                        if sum_weight == 0:
                            gsk_zone[node] = 1
                        else:
                            gsk_zone[node] = self.nodes.gsk[node]/sum_weight
                    gsk_dict[zone] = gsk_zone
                ## Convert to GSK Array
                gsk_array = np.zeros((len(self.nodes), len(self.nodes.zone.unique())))
                gsk = self.nodes.zone.to_frame()
                for zone in list(self.nodes.zone.unique()):
                    gsk[zone] = [0]*len(self.nodes)
                    gsk[zone][gsk.zone == zone] = list(gsk_dict[zone].values())
                gsk = gsk.drop(["zone"], axis=1)
                gsk_array = gsk.values
                if option == "array":
                    return gsk_array
                else:
                    return gsk_dict
            except:
                self.logger.exception('error:update_gsk')

    def create_zonal_Ab_old(self):
        """
        Create Zonal ptdf -> creates both positive and negative line
        restrictions or ram. Depending on flow != 0
        """
        try:
            self.logger.info("Creating zonal Ab")
            gsk = self.update_gsk(option="array")

            # Calculate zonal ptdf based on ram -> (if current flow is 0 the
            # zonal ptdf is based on overall
            # avalable line capacity (l_max)), ram is calculated for every n-1
            # ptdf matrix to ensure n-1 security constrained FB Domain
            # The right side of the equation has to be positive ,
            # therefore the distingtion.

            ## create full matrix
            length = len(self.lines)*(len(self.lines)+1)*2
            width = len(self.nodes.zone.unique())
            A_zonal = np.zeros((length, width), dtype=np.float64)
            b_zonal = np.zeros((length,), dtype=np.float64)

            # N-0
            ram_array  = self.grid_object.update_ram(self.grid_object.ptdf, option="array")
            z_ptdf = np.dot(self.grid_object.ptdf, gsk)
            r = range(0, 2*len(self.lines.index))
            A_zonal[r, :] = np.vstack([z_ptdf, -z_ptdf])
            b_zonal[r] = np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0)
            # All other N-1 Matrices
            for idx in range(1, len(self.lines.index)+1):
                ptdf = self.grid_object.create_n_1_ptdf_outage(idx-1)
                ram_array = self.grid_object.update_ram(ptdf, option="array")
                z_ptdf = np.dot(ptdf, gsk)
                r = range(idx*2*len(self.lines.index), (idx + 1)*2*len(self.lines.index))
                A_zonal[r, :] = np.vstack([z_ptdf, -z_ptdf])
                b_zonal[r] = np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0)

            self.logger.info("Done!")
            self.A = A_zonal
            self.b = b_zonal
            return A_zonal, b_zonal
        except:
                self.logger.exception('error:create_zonal_ptdf')

    def create_zonal_Ab(self):
        """
        Create Zonal ptdf -> creates both positive and negative line
        restrictions or ram. Depending on flow != 0
        """
        try:
            self.logger.info("Creating zonal Ab")
            # Calculate zonal ptdf based on ram -> (if current flow is 0 the
            # zonal ptdf is based on overall
            # avalable line capacity (l_max)), ram is calculated for every n-1
            # ptdf matrix to ensure n-1 security constrained FB Domain
            # The right side of the equation has to be positive ,
            # therefore the distingtion.

            ## create full matrix
#            if "cb" in self.lines.columns:
#                select_lines = self.lines.index[self.lines.cb]
            if "cnec" in self.lines.columns:
                select_lines = self.lines.index[self.lines.cnec]
            else:
                select_lines = self.lines.index

            lines_indecies = [self.lines.index.get_loc(line) for line in select_lines]

            gsk = self.load_gsk()
            frm_fav = []
            for line in select_lines:
                if line in self.frm_fav.columns:
                    frm_fav.append(self.frm_fav[line][self.timestep])
                else:
                    frm_fav.append(0)

            length = (len(select_lines))*(len(self.lines.index) + 1)*2
            width = len(self.nodes.zone.unique())
            A_zonal = np.zeros((length, width), dtype=np.float64)
            b_zonal = np.zeros((length,), dtype=np.float64)

            for idx, line_idx in enumerate(lines_indecies):
                ptdf = self.grid_object.create_n_1_ptdf_line(line_idx)
                flow = np.dot(ptdf, self.nodes.net_injection.values)
                z_ptdf = np.dot(ptdf, gsk)
                r = range(idx*2*(len(self.lines.index)+1), (idx + 1)*2*(len(self.lines.index)+1))
                A_zonal[r, :] = np.vstack([z_ptdf, -z_ptdf])
                b_zonal[r] = np.concatenate([np.subtract(self.lines.maxflow.iloc[line_idx] - frm_fav[idx], flow),
                                             -np.subtract(-self.lines.maxflow.iloc[line_idx] + frm_fav[idx], flow)], axis=0)
            self.logger.info("Done!")
            self.A = A_zonal
            self.b = b_zonal
            return A_zonal, b_zonal
        except:
                self.logger.exception('error:create_zonal_ptdf')

    def create_zonal_Ab_file(self, grid_object):
        """ Storing Matrix A and Vector b to disk to save memory"""
        # set up for Ab file
        # dtype = np.array(grid_object.ptdf, dtype=np.float16).dtype
        dtype = np.dtype("Float64")
        shape = len(list(self.nodes.zone.unique()))
        expectedrows = len(self.lines)*(len(self.lines)+1)*2

        ## Init PyTables
        hdf5_path = self.wdir.joinpath("temp_data/zonal_Ab_expandeble_compressed.hdf5")
        hdf5_file = tables.open_file(str(hdf5_path), mode='w')
        filters = tables.Filters(complevel=1, complib='zlib')
        zonal_A_storage = hdf5_file.create_earray(hdf5_file.root, 'A',
                                                  tables.Atom.from_dtype(dtype),
                                                  shape=(0, shape),
                                                  filters=filters,
                                                  expectedrows=expectedrows)

        zonal_b_storage = hdf5_file.create_earray(hdf5_file.root, 'b',
                                                  tables.Atom.from_dtype(dtype),
                                                  shape=(0,),
                                                  filters=filters,
                                                  expectedrows=expectedrows)
        gsk = self.update_gsk(option="array")
        ## N-0 Matrix
        ram_array = grid_object.update_ram(grid_object.ptdf, option="array")
        z_ptdf = np.dot(grid_object.ptdf, gsk)
        zonal_A_storage.append(np.vstack([z_ptdf, -z_ptdf]))
        zonal_b_storage.append(np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0))

        for idx, line in enumerate(grid_object.lines.index):
            ptdf = grid_object.create_n_1_ptdf_outage(idx)
            ram_array = grid_object.update_ram(ptdf, option="array")
            z_ptdf = np.dot(ptdf, gsk)
            zonal_A_storage.append(np.vstack([z_ptdf, -z_ptdf]))
            zonal_b_storage.append(np.concatenate([ram_array[:, 0], -ram_array[:, 1]], axis=0))

        hdf5_file.close()
        ## To be clear
        zonal_Ab_file_path = str(hdf5_path)
        return zonal_Ab_file_path

    def retrieve_zonal_Ab(self, selection):
        """getting the rows/values from A,b, input either array/list with indices, range or index"""
        Ab_file = tables.open_file(self.zonal_Ab_file_path, mode='r')
        A = Ab_file.root.A[selection, :]
        b = Ab_file.root.b[selection]
        Ab_file.close()
        if isinstance(b, np.ndarray):
            b = b.reshape(len(b), 1)
        return A, b

    def create_Ab_from_jao_data(self, timestep="t0001"):
#        self = fbmc
        self.logger.info(f"Using Jao Data for timestep {timestep}")
        jao_data_raw = pd.read_csv(self.wdir.joinpath("data/JAO_complete_CBCO.csv"), sep=",", encoding='ISO8859')
        jao_data_raw.CalendarDate = pd.DatetimeIndex(jao_data_raw.CalendarDate)
        jao_data_raw.CalendarHour = pd.to_timedelta(jao_data_raw.CalendarHour, unit="h")
        jao_data_raw["utc_timestamp"] = jao_data_raw.CalendarDate + jao_data_raw.CalendarHour
        jao_data_raw = jao_data_raw[["utc_timestamp", "DEAT", "BE", "FR", "NL", 'FAV','FRM','Fmax', 'Fref', "RemainingAvailableMargin"]]
        jao_data_raw.rename(columns={'DEAT': 'DE', "RemainingAvailableMargin": "RAM"}, inplace=True)

        timesteps = jao_data_raw.utc_timestamp.unique()
        timestep_dict = {timesteps[idx]: 't'+ "{0:0>4}".format(idx + 1) for idx in range(0, len(timesteps))}
        jao_data_raw["timestep"] = None
        for t in timestep_dict:
            jao_data_raw.timestep[jao_data_raw.utc_timestamp == t] = timestep_dict[t]
#        jao_data_raw.set_index("utc_timestamp", inplace=True)

        cbco_df = jao_data_raw[jao_data_raw.timestep == timestep]
        length = len(cbco_df.index)
        width = len(self.nodes.zone.unique())
        A_zonal = np.zeros((length, width), dtype=np.float64)
        b_zonal = np.zeros((length,), dtype=np.float64)

        for idx, zone in enumerate(list(self.nodes.zone.unique())):
            if zone in cbco_df.columns:
                A_zonal[:,idx] = cbco_df[zone].values
        b_zonal[:] =  cbco_df.RAM.values

        self.A = A_zonal
        self.b = b_zonal
        self.gsk_strat = "jao"
        self.timestep = timestep

        return A_zonal, b_zonal

    def create_fbmc_equations(self, domain_x=None, domain_y=None, gsk_sink=None):
            """
            from zonal ptdf calculate linear equations ax = b to plot the FBMC domain
            nodes/Zones that are not part of the 2D FBMC are summerized using GSK sink
            """
            try:
                domain_x = domain_x or []
                domain_y = domain_y or []
                gsk_sink = gsk_sink or []

                list_zones = list(self.nodes.zone.unique())
                if not (isinstance(self.A, np.ndarray) and isinstance(self.b, np.ndarray)):
                    A, b = self.create_zonal_Ab()
                else:
                    self.logger.info("Using previous Ab")
                A = self.A
                b = self.b
                self.logger.info("Creating fbmc equations...")
                if len(domain_x) == 1:
                    domain_idx = [list_zones.index(z) for z in domain_x + domain_y]
                    sink_idx = [list_zones.index(z) for z in gsk_sink.keys()]
                    sink_values = np.array([gsk_sink[z] for z in gsk_sink.keys()])
                    A = np.vstack([A[:, domain] - np.dot(A[:, sink_idx], sink_values) for domain in domain_idx]).T

                elif len(domain_x) == 2:
                    domain_idx = [[list_zones.index(z[0]), list_zones.index(z[1])] for z in [domain_x, domain_y]]
                    A = np.vstack([np.dot(A[:, domain], np.array([1, -1])) for domain in domain_idx]).T

                #Clean reduce Ax=b only works if b_i != 0 for all i,
                #which should be but sometimes wierd stuff comes up
                #Therefore if b == 0, b-> 1 (or something small>0)
                if not (b > 0).all():
                    b[(b < 0)] = 0.1
                    self.logger.debug('some b is not right (possibly < 0) and \
                                       implies that n-1 is not given!')
                self.logger.info("Done!")
                return(A, b)
            except:
                self.logger.exception('error:create_eq_list_zptdf')

    def reduce_ptdf(self, A, b):
        """
        Given an system Ax = b, where A is a list of ptdf and b the corresponding ram
        Reduce will find the set of ptdf equations which constrain the solution domain
        (which are based on the N-1 ptdfs)
        """
        self.logger.info("Reducing Ab")
        A = np.array(A, dtype=np.float)
        b = np.array(b, dtype=np.float).reshape(len(b), 1)
        D = A/b
        k = ConvexHull(D, qhull_options="QJ")
        self.logger.info("Done!")
        return k.vertices

    def create_domain_plot(self, A, b, cbco_index):
        """Creates coordinates for the FBMC Domain
        """
        try:
#            cbco_index = cbco_plot
            # create 2D equation from zonal ptdfs
            # This creates Domain X-Y
            A = np.take(np.array(A), cbco_index, axis=0)
            b = np.take(np.array(b), cbco_index, axis=0)
            Ab = np.concatenate((np.array(A), np.array(b).reshape(len(b), 1)), axis=1)

            self.logger.debug(f"Number of CBCOs {len(Ab)} in plot_domain")

            # Calculate two coordinates for a line plot -> Return X = [X1;X2], Y = [Y!,Y2]
            x_upper = int(max(b)*20)
            x_lower = -x_upper
            plot_equations = []
            for index in range(0, len(Ab)):
                xcoord = []
                ycoord = []
#                for idx in range(-10000, 10001, 20000):
                for idx in range(x_lower, x_upper +1, (x_upper - x_lower)):
                    if Ab[index][1] != 0:
                        ycoord.append((Ab[index][2] - idx*(Ab[index][0])) / (Ab[index][1]))
                        xcoord.append(idx)
                    elif Ab[index][0] != 0:
                        ycoord.append(idx)
                        xcoord.append((Ab[index][2] - idx*(Ab[index][1])) / (Ab[index][0]))
                plot_equations.append([xcoord, ycoord])

            return plot_equations
        except:
            self.logger.exception('error:plot_domain')

    def plot_vertecies_of_inequalities(self, domain_x, domain_y, gsk_sink):
        """Plot Vertecies Representation of FBMC Domain"""
        self.nodes.net_injection = 0
        contingency = self.n_1_ptdf
        gsk_sink = gsk_sink or {}
        list_zonal_ptdf = self.create_zonal_ptdf(contingency)

        A, b = self.create_eq_list_zptdf(list_zonal_ptdf, domain_x, domain_y, gsk_sink)
        cbco_index = self.reduce_ptdf(A, b)

        full_indices = np.array([x for x in range(0,len(A))])
        # only plot a subset of linear inequalities that are not part of the load flow domain if A too large
        if len(A) > 1e3:
            relevant_subset = np.append(cbco_index, np.random.choice(full_indices, int(1e3), replace=False))
        else:
            relevant_subset = full_indices

        A = np.array(A)
        b = np.array(b).reshape(len(b), 1)

        vertecies_full = np.take(A, relevant_subset, axis=0)/np.take(b, relevant_subset, axis=0)
        vertecies_reduces = np.take(A, cbco_index, axis=0)/np.take(b, cbco_index, axis=0)

        x_max, x_min, y_max, y_min = tools.find_xy_limits([[vertecies_reduces[:,0], vertecies_reduces[:,1]]])

        fig = plt.figure()
        ax = plt.subplot()

        scale = 1.2
        ax.set_xlim(x_min*scale, x_max*scale)
        ax.set_ylim(y_min*scale, y_max*scale)

        for point in vertecies_full:
            ax.scatter(point[0], point[1], c='lightgrey')
        for point in vertecies_reduces:
            ax.scatter(point[0], point[1], c='r')
        return fig

    def get_xy_hull(self, A, b, cbco_index):
        """get x,y coordinates of the FBMC Hull"""
        try:
            # ptdf_x * X + ptdf_y *Y = B
            # Or in Matrix Form A*x = b where X = [X;Y]
            ptdf = np.take(A, cbco_index, axis=0)
            ram = np.take(b, cbco_index, axis=0)
            self.logger.info(f"Number of CBCOs {len(ram)} in get_xy_hull")
            ### Find all intersections between CO
            intersection_x = []
            intersection_y = []
            for idx in range(0, len(ptdf)):
                for idy in range(0, len(ptdf)):
                    ### x*ptdf_A0 +  y*ptdf_A1 = C_A ----- x*ptdf_B0 + y*ptdf_B1 = C_B
                    ### ptdf[idx,0] ptdf[idx,1] = ram[idx] <-> ptdf[idy,0] ptdf[idy,1] = ram[idy]
                    if idx != idy:
                            # A0 close to 0
                        if np.square(ptdf[idx, 0]) < 1E-9 and np.square(ptdf[idy, 0]) > 1E-9:
                            intersection_x.append((ram[idy]*ptdf[idx, 1] - ram[idx]*ptdf[idy, 1])\
                                     /(ptdf[idx, 1]*ptdf[idy, 0]))
                            intersection_y.append(ram[idx]/ptdf[idx, 1])
                            ## A1 close to 0
                        elif np.square(ptdf[idx, 1]) < 1E-9 and np.square(ptdf[idy, 1]) > 1E-9:
                            intersection_x.append(ram[idx]/ptdf[idx, 0])
                            intersection_y.append((ram[idy]*ptdf[idx, 0] - ram[idx]*ptdf[idy, 0]) \
                                     /(ptdf[idx, 0]*ptdf[idy, 1]))

                        elif (ptdf[idx, 1]*ptdf[idy, 0] - ptdf[idy, 1]*ptdf[idx, 0]) != 0 \
                        and (ptdf[idx, 0]*ptdf[idy, 1] - ptdf[idy, 0]*ptdf[idx, 1]):
                            intersection_x.append((ram[idx]*ptdf[idy, 1] - ram[idy]*ptdf[idx, 1]) \
                                    / (ptdf[idx, 0]*ptdf[idy, 1] - ptdf[idy, 0]*ptdf[idx, 1]))
                            intersection_y.append((ram[idx]*ptdf[idy, 0] - ram[idy]*ptdf[idx, 0]) \
                                    / (ptdf[idx, 1]*ptdf[idy, 0] - ptdf[idy, 1]*ptdf[idx, 0]))

            hull_x = []
            hull_y = []
            ### Filter intersection points for those which define the FB Domain
            for idx in range(0, len(intersection_x)):
                temp = 0
                for idy in range(0, len(ptdf)):
                    if ptdf[idy, 0]*intersection_x[idx] +\
                        ptdf[idy, 1]*intersection_y[idx] <= ram[idy]*1.0001:
                        temp += 1
                    if temp >= len(ptdf):
                        hull_x.append(intersection_x[idx])
                        hull_y.append(intersection_y[idx])

            ### Sort them Counter Clockwise to plot them
            list_coord = []
            for idx in range(0, len(hull_x)):
                radius = np.sqrt(np.power(hull_y[idx], 2) + np.power(hull_x[idx], 2))
                if hull_x[idx] >= 0 and hull_y[idx] >= 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])
                elif hull_x[idx] < 0 and hull_y[idx] > 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       180 - np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])
                elif hull_x[idx] <= 0 and hull_y[idx] <= 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       180 - np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])
                elif hull_x[idx] > 0 and hull_y[idx] < 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       360 + np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])

            from operator import itemgetter
            list_coord = sorted(list_coord, key=itemgetter(2))
            ## Add first element to draw complete circle
            list_coord.append(list_coord[0])
            list_coord = np.array(list_coord)
            list_coord = np.round(list_coord, decimals=3)
            unique_rows_idx = [x for x in range(0, len(list_coord)-1) \
                               if not np.array_equal(list_coord[x, 0:2], list_coord[x+1, 0:2])]
            unique_rows_idx.append(len(list_coord)-1)
            list_coord = np.take(list_coord, unique_rows_idx, axis=0)
            return(list_coord[:, 0], list_coord[:, 1], intersection_x, intersection_y)
        except:
            self.logger.exception('error:get_xy_hull')

    def plot_fbmc(self, domain_x, domain_y, gsk_sink=None, reduce=False):
        """
        Combines previous functions to actually plot the FBMC Domain with the
        hull
        """
        try:
#            self = fbmc
            gsk_sink = gsk_sink or {}
#            domain_x = ["DE"]
#            domain_y = ["FR"]
            # self.nodes.net_injection = 0
            A, b = self.create_fbmc_equations(domain_x, domain_y, gsk_sink)
            # Reduce
            cbco_index = self.reduce_ptdf(A, b)

            full_indices = np.array([x for x in range(0,len(A))])
            if not reduce: # plot only the reduced set or more constraints
                # Limit the number of constraints to 3*number of lines (pos and neg)
                if len(A) > 5e3:
                    cbco_plot_indices = np.append(cbco_index,
                                                  np.random.choice(full_indices,
                                                                   size=int(5e3),
                                                                   replace=False))
                else:
                    cbco_plot_indices = full_indices

            plot_equations = self.create_domain_plot(A, b, cbco_plot_indices)

            hull_plot_x, hull_plot_y, hull_coord_x, hull_coord_y = self.get_xy_hull(A, b, cbco_index)

            xmax, xmin, ymax, ymin = tools.find_xy_limits([[hull_plot_x, hull_plot_y]])

            fbmc_plot = FBMCDomain({"gsk_strat": self.gsk_strat, "timestep": self.timestep,
                                   "domain_x": domain_x, "domain_y": domain_y},
                                   plot_equations,
                                   {"hull_plot_x": hull_plot_x, "hull_plot_y": hull_plot_y,
                                   "hull_coord_x": hull_coord_x, "hull_coord_y": hull_coord_y},
                                   xmax, xmin, ymax, ymin)

            self.fbmc_plots[fbmc_plot.title] = fbmc_plot

        except:
            self.logger.exception('error:plot_fbmc', sys.exc_info()[0])

