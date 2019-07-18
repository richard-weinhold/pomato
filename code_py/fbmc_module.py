import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import tools as tools


# kriterien für cb auswahl
# # n-0 last
# # lodf filter vll die ersten 10
# #
# # häufigkeit als teil der domain
# #
# # 3d plot

class FBMCDomain(object):
    """Class to store all relevant information of an FBMC Plot"""
    def __init__(self, plot_information, plot_equations, hull_information, xy_limits, domain_data):

        self.gsk_strategy = plot_information["gsk_strategy"]
        self.timestep = plot_information["timestep"]
        self.domain_x = plot_information["domain_x"]
        self.domain_y = plot_information["domain_y"]

        self.title = self.timestep + "_" + self.gsk_strategy

        self.plot_equations = plot_equations
        self.hull_information = hull_information

        self.x_max = xy_limits["x_max"]
        self.x_min = xy_limits["x_min"]
        self.y_max = xy_limits["y_max"]
        self.y_min = xy_limits["y_min"]
        self.domain_data = domain_data

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
        ax.set_xlim(self.x_min*scale, self.x_max*scale)
        ax.set_ylim(self.y_min*scale, self.y_max*scale)

        title = 'FBMC Domain between: ' + "-".join(self.domain_x) + ' and ' + "-".join(self.domain_y) \
                + '\n Number of CBCOs: ' + str(len(hull_plot_x)-1) \
                + "\n GSK Strategy: " + self.gsk_strategy + " - Timestep: " + self.timestep

        for elem in self.plot_equations:
            ax.plot(elem[0], elem[1], c='lightgrey', ls='-')

        ax.plot(hull_plot_x, hull_plot_y, 'r--', linewidth=2)
        ax.set_title(title)
        ax.scatter(hull_coord_x, hull_coord_y)
        fig.savefig(str(folder.joinpath(f"FBMC_{self.title}.png")))
        fig.clf()


class FBMCModule(object):
    """ Class to do all calculations in connection with cbco calculation"""
    def __init__(self, wdir, grid_object, results):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.FBMCModule')
        self.logger.info("Initializing the FBMCModule....")

        self.wdir = wdir

        self.grid = grid_object
        self.nodes = grid_object.nodes
        self.lines = grid_object.lines

        self.flowbased_region = ["DE", "FR", "NL", "BE", "LU"]

        self.gsk_strategy = "gmax"
        self.timestep = "t0001"

        self.results = results
        self.fbmc_plots = {}
        self.nodal_fbmc_ptdf, self.domain_info = self.create_fbmc_info()
        
        # A, b saved, for recreation of multiple plots with the same
        # configuration of gsk_strat and timestep
        self.A = None
        self.b = None

        # set-up: dont show the graphs when created
        plt.ioff()
        plt.close("all")
        self.logger.info("FBMCModule  Initialized!")


    def save_worker(self, arg):
        """ arg[0] = plot, arg[1] = folder"""
        # self.logger.info(f"Plotting Domain of {arg[0]}")
        print(f"Plotting Domain of {arg[0]} in folder {arg[1]}")
        self.fbmc_plots[arg[0]].plot_fbmc_domain(arg[1])
        print(f"Done {arg[0]}")

    def save_all_domain_plots(self, folder, set_xy_limits=True):
        if set_xy_limits:
            self.set_xy_limits_forall_plots()
        plt.close("all")
        for plot in self.fbmc_plots:
            arg = [plot, folder]
            self.save_worker(arg)

    def save_all_domain_info(self, folder, name_suffix=""):

        domain_info = pd.concat([self.fbmc_plots[plot].domain_data for plot in self.fbmc_plots])
        # oder the columns
        columns = ["timestep", "gsk_strategy", "cb", "co"]
        columns.extend(list(self.nodes.zone.unique()))
        columns.extend(["ram", "in_domain"])
        domain_info = domain_info[columns]

        mask = domain_info[['cb','co']].isin({'cb': domain_info[domain_info.in_domain].cb.unique(), 
                                              'co': domain_info[domain_info.in_domain].co.unique()}).all(axis=1)

        self.logger.info("Saving domain info as csv")
        domain_info[domain_info.in_domain].to_csv(folder.joinpath("domain_info" + name_suffix + ".csv"))
        domain_info[mask].to_csv(folder.joinpath("domain_info_full" + name_suffix + ".csv"))
        # domain_info[domain_info.in_domain].to_csv(folder.joinpath("domain_info_domain" + name_suffix + ".csv"))
        return domain_info

    def update_plot_setup(self, timestep, gsk_strategy):
        self.logger.info("Setting Net Injection and Updating Ab Matrix")
        self.gsk_strategy, self.timestep = gsk_strategy, timestep
        ## recalculating Ab
        self.create_zonal_Ab()

    def set_xy_limits_forall_plots(self):
        """For each fbmc plot object, set x and y limits"""

        x_min = min([self.fbmc_plots[plot].x_min for plot in self.fbmc_plots])
        x_max = max([self.fbmc_plots[plot].x_max for plot in self.fbmc_plots])
        y_min = min([self.fbmc_plots[plot].y_min for plot in self.fbmc_plots])
        y_max = max([self.fbmc_plots[plot].y_max for plot in self.fbmc_plots])

        for plots in self.fbmc_plots:
            self.logger.info(f"Resetting x and y limits for {self.fbmc_plots[plots].title}")
            self.fbmc_plots[plots].x_min = x_min
            self.fbmc_plots[plots].x_max = x_max
            self.fbmc_plots[plots].y_min = y_min
            self.fbmc_plots[plots].y_max = y_max

    def create_gsk(self, option="flat"):
        """returns GSK, either flat or gmax"""

        gsk = pd.DataFrame(index=self.nodes.index)
        conv_fuel = ['uran', 'lignite', 'hard coal', 'gas', 'oil', 'hydro', 'waste']
        condition = self.results.data.plants.fuel.isin(conv_fuel)&(self.results.data.plants.tech != "psp")
        gmax_per_node = self.results.data.plants.loc[condition, ["g_max", "node"]].groupby("node").sum()

        for zone in self.results.data.zones.index:
            nodes_in_zone = self.nodes.index[self.nodes.zone == zone]
            gsk[zone] = 0
            gmax_in_zone = gmax_per_node[gmax_per_node.index.isin(nodes_in_zone)]
            if option == "gmax":
                if not gmax_in_zone.empty:
                    gsk_value = gmax_in_zone.g_max/gmax_in_zone.values.sum()
                    gsk.loc[gsk.index.isin(gmax_in_zone.index), zone] = gsk_value
                else:
                    gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)

            elif option == "flat":
                gsk.loc[gsk.index.isin(nodes_in_zone), zone] = 1/len(nodes_in_zone)

        return gsk.values


    def return_critical_branches(self, threshold=5e-2):
        gsk = self.create_gsk("gmax")
        zonal_ptdf = np.dot(self.grid.ptdf, gsk)
        zonal_ptdf_df = pd.DataFrame(index=self.lines.index, columns=self.results.data.zones.index, data=zonal_ptdf)
        z2z_ptdf_df = pd.DataFrame(index=self.lines.index)
        for z in self.flowbased_region:
            for zz in self.flowbased_region:
                z2z_ptdf_df["-".join([z, zz])] = zonal_ptdf_df[z] - zonal_ptdf_df[zz]

        critical_branches = list(z2z_ptdf_df.index[np.any(z2z_ptdf_df.abs() > threshold, axis=1)])

        condition_cross_border = self.nodes.zone[self.lines.node_i].values != self.nodes.zone[self.lines.node_j].values
        condition_flowbased_region = self.nodes.zone[self.lines.node_i].isin(self.flowbased_region).values & \
                                     self.nodes.zone[self.lines.node_j].isin(self.flowbased_region).values

        cross_border_lines = list(self.lines.index[condition_cross_border&condition_flowbased_region])
        total_cbs = list(set(critical_branches + cross_border_lines))

        self.logger.info("Number of Critical Branches: %d, \
                          Number of Cross Border lines: %d, \
                          Total Number of CBs: %d", len(critical_branches), len(cross_border_lines), len(total_cbs))

        return total_cbs



    def create_fbmc_info(self, lodf_sensitivity=20e-2):

        """
        create ptdf, determine CBs
        """

        self.lines["cb"] = False
        critical_branches = self.return_critical_branches(threshold=5e-2)
        self.lines.loc[self.lines.index.isin(critical_branches), "cb"] = True

        select_lines = self.lines.index[(self.lines["cb"])&(self.lines.contingency)]

        full_ptdf = []
        label_lines, label_outages = [], []

        for idx, line in enumerate(select_lines):
            outages = list(self.grid.lodf_filter(line, lodf_sensitivity))
            tmp_ptdf = np.vstack([self.grid.create_n_1_ptdf_cbco(line,o) for o in outages])
            full_ptdf.extend([tmp_ptdf, -tmp_ptdf])
            label_lines.extend([line for i in range(0, 2*len(outages))])
            label_outages.extend(outages*2)

        nodal_fbmc_ptdf = np.concatenate(full_ptdf).reshape(len(label_lines), len(list(self.nodes.index)))

        domain_info = pd.DataFrame(columns=list(self.results.data.zones.index))
        domain_info["cb"] = label_lines
        domain_info["co"] = label_outages

        return nodal_fbmc_ptdf, domain_info

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
            # The right side of the equation has to be positive
            

            frm_fav = pd.DataFrame(index=self.domain_info.cb.unique())
            frm_fav["value"] = self.lines.maxflow[frm_fav.index]*0.2

            injection = self.results.INJ.INJ[self.results.INJ.t == self.timestep].values


            f_ref_base_case = np.dot(self.nodal_fbmc_ptdf, injection)
            gsk = self.create_gsk(self.gsk_strategy)
            zonal_fbmc_ptdf = np.dot(self.nodal_fbmc_ptdf, gsk)

            # F Day Ahead (eigentlich mit LTNs)
            net_position = self.results.net_position()
            net_position.loc[:, ~net_position.columns.isin(self.flowbased_region)] = 0

            f_da = np.dot(zonal_fbmc_ptdf, net_position.loc[self.timestep].values)
            f_ref_nonmarket = f_ref_base_case - f_da

            ram = np.subtract(self.lines.maxflow[self.domain_info.cb] - frm_fav.value[self.domain_info.cb],
                                  f_ref_nonmarket).values

            ram = ram.reshape(len(ram), 1)

            # if any(ram < 0):
            #     self.logger.warning("%d RAMs are <0", sum(ram<0))
            #     ram[ram<150] = 10000



            self.domain_info[list(self.results.data.zones.index)] = zonal_fbmc_ptdf
            self.domain_info["ram"] = ram
            self.domain_info["timestep"] = self.timestep
            self.domain_info["gsk_strategy"] = self.gsk_strategy

            self.logger.info("Done!")

            self.A = zonal_fbmc_ptdf
            self.b = ram

            return zonal_fbmc_ptdf, ram
        except:
                self.logger.exception('error:create_zonal_ptdf')


    def create_fbmc_equations(self, domain_x=None, domain_y=None, gsk_sink=None):
            """
            from zonal ptdf calculate linear equations ax = b to plot the FBMC domain
            nodes/Zones that are not part of the 2D FBMC are summerized using GSK sink
            """
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
                self.logger.warning('some b is not right (possibly < 0)')

            return(A, b)


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
        """
        Creates coordinates for the FBMC Domain
        create 2D equation from zonal ptdfs for domain x and y
        """
        A = np.take(np.array(A), cbco_index, axis=0)
        b = np.take(np.array(b), cbco_index, axis=0)
        Ab = np.concatenate((np.array(A), np.array(b).reshape(len(b), 1)), axis=1)

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

        relevant_subset = cbco_index
        A = np.array(A)
        b = np.array(b).reshape(len(b), 1)

        vertecies_full = np.take(A, relevant_subset, axis=0)/np.take(b, relevant_subset, axis=0)
        vertecies_reduces = np.take(A, cbco_index, axis=0)/np.take(b, cbco_index, axis=0)

        xy_limits = tools.find_xy_limits([[vertecies_reduces[:,0], vertecies_reduces[:,1]]])

        fig = plt.figure()
        ax = plt.subplot()

        scale = 1.2
        ax.set_xlim(xy_limits["x_min"]*scale, xy_limits["x_max"]*scale)
        ax.set_ylim(xy_limits["y_min"]*scale, xy_limits["y_max"]*scale)

        for point in vertecies_full:
            ax.scatter(point[0], point[1], c='lightgrey')
        for point in vertecies_reduces:
            ax.scatter(point[0], point[1], c='r')
        return fig

    def get_xy_hull(self, A, b, cbco_index):
        """get x,y coordinates of the FBMC Hull"""
        # ptdf_x * X + ptdf_y *Y = B
        # Or in Matrix Form A*x = b where X = [X;Y]
        ptdf = np.take(A, cbco_index, axis=0)
        ram = np.take(b, cbco_index, axis=0)

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

    def plot_fbmc(self, domain_x, domain_y, gsk_sink=None, reduce=False):
        """
        Combines previous functions to actually plot the FBMC Domain with the
        hull
        """
        gsk_sink = gsk_sink or {}
        A, b = self.create_fbmc_equations(domain_x, domain_y, gsk_sink)
        # Reduce
        cbco_index = self.reduce_ptdf(A, b)
        self.domain_info["in_domain"] = False
        self.domain_info.loc[self.domain_info.index.isin(cbco_index), "in_domain"] = True

        self.logger.info(f"Number of CBCOs %d", len(cbco_index))

        full_indices = np.array([x for x in range(0,len(A))])
        
        # Limit the number of constraints to 5000
        threshold = 5e3
        if len(A) > threshold:
            cbco_plot_indices = np.append(cbco_index,
                                          np.random.choice(full_indices,
                                                           size=int(threshold),
                                                           replace=False))
        else:
            cbco_plot_indices = full_indices

        plot_equations = self.create_domain_plot(A, b, cbco_plot_indices)
        hull_plot_x, hull_plot_y, hull_coord_x, hull_coord_y = self.get_xy_hull(A, b, cbco_index)
        self.logger.info(f"Number of CBCOs defining the domain %d", len(hull_plot_x) - 1)


        xy_limits = tools.find_xy_limits([[hull_plot_x, hull_plot_y]])

        fbmc_plot = FBMCDomain({"gsk_strategy": self.gsk_strategy, 
                                "timestep": self.timestep,
                                "domain_x": domain_x, 
                                "domain_y": domain_y
                                },
                               plot_equations,
                               {"hull_plot_x": hull_plot_x, 
                                "hull_plot_y": hull_plot_y,
                                "hull_coord_x": hull_coord_x, 
                                "hull_coord_y": hull_coord_y
                                },
                               xy_limits,
                               self.domain_info.copy())

        self.fbmc_plots[fbmc_plot.title] = fbmc_plot

