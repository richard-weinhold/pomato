"""

THIS IS FBMC Module
INPUT: DATA
OUTPUT: RESULTS

"""

import logging
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import spatial
import tools as tools


# kriterien für cb auswahl
# # n-0 last
# # lodf filter vll die ersten 10
# #
# # häufigkeit als teil der domain
# #
# # 3d plot

class FBMCDomain():
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
        axis = plt.subplot()
        scale = 1.05
        axis.set_xlim(self.x_min*scale, self.x_max*scale)
        axis.set_ylim(self.y_min*scale, self.y_max*scale)

        title = 'FBMC Domain between: ' + "-".join(self.domain_x) \
                + ' and ' + "-".join(self.domain_y) \
                + '\n Number of CBCOs: ' + str(len(hull_plot_x)-1) \
                + "\n GSK Strategy: " + self.gsk_strategy \
                + " - Timestep: " + self.timestep

        for elem in self.plot_equations:
            axis.plot(elem[0], elem[1], c='lightgrey', ls='-')

        axis.plot(hull_plot_x, hull_plot_y, 'r--', linewidth=2)
        axis.set_title(title)
        axis.scatter(hull_coord_x, hull_coord_y)
        fig.savefig(str(folder.joinpath(f"FBMC_{self.title}.png")))
        fig.clf()


class FBMCModule():
    """ Class to do all calculations in connection with cbco calculation"""
    def __init__(self, wdir, grid_object, results, cbco_list=None):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.FBMCModule')
        self.logger.info("Initializing the FBMCModule....")

        self.wdir = wdir

        self.grid = grid_object
        self.nodes = grid_object.nodes
        self.lines = grid_object.lines

        self.flowbased_region = ["DE", "FR", "NL", "BE", "LU"]
        self.cbco_list = cbco_list

        self.results = results
        self.fbmc_plots = {}
        self.nodal_fbmc_ptdf, self.domain_info = self.create_fbmc_info()

        # set-up: dont show the graphs when created
        plt.ioff()
        plt.close("all")
        self.logger.info("FBMCModule  Initialized!")


    def save_all_domain_plots(self, folder):
        """Saving all the FBMC Plots"""
        self.set_xy_limits_forall_plots()
        plt.close("all")
        for plot in self.fbmc_plots:
            self.logger.info("Plotting Domain of %s in folder %s", plot, folder)
            self.fbmc_plots[plot].plot_fbmc_domain(folder)
            self.logger.info("Done!")

    def save_all_domain_info(self, folder, name_suffix=""):
        """Save the FBMC Domain Info (=Data) """
        domain_info = pd.concat([self.fbmc_plots[plot].domain_data for plot in self.fbmc_plots])
        # oder the columns
        columns = ["timestep", "gsk_strategy", "cb", "co"]
        columns.extend(list(self.nodes.zone.unique()))
        columns.extend(["ram", "in_domain"])
        domain_info = domain_info[columns]

        mask_dict = {'cb': domain_info[domain_info.in_domain].cb.unique(),
                     'co': domain_info[domain_info.in_domain].co.unique()}
        mask = domain_info[['cb', 'co']].isin(mask_dict).all(axis=1)

        self.logger.info("Saving domain info as csv")

        filename = folder.joinpath("domain_info" + name_suffix + ".csv")
        domain_info[domain_info.in_domain].to_csv(filename)

        domain_info[mask].to_csv(folder.joinpath("domain_info_full" + name_suffix + ".csv"))
        return domain_info

    def set_xy_limits_forall_plots(self):
        """For each fbmc plot object, set x and y limits"""

        x_min = min([self.fbmc_plots[plot].x_min for plot in self.fbmc_plots])
        x_max = max([self.fbmc_plots[plot].x_max for plot in self.fbmc_plots])
        y_min = min([self.fbmc_plots[plot].y_min for plot in self.fbmc_plots])
        y_max = max([self.fbmc_plots[plot].y_max for plot in self.fbmc_plots])

        for plots in self.fbmc_plots:
            self.logger.info("Resetting x and y limits for %s", self.fbmc_plots[plots].title)
            self.fbmc_plots[plots].x_min = x_min
            self.fbmc_plots[plots].x_max = x_max
            self.fbmc_plots[plots].y_min = y_min
            self.fbmc_plots[plots].y_max = y_max

    def create_gsk(self, option="flat"):
        """returns GSK, either flat or gmax"""

        gsk = pd.DataFrame(index=self.nodes.index)
        conv_fuel = ['uran', 'lignite', 'hard coal', 'gas', 'oil', 'hydro', 'waste']
        condition = (self.results.data.plants.fuel.isin(conv_fuel)) & \
                    (self.results.data.plants.tech != "psp")
        gmax_per_node = self.results.data.plants.loc[condition, ["g_max", "node"]] \
                        .groupby("node").sum()

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

    def return_critical_branches(self, threshold=5e-2, gsk_strategy="gmax"):

        self.logger.info("List of CBs is generated from zone-to-zone PTDFs with:")
        self.logger.info("GSK Strategy: %s, Threshold: %d percent", gsk_strategy, threshold*100)

        gsk = self.create_gsk(gsk_strategy)
        zonal_ptdf = np.dot(self.grid.ptdf, gsk)
        zonal_ptdf_df = pd.DataFrame(index=self.lines.index,
                                     columns=self.results.data.zones.index,
                                     data=zonal_ptdf)

        z2z_ptdf_df = pd.DataFrame(index=self.lines.index)
        for zone in self.flowbased_region:
            for zzone in self.flowbased_region:
                z2z_ptdf_df["-".join([zone, zzone])] = zonal_ptdf_df[zone] - zonal_ptdf_df[zzone]

        critical_branches = list(z2z_ptdf_df.index[np.any(z2z_ptdf_df.abs() > threshold, axis=1)])

        condition_cross_border = self.nodes.zone[self.lines.node_i].values != \
                                 self.nodes.zone[self.lines.node_j].values

        cond_fb_region = self.nodes.zone[self.lines.node_i].isin(self.flowbased_region).values & \
                         self.nodes.zone[self.lines.node_j].isin(self.flowbased_region).values

        cross_border_lines = list(self.lines.index[condition_cross_border&cond_fb_region])
        total_cbs = list(set(critical_branches + cross_border_lines))

        self.logger.info("Number of Critical Branches: %d, \
                         Number of Cross Border lines: %d, \
                         Total Number of CBs: %d",
                         len(critical_branches), len(cross_border_lines), len(total_cbs))

        return total_cbs

    def create_fbmc_info(self, lodf_sensitivity=5e-2):

        """
        create ptdf, determine CBs
        """

        if isinstance(self.cbco_list, pd.DataFrame):
            base_cb = list(self.cbco_list.cb[self.cbco_list.co == "basecase"])

            index_position = [self.lines.index.get_loc(line) for line in base_cb]
            base_ptdf = self.grid.ptdf[index_position, :]
            full_ptdf = [base_ptdf, -base_ptdf]
            label_lines = list(base_cb)+list(base_cb)
            label_outages = ["basecase" for line in label_lines]

            self.cbco_list = self.cbco_list[~(self.cbco_list.co == "basecase")]
            select_lines = []
            select_outages = {}
            for line in self.cbco_list.cb.unique():
                select_lines.append(line)
                select_outages[line] = list(self.cbco_list.co[self.cbco_list.cb == line])

        else:
            self.lines["cb"] = False
            critical_branches = self.return_critical_branches(threshold=5e-2)
            self.lines.loc[self.lines.index.isin(critical_branches), "cb"] = True

            select_lines = self.lines.index[(self.lines["cb"])&(self.lines.contingency)]
            select_outages = {}
            for line in select_lines:
                select_outages[line] = list(self.grid.lodf_filter(line, lodf_sensitivity))

            index_position = [self.lines.index.get_loc(line) for line in select_lines]
            base_ptdf = self.grid.ptdf[index_position, :]
            full_ptdf = [base_ptdf, -base_ptdf]
            label_lines = list(select_lines)+list(select_lines)
            label_outages = ["basecase" for line in label_lines]

        for line in select_lines:
            outages = select_outages[line]
            tmp_ptdf = np.vstack([self.grid.create_n_1_ptdf_cbco(line, out) for out in outages])
            full_ptdf.extend([tmp_ptdf, -tmp_ptdf])
            label_lines.extend([line for i in range(0, 2*len(outages))])
            label_outages.extend(outages*2)

        nodal_fbmc_ptdf = np.concatenate(full_ptdf)
        nodal_fbmc_ptdf = nodal_fbmc_ptdf.reshape(len(label_lines), len(list(self.nodes.index)))

        domain_info = pd.DataFrame(columns=list(self.results.data.zones.index))
        domain_info["cb"] = label_lines
        domain_info["co"] = label_outages

        return nodal_fbmc_ptdf, domain_info

    def create_flowbased_ptdf(self, gsk_strategy, timestep):
        """
        Create Zonal ptdf -> creates both positive and negative line
        restrictions or ram. Depending on flow != 0
        """
        self.logger.info("Creating zonal Ab")
        # Calculate zonal ptdf based on ram -> (if current flow is 0 the
        # zonal ptdf is based on overall
        # avalable line capacity (l_max)), ram is calculated for every n-1
        # ptdf matrix to ensure n-1 security constrained FB Domain
        # The right side of the equation has to be positive

        frm_fav = pd.DataFrame(index=self.domain_info.cb.unique())
        frm_fav["value"] = self.lines.maxflow[frm_fav.index]*0

        injection = self.results.INJ.INJ[self.results.INJ.t == timestep].values


        f_ref_base_case = np.dot(self.nodal_fbmc_ptdf, injection)
        gsk = self.create_gsk(gsk_strategy)
        zonal_fbmc_ptdf = np.dot(self.nodal_fbmc_ptdf, gsk)

        # F Day Ahead (eigentlich mit LTNs)
        net_position = self.results.net_position() * 0.75
        net_position.loc[:, ~net_position.columns.isin(self.flowbased_region)] = 0

        f_da = np.dot(zonal_fbmc_ptdf, net_position.loc[timestep].values)
        # f_ref_nonmarket = f_ref_base_case
        f_ref_nonmarket = f_ref_base_case - f_da

        ram = np.subtract(self.lines.maxflow[self.domain_info.cb] - \
                          frm_fav.value[self.domain_info.cb],
                          f_ref_nonmarket).values

        ram = ram.reshape(len(ram), 1)

        if any(ram < 0):
            self.logger.warning("%d RAMs are <0", sum(ram<0))
            # ram[ram<150] = 10000


        self.domain_info[list(self.results.data.zones.index)] = zonal_fbmc_ptdf
        self.domain_info["ram"] = ram
        self.domain_info["timestep"] = timestep
        self.domain_info["gsk_strategy"] = gsk_strategy

        self.logger.info("Done!")

        # return A, b
        return zonal_fbmc_ptdf, ram

    def create_fbmc_equations(self, domain_x, domain_y, A, b):
        """
        from zonal ptdf calculate linear equations ax = b to plot the FBMC domain
        nodes/Zones that are not part of the 2D FBMC are summerized using GSK sink
        """
        self.logger.info("Creating fbmc equations...")
        list_zones = list(self.nodes.zone.unique())
        if len(domain_x) == 2:
            domain_idx = [[list_zones.index(zone[0]),
                           list_zones.index(zone[1])] for zone in [domain_x, domain_y]]
            A = np.vstack([np.dot(A[:, domain], np.array([1, -1])) for domain in domain_idx]).T
        else:
            self.logger.warning("Domains not set in the right way!")
            raise

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

    def plot_fbmc(self, domain_x, domain_y, gsk_strategy, timestep):
        """
        Combines previous functions to actually plot the FBMC Domain with the
        hull
        """

        A, b = self.create_flowbased_ptdf(gsk_strategy, timestep)
        A, b = self.create_fbmc_equations(domain_x, domain_y, A, b)
        # Reduce
        cbco_index = self.reduce_ptdf(A, b)
        self.domain_info["in_domain"] = False
        self.domain_info.loc[self.domain_info.index.isin(cbco_index), "in_domain"] = True
        self.logger.info("Number of CBCOs %d", len(cbco_index))

        # Limit the number of constraints to 5000
        full_indices = np.array([x for x in range(0,len(A))])
        threshold = 5e3
        if len(A) > threshold:
            self.logger.info("Plot limited to %d constraints plotted", threshold)
            cbco_plot_indices = np.append(cbco_index,
                                          np.random.choice(full_indices,
                                                           size=int(threshold),
                                                           replace=False))
        else:
            cbco_plot_indices = full_indices

        plot_equations = self.create_domain_plot(A, b, cbco_plot_indices)
        hull_plot_x, hull_plot_y, hull_coord_x, hull_coord_y = self.get_xy_hull(A, b, cbco_index)
        xy_limits = tools.find_xy_limits([[hull_plot_x, hull_plot_y]])

        self.logger.info("Number of CBCOs defining the domain %d", len(hull_plot_x) - 1)
        fbmc_plot = FBMCDomain({"gsk_strategy": gsk_strategy,
                                "timestep": timestep,
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
