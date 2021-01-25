
import datetime as dt
import logging
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy import spatial
import progress
progress.HIDE_CURSOR, progress.SHOW_CURSOR = '', ''
from progress.bar import Bar

import pomato.tools as tools
from pomato.grid import GridModel
from pomato.fbmc import FBMCModule

class FBDomain():
    """Individual FB Domain plot. 

    This class bundles all data of an individual domain plot
    and the plotting functionality. It gets instantiated 
    into the *fbmc_plots* of the :class:`~FBDomainPlots`
    
    Parameters
    ----------
    domain_information : dict 
        Dictionary with general domain information. 
    domain_equations : array
        Equations to be plottet in the domain.
    feasible_region_vertices : array
        Array of the domain's feasible region vertices.
    domain_data : pandas.DataFrame
        The raw data from which the plot is derived.
    ntc : pd.DataFrame
        NTC, which can be optionally included in the domain plot. 
    """ 
    def __init__(self, domain_information, domain_equations, feasible_region_vertices, domain_data, ntc):   
        self.gsk_strategy = domain_information["gsk_strategy"]
        self.timestep = domain_information["timestep"]
        self.domain_x = domain_information["domain_x"]
        self.domain_y = domain_information["domain_y"]
        
        if domain_information["filename_suffix"]:
            self.title = self.timestep + "_" + self.gsk_strategy \
                         + "_" + domain_information["filename_suffix"]
        else:
            self.title = self.timestep + "_" + self.gsk_strategy

        self.domain_equations = domain_equations
        self.feasible_region_vertices = feasible_region_vertices
        self.ntc = ntc.set_index(["zone_i", "zone_j"])

        self.x_max, self.y_max = np.amax(feasible_region_vertices, axis=0)
        self.x_min, self.y_min = np.amin(feasible_region_vertices, axis=0)

        self.domain_data = domain_data
        # set-up: don't show the graphs when created
        plt.ioff()

    def create_fbmc_figure(self, include_ntc):
        """Plot the domain and return the figure.

        Parameters
        ----------
        include_ntc : bool
            Include NTC into FBMC domain.
        """        
        # Setup
        fig, axis = plt.figure(), plt.subplot()
        scale = 2

        title = 'FBMC Domain between: ' + "-".join(self.domain_x) \
                + ' and ' + "-".join(self.domain_y) \
                + '\nNumber of CBCOs: ' + str(len(self.feasible_region_vertices[:, 0]) - 1) \
                + "\nGSK Strategy: " + self.gsk_strategy \
                + " - Timestep: " + self.timestep
        
        n0_lines = []
        n1_lines = []  
        tmp = self.domain_data.reset_index(drop=True)
        for i in tmp[tmp.co == "basecase"].index:
            n0_lines.append([(self.domain_equations[i][0][0], self.domain_equations[i][1][0]),
                             (self.domain_equations[i][0][1], self.domain_equations[i][1][1])])
        for i in tmp[tmp.co != "basecase"].index:
            n1_lines.append([(self.domain_equations[i][0][0], self.domain_equations[i][1][0]),
                             (self.domain_equations[i][0][1], self.domain_equations[i][1][1])])
            
        n1_segments = LineCollection(n1_lines, linewidths=1, alpha=0.6, colors="lightgrey", linestyle='-')
        n0_segments = LineCollection(n0_lines, linewidths=1.5, alpha=1, colors="dimgrey", linestyle='-')
        axis.add_collection(n1_segments)
        axis.add_collection(n0_segments)
        
        axis.plot(self.feasible_region_vertices[:, 0], self.feasible_region_vertices[:, 1],
                  'r--', linewidth=1.5, zorder=3)

        legend = [Line2D([0], [0], color="r", lw=2, ls='--'),
                  Line2D([0], [0], color="dimgrey", lw=2),
                  Line2D([0], [0], color="lightgrey", lw=2)]
        legend_text = ['Flow Based Domain', 'N-0 Constraints', 'N-1 Constraints']
        if include_ntc and not self.ntc.empty:
            # Include NTC as box in the domain plot
            ntc_x_pos = self.ntc.loc[(self.domain_x[0], self.domain_x[1]), "ntc"]
            ntc_x_neg = self.ntc.loc[(self.domain_x[1], self.domain_x[0]), "ntc"]
            ntc_y_pos = self.ntc.loc[(self.domain_y[0], self.domain_y[1]), "ntc"]
            ntc_y_neg = self.ntc.loc[(self.domain_y[1], self.domain_y[0]), "ntc"]
            ntc_plot = [[[ntc_x_pos, ntc_x_pos], [ntc_y_pos, -ntc_y_neg]],
                        [[-ntc_x_neg, -ntc_x_neg], [ntc_y_pos, -ntc_y_neg]],
                        [[ntc_x_pos, -ntc_x_neg], [ntc_y_pos, ntc_y_pos]],
                        [[ntc_x_pos, -ntc_x_neg], [-ntc_y_neg, -ntc_y_neg]]]
            for elem in ntc_plot:
                axis.plot(elem[0], elem[1], c='blue', ls='--')
            
            # expand x min/max, y min/max to accommodate the NTC box
            self.x_min, self.x_max = min(self.x_min, -ntc_x_neg), max(self.x_max, ntc_x_pos)
            self.y_min, self.y_max = min(self.y_min, -ntc_y_neg), max(self.y_max, ntc_y_pos)
            legend.append(Line2D([0], [0], color="b", lw=2, ls='--'))
            legend_text.append("NTC Domain")
        
        axis.set_xlim(self.x_min*scale, self.x_max*scale)
        axis.set_ylim(self.y_min*scale, self.y_max*scale)
        axis.set_title(title)
        axis.legend(legend, legend_text, bbox_to_anchor=(0,-0.1,1,0), loc="upper left",
                                         mode="expand", borderaxespad=0, ncol=3)
        
        plt.tight_layout(rect=[0,0,1,1])
        return fig

    def save_fbmc_domain(self, folder, include_ntc=False):
        """Save fb domain plot to the specified folder. 
        
        Parameters
        ----------
        folder : pathlib.Path
            Destination directory.  
        include_ntc : bool, optional
            Include the NTC box in the domain plot, by default False
        """
        fig = self.create_fbmc_figure(include_ntc=include_ntc) 
        fig.savefig(str(folder.joinpath(f"FBMC_{self.title}.png")))
        fig.clf()


class FBDomainPlots():
    """Create FB domain plots based on flowbased paramters.
    
    This module creates 2D plots of the flow based domain, derived from the 
    FB paramterters created by :meth:`~pomato.fbmc.FBMCModule.create_flowbased_parameters`.

    The FB parameters are zonal PTDF and RAMs for each timestep, the number of zones
    defines the width of the matrix, the length is determined by the number of lines
    defined as cb (critical branch) and co (critical outages). 

    To create a 2D plot, the x and y axis represent commercial exchange between two market areas, 
    thereby the whole system remains balanced for each point in the graph. 
    The methods create the necessary data for each domain plot and stores them as an instance of 
    the :class:`~FBDomain` in the *fbmc_plots* attribute. 
    
    Parameters
    ----------
    wdir : pathlib.Path
        POMATO working directory.  
    data : :class:`~pomato.data.DataManagement`
        Instance of POMATO data management. 
    flowbased_parameters : pd.DataFrame 
        FB parameters, as derived from :class:`~pomato.fbmc.FBMCModule`.
    """  
    def __init__(self, data, flowbased_parameters):
        # Import Logger
        self.logger = logging.getLogger('log.pomato.visualization.FBDomainPlots')
        self.logger.info("Initializing FBDomainPlots....")
        self.data = data
        self.fbmc_plots = {}
        self.flowbased_parameters = flowbased_parameters
        
        # set-up: don't show the graphs when created
        plt.ioff()
        plt.close("all")
        self.logger.info("FBMCModule  Initialized!")

    def save_all_domain_plots(self, folder, include_ntc=False):
        """Saving all the FBMC Plots.

        Parameters
        ----------
        folder : pathlib.Path
            Destination folder.
        include_ntc : bool, optional
            Include the NTC box in the domain plot, by default False
        """        
        self.set_xy_limits_forall_plots()
        plt.close("all")
        self.logger.info("Saving %s FB domains to folder %s", str(len(self.fbmc_plots)), folder)

        bar = Bar('Processing', max=len(self.fbmc_plots), 
                  check_tty=False, hide_cursor=True)
        for plot in self.fbmc_plots:
            self.fbmc_plots[plot].save_fbmc_domain(folder, include_ntc)
            plt.close("all")
            bar.next()
        # bar.finish()

        self.logger.info("Plotting Domains as .gif")    
        plot_types = set(["_".join(plot.split("_")[1:]) for plot in self.fbmc_plots])
        timesteps = sorted(set([plot.split("_")[0] for plot in self.fbmc_plots]))
        gif_name = "_".join([plot.split("_")[-1] for plot in plot_types])
    
        gif_path = str(folder.joinpath(f"{gif_name}.gif"))
        with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
            for t in timesteps:
                filenames = [f"FBMC_{t}_{plot}.png" for plot in plot_types]
                imgs = [imageio.imread(str(folder.joinpath(filename))) for filename in filenames]
                writer.append_data(np.hstack([img for img in imgs]))

    def save_all_domain_info(self, folder, name_suffix=""):
        """Save the all domain info. 
        
        The domain info contains the raw plot data. This method concatenates
        all data of all generatied plots and saves it to the designated folder. 

        Parameters
        ----------
        folder : pathlib.Path
            Destination folder.
        name_suffix : str, optional
            Optionally include a name suffix.
        """
        domain_info = pd.concat([self.fbmc_plots[plot].domain_data for plot in self.fbmc_plots])
        # oder the columns
        columns = ["timestep", "gsk_strategy", "cb", "co"]
        columns.extend(list(self.data.nodes.zone.unique()))
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
        self.logger.info("Resetting x and y limits for all domain plots")
        x_min = min([self.fbmc_plots[plot].x_min for plot in self.fbmc_plots])
        x_max = max([self.fbmc_plots[plot].x_max for plot in self.fbmc_plots])
        y_min = min([self.fbmc_plots[plot].y_min for plot in self.fbmc_plots])
        y_max = max([self.fbmc_plots[plot].y_max for plot in self.fbmc_plots])

        for plots in self.fbmc_plots:
            self.fbmc_plots[plots].x_min = x_min
            self.fbmc_plots[plots].x_max = x_max
            self.fbmc_plots[plots].y_min = y_min
            self.fbmc_plots[plots].y_max = y_max

    def domain_feasible_region(self, A, b):
        """Determining the feasible region of the FB domain, utilizing a convexhull algorithm.
        
        Based on the 2D input argument A and vector v, the convex hull is used to find the linear
        inequations that define the inner feasible region of the domain. 
        
        Returns
        -------
        indices : array
            Indices of A,b that define the domain's feasible region. 

        """
        A = np.array(A, dtype=np.float)
        b = np.array(b, dtype=np.float).reshape(len(b), 1)
        D = A/b
        k = spatial.ConvexHull(D, qhull_options="QJ") #pylint: disable=no-member
        return k.vertices

    def zonal_ptdf_projection(self, domain_x, domain_y, A):
        """The zonal PTDF has to be projected into 2D to be visualized as a domain plot. 

        The input matrix A is projected into 2D. The x,y axis represent commercial exchange 
        between two market areas define in the domain_x and domain_y input arguments, 
        therefore are net zero.  

        """
        list_zones = list(self.data.nodes.zone.unique())
        domain_idx = [[list_zones.index(zone[0]),
                        list_zones.index(zone[1])] for zone in [domain_x, domain_y]]
        A = np.vstack([np.dot(A[:, domain], np.array([1, -1])) for domain in domain_idx]).T
        return A

    def create_domain_plot(self, A, b, indices):
        """Create linear equations of the FB domain. 
        
        Create 2D equation from the 2D projection of the zonal PTDF, suitable to for a 
        matplotlib line plot in the form axis.plot(plot_equations[i][0], plot_equations[i][1])
        for each linear inequation that represents a specific line under contingency.

        The indices represent a subset of equations to be plottet if the size of A is too high. 
        
        Parameters
        ----------
        A : np.array
            Projected zonal PTDF with width 2. 
        b : np.array
            Vector of RAMs
        indices : list-like
            List of indices that compose the domain plot.

        Returns
        -------
        plot_equations : list of [[x1;x2],[y1;y2]]
            Each plot consists of two x and y coordinates.

        """

        A = np.take(np.array(A), indices, axis=0)
        b = np.take(np.array(b), indices, axis=0)
        Ab = np.concatenate((np.array(A), np.array(b).reshape(len(b), 1)), axis=1)

        # Calculate two coordinates for a line plot -> Return X = [X1;X2], Y = [Y1,Y2]
        x_upper = int(max(b)*10)
        x_lower = -x_upper
        plot_equations = []
        for index in range(0, len(Ab)):
            x_coordinates = []
            y_coordinates = []
            # for idx in range(-10000, 10001, 20000):
            for idx in range(x_lower, x_upper +1, (x_upper - x_lower)):
                if Ab[index][1] != 0:
                    y_coordinates.append((Ab[index][2] - idx*(Ab[index][0])) / (Ab[index][1]))
                    x_coordinates.append(idx)
                elif Ab[index][0] != 0:
                    y_coordinates.append(idx)
                    x_coordinates.append((Ab[index][2] - idx*(Ab[index][1])) / (Ab[index][0]))
            plot_equations.append([x_coordinates, y_coordinates])
        return plot_equations

    def create_feasible_region_vertices(self, A, b, feasible_region_indices):
        """Calculate vertices of the FB domain feasible region.

        To plot the feasible region of the domain, this method calculates all intersections 
        of the linear inequalities A x <= b that make up the domain, filters the vertices that 
        are part of the domain (and not intersection more outward) and sorts them clockwise, 
        so that matplotlib can create a continuosly connected domain plot (and not a star shape). 
        This method is pretty high up on the list to be rewritten :/ 

        Parameters
        ----------
        A : np.array
            Projected zonal PTDF with width 2. 
        b : np.array
            Vector of RAMs.
        feasible_region_indices : list
            List of indices of linear inequations that compose the FB domain 
            feasible region.

        Returns
        -------
        vertices : array of width 2 and length of however many constraints make 
            up the domain/feasible region.
        all_intersections : [list x-coordinates, list y-coordinates] 
            Coordinates of all intersections of CBCOs that make up the domain.
        """        
        # ptdf_x * X + ptdf_y * Y = B
        # Or in Matrix Form A*x = b where X = [X;Y]
        ptdf = np.take(A, feasible_region_indices, axis=0)
        ram = np.take(b, feasible_region_indices, axis=0)

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

        vertices_x = []
        vertices_y = []
        ### Filter intersection points for those which define the FB Domain
        for idx in range(0, len(intersection_x)):
            temp = 0
            for idy in range(0, len(ptdf)):
                if ptdf[idy, 0]*intersection_x[idx] +\
                    ptdf[idy, 1]*intersection_y[idx] <= ram[idy]*1.0001:
                    temp += 1
                if temp >= len(ptdf):
                    vertices_x.append(intersection_x[idx])
                    vertices_y.append(intersection_y[idx])

        ### Sort them Counter Clockwise to plot them
        vertices_sorted = []
        for idx in range(0, len(vertices_x)):
            radius = np.sqrt(np.power(vertices_y[idx], 2) + np.power(vertices_x[idx], 2))
            if vertices_x[idx] >= 0 and vertices_y[idx] >= 0:
                vertices_sorted.append([vertices_x[idx], vertices_y[idx],
                                   np.arcsin(vertices_y[idx]/radius)*180/(2*np.pi)])
            elif vertices_x[idx] < 0 and vertices_y[idx] > 0:
                vertices_sorted.append([vertices_x[idx], vertices_y[idx],
                                   180 - np.arcsin(vertices_y[idx]/radius)*180/(2*np.pi)])
            elif vertices_x[idx] <= 0 and vertices_y[idx] <= 0:
                vertices_sorted.append([vertices_x[idx], vertices_y[idx],
                                   180 - np.arcsin(vertices_y[idx]/radius)*180/(2*np.pi)])
            elif vertices_x[idx] > 0 and vertices_y[idx] < 0:
                vertices_sorted.append([vertices_x[idx], vertices_y[idx],
                                   360 + np.arcsin(vertices_y[idx]/radius)*180/(2*np.pi)])
        from operator import itemgetter
        vertices_sorted = sorted(vertices_sorted, key=itemgetter(2))
        
        ## Add first element to draw complete circle
        vertices_sorted.append(vertices_sorted[0])
        vertices_sorted = np.array(vertices_sorted)
        vertices_sorted = np.round(vertices_sorted, decimals=3)
        unique_rows_idx = [x for x in range(0, len(vertices_sorted)-1) \
                           if not np.array_equal(vertices_sorted[x, 0:2], vertices_sorted[x+1, 0:2])]
        unique_rows_idx.append(len(vertices_sorted)-1)
        vertices_sorted = np.take(vertices_sorted, unique_rows_idx, axis=0)
    
        return vertices_sorted[:, [0,1]], [intersection_x, intersection_y]


    def generate_flowbased_domains(self, domain_x, domain_y, filename_suffix=None):
        """Create FB domains for all timesteps of the supplied FB parameters.
        
        This method is based on :meth:`~generate_flowbased_domain`, which create the domain 
        plot for a specific timestep using the same arguments.       
        """
        timesteps = self.flowbased_parameters.timestep.unique()
        bar = Bar('Processing', max=len(timesteps), 
                  check_tty=False, hide_cursor=True)
        for timestep in timesteps:
            self.generate_flowbased_domain(domain_x, domain_y, timestep, filename_suffix)
            bar.next()
        # bar.finish()

    def generate_flowbased_domain(self, domain_x, domain_y, timestep, filename_suffix=None):
        """Create FB Domain for specified zones and timesteps. 
        
        Parameters
        ----------
        domain_x : 2-element, list-like 
            Two-element list-like of market areas whose commercial exchange is depicted on the 
            x-axis, where positive values indicate a commercial exchange from element one to 
            element two. 
        domain_y : 2-element, list-like 
            Analogue to *domain_x*, just for the y-axis of the 2 dimensional plot.
        timestep : string, 
            Timestep for which the domain is generated. 
        filename_suffix : string, optional
            Optionally append to the resulting filename a suffix that makes it easier to 
            identify when domains for more scenarios are created, by default None.
        """
        
        domain_info = self.flowbased_parameters.loc[self.flowbased_parameters.timestep == timestep]
        domain_info = domain_info[~(domain_info[self.data.zones.index] == 0).all(axis=1)].reset_index()

        A = domain_info.loc[:, list(self.data.zones.index)].values
        b = domain_info.loc[:, "ram"].values

        # Checks 
        if not len(domain_x) == len(domain_y) == 2:
            raise AttributeError("Attributes domain_x, domain_y must have 2 elements")
        if not all(b >= 0):
            raise ValueError("Not all RAM >= 0, check FB paramters.")
        if not isinstance(self.flowbased_parameters, pd.DataFrame):
            raise AttributeError("No precalculated flow based parameters available, run create_flowbased_parameters with basecase and GSK")
        if not len(self.flowbased_parameters[self.flowbased_parameters.timestep == timestep].gsk_strategy.unique()) == 1:
            raise AttributeError("Multiple GSK Strategies in flow based parameters, slice first!")
        else:
            gsk_strategy = self.flowbased_parameters[self.flowbased_parameters.timestep == timestep].gsk_strategy.unique()[0]

        A = self.zonal_ptdf_projection(domain_x, domain_y, A)
        feasible_region_indices = self.domain_feasible_region(A, b)
        domain_info["in_domain"] = False
        domain_info.loc[domain_info.index.isin(feasible_region_indices), "in_domain"] = True

        # Limit the number of constraints plottet to a threshold
        threshold = int(1e3)
        if len(A) > threshold:
            self.logger.debug("Plot limited to %d constraints plotted", threshold)
            random_choice = np.random.choice(domain_info.index, size=threshold, replace=False)
            n_0_indices = domain_info.index[domain_info.co == "basecase"].values
            plot_indices = np.sort(np.unique(np.hstack([feasible_region_indices, random_choice, n_0_indices])))
            domain_info = domain_info.loc[plot_indices, :]
        else:
            plot_indices = domain_info.index

        plot_equations = self.create_domain_plot(A, b, plot_indices)
        feasible_region_vertices, _ = self.create_feasible_region_vertices(A, b, feasible_region_indices)
        
        self.logger.debug("Number of CBCOs defining the domain %d", len(feasible_region_vertices[:, 0]) - 1)

        plot_information = {"gsk_strategy": gsk_strategy, "timestep": timestep,
                            "domain_x": domain_x, "domain_y": domain_y,
                            "filename_suffix": filename_suffix}
        
        fbmc_plot = FBDomain(plot_information, plot_equations, feasible_region_vertices,
                               domain_info.copy(), self.data.ntc)

        self.fbmc_plots[fbmc_plot.title] = fbmc_plot
