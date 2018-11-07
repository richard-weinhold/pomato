"""
GRID Model
"""
import sys
import logging
import numpy as np
import pandas as pd
import tables

#import tools as tools
from cbco_module import CBCOModule
#from scipy.spatial import ConvexHull
#from sklearn.decomposition import PCA

class GridModel(object):
    """GRID Model Class"""
    numpy_settings = np.seterr(divide="raise")
    def __init__(self, wdir):
        self.logger = logging.getLogger('Log.MarketModel.GridModel')
#        self.logger.info("Initializing GridModel..")
        self.is_empty = True
        self.wdir = wdir

    def build_grid_model(self, nodes, lines):
        try:
            # import logger
            self.logger.info("Initializing GridModel..")
            self.nodes = nodes
            self.lines = lines
            if not any(self.nodes.slack):
                self.logger.warning("No slack detected, setting first node as slack!")
                self.nodes.slack[0] = True

            self.mult_slack = bool(len(nodes.index[nodes.slack]) > 1)
            if self.mult_slack:
                self.logger.warning("Multiple Slacks Detected!")

            self.logger.info("Calculating PTDF and PSDF Matrices!")
            self.ptdf = self.create_ptdf_matrix()
            self.psdf = self.create_psdf_matrix()
#

            self.check_grid_topology()
            self.logger.info("Calculating LODF and the N-1 Matrices!")
            self.lodf = self.create_lodf_matrix()
            ### Precalc and store all N-1 PTDFs
#            self.n_1_ptdf_path = self.create_and_store_n_1_ptdf()

            self.cbco_index = None
            self.add_cbco = None
            self._cbco_option = "convex_hull" ## or gams
            self.is_empty = False

            self.logger.info("GridModel initialized!")
            self.cbco_module = None

        except:
            self.logger.exception("Error in GridModel!")

    def __getstate__(self):
        """
        Method to remove logger attribute from __dict__
        needed when pickeled
        """
        d = dict(self.__dict__)
        del d["logger"]
        del d["cbco_module"]
        return d

    def __setstate__(self, d):
        """
        Method updates self with modified __dict__ without logger
        needed when pickeled
        """
        self.__dict__.update(d) # I *think* this is a safe way to do it

    def check_grid_topology(self):
        """Checking grid topology for radial nodes and lines"""
        self.logger.info("Checking Grid Topology...")

        radial_nodes = []
        for node in self.nodes.index:
            if len(self.lines[(self.lines.node_i == node)|(self.lines.node_j == node)]) < 2:
                radial_nodes.append(node)

        radial_lines = []
        for idx, line in enumerate(self.lines.index):
            tmp = np.abs(self.ptdf[idx, :])
            tmp = np.around(tmp, decimals=3)
            if 1 in tmp:
                radial_lines.append(line)
            elif self.lines.at["l1", "b"] == 0:
                radial_lines.append(line)

        tmp = self.lines[((self.lines.node_i.isin(radial_nodes))| \
                          (self.lines.node_j.isin(radial_nodes)))& \
                          (self.lines.contingency)]
        if not tmp.empty:
            self.logger.info("Radial nodes are part of the contingency: " + \
                             ", ".join(list(tmp.index)))
            self.lines.contingency[((self.lines.node_i.isin(radial_nodes))| \
                                    (self.lines.node_j.isin(radial_nodes)))& \
                                    (self.lines.contingency)] = False
            self.logger.info("Contingency of radial nodes is set to False")

        tmp = self.lines.contingency[(self.lines.index.isin(radial_lines))& \
                                     (self.lines.contingency)]
        if not tmp.empty:
            self.logger.info("Radial lines are part of the contingency: " + \
                             ", ".join(list(tmp.index)))
            self.lines.contingency[(self.lines.index.isin(radial_lines))& \
                                   (self.lines.contingency)] = False
            self.logger.info("Contingency of radial lines is set to False")

    def loss_of_load(self, list_nodes):
        """
        see if loss of load breaches security domain
        input in the form list_nodes = [ ["n1","n2"], ["n1"], ["n2","n5","n7"]]
        """
        # get slack zones, loss of load is distributed equally in slack zone
        if self.mult_slack:
            slack_zones = self.slack_zones()
        else:
            slack_zones = [list(self.nodes.index)]
        # update injection vector
        for nodes in list_nodes:
            inj = self.nodes.net_injection.copy()
            for node in nodes:
                sz_idx = [x for x in range(0, len(slack_zones)) if node in slack_zones[x]][0]
                inj[inj.index.isin(slack_zones[sz_idx])] += inj[node]/(len(slack_zones[sz_idx])-1)
                inj[node] = 0
            #calculate resulting line flows
            flow = np.dot(self.ptdf, inj)
            f_max = self.lines.maxflow.values
            if self.lines.index[abs(flow) > f_max].empty:
                self.logger.info("The loss of load at nodes: " + ", ".join(nodes) +
                                 "\nDoes NOT cause a security breach!")
            else:
                self.logger.info("The loss of load at nodes: " + ", ".join(nodes) +
                                 "\nCauses a security breach at lines: \n" +
                                 ", ".join(self.lines.index[abs(flow) > f_max]))

    def check_n_1(self):
        """Check N-1 security for injections in self.nodes"""
        overloaded_lines = {}
        n_1_flow = self.n_1_flows()
        for outage in n_1_flow: # Outage
            # compare n-1flow vector with maxflow vector -> bool vector of overloaded lines
            for ov_line in self.lines.index[np.abs(n_1_flow[outage])>self.lines.maxflow.values*1.05]:
                overloaded_lines[len(overloaded_lines)] = {'Line': ov_line, 'Outage': outage,
                                                            'Flow': n_1_flow[outage][self.lines.index.get_loc(ov_line)],
                                                            'maxflow': self.lines.maxflow[ov_line]}
        return overloaded_lines

    def check_n_1_for_marketresult(self, injections, timeslice=None, threshold=3):
        """
        Checks Market Result for N-1 Security, optional timeslice as str,
        optional threshhold for overloaded lines from which further check is cancelled
        injections dataframe from gms method gams_symbol_to_df
        """
        timeslice = timeslice or ['t'+ "{0:0>4}".format(x+1) \
                                  for x in range(0, len(injections.t.unique()))]
        self.logger.info(f"Run N-1 LoadFlow Check from {timeslice[0]} to {timeslice[-1]}")

        all_overloaded_lines = {}
        nr_overloaded_lines = 0
        for i, time in enumerate(timeslice):
            ## Generate Fancy Progressbar
            sys.stdout.write("\r[%-35s] %d%%  - Overloaded Lines in %d Timesteps" % \
                             ('='*int(i*35/len(timeslice)),
                              int(i*101/len(timeslice)), nr_overloaded_lines))
            sys.stdout.flush()
            #Exit if more than 10 N-1 Breaches are detected
            if nr_overloaded_lines > threshold:
                break
                print("\n")
                self.logger.error(f"More than {threshold} N-1 breaches!")

            net_injections = injections[injections.t == time]
            # sort by the same order of nodes as self.nodes.index
            net_injections = net_injections.set_index("n").reindex(self.nodes.index).reset_index()

            self.update_net_injections(net_injections.INJ.values)
            overloaded_lines = self.check_n_1()
            if overloaded_lines != {}:
                all_overloaded_lines[time] = overloaded_lines
                nr_overloaded_lines += 1
        self.logger.info(f"Check Done: {nr_overloaded_lines} Overloaded Lines")

        return all_overloaded_lines

    def grid_representation(self, option, ntc, f_ref=None, precalc_filename=None, add_cbco=None):
        """Bundle all relevant grid information in one dict for the market model"""
        grid_rep = {}
        grid_rep["option"] = option
        grid_rep["mult_slacks"] = self.mult_slack
        grid_rep["slack_zones"] = self.slack_zones()
        if option == "nodal":
            ptdf_dict = {}
            for idx, line in enumerate(self.lines.index):
                ptdf_dict[line + "_pos"] = {"ptdf": list(np.around(self.ptdf[idx,:], decimals=5)), "ram": self.lines.maxflow[line]}
                ptdf_dict[line + "_neg"] = {"ptdf": list(-np.around(self.ptdf[idx,:], decimals=5)), "ram": self.lines.maxflow[line]}
            grid_rep["cbco"] = ptdf_dict
        elif option == "ntc":
            grid_rep["ntc"] = ntc

        elif "cbco" in option.split("_"):
            if not self.cbco_module:
                self.cbco_module = CBCOModule(self.wdir, self)
            if precalc_filename:
                self.cbco_module.main(use_precalc=precalc_filename, only_convex_hull=False)
            else:
                self.cbco_module.main(only_convex_hull=True)

            if add_cbco:
                self.cbco_module.add_to_cbco_index(self.cbco_module.return_index_from_cbco(add_cbco))

            info, cbco = self.cbco_module.return_cbco()
            grid_rep["info"] = info
            grid_rep["cbco"] = cbco

        elif option == "d2cf":
            ptdf_dict = {}
            for line in self.lines.index[self.lines.cb]:
#            for line in self.lines.index:
                idx = self.lines.index.get_loc(line)
                ptdf_dict[line] = {}
                ptdf_dict[line]["ptdf"] = list(np.around(self.ptdf[idx,:], decimals=5))
                ptdf_dict[line]["f_max"] = int(self.lines.maxflow[line])
                if self.lines.cnec[line]:
                    ptdf_dict[line]["f_ref"] = f_ref[line].to_dict()
            grid_rep["cbco"] = ptdf_dict
        return grid_rep

    def slack_zones(self):
        """
        returns number of nodes that are part of a synchronous area
        for each slack defined
        """
        ## Creates Slack zones, given that the given slacks are well suited
        ## Meaning one slack per zone, all zones have a slack.
        # Method: slack -> find Line -> look at ptdf
        # all non-zero elementes are in slack zone.
        slacks = self.nodes.index[self.nodes.slack]
        slack_zones = {}
        for slack in slacks:
            slack_line = self.lines.index[(self.lines.node_i == slack) \
                                          |(self.lines.node_j == slack)][0]
            line_index = self.lines.index.get_loc(slack_line)
            pos = self.ptdf[line_index, :] != 0
            tmp = list(self.nodes.index[pos])
            tmp.append(slack)
            slack_zones[slack] = tmp
        return slack_zones

    def check_n_0(self, flow):
        """ Check N-0 Loadflow"""
        self.logger.info("Run N-0 LoadFlow Check")
        for l_idx, elem in enumerate(flow):
            if elem > self.lines.maxflow[l_idx]*1.01:
                self.logger.info(self.lines.index[l_idx] + ' is above max capacity')
            elif elem < -self.lines.maxflow[l_idx]*1.01:
                self.logger.info(self.lines.index[l_idx] + ' is below min capacity')

    def max_n_1_flow_per_line(self):
        """ returns the maximum n-1 flow per line, preserving flow direction"""
        n_1_flows = self.n_1_flows(option="lines")
        return [n_1_flows[l][np.argmax(np.abs(n_1_flows[l]))] for l in self.lines.index]

    def n_1_flows(self, option="outage"):
        """ returns N-1 Flows, either by outage or by line"""
        # Outate Line -> resulting Flows on all other lines
        injections = self.nodes.net_injection.values
        n_1 = []
        for i, _ in enumerate(self.lines.index):
            n_1_ptdf = self.create_n_1_ptdf_outage(i)
            n_1.append(np.dot(n_1_ptdf, injections))
        n_1_stack = np.vstack(n_1)
        n_1_flows = {}
        if option == "outage":
            for i, outage in enumerate(self.lines.index):
                n_1_flows[outage] = n_1_stack[i, :]
        else:
            for i, line in enumerate(self.lines.index):
                n_1_flows[line] = n_1_stack[:, i]
        return n_1_flows

#    def n_1_flows_multiprocess(list_inj_lines):
#        obj = list_inj_lines[2]
#        injections = list_inj_lines[0]
#        line = list_inj_lines[1]
#        flows_t = []
#        n_1_ptdf = obj.create_n_1_ptdf_outage(line)
#        for t in np.unique(injections.t.values):
#            flows_t.append(np.dot(n_1_ptdf, injections.INJ[injections.t == t].values))
#        return line, flows_t

    def n_1_flows_timeseries(self, injections):
        """ returns the n-1 flows for an injection timeseries (from the market model)
        """
#        injections = INJ
#        injections = injections[injections.t.isin(["t0001", "t0002", "t0003"])]
#        self = mato.grid

#        %%timeit -r 1 -n 1
        flows_t = {t: [] for t in np.unique(injections.t.values)}
        for i in self.lines.index:
            n_1_ptdf = self.create_n_1_ptdf_outage(i)
#            print(i)
            for t in np.unique(injections.t.values):
                flows_t[t].append(np.dot(n_1_ptdf, injections.INJ[injections.t == t].values))


        n_1_flows = pd.DataFrame(index=self.lines.index)
        for t in np.unique(injections.t.values):
#            print(t)
            n_1_flows_tmp = np.vstack(flows_t[t])
            max_values = np.argmax(np.abs(n_1_flows_tmp), axis=0)
            n_1_flows[t] = [n_1_flows_tmp[column, row] for row, column in enumerate(max_values)]

        return n_1_flows
##
#        from multiprocessing.pool import ThreadPool as Pool
#        %%timeit -r 1 -n 1
#        pool = Pool(8)
#        input_par = [[INJ, line, self] for line in self.lines.index]
#        for result in pool.imap_unordered(n_1_flows_multiprocess, input_par):
#            print(result[0])



    def update_flows(self):
        """update flows in self.lines"""
        flows = self.n_0_flows()
        self.lines.flow = flows
        return flows

    def n_0_flows(self):
        """ returns flows, based in the net injection in the nodes attribute"""
        return np.dot(self.ptdf, self.nodes.net_injection.values)

    def n_0_flows_timeseries(self, injections):
        """ returns flows as a dataframe for the timesteps in the injections input"""
        n_0_flows = pd.DataFrame(index=self.lines.index)
        for t in np.unique(injections.t.values):
            n_0_flows[t] = np.dot(self.ptdf, injections.INJ[injections.t == t].values)
        return n_0_flows

    def update_net_injections(self, net_inj):
        """updates net injection in self.nodes from list"""
        try:
            self.nodes.net_injection = net_inj
        except ValueError:
            self.logger.exception("invalid net injections provided")

    def create_incedence_matrix(self):
        """Create incendence matrix"""
        incedence = np.zeros((len(self.lines), len(self.nodes)))
        for i, elem in enumerate(self.lines.index):
            incedence[i, self.nodes.index.get_loc(self.lines.node_i[elem])] = 1
            incedence[i, self.nodes.index.get_loc(self.lines.node_j[elem])] = -1
        return incedence

    def create_susceptance_matrices(self):
        """ Create Line (Bl) and Node (Bn) susceptance matrix """
        suceptance_vector = self.lines.b
        incedence = self.create_incedence_matrix()
        susceptance_diag = np.diag(suceptance_vector)
        line_susceptance = np.dot(susceptance_diag, incedence)
        node_susceptance = np.dot(np.dot(incedence.transpose(1, 0), susceptance_diag), incedence)
        return(line_susceptance, node_susceptance)

    def create_ptdf_matrix(self):
        """ Create ptdf Matrix """
#        self = mato_ieee.grid
        try:
            #Find slack
            slack = list(self.nodes.index[self.nodes.slack])
            slack_idx = [self.nodes.index.get_loc(s) for s in slack]
            line_susceptance, node_susceptance = self.create_susceptance_matrices()
            #Create List without the slack and invert it
            list_wo_slack = [x for x in range(0, len(self.nodes.index)) \
                            if x not in slack_idx]

            node_susceptance_wo_slack = node_susceptance[np.ix_(list_wo_slack, list_wo_slack)]
#            inv = np.linalg.pinv(node_susceptance_wo_slack)
            inv = np.linalg.inv(node_susceptance[np.ix_(list_wo_slack, list_wo_slack)])
            #sort slack back in to get nxn
            node_susc_inv = np.zeros((len(self.nodes), len(self.nodes)))
            node_susc_inv[np.ix_(list_wo_slack, list_wo_slack)] = inv
            #calculate ptdfs
            ptdf = np.dot(line_susceptance, node_susc_inv)
            return ptdf
        except:
            self.logger.exception('error:create_ptdf_matrix')

    def create_psdf_matrix(self):
        """
        Calculate psdf (phase-shifting distribution matrix, LxLL)
        meaning the change of p.u. loadflow
        on a line LL through a phase-shifting by 1rad at line L
        """
        line_susceptance, _ = self.create_susceptance_matrices()
        psdf = np.diag(self.lines.b) - np.dot(self.ptdf, line_susceptance.T)
        return psdf

    def shift_phase_on_line(self, shift_dict):
        """
        Shifts the phase on line l by angle a (in rad)
        Recalculates the ptdf matrix and replaces it as the attribute
        """
        shift = np.zeros(len(self.lines))
        for line in shift_dict:
            shift[self.lines.index.get_loc(line)] = shift_dict[line]
        # recalc the ptdf
        shift_matrix = np.multiply(self.psdf, shift)
        self.ptdf += np.dot(shift_matrix, self.ptdf)
        # subsequently recalc n-1 ptdfs
        self.lodf = self.create_lodf_matrix()
#        self.n_1_ptdf = self.create_n_1_ptdf()

    def create_lodf_matrix_old(self):
        """ Load outage distribution matrix -> Line to line sensitivity """
        try:
            lodf = np.zeros((len(self.lines), len(self.lines)))
            incedence = self.create_incedence_matrix()
            for idx, _ in enumerate(self.lines.index):
                for idy, line in enumerate(self.lines.index):
                    ## Exception for lines that are not in the contingency
                    if line in self.lines.index[~self.lines.contingency]:
                        lodf[idx, idy] = 0
                    elif idx == idy:
                        lodf[idx, idy] = -1
                    else:
                        lodf[idx, idy] = (np.dot(incedence[idy, :], self.ptdf[idx, :]))/ \
                                       (1-np.dot(incedence[idy, :], self.ptdf[idy, :]))
            return lodf
        except:
            self.logger.exception("error in create_lodf_matrix ", sys.exc_info()[0])
            raise ZeroDivisionError('LODFError: Check Slacks, radial Lines/Nodes')

    def create_lodf_matrix(self):
        """ Load outage distribution matrix -> Line to line sensitivity """
        try:
            incedence = self.create_incedence_matrix()
            H = np.dot(self.ptdf, incedence.T)
            h = np.diag(H).reshape(len(self.lines), 1)
            ## Avoid division by zero because of radial nodes and lines
            con = self.lines.contingency.values.astype(int).reshape(len(self.lines), 1)
            h = np.multiply(h,con)
            lodf = np.divide(H, (np.ones((len(self.lines), len(self.lines))) - np.dot(np.ones((len(self.lines), 1)), h.T)))
            lodf = lodf - np.diag(np.diag(lodf)) - np.eye(len(self.lines), len(self.lines))
            # explicitly set line-line sensitivity to 0 for contingency==False
            lodf = np.multiply(lodf, np.ones((len(self.lines),1))*self.lines.contingency.values)
            return lodf
        except:
            self.logger.exception("error in create_lodf_matrix ", sys.exc_info()[0])
            raise ZeroDivisionError('LODFError: Check Slacks, radial Lines/Nodes')

    def create_n_1_ptdf_line(self, line_idx):
        try:
            n_1_ptdf_line =  np.vstack([self.ptdf[line_idx, :]] + \
                                       [self.ptdf[line_idx, :] + np.dot(self.lodf[line_idx, outage], self.ptdf[outage, :]) \
                                        for outage in range(0, len(self.lines))])
            return n_1_ptdf_line
        except:
            self.logger.exception('error:create_n_1_ptdf')

    def create_n_1_ptdf_outage(self, outage):
        """
        Create N-1 ptdfs - For each line add the resulting ptdf to list contingency
        first ptdf is N-0 ptdf
        """
        try:
            if not isinstance(outage, int):
                outage = self.lines.index.get_loc(outage)
            n_1_ptdf = np.array(self.ptdf, dtype=np.float) + np.vstack([np.dot(self.lodf[lx, outage], self.ptdf[outage, :]) for lx in range(0, len(self.lines))])
            return n_1_ptdf
        except:
            self.logger.exception('error:create_n_1_ptdf')


    def create_and_store_n_1_ptdf(self):
        """
        Create and store N-1 ptdfs to save ram- For each line add the resulting ptdf to list contingency
        first ptdf is N-0 ptdf
        """
        dtype = np.dtype("Float16")
        shape = self.ptdf.shape[-1]
        expectedrows = len(self.lines)*(len(self.lines)+1)

        ## Init PyTables
        hdf5_path = self.wdir.joinpath("temp_data/n_1_ptdf.hdf5")
        hdf5_file = tables.open_file(str(hdf5_path), mode='w')
        filters = tables.Filters(complevel=0, complib='zlib')
        n_1_storgae = hdf5_file.create_earray(hdf5_file.root, 'n_1_ptdf',
                                              tables.Atom.from_dtype(dtype),
                                              shape=(0, shape),
                                              filters=filters,
                                              expectedrows=expectedrows)
        for idx, line in enumerate(self.lines.index):
            ptdf = self.create_n_1_ptdf_outage(idx)
            n_1_storgae.append(ptdf)

        hdf5_file.close()
        ## To be clear
        n_1_filepath = str(hdf5_path)
        return n_1_filepath

    def create_all_n_1_ptdf(self):
        """
        Create N-1 ptdfs - For each line add the resulting ptdf to list contingency
        first ptdf is N-0 ptdf
        """
        try:
            contingency = [np.array(self.ptdf, dtype=np.float)] +                                       \
                          [self.ptdf + np.vstack([np.dot(self.lodf[lx, outage], self.ptdf[outage, :])   \
                          for lx in range(0, len(self.lines))])                                         \
                          for outage in range(0, len(self.lines))]
            return contingency
        except:
            self.logger.exception('error:create_n_1_ptdf')

    def create_n_1_ptdf_old(self):
        """
        Create N-1 ptdfs - For each line add the resulting ptdf to list contingency
        first ptdf is N-0 ptdf
        """
        try:
            # add N-0 ptdf
            contingency = [np.array(self.ptdf, dtype=np.float)]
            # add ptdf for every CO -> N-1 ptdf (list with lxn matrices with
            #length l+1 (number of lines plus N-0 ptdf))
            for l0_idx, _ in enumerate(self.lines.index):
                n_1_ptdf = np.zeros((len(self.lines), len(self.nodes)))
                for l1_idx, _ in enumerate(self.lines.index):
                    n_1_ptdf[l1_idx, :] = self.ptdf[l1_idx, :] + \
                                          np.dot(self.lodf[l1_idx, l0_idx],
                                                 self.ptdf[l0_idx, :])
                contingency.append(np.array(n_1_ptdf, dtype=np.float))

            return contingency
        except:
            self.logger.exception('error:create_n_1_ptdf')

    def update_ram(self, ptdf, option="dict"):
        """
        Update ram based on Lineflows from netinjections
        option to return either array or dict
        (array used in cbco to make things faster)
        """
        injections = self.nodes.net_injection.values
        ram = []
        if option == "array":
            for idx, line in enumerate(ptdf):
                pos = self.lines.maxflow[idx] - np.dot(line, injections)
                neg = -self.lines.maxflow[idx] - np.dot(line, injections)
                if pos < 0:
                    ram.append([0.1, neg])
                elif neg > 0:
                    ram.append([pos, 0.1])
                else:
                    ram.append([pos, neg])
            ram = np.asarray(ram)
        else:
            for idx, line in enumerate(ptdf):
                pos = self.lines.maxflow[idx] - np.dot(line, injections)
                neg = -self.lines.maxflow[idx] - np.dot(line, injections)
                if pos < 0:
                    ram.append({'pos': 0.1, 'neg': neg})
                elif neg > 0:
                    ram.append({'pos': pos, 'neg': 0.1})
                else:
                    ram.append({'pos': pos, 'neg': neg})
        return ram

    def slack_zones_index(self):
        """returns the indecies of nodes per slack_zones
        (aka control area/synchronious area) in the A matrix"""
        slack_zones = self.slack_zones()
        slack_zones_idx = []
        for slack in slack_zones:
            slack_zones_idx.append([self.nodes.index.get_loc(node) \
                                    for node in slack_zones[slack]])
        slack_zones_idx.append([x for x in range(0, len(self.nodes))])
        return slack_zones_idx

    def contingency_Ab(self, option, contingency=None):
        """ Bring the N-1 PTDF list in the form of inequalities A x leq b
            where A are the ptdfs, b the ram and x the net injections
            returns lists
        """
        contingency = contingency or self.n_1_ptdf
        if option == 'zonal':
            zonal_contingency = self.create_zonal_ptdf(contingency)
            A = []
            b = []
            for i, equation in enumerate(zonal_contingency):
                ### Check if RAM > 0
                if equation[-1] != 0:
                    A.append(equation[:-1])
                    b.append(equation[-1])
                else:
                    self.logger.debug('zonal:cbco not n-1 secure')
                    A.append(equation[:-1])
                    b.append(1)
        else:
            A = np.vstack((np.vstack([ptdf, -ptdf]) for ptdf in contingency))
            ram_array = self.update_ram(contingency[0], option="array")
            b = np.hstack(np.concatenate([ram_array[:, 0], -ram_array[:, 1]],
                                         axis=0) for i in range(0,len(contingency)))

        return np.array(A, dtype=np.float32), np.array(b, dtype=np.float32)

    def lineloading_timeseries(self, injections, line):
        """
        Plots Line Loading (N-0; N-1) for all timeslices in inj dataframe
        inj dataframe from gms method gams_symbol_to_df
        """
        sys.stdout.write("\n")
        sys.stdout.flush()
        line_loadings = {}

        for i, time in enumerate(injections.t.unique()):
            ## Generate Fancy Progressbar
            sys.stdout.write("\r[%-35s] %d%%" % \
                             ('='*int(i*35/len(injections.t.unique())),
                              int(i*101/len(injections.t.unique()))))
            sys.stdout.flush()
            self.update_net_injections(injections.INJ[injections.t == time].values)
            self.update_flows()
            flow_n0 = abs(self.lines.flow[line])/self.lines.maxflow[line]
            flow_n0_20 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.2
            flow_n0_40 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.4
            flow_n0_60 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.6
            flow_n0_80 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.8
            flow_n0_100 = abs(self.lines.flow[line])/self.lines.maxflow[line]*2
            flow_n1 = self.n_1_flows(option="lines")
            flow_max_n1 = max(abs(flow_n1[line]))/self.lines.maxflow[line]
            line_loadings[time] = {"N-0": flow_n0, "N-1": flow_max_n1,
#                                   "N-1 + 20%": flow_n0_20, "N-1 + 40%": flow_n0_40,
#                                   "N-1 + 60%": flow_n0_60, "N-1 + 80%": flow_n0_80,
#                                   "N-1 + 100%": flow_n0_100
                                   }

        return pd.DataFrame.from_dict(line_loadings, orient="index")
