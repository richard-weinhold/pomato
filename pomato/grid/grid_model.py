"""Grid Model of POMATO"""
import sys
import logging
import numpy as np
import pandas as pd

class GridModel():
    """Grid Model of POMATO

    This module provides the grid functionality to POMATO. And is initialized as
    an attribute to the POMATO instance when data is (successfully) loaded.

    This includes:
        - Calculation of the ptdf matrix (power transfer distribution factor),
          used in linear power flow analysis.
        - Calculation of the lodf matrix (load outage distribution factor), to
          account for contingencies.
        - Validation of the input topology based on the nodes and lines data. This
          includes verification of the chosen slacks/reference nodes and setting
          of multiple slacks if needed. Also providing the information about which
          nodes should be balanced through which slack.
        - Validation of possible contingencies. This means that lines that disconnect
          nodes or groups of nodes cannot be considered as a contingency.
        - A selection of methods that allow for contingency analysis by obtaining
          N-1 ptdf matrices by lines, outages or with a sensitivity filter.

    This module is initialized solely with nodes and lines data. It purposely
    does not contain additional data or results, as analysis tasks line power
    flow calculation are done in respective modules
    :obj:`~pomato.data.ResultProcessing`, :obj:`~pomato.cbco.CBCOModule` or
    :obj:`~pomato.fbmc.FBMCModule`. This module solely provides the tools for
    this analysis and the corresponding means of validation to provide a robust
    expirience. The main functionality of contingency analysis is embeded in
    :meth:`~create_filtered_n_1_ptdf` which reappears in similar form in the
    other contingency related as well.

    The initialization does the following:
        - set nodes/lines as attributes.
        - check if slacks are set.
        - calculate ptdf (and psdf) matrix.
        - check if topology contains radial lines/nodes, remove from contingency.
        - calculate lodf matrix.


    Parameters
    ----------
    nodes : DataFrame
        Nodes data, initialized as attribute of GridModel.
    lines : DataFrame
        Lines data, initialized as attribute of GridModel.

    Attributes
    ----------
    nodes : DataFrame
        Nodes table, which .
    lines : DataFrame
        Lines table from data.
    ptdf : np.ndarray
        ptdf (power transfer distribution factor) matrix :math:`(L \\times N)`.
    psdf : np.ndarray
        psdf (phase shifting distribution factor) matrix :math:`(L \\times L)`.
    lodf : np.ndarray
        lodf (load outage distribution factor) matrix :math:`(L \\times L)`.
    """

    numpy_settings = np.seterr(divide="raise")

    def __init__(self, nodes, lines):
        self.logger = logging.getLogger('Log.MarketModel.GridModel')
        self.logger.info("Initializing GridModel..")

        self.nodes = nodes
        self.lines = lines
        self.check_slack()
        self.logger.info("Calculating PTDF and PSDF Matrices!")
        self.ptdf = self.create_ptdf_matrix()
        self.psdf = self.create_psdf_matrix()
        self.check_grid_topology()
        self.logger.info("Calculating LODF Matrix!")
        self.lodf = self.create_lodf_matrix()
        self.logger.info("GridModel initialized!")

    def check_slack(self):
        """Check slack configuration from input data.

        By checking:
            - If no slack is set, first node is set as slack
            - Unconnected nodes as additional slacks
            - Logs info if > 1 slacks in data

        """
        if not any(self.nodes.slack):
            self.logger.warning("No slack detected, setting first node as slack!")
            self.nodes.loc[self.nodes.index[0], "slack"] = True

        condition = ~(self.nodes.index.isin(self.lines.node_i) |
                      self.nodes.index.isin(self.lines.node_j))
        if any(condition):
            self.logger.warning(f"{sum(condition)} unconnected nodes detected, "
                                "set as slacks!")
        self.nodes.loc[condition, "slack"] = True

        self.mult_slack = bool(len(self.nodes.index[self.nodes.slack]) > 1)
        if self.mult_slack:
            self.logger.info("Multiple Slacks Detected!")

    def check_grid_topology(self):
        """Check grid topology for radial nodes and lines.

        Based on the topology, implicitly available in the ptdf matrix, this
        method find radial lines/nodes and sets the contingency attribute
        of the grid.lines accordingly.

        If a radial line, or a line connecting a radial node, outs it disconnects
        the network. This means, that the disconnected node(s) cannot be balanced
        though the slack. Mathematically this causes a division by zero in the
        lodf matrix calculation.
        """
        self.logger.info("Checking Grid Topology...")

        radial_nodes = []
        for node in self.nodes.index:
            if len(self.lines[(self.lines.node_i == node) | (self.lines.node_j == node)]) < 2:
                radial_nodes.append(node)

        radial_lines = []
        for idx, line in enumerate(self.lines.index):
            tmp = np.abs(self.ptdf[idx, :])
            tmp = np.around(tmp, decimals=3)
            if 1 in tmp:
                radial_lines.append(line)
#           elif self.lines.at["l1", "b"] == 0:
#               radial_lines.append(line)

        condition = (self.lines.node_i.isin(radial_nodes)) | \
                    (self.lines.node_j.isin(radial_nodes)) & self.lines.contingency

        if not self.lines[condition].empty:
            self.logger.info("Radial nodes are set as contingency:")
            self.lines.loc[condition, "contingency"] = False
            self.logger.info("Contingency of %d lines is set to false", len(self.lines.index[condition]))

        condition = self.lines.index.isin(radial_lines) & self.lines.contingency
        if not self.lines.contingency[condition].empty:
            self.logger.info("Radial lines are set as contingency:")
            self.lines.loc[condition, "contingency"] = False
            self.logger.info("Contingency of %d lines is set to false", len(self.lines.index[condition]))

        self.logger.info("Total number of Lines: %d", len(self.lines.index))
        self.logger.info("Total number of contingencies: %d", len(self.lines[self.lines.contingency]))

    def slack_zones(self):
        """Return nodes that are balanced through each slack.

        In other words, if there are multiple disconnected networks, this
        returns the informations which node is balanced through which slack.

        Returns
        -------
            slack_zones : dict(slack, list(node.index))
                Dictionary containing slack as keys and list of the nodes
                indices that are balanced though the slack.
        """
        # Creates Slack zones, given that the given slacks are well suited
        # Meaning one slack per zone, all zones have a slack.
        # Method: slack -> find Line -> look at ptdf
        # all non-zero elements are in slack zone.
        slacks = self.nodes.index[self.nodes.slack]
        slack_zones = {}
        for slack in slacks:
            condition = (self.lines.node_i == slack) | (self.lines.node_j == slack)
            if self.lines.index[condition].empty:
                slack_zones[slack] = [slack]
            else:
                slack_line = self.lines.index[condition][0]
                line_index = self.lines.index.get_loc(slack_line)
                pos = self.ptdf[line_index, :] != 0
                tmp = list(self.nodes.index[pos])
                tmp.append(slack)
                slack_zones[slack] = tmp

        return slack_zones

    def create_incidence_matrix(self):
        """Create incidence matrix from *lines* and *nodes* attributes.

        Returns
        -------
        incidence : np.ndarray
            Incidence matrix.
        """
        incidence = np.zeros((len(self.lines), len(self.nodes)))
        for i, elem in enumerate(self.lines.index):
            incidence[i, self.nodes.index.get_loc(self.lines.node_i[elem])] = 1
            incidence[i, self.nodes.index.get_loc(self.lines.node_j[elem])] = -1
        return incidence

    def create_susceptance_matrices(self):
        """Create Line (Bl) and Node (Bn) susceptance matrix.

        Returns
        -------
        line_susceptance : np.ndarray
            Line susceptance matrix.
        node_susceptance : np.ndarray
            Node susceptance matrix.
        """
        suceptance_vector = self.lines.b
        incidence = self.create_incidence_matrix()
        susceptance_diag = np.diag(suceptance_vector)
        line_susceptance = np.dot(susceptance_diag, incidence)
        node_susceptance = np.dot(np.dot(incidence.transpose(1, 0), susceptance_diag), incidence)
        return line_susceptance, node_susceptance

    def create_ptdf_matrix(self):
        """Create ptdf matrix.

        The ptdf matrix represents a note to line sensitivity, essentially a
        linear mapping of how nodal power injections distribute within the
        network.

        The ptdf matrix is calculated based on the topology and line parameters
        (i.e. line susceptance).
        """
        try:
            # Find slack
            slack = list(self.nodes.index[self.nodes.slack])
            slack_idx = [self.nodes.index.get_loc(s) for s in slack]
            line_susceptance, node_susceptance = self.create_susceptance_matrices()
            # Create List without the slack and invert it
            list_wo_slack = [x for x in range(0, len(self.nodes.index)) if x not in slack_idx]

            node_susceptance_wo_slack = node_susceptance[np.ix_(list_wo_slack, list_wo_slack)]
            inv = np.linalg.inv(node_susceptance_wo_slack)
            # sort slack back in to get nxn
            node_susc_inv = np.zeros((len(self.nodes), len(self.nodes)))
            node_susc_inv[np.ix_(list_wo_slack, list_wo_slack)] = inv
            # calculate ptdfs
            ptdf = np.dot(line_susceptance, node_susc_inv)
            return ptdf
        except:
            self.logger.exception('error:create_ptdf_matrix')

    def create_psdf_matrix(self):
        """Calculate psdf (phase-shifting distribution matrix, LxLL).

        A psdf at position (l,ll) represents the change in power flow on
        line ll caused by a phase-shift of 1 [rad] on line l.

        """
        line_susceptance, _ = self.create_susceptance_matrices()
        psdf = np.diag(self.lines.b) - np.dot(self.ptdf, line_susceptance.T)
        return psdf

    def shift_phase_on_line(self, phase_shift):
        """Shifts the phase on line l by angle a (in rad).

        Recalculates the ptdf matrix. This is a static representation of a
        phase shift rather than a dynamic (and useful one).

        Parameters
        ----------
        phase_shift : dict
            dict with line as key, phase shift [rad] as value.
        """
        shift = np.zeros(len(self.lines))
        for line in phase_shift:
            shift[self.lines.index.get_loc(line)] = phase_shift[line]
        # recalc ptdf and lodf
        shift_matrix = np.multiply(self.psdf, shift)
        self.ptdf += np.dot(shift_matrix, self.ptdf)
        self.lodf = self.create_lodf_matrix()

    def create_lodf_matrix(self):
        """Create ptdf matrix.

        The lodf matrix represents a line to line sensitivity in the case of
        an outage. In other words, how does the flow on a line distribute on
        other lines in the case of an outage.

        The ZeroDivisionError is important to catch as it indicates that the
        slack is not chosen correctly.
        """
        try:
            incidence = self.create_incidence_matrix()
            H = np.dot(self.ptdf, incidence.T)
            h = np.diag(H).reshape(len(self.lines), 1)
            # Avoid division by zero because of radial nodes and lines
            con = self.lines.contingency.values.astype(int).reshape(len(self.lines), 1)
            h = np.multiply(h, con)
            lodf = np.divide(H,
                    (np.ones((len(self.lines), len(self.lines))) \
                        - np.dot(np.ones((len(self.lines), 1)), h.T)))

            lodf = lodf - np.diag(np.diag(lodf)) - np.eye(len(self.lines), len(self.lines))
            # explicitly set line-line sensitivity to 0 for contingency==False
            lodf = np.multiply(lodf, np.ones((len(self.lines), 1))*self.lines.contingency.values)
            return lodf

        except:
            self.logger.exception("error in create_lodf_matrix ", sys.exc_info()[0])
            raise ZeroDivisionError('LODFError: Check Slacks, radial Lines/Nodes')

    def lodf_filter(self, line, sensitivity=5e-2, as_index=False):
        """Return outages that impact the specified line with more that the specified sensitivity.

        Contingency analysis relies on making sure line capacities are not
        violated in the case of an outage. This methods returns the lines that,
        in the worst case of outage, impact the specified line more than the
        specified sensitivity.

        For example this methods returns:
            - for a line l, and sensitivity of 5%
            - a list of all outages
            - which impact line l with more that 5% of its capacity
            - in the worst case outage, i.e. when fully loaded.

        Parameters
        ----------
        line : lines.index, int
            line index (DataFrame index or integer).
        sensitivity : float, optional
            The sensitivity defines the threshold
        as_index : bool, optional
            Bool to indicate whether to return int index.

        Returns
        -------
        outages : list
            list of outages that have a significant (sensitivity) impact
            on line in case of the worst outage.
        """
        if not isinstance(line, int):
            line = self.lines.index.get_loc(line)

        cond = abs(np.multiply(self.lodf[line], self.lines.maxflow.values)) >= \
            sensitivity*self.lines.maxflow[line]

        if as_index:
            return [self.lines.index.get_loc(line) for line in self.lines.index[cond]]
        else:
            return self.lines.index[cond]

    def create_n_1_ptdf_line(self, line):
        """Create N-1 ptdf for one specific line and all other lines as outages.

        Parameters
        ----------
        line : lines.index, int
            line index (DataFrame index or integer).

        Returns
        -------
        ptdf : np.ndarray
            Returns ptdf matrix (LxN) where for a N-dim vector of nodal
            injections INJ, the dot-product :math:`PTDF \\cdot INJ` results
            in the flows on the specified line for each other line as outages.
        """
        try:
            if not isinstance(line, int):
                line = self.lines.index.get_loc(line)

            n_1_ptdf_line = np.vstack([self.ptdf[line, :]] +
                                      [self.ptdf[line, :] +
                                       np.dot(self.lodf[line, outage], self.ptdf[outage, :])
                                       for outage in range(0, len(self.lines))])
            return n_1_ptdf_line
        except:
            self.logger.exception('error:create_n_1_ptdf_cb')

    def create_n_1_ptdf_outage(self, outage):
        """Create N-1 ptdf for all lines under a specific outage..

        Parameters
        ----------
        outage : lines.index, int
            line index (DataFrame index or integer).

        Returns
        -------
        ptdf : np.ndarray
            Returns ptdf matrix (LxN) where for a N-dim vector of nodal
            injections INJ, the dot-product :math:`PTDF \\cdot INJ`
            results in the flows on each line under the specified outage.
        """
        try:
            if not isinstance(outage, int):
                outage = self.lines.index.get_loc(outage)
            n_1_ptdf = np.array(self.ptdf, dtype=np.float) + \
                np.vstack([np.dot(self.lodf[lx, outage], self.ptdf[outage, :])
                           for lx in range(0, len(self.lines))])
            return n_1_ptdf

        except:
            self.logger.exception('error:create_n_1_ptdf_co')

    def create_n_1_ptdf_cbco(self, line, outage):
        """Create N-1 ptdf for one specific line and one specific outage.

        Parameters
        ----------
        line : lines.index, int
            Line index (DataFrame index or integer).
        outage : lines.index, int
            Line index (DataFrame index or integer).

        Returns
        -------
        ptdf : np.ndarray
            Returns ptdf matrix (1xN) where for a N-dim vector of nodal
            injections INJ, the dot-product :math:`PTDF \\cdot INJ` results
            in the flow on the specified line under the specified outage.
        """
        try:
            if not isinstance(outage, int):
                outage = self.lines.index.get_loc(outage)
            if not isinstance(line, int):
                line = self.lines.index.get_loc(line)

            n_1_ptdf_cbco = self.ptdf[line, :] + np.dot(self.lodf[line, outage],
                                                        self.ptdf[outage, :])
            return n_1_ptdf_cbco
        except:
            self.logger.exception('error:create_n_1_ptdf_cbco')

    def create_filtered_n_1_ptdf(self, sensitivity=5e-2):
        """Create a N-1 ptdf/info containing all lines under outages with significant impact.

        Create a ptdf that covers the N-0 ptdf (with the outage indicated as
        *basecase* in the return DataFrame) and additionally for each line
        the ptdf that considers outages with significant impact based on
        the method :meth:`~lodf_filter`.
        The methods returns a DataFrame with the resulting ptdf matrix including
        information which lines/outages make up each row.

        This methodology is extremely helpful for contingency analysis where
        the resulting ptdf matrix, and therefore the resulting optimization
        problem, becomes prohibitive large. We have shown that even a
        sensitivity of 1% reduces the size of the resulting ptdf matrix by 95%.

        See `Fast Security-Constrained Optimal Power Flow through Low-Impact
        and Redundancy Screening <https://arxiv.org/abs/1910.09034>`_ for more
        detailed information.

        Parameters
        ----------
        sensitivity : float, optional
            The sensitivity defines the threshold from which outages are
            considered critical. A outage that can impact the lineflow,
            relative to its maximum capacity, more than the sensitivity is
            considered critical.

        Returns
        -------
        ptdf : DataFrame
            Returns DataFrame, each row represents a line (cb, critical branch)
            under an outage (co, critical outage) with ptdf for each node and
            the available capacity (ram, remaining available margin) which is
            equal to the line capacity (but does not have to).
        """
        try:
            A = [self.ptdf]
            label_lines = list(self.lines.index)
            label_outages = ["basecase" for i in range(0, len(self.lines.index))]

            for idx, line in enumerate(self.lines.index[self.lines.contingency]):
                outages = list(self.lodf_filter(line, sensitivity))
                label_lines.extend([line for i in range(0, len(outages))])
                label_outages.extend(outages)

            # estimate size of array = nr_elements * bits per element (float32) / (8 * 1e6) MB
            estimate_size = len(label_lines)*len(self.nodes.index)*32/(8*1e6)
            self.logger.info(f"Estimated size in RAM for A is: {estimate_size} MB")
            if estimate_size > 3000:
                raise

            for idx, line in enumerate(self.lines.index[self.lines.contingency]):
                outages = list(self.lodf_filter(line, sensitivity))
                tmp_ptdf = np.vstack([self.create_n_1_ptdf_cbco(line, o) for o in outages])
                A.append(tmp_ptdf)

            A = np.concatenate(A).reshape(len(label_lines), len(list(self.nodes.index)))
            b = self.lines.maxflow[label_lines].values.reshape(len(label_lines), 1)

            df_info = pd.DataFrame(columns=list(self.nodes.index), data=A)
            df_info["cb"] = label_lines
            df_info["co"] = label_outages
            df_info["ram"] = b
            df_info = df_info[["cb", "co", "ram"] + list(list(self.nodes.index))]
            return A, b, df_info
        except:
            self.logger.exception('error:create_n_1_ptdf')

    def slack_zones_index(self):
        """Return the integer indices for each node per slack_zones."""
        slack_zones = self.slack_zones()
        slack_zones_idx = []
        for slack in slack_zones:
            slack_zones_idx.append([self.nodes.index.get_loc(node)
                                    for node in slack_zones[slack]])
        slack_zones_idx.append([x for x in range(0, len(self.nodes))])
        return slack_zones_idx
