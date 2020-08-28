import logging
import random
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato

# pylint: disable-msg=E1101
class TestPomatoGrid(unittest.TestCase):
    
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")
        self.options = pomato.tools.default_options()
        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)
        self.data.load_data('data_input/pglib_opf_case118_ieee.m')
        self.grid  = pomato.grid.GridTopology()
        self.grid.calculate_parameters(self.data.nodes, self.data.lines)

    def test_init(self):
        self.assertAlmostEqual(np.shape(self.grid.ptdf), (186, 118))
        self.assertEqual(sum(self.grid.lines.contingency), 177)
        
        self.assertFalse((self.grid.ptdf == np.nan).any())
        self.assertFalse((self.grid.lodf == np.nan).any())

    def nodal_balance(self, flow, inj):
        A = self.grid.create_incidence_matrix()
        nodal_balance = np.dot(flow, A) - inj
        np.testing.assert_allclose(nodal_balance, 0, atol=1e-3)

    def test_line_flow(self):
        # Injection of 100 at node 10
        node_in = self.grid.nodes.index.get_loc("n10")
        node_out = self.grid.nodes.index.get_loc("n112")

        inj = np.zeros((118, ))
        inj[node_in] = 100
        inj[node_out] = -100

        flow = np.dot(self.grid.ptdf, inj)

        # Flow ob line 6 and 8 should be 100
        line8 = self.grid.lines.index.get_loc("l8")
        line6 = self.grid.lines.index.get_loc("l6")
        self.assertAlmostEqual(abs(flow[line8]), 100)
        self.assertAlmostEqual(abs(flow[line6]), 100)

        self.nodal_balance(flow, inj)

    def test_contingency_flow(self):
        node_in = self.grid.nodes.index.get_loc("n10")
        node_out = self.grid.nodes.index.get_loc("n112")

        inj = np.zeros((118, ))
        inj[node_in] = 100
        inj[node_out] = -100
        pre_contingency_flow = np.dot(self.grid.ptdf, inj)
        
        # Test Outage of line with contingency==false
        outage = self.grid.lines.index.get_loc("l8")
        ptdf_l8 = self.grid.create_n_1_ptdf_outage(outage)
        post_contingency_flow = np.dot(ptdf_l8, inj)
        np.testing.assert_array_equal(pre_contingency_flow, post_contingency_flow)

        # Test Outage of line with contingency==true
        # l0 is outaged, which affects l1
        outage = self.grid.lines.index.get_loc("l0")
        affected_line = self.grid.lines.index.get_loc("l1")

        ptdf_l0 = self.grid.create_n_1_ptdf_outage(outage)
        post_contingency_flow = np.dot(ptdf_l0, inj)

        np.testing.assert_equal(post_contingency_flow[outage], 0)
        np.testing.assert_almost_equal(post_contingency_flow[affected_line], 
                                       pre_contingency_flow[affected_line] + pre_contingency_flow[outage])

        self.nodal_balance(post_contingency_flow, inj)

    def test_multiple_contingencies(self):
        node_in1 = self.grid.nodes.index.get_loc("n10")
        node_in2 = self.grid.nodes.index.get_loc("n4")
        node_out = self.grid.nodes.index.get_loc("n112")

        inj = np.zeros((118, ))
        inj[node_in1] = 100
        inj[node_in2] = 100
        inj[node_out] = -200

        pre_contingency_flow = np.dot(self.grid.ptdf, inj)
        self.nodal_balance(pre_contingency_flow, inj)

        ptdf_l10l9 = self.grid.create_n_1_ptdf_outage(["l9", "l10"])      
        post_contingency_flow = np.dot(ptdf_l10l9, inj)
        self.nodal_balance(post_contingency_flow, inj)

        line2 = self.grid.lines.index.get_loc("l2")
        self.assertAlmostEqual(abs(post_contingency_flow[line2]), 100)

        # compare to manual multiple contingencies
        self.grid.ptdf = self.grid.create_n_1_ptdf_outage(9)
        lodf_9 = self.grid.create_n_1_lodf_matrix()
        ptdf_9_10 = self.grid.ptdf + np.dot(lodf_9[:, 10].reshape(len(self.grid.lines), 1), 
                                            self.grid.ptdf[10, :].reshape(1, len(self.grid.nodes)))
                                            
        np.testing.assert_almost_equal(ptdf_l10l9, ptdf_9_10)


    def test_combined_contingencies_sensitivity(self):
        tmp = self.grid.create_contingency_groups(0.4)
        self.assertTrue(all([outage in tmp[outage] for outage in tmp]))

    def test_combined_contingencies(self):
        node_in = self.grid.nodes.index.get_loc("n10")
        node_out = self.grid.nodes.index.get_loc("n112")

        inj = np.zeros((118, ))
        inj[node_in] = 100
        inj[node_out] = -100

        # pre_contingency_flow = np.dot(self.grid.ptdf, inj)
        outages = [outage for outage in self.grid.combined_contingencies if len(self.grid.combined_contingencies[outage]) > 1]
        outage = outages[0]
        outages = self.grid.combined_contingencies[outage]

        c_ptdf = self.grid.create_n_1_ptdf_outage(outage)
        post_contingency_flow = np.dot(c_ptdf, inj)

        for line in outages:
            self.assertAlmostEqual(post_contingency_flow[self.grid.lines.index.get_loc(line)], 0)

if __name__ == '__main__':
    unittest.main()
