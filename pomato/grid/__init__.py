"""The Grid Model of POMATO

This module provides the grid functionality to POMATO. And is initialied as
an attribute to the POMATO instance when data is (sucessfully) loaded.

This includes:
    - Calculation of the ptdf matrix (power transfer distribution factor),
      used in linear power flow analysis.
    - Calculation of the lodf matrix (load outage distribution factor), to
      account for contingencies.
    - Validation of the input topology based on the nodes and lines data. This
      includes verificatiion of the chosen slacks/reference nodes and setting
      of multiple slacks if needed. Also providing the inforamtion about which
      nodes should be balanced through which slack.
    - Validation of possible contingencies. This means that lines that disconnect
      nodes or groups of nodes cannot be considered as a contingency.
    - A selection of methods that allow for contingency analsys by obtaining
      N-1 ptdf matrices by lines, outages or with a sensitivity filter.
"""

from pomato.grid.grid_model import GridModel