"""The Grid Model of POMATO

These two modules provide the grid functionality to POMATO. And is initialied as
an attribute to the POMATO instance when data is (successfully) loaded.

The functionality of the GridTopology include:

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


The purpose of the GridModel is to create a usable grid representation for
the market model. This module acts as a combinator of the data and grid modules
and allow to easily change grid representation for the market model.

There are currently the following options:
    * dispatch : Not grid representation at all, meaning that the whole market
      is cleared with uniform pricing i.e. as a copper plate.
    * ntc : Constraining zonal exchange with net transfer capacities. These are
      should be part of the input data.
    * nodal : Nodal pricing through linear power flow inform of the ptdf
      matrix. This is also denoted as optimal power flow (OPF), where each
      nodal injection is constrained by all lines in the network.
    * zonal : Representation through a zonal ptdf. This implementation uses
      the nodal ptdf matrix and an assumptions of how the zonal net position is
      distributed across all nodes. This assumptions is often denoted as
      node-to-zone mapping, participation actor or generation shift key (gsk).

      There are currently two options available:
          - gmax : It is assumed that nodes participate in the net position
            proportional to the conventional capacity installed.
          - flat : all nodes participate equally.


Additionally, this model can provide two N-1 grid representations which allow
to preemptively represent the N-1 criterion, this is the main feature and
the main reason why its organized in a module.

The two options are:
    - cbco_nodal :  A nodal representation including contingencies. This is
      often denoted as security constrained optimal power flow (SCOPF). This
      setting comes with multiple methods that enable pomato to find the
      smallest set of constraints to guaranty SCOPF. These are scribed in the
      methods of :class:`~pomato.grid.GridTopology` and in the corresponding
      paper in the *See Also* section below.
    - cbco_zonal : Creating a zonal version of the security constrained
      nodal representation, similar to the zonal options. Analog to the
      zonal option, one of the two available gsks is chosen.

The grid representation with contingencies come with additional methods that
filter and reduce the number of constraints.

There are multiple options available:
    - full : no constraints are removed. This leads to a very large ptdf matrix
      and is generally not recommended. For example the IEEE 118 bus case
      study includes 186 branches, therefore the N-1 ptdf matrix would
      consists of a 34.596 x 118 ptdf matrix. So even a very small network
      will cause significant number of constraints.
    - clarkson/clarkson_base : Will reduce the number constraints significantly.
      Takes a fairly long time for large networks (a few hours for the 995
      line DE case study, with gurobi).

The options which grid representation is generated is ["optimization"]["type"]
and option for the reduction level is under ["grid"]["redundancy_removal_option"].

Corresponding paper to reduction algorithm used in POMATO: `Fast
Security-Constrained Optimal Power Flow through Low-Impact and Redundancy
Screening <https://arxiv.org/abs/1910.09034>`_.

"""

from pomato.grid.grid_topology import GridTopology
from pomato.grid.grid_model import GridModel