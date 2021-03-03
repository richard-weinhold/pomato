
.. _line_parameters:

Line Parameters for linear power flow
-------------------------------------
Power flow calculations rely on many parameters, related to the existing infrastructure of lines
(branches), nodes (buses) and transformers. Detailed information is scarce and especially in the
composition of open energy models, it is required to abstract and interpolate the existing data to
guess the electric properties as accurate as needed. While for applications in electrical
engineering, almost all parameters are required and there is almost no way around detailed data
collection, for linearized power flow calculations, like in pomato, only a subset of electric
properties are relevant for the calculations. This allows to extrapolate the needed data with a
higher degree of freedom, which does not mean the derivation process shouldn't be rooted on a solid
electrical engineering foundation but allows to work with substantial larger, automatically
generated, datasets that are not available otherwise. 

In the following sections we document the process for the data set used in POMATO for the purpose of
transparency and to iron out mistakes or inaccuracies along the way. 

Original Data
*************

The data originates from the `ENTSO-E Gridmap <https://www.entsoe.eu/data/map/>`_ which was
scraped using the GridKit utility [2]_ and is currently forked and maintained
within the `PyPSA Project <www.github.com/PyPSA/GridKit>`_.

The data comes with very limited information, as it is solely based on geographic information. The
following data is included: 

+--------------+------------------------------------------------------------------------+
| nodes        | node_id, station_id, voltage, dc, terminal, symbol, position (let/lon) |
+--------------+------------------------------------------------------------------------+
| lines        | line_id, node_i, node_j, voltage, circuits, length, underground        |
+--------------+------------------------------------------------------------------------+
| dclines      | dcline_id, node_i, node_i, length, underground                         |
+--------------+------------------------------------------------------------------------+
| transformers | tranformer_id, node_i, node_i                                          |
+--------------+------------------------------------------------------------------------+

Stations can consist of multiple nodes (buses), and will ultimately be omitted. Transformers are not
actually part of the dataset, but extrapolated based on the substations and their nodes. 

Some statistics regarding the data: 

.. table:: GridKit Data Summary


    +--------------+---------+-------+------+-----+
    |              |         | Total | CWE  | DE  |
    +--------------+---------+-------+------+-----+
    | nodes        |         | 8011  | 1612 | 604 |
    +--------------+---------+-------+------+-----+
    | lines        | voltage | 9856  | 1875 | 663 |
    +--------------+---------+-------+------+-----+
    |              | 132     |       | 10   | 9   |
    +--------------+---------+-------+------+-----+
    |              | 220     |       | 1076 | 211 |
    +--------------+---------+-------+------+-----+
    |              | 380     |       | 789  | 443 |
    +--------------+---------+-------+------+-----+
    | transformers |         | 1061  | 213  | 96  |
    +--------------+---------+-------+------+-----+
    | dclines      |         | 51    | 19   | 13  |
    +--------------+---------+-------+------+-----+


Required Data
*************

Pomato remains in the realm of linear programming, therefore the only parameter that is required is
the imaginary part of the complex impedance :math:`Z = R + jX` of a network element, the reactance $X$
which relates approximately to the susceptance :math:`B \approx \frac{1}{X}` which is the only remaining
parameter of the linear power equations.

This parameter is dependant on multiple factors, e.g. materials of the line, configuration of
circuits, voltage level and many more. While it is possible to research these properties for a
individual line, the effort is unreasonable for an entire system and given the purpose of a techno
economic analysis. The next sections document the process of using default parameters
to find reasonable assumptions for impedances of network elements.  

Lines
*****

For transmission lines [1]_ provides a detailed overview of how parameters
manifest depending on material- and construction properties. The level of detail is however far too
great to be useful in this application as we hardly know anything about the transmission lines. 

Table 9.7 however provides parameters for voltage levels 110KV to 380kV in single and double line
configurations for the commonly used overhead line tower (Donaumast). This table combines many
assumptions that are difficult to make: type of ropes used (German ``Beseilung``), mounting and bundle
configuration. The following table contains the parameters. Note: for consistency always the single
circuit values are used. The ropes are assumed to be Al/St-Ropes of type 243-A1/39-St1A with a
nominal current :math:`I_{nom}` of 645 as per Table 9.2 in [1]_.

.. table:: Exemplary overhead line parameters, from Table 9.2 in [1]_. 

    +---------+---------------+---------------------------------------------------------+-----------------+
    | Voltage | Configuration | Positive Sequence Impedance                             | :math:`I_{nom}` |
    +---------+---------------+----------------------------+----------------------------+-----------------+
    | kV      |               | r_per_km :math:`\Omega`/km | x_per_km :math:`\Omega`/km | A               |
    +---------+---------------+----------------------------+----------------------------+-----------------+
    | 110     | Single Rope   | 0.12                       | 0.387                      | 645             |
    +---------+---------------+----------------------------+----------------------------+-----------------+
    | 220     | 2-Bundle      | 0.06                       | 0.301                      | 1290            |
    +---------+---------------+----------------------------+----------------------------+-----------------+
    | 380     | 4-Bundle      | 0.03                       | 0.246                      | 2580            |
    +---------+---------------+----------------------------+----------------------------+-----------------+


Based on these values and a line length $l$ in km the apparent power for a transmission the thermal
capacity, resistance and reactance are calculated as:

.. math::

    S &= \sqrt{3} \cdot U_d \cdot I_{nom} \\
    x &= x\_per\_km \cdot l \\
    r &= r\_per\_km \cdot l \\


In linear power flow only active power is accounted for. While also representing a strong assumption
the full apparent power is used as thermal capacity and therefore as limit to active power in the
calculations. 

The values for reactance and resistance in per unit (p.u.), in reference to :math:`S_B = 1`MVA and the
nominal line voltage $U_d$ become:

.. math::

    z_{base} &= U_d^2 / S_B \\
    x\_pu &= x / z_{base}   \\
    r\_pu &= r / z_{base}

Cables
******

The dataset includes an underground identifier, which allows to categorize transmission lines as
cables. Given that cables have different parameters, especially regarding their impedance it makes
sense to include that information. However, the line parameters are highly dependant on operating
temperature, which is impossible to know or too specific to assume. Generally cables would provide
lower impedance than overhead but lines, but their operation differs significantly due to high
capacity to earth and difficult heat dispersion through the isolation. At this point we prefer using
the types for overhead lines. This needs more input. 

Transformers
************

Transformers are used to connect nodes of different voltage levels which share the same substation.
All transformers in the dataset connect two voltage levels. The relevant types are 110kV/220kV,
110kV/380kV and 220kV/380kV. Transformers are modeled as lines in pomato, and linear power flow in
general, where the impedance can be calculated using short circuit voltages and the ohmic voltage
drop (:math:`u_{kr}`, :math:`u_{Rr}`$  in [1]_,  :math:`v_{sc}`, :math:`v_{scr}` in the `PyPSA
documentation <www.pypsa.readthedocs.io/en/latest/components.html#transformer-types>`_ (at least to
my understanding). These parameters are from from the pyPSA project [3]_ as `standard types
<www.github.com/PyPSA/PyPSA/blob/master/pypsa/standard_types/transformer_types.csv>`_ mostly
equivalent to the ones used in pandapower [4]_ and SimBench [5]_, the latter included higher rated
transformers and a 220/380 type. The following table contains the relevant types, with reference. 

.. table:: Transformer parameters from various sources. 

    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    |                    | :math:`S_n` | :math:`U_H` | :math:`U_L` | :math:`u_{kr}` | :math:`u_{Rr}` | Source           |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    |                    | MVA         | kV          | kV          | %              | %              |                  |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    | 160 MVA 380/110 kV | 160         | 110         | 380         | 12.2           | 0.25           | pandapower/pyPSA |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    | 100 MVA 220/110 kV | 100         | 110         | 220         | 12             | 0.26           | pandapower/pyPSA |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    | 300MVA220/110      | 300         | 110         | 220         | 12             | 0.128          | SimBench         |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    | 350MVA380/110      | 350         | 110         | 380         | 22             | 0.257          | SimBench         |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+
    | Typx380/220        | 600         | 220         | 380         | 18.5           | 0.25           | SimBench         |
    +--------------------+-------------+-------------+-------------+----------------+----------------+------------------+

Given these parameters, we can calculate the transformer impedance following equations (8.3 - 8.5)
from [1]_: 

.. math::

    z &=  \dfrac{u_{kr} \cdot U_H^2}{100 \cdot S_n} \\
    r &=  \dfrac{u_{Rr} \cdot U_H^2}{100 \cdot S_n} \\
    x &= \sqrt{z^2 - r^2}


The respective p.u. values are obtained with rated power $S_n$ in reference to the base $S_B = 1$MVA:

.. math::

    x\_pu &= x \cdot S_B / S_n   \\
    r\_pu &= r \cdot S_B / S_n 

DC Lines 
********

DC lines represent active network elements are do not interact with the parameterization of linear
power flow. Therefore they require no parametrization for power flow calculations. The rated
capacity is not included in the data, but given the limited amount of elements and ease of research,
these values can be manually added. 


Validation
**********

To validate the parameters we can look into the static grid models that are published on TSO
websites. For example the German TSO `50Hertz
<www.50hertz.com/de/Transparenz/Kennzahlen/Netzdaten/StatischesNetzmodell>`_. publishes the data for
their system including nominal current, nominal voltage and impedance for each element. The
following table shows two lines (220 and 380kV) and two transformers. 

.. table:: Public data from 50Hertz's static gridmodel. 

    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    |             |                          | :math:`U_n` | L    | :math:`I_r` | R1     | X1      |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    | Line        | Redwitz - Remptendorf    | 380         | 56.0 | 3600        | 1.6526 | 14.9690 |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    | Line        | Neuenhagen - Marzahn     | 380         | 16.9 | 2400        | 0.7606 | 4.3135  |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    | Line        | Neuenhagen - Hennigsdorf | 220         | 45.9 | 1070        | 3.1905 | 13.1680 |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    |             |                          | Ur1         | Ur2  | Sr          | R1     | X1      |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    | Transformer | Wolmirstedt              | 400         | 231  | 400         | 0.9171 | 63.1933 |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+
    | Transformer | Röhrsdorf                | 380         | 231  | 800         | 0.3328 | 22.5167 |
    +-------------+--------------------------+-------------+------+-------------+--------+---------+

Given the parameters from the previous sections and the dataset which does include length l and
voltage level :math:`U_d`, the nominal current :math:`I_d` we would estimate the
following resistance and reactance.  

.. table:: Calculated (estimated) parameters for a sample of network elements. 

    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+
    |              |                           |  :math:`U_d` | :math:`l` |  :math:`I_{nom}` |  r        |  x       |
    +==============+===========================+==============+===========+==================+===========+==========+
    |  Line        |  Redwitz - Remptendorf    |  380         |  57.7     |  2580            |  1.73027  |  14.1882 |
    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+
    |  Line        |  Neuenhagen - Marzahn     |  380         |  17.5     |  2580            |  0.525349 |  4.30786 |
    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+
    |  Line        |  Neuenhagen - Hennigsdorf |  220         |  70.7     |  1290            |  4.24497  |  21.2956 |
    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+
    |                                          |  Ur1         |  Ur2      |  Sr              |  r        |  x       |
    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+
    |  Transformer |  Wolmirstedt              |  380         |  220      |  600             |  0.601667 |  44.5193 |
    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+
    |  Transformer |  Röhrsdorf                |  380         |  220      |  600             |  0.601667 |  44.5193 |
    +--------------+---------------------------+--------------+-----------+------------------+-----------+----------+


The comparison shows that line parameters are fairly accurate in terms of impedance, given that the
length is accurate (which it isn't in the Henningsdorf line) but rather imprecise in term of nominal
voltage and therefore capacity. This is no surprise as the nominal current that depends on how lines
are mounted and there are huge differences. For example the Redwitz line is a know congestion,
therefore it contains larger bundles/stronger ropes than other 380 lines. 

Similarly the transformer parameters are in the ballpark but not super accurate. Again, the
differences between transformers of the same type, namely differences in rated current yield large
differences in capacity and impedances. 

However, given that all parameters are derived from a handful of standard types, the results are
satisfactory. More precise calibration, based on the static grid models or specific information is
always possible. 



.. [1] **Oeding, D. and B.R. Oswald** (2016). Elektrische Kraftwerke und Netze. Springer Berlin
    Heidelberg. doi: 10.1007/978-3-642-19246-3

.. [2] **Wiegmans,  Bart**  (2016). GridKit extract of ENTSO-E interactive map
    doi:10.5281/zenodo.55853

.. [3] **Brown, T., J. Horsch, and D. Schlachtberger** (2018). PyPSA: Python for PowerSystem Analysis. 
    Journal of Open Research Software. doi: 10.5334/jors.188. eprint:1707.09913

.. [4] **Thurner, L. et al.** (2018). pandapower - an Open Source Python Tool for Convenient  Modeling,  Analysis and  Optimization of  Electric  Power  Systems
    IEEE Transactions on Power Systems. doi: 10.1109/TPWRS.2018.2829021
    
.. [5] **Meinecke, Steffen et al.** (2020). SimBench—A Benchmark Dataset of Electric Power Systems to Compare Innovative Solutions based on Power Flow Analysis. 
    Energies. doi: 10.3390/en13123290

.. [6] **Müller, Ulf Philipp et al.** (2018) The eGo grid model: An open source approach towards a model of German high and extra-high voltage power grid. 
    Journal of Physics. doi: 10.1088/1742-6596/977/1/012003
