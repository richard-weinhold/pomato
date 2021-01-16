.. _model_data:

Input and Model Data
--------------------

Pomato represents a electricity market model, also denoted as dispatch problem, that needs a certain 
minimal dataset to run but can potentially accommodate a wide range of input data. 
 
Therefore, a set of input parameters is defined as *model_structure* which represents the data that 
is eventually used in the market model. All data part of the  *model_structure* is initialized 
as a pandas DataFrame attribute to the :obj:`~pomato.data.DataManagement` class. Default values 
exists for attributes that are optional or are automatically set in the default case.

Note, not all of these data has to included for all models. The IEEE case study for examples does not 
include district heating, dclines or electricity storages. However the *model_structure* defines the 
data (i.e. nodes, lines etc.) and their attributes and they can remain empty. 

Also note that this data does not include data like fuel or technology. This type of data is not strictly 
necessary for a model run, therefore is not part of the *model structure*. However, it is fairly obvious 
that this kind of data is of high value for post processing of the model results. 
Therefore the input data does not have to be the same as the necessarily exactly the same as the 
*model structure* but has to contain the required data to run the desired type of model. 

The full list of all data that can be used directly in the market model can be found here as a 
:download:`json <_static/files//model_structure.json>` including their attributes, attribute types and  
default values. 


nodes
*****

Network nodes. Besides the index, the corresponding zone has to be set and has to match existing 
zone indices. The slack attribute is important for the calculation of the PTDF matrix and in case
of multiple non-synchronized sub-networks, each partition has to have a slack node. These are however
automatically checked and set in the initialization of the :obj:`~pomato.grid.GridModel`.

.. table::
    :align: left

    +---------+-------------+---------+
    |         | type        | default |
    +=========+=============+=========+
    | index   | any         |         |
    +---------+-------------+---------+
    | zone    | zones.index |         |
    +---------+-------------+---------+
    | slack   | bool        | False   |
    +---------+-------------+---------+
    | voltage | float64     | 220.0   |
    +---------+-------------+---------+

lines
*****
.. table::
    :align: left

    +-------------+-------------+---------+
    |             | type        | default |
    +=============+=============+=========+
    | index       | any         |         |
    +-------------+-------------+---------+
    | node_i      | nodes.index |         |
    +-------------+-------------+---------+
    | node_j      | nodes.index |         |
    +-------------+-------------+---------+
    | x           | float64     |         |
    +-------------+-------------+---------+
    | r           | float64     |         |
    +-------------+-------------+---------+
    | x_pu        | float64     |         |
    +-------------+-------------+---------+
    | x_pu        | float64     |         |
    +-------------+-------------+---------+
    | capacity    | float64     |         |
    +-------------+-------------+---------+
    | contingency | bool        | True    |
    +-------------+-------------+---------+


zones
*****

The definition of market areas is solely defined by the network nodes that are part of it.

.. table::
    :align: left

    +-------------+-------------+---------+
    |             | type        | default |
    +=============+=============+=========+
    | index       | any         |         |
    +-------------+-------------+---------+



plants
******

.. table::
    :align: left

    +------------------+-----------------+---------+
    |                  | type            | default |
    +==================+=================+=========+
    | index            | any             |         |
    +------------------+-----------------+---------+
    | node             | nodes.index     |         |
    +------------------+-----------------+---------+
    | mc_el            | float64         |         |
    +------------------+-----------------+---------+
    | mc_heat          | float64         | 0       |
    +------------------+-----------------+---------+
    | g_max            | float64         |         |
    +------------------+-----------------+---------+
    | h_max            | float64         | 0       |
    +------------------+-----------------+---------+
    | eta              | float64         | 1       |
    +------------------+-----------------+---------+
    | plant_type       | any             |         |
    +------------------+-----------------+---------+
    | storage_capacity | float64         |         |
    +------------------+-----------------+---------+
    | heatarea         | heatareas.index |         |
    +------------------+-----------------+---------+



availability
************

time-dependant capacity factor for plants like wind turbines

.. table::
    :align: left

    +--------------+--------------+---------+
    |              | type         | default |
    +==============+==============+=========+
    | index        | any          |         |
    +--------------+--------------+---------+
    | timestep     | any          |         |
    +--------------+--------------+---------+
    | plant        | plants.index |         |
    +--------------+--------------+---------+
    | availability | float64      |         |
    +--------------+--------------+---------+


dclines
*******

.. table::
    :align: left

    +---------+-------------+---------+
    |         | type        | default |
    +=========+=============+=========+
    | index   | any         |         |
    +---------+-------------+---------+
    | node_i  | nodes.index |         |
    +---------+-------------+---------+
    | node_j  | nodes.index |         |
    +---------+-------------+---------+
    | capacity | float64     |         |
    +---------+-------------+---------+


demand_el
*********

electricity demand_el

.. table::
    :align: left

    +-----------+-------------+---------+
    |           | type        | default |
    +===========+=============+=========+
    | index     | any         |         |
    +-----------+-------------+---------+
    | timestep  | any         |         |
    +-----------+-------------+---------+
    | node      | nodes.index |         |
    +-----------+-------------+---------+
    | demand_el | float       |         |
    +-----------+-------------+---------+



ntc
***

net transfer capacities

.. table::
    :align: left

    +--------+-------------+---------+
    |        | type        | default |
    +========+=============+=========+
    | index  | any         |         |
    +--------+-------------+---------+
    | zone_i | zones.index |         |
    +--------+-------------+---------+
    | zone_j | zones.index |         |
    +--------+-------------+---------+
    | ntc    | float64     |         |
    +--------+-------------+---------+


net_export
**********

nodal injections representing exchange with non-model market areas

.. table::
    :align: left

    +------------+-------------+---------+
    |            | type        | default |
    +============+=============+=========+
    | index      | any         |         |
    +------------+-------------+---------+
    | timestep   | any         |         |
    +------------+-------------+---------+
    | node       | nodes.index |         |
    +------------+-------------+---------+
    | net_export | float64     |         |
    +------------+-------------+---------+

inflows
*******
inflows into hydro storages

.. table::
    :align: left

    +----------+--------------+---------+
    |          | type         | default |
    +==========+==============+=========+
    | index    | any          |         |
    +----------+--------------+---------+
    | timestep | any          |         |
    +----------+--------------+---------+
    | plant    | plants.index |         |
    +----------+--------------+---------+
    | inflow   | float64      |         |
    +----------+--------------+---------+

net_position
************

net position for market areas

.. table::
    :align: left

    +--------------+-------------+---------+
    |              | type        | default |
    +==============+=============+=========+
    | index        | any         |         |
    +--------------+-------------+---------+
    | timestep     | any         |         |
    +--------------+-------------+---------+
    | zone         | zones.index |         |
    +--------------+-------------+---------+
    | net_position | float64     |         |
    +--------------+-------------+---------+


heatareas
*********

district heating networks 

.. table::
    :align: left

    +-------------+-------------+---------+
    |             | type        | default |
    +=============+=============+=========+
    | index       | any         |         |
    +-------------+-------------+---------+

demand_h
********

district heating demand
	
.. table::
    :align: left
		
    +-----------+-----------------+---------+
    |           | type            | default |
    +===========+=================+=========+
    | index     | any             |         |
    +-----------+-----------------+---------+
    | timestep  | any             |         |
    +-----------+-----------------+---------+
    | heatarea  | heatareas.index |         |
    +-----------+-----------------+---------+
    | demand_el | float           |         |
    +-----------+-----------------+---------+
