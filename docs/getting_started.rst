Getting Started
***************

This guide aims to introduce all core functionalities of pomato in a tutorial like structure. It 
covers set up, folder/data/model structure and introduces the current workflow and targeted use case 
by running the DE case study. 

Prerequisites
=============

A successful execution of pomato requires three components: Folder structure, valid options and 
correctly formatted input data. While there is some flexibility for each of these components, they 
must abide a general structure that is layed out in the next three subsections. 

While it is generally a good idea to have the program dynamically adopt to all possible inputs, but 
because pomato performs a very specific function it simply requires a certain parametrization to work. 
We have opted to be fairly strict when it comes to definition of data structure and input parameters, 
but made sure the definitions are easily accessible to potentially adopt pomato to other applications. 

Folder and Data Structure
-------------------------

Running pomato requires besides input data, temporary directories to store data and results from the 
market model or redundancy removal. The idea is to have a working folder to run the model including 
all relevant inputs and store outputs and results. The folder structure is defined as follows and will
be automatically generated when running the model: 

::

    working_directory
    ├── profiles/
    ├── logs/
    ├── data_input/
    ├── data_output/
    ├── data_temp/
        ├── bokeh_files/
        └── julia_files/
             ├── cbco_data/
             ├── results/
             └── data/

Input and Model Data
--------------------

Pomato represents a electricity market model, also denoted as dispatch problem, that needs a certain 
minimal dataset to run but can potentially accommodate a wide range of input data. 
 
Therefore, a set of input parameters is defined as *model_structure* which represents the data that 
is eventually used in the market model. All data part of the  *model_structure* is initialized 
as a pandas DataFrame attribute to the DataManagement class. 
Namely the *model_structure* is: 

- nodes: network nodes
- zones: market areas
- lines: transmission lines
- plants: power plants
- availability: time-dependant capacity factor for plants like wind turbines
- dclines: DC lines
- demand_el: electricity demand_el
- ntc: net transfer capacities
- net_export: nodal injections representing exchange with non-model market areas
- inflows: inflows into hydro storages
- net_position: net position for market areas
- demand_h: district heating demand
- heatareas: district heating networks 
			
Note, not all of these data has to included for all models. The IEEE case study for examples does not 
include district heating, dclines or electricity storages. However the *model_structure* defines the 
data (i.e. nodes, lines etc.) and their attributes and they can remain empty. 

For example the *model_structure* for *plants* data looks like this:  

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
| plant_type       | any             |   " "   |
+------------------+-----------------+---------+
| storage_capacity | float64         |         |
+------------------+-----------------+---------+
| heatarea         | heatareas.index |         |
+------------------+-----------------+---------+

Note that this data does not include data like fuel or technology. This type of data is not strictly 
necessary for a model run, therefore is not part of the *model structure*. However, it is fairly obvious 
that this kind of data is of high value for post processing of the model results. 
Therefore the input data does not have to be the same as the necessarily exactly the same as the 
*model structure* but has to contain the required data to run the desired type of model. 

The full list of all data that can be used directly in the market model can be found here as a 
:download:`json <_static/model_structure.json>` including their attributes, attribute types and  
default values. 

Options
-------

With a valid data set the model is able to run different configurations of the market model e.g. 
including transmission lines in the market clear with/without redispatch.

All options are collected in the options attribute of pomato and can be initialized from a json file
located in the ``/profiles`` folder.

The option file does not have to include every possible options. Not specified options are set with 
a default value. The default options are available as a method ``pomato.tools.default_options()``.

The options are divided into three sections: Optimization, Grid, Data and are as follows:

- Optimization:
   - type: ntc,
   - model_horizon:
   - heat_model
   - constrain_nex
   - timeseries: split, market_horizon, redispatch_horizon
   - redispatch: include zonal_redispatch zones cost
   - curtailment: include, cost
   - chance_constrained: include, fixed_alpha, cc_res_mw, alpha_plants_mw
   - parameters: storage_start
   - infeasibility:
   - plant_types:
- Grid:
   - cbco_option: full
   - precalc_filename: 
   - sensitivity: 
   - capacity_multiplier: 
   - preprocess: 
   - gsk:
   - minram:
- Data: 
   - data_type:
   - stacked: 
   - process: 
   - process_input: False
   - unique_mc: False
   - round_demand: True
   - default_efficiency: 0.5
   - default_mc: 200
   - co2_price: 20

Model Formulation
-----------------

.. math::

   a_t &= 1 &\forall t \in T \\
   b_t &= 1 - 3b \cdot A_{t,z} \quad &\forall t \in T \\

Which is important 

Model Structure
---------------

Running the Model
=================

The intention is to run the model from a self made script that includes data read-in, model run 
and result analysis using the provided functionality from pomato which can be run from console or 
within any IDE. This step-by-step guide is part of the ``/examples`` folder which can be copied as 
working folder.  

With the following option file we can run the model: 

.. literalinclude:: _static/de_doc.json
  :language: JSON

As indicated in section `Options`_  

.. code-block:: python

   from pomato import POMATO
   mato = POMATO(wdir=Path(/your_working_directory/), 
                 options_file="profiles/de.json")
   mato.load_data('data_input/dataset_de.zip')
   mato.run_market_model()


This is the result 

.. raw:: html
   :file: _static/bokeh_market_result.html


rerun 

.. code-block:: python

   mato.options["optimization"]["redispatch"]["include"] = True
   mato.options["optimization"]["redispatch"]["zones"] = ["DE"]
   mato.options["optimization"]["redispatch"]["cost"] = 50
   mato.create_grid_representation()
   mato.run_market_model()


.. raw:: html
   :file: _static/bokeh_redispatch.html
