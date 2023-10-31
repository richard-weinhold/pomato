
.. _running_pomato:

Running POMATO
==============

The intention is to run the model from a self-made script that includes data read-in, model run 
and result analysis using the provided functionality from pomato which can be run from console or 
within any IDE. To illustrate the core functionality of pomato this guide comprises of two case 
studies illustrating different aspects. First, we use the IEEE 118 bus network to run different 
grid representations, including SCOPF utilizing the RedundancyRemoval. In the second case study, we
look into a more to-live model of the German electricity system. Modeling the process of market 
clearing with redispatch. 

Both guides go through the code step-by-step. Having an IDE like Spyder can make this process quite 
intuitive, but running the full script from console does also work. 

IEEE Case Study
---------------

Pomato is out of the box compatible with all IEEE test cases and this case study takes a look into 
the 118 bus network. The data is available under open license at 
`https://power-grid-lib.github.io/ <https://power-grid-lib.github.io/>`_, we have added coordinates
in a separate file, to allow geo-plotting the result. See the */examples/data_input* folder for 
reference. 

The model is initialized by running the following commands. Note that pomato will automatically 
recognize the ieee data format and import it correctly. After the import all data is available 
though the :obj:`~pomato.data.DataManagement` module initialized in :code:`mato.data`. 

.. code-block:: python

   from pomato import POMATO
   mato = POMATO(wdir=Path(your_working_directory),
                 options_file='profiles/ieee118.json')
   mato.load_data('data_input/pglib_opf_case118_ieee.m')

The option file in this example is as follows and indicates a dispatch model, i.e. "copper-plate", 
for a single timestep, IEEE usually only consist of a demand snapshot and Infeasibility variable with 
an upper bound of 20. 

.. literalinclude:: _static/files/ieee_doc.json
  :language: JSON

Fist, the model is run with these settings and after the model has concluded we analyse the resulting 
power flows. The results are initialized in :code:`mato.data.results` with the name of the result 
folder, which can be accessed via the :code:`mato.market_model.result_folders` attribute.

.. code-block:: python
   
   mato.options["type"] = "dispatch"
   mato.options["title"] = "Uniform Pricing"
   mato.create_grid_representation()
   mato.run_market_model()

   result_name = mato.market_model.result_folders[0]
   result = mato.data.results[result_name.name]

   # Check Overloaded Lines for N-0 and N-1 contingency cases.
   df1, df2 = result.overloaded_lines_n_0()
   df3, df4 = result.overloaded_lines_n_1()

   print("Number of overloaded lines (Dispatch): ", len(df1))
   print("Number of overloaded lines N-1 (Dispatch): ", len(df3))

The results should show overload in the N-0, as well as N-1 case. To account for line capacities 
the optimization type is changed to "nodal" and the model is re-run. 

.. code-block:: python
   
   mato.options["type"] = "nodal"
   mato.options["title"] = "Nodal Pricing"

   mato.create_grid_representation()
   mato.run_market_model()

   result_name = mato.market_model.result_folders[0]
   result = mato.data.results[result_name.name]

   # Check Overloaded Lines for N-0 and N-1 contingency cases.
   df1, df2 = result.overloaded_lines_n_0()
   df3, df4 = result.overloaded_lines_n_1()

   print("Number of overloaded lines (Nodal): ", len(df1))
   print("Number of overloaded lines N-1 (Nodal): ", len(df3))

The results should no longer contain overloaded lines in the N-0 case, however overloads in the case
of contingencies. The Results method :meth:`pomato.data.Results.overloaded_lines_n_1()` returns the overloaded 
line/contingency pairs. 

To also include contingencies, known as security constrained optimal power flow (SCOPF), the 
optimization type has to be changed to "cbco_nodal". The "redundancy_removal_option" defines the reduction algorithm
where the option "full" will perform no reduction and options "redundancy_removal" and "conditional_redundancy_removal" will 
reduce the power flow constraints to a minimal set. 

Therefore the option "conditional_redundancy_removal" will invoke the RedundancyRemoval algorithm and yield a set
of 540 constraints that guarantee SCOPF. In comparison, the full PTDF, without RedundancyRemoval or
Impact Screening will have a length of approx. 26.000 lines/outages. Given the small network of 118 
buses and a single timestep, this would still be solvable, but scaling the problem will quickly 
increase complexity prohibitively. 

.. code-block:: python

   mato.options["type"] = "cbco_nodal"
   mato.options["title"] = "SCOPF"

   mato.options["grid"]["redundancy_removal_option"] = "conditional_redundancy_removal"

   mato.create_grid_representation()

   # # Update the model data
   mato.update_market_model_data()
   mato.run_market_model()

   # # Check for overloaded lines (Should be none for N-0 and N-1)
   result_folder = mato.market_model.result_folders[0]
   result = mato.data.results[result_folder.name]

   df1, df2 = result.overloaded_lines_n_0()
   df3, df4 = result.overloaded_lines_n_1()

   print("Number of overloaded lines (SCOPF): ", len(df1))
   print("Number of overloaded lines N-1 (SCOPF): ", len(df3))

As expected, there are no overloads in the N-0 or N-1 case, as the problem represents a SCOPF. 

DE Case Study
-------------

In this example the german system is modeled, first as a single market clearing based on the 
options stored in a *.json* file and then in a second step including redispatch and altered options. 

The data for the underlying data comes from multiple sources and is compiled using the `Pomato Data
<https://github.com/richard-weinhold/PomatoData/>`_ repository: 

   - Conventional power plant data is taken from the 
     `Open Power System Data <https://open-power-system-data.org/>`_ Platform (OPSD).
   - Geographic information is used from the `ExtremOS <https://opendata.ffe.de/project/extremos/>`_
     project of Forschungsstelle für Energiewirtschaft (FfE) and their FfE Open Data Portal.   
   - Wind and PV capacities are distributed using the NUTS3 Potentials from FfE. 
   - Future capacities are taken from results of the large scale energy system model `AnyMod
     <https://github.com/leonardgoeke/AnyMOD.jl>`_.
   - NUTS3 availability timeseries for wind and solar are generated using the `atlite
     <https://github.com/PyPSA/atlite>`_, package. Offshore availability based on EEZ regions of FfE.
   - The grid data comes from the `GridKit <https://github.com/bdw/GridKit>`_ project, more
     specifically 
     `PyPSA/pypsa-eur <https://github.com/PyPSA/pypsa-eur/tree/master/data/entsoegridkit>`_ fork,
     which contains more recent data. 
   - Generation costs are based on data from `ELMOD-DE <http://www.diw.de/elmod>`_ which is 
     openly available and described in detail in `DIW DataDocumentation 83 
     <https://www.diw.de/documents/publikationen/73/diw_01.c.528927.de/diw_datadoc_2016-083.pdf>`_ 
   - Hydro Plants are taken from the `JRC Hydro-power plants database
     <https://github.com/energy-modelling-toolkit/hydro-power-database>`_ and inflows are determined
     using the `atlite <https://github.com/PyPSA/atlite>`_ hydro capabilities and scaled using
     annual generation.
   - Load, commercial exchange from ENTSO-E Transparency platform [9]

This step-by-step guide is part of the ``/examples`` folder which can be copied as working folder.  

.. code-block:: python

   from pomato import POMATO
   mato = POMATO(wdir=Path(your_working_directory),
                 options_file='profiles/de.json')
   mato.load_data('data_input/DE_2020.zip')

The model is initialized with the lines above. First an instance of pomato is created, then data 
is loaded. The option file, part of the initialization looks like this:

.. literalinclude:: _static/files/de_doc.json
  :language: JSON

As indicated in section :ref:`sec-options` this will set-up pomato for a model clearing without a network
representation ("copper plate") for the model horizon from hour 0 to 48. Additionally, plants where 
plant type is either "hydro_res" or "hydro_psp" are considered storages and plants of type 
"wind onshore", "wind offshore" or "solar" have an availability timeseries attached to them. 
Infeasibility is allowed on the electricity energy balance with costs and upper bound of 1000. The 
data for "demand_el", "net_export" and "availability" are stored in the excel/zip archive in wide/stacked
format. Therefore the option is set so they are restructured when read in. 

After the initialization and data read is is (successfully) done, all data is available to the used 
through the data class. For example:

.. code-block:: python

   mato.data.plants[["g_max", "fuel"]].groupby("fuel").sum()

will aggregate the installed capacity by fuel.

With the following command the model is run and after the market result can be accessed through the 
:code:`mato.data.results` dictionary, where all results will be instantiated. Since there will only 
be one element present, it can directly be assigned. 

.. code-block:: python

   mato.run_market_model()
   market_result = next(iter(mato.data.results.values()))

The instantiation of results within the :obj:`~pomato.data.DataManagement` module allows for easy 
analysis of the model results. 

.. code-block:: python

   # Standard plots that visualize
   # the market result
   market_result.default_plots()
   
   # N-0, N-1 Flows
   market_result.n_0_flow()
   market_result.n_1_flow()

   # N-0 Overloads, returning two DataFrames
   # 1) Aggregated statistics 
   # 2) power flow on overloaded lines
   df1, df2 = market_result.overloaded_lines_n_0()

Besides the predefined function in :obj:`~pomato.data.Results` the results can be manually
accessed easily, like merging the plant list with the generation variable and subsequent calculation 
of utilization per *plant_type*:

.. code-block:: python

   gen = pd.merge(mato.data.plants, market_result.G, 
                  left_index=True, right_on="p")
   util = gen[["plant_type", "g_max", "G"]].groupby("plant_type").sum()
   # Print Utilization by Plant Type
   print(util.G / util.g_max)

The visualization functionality of pomato allows to create a plots utilizing the plotly package. For
example the generation can be visualized via the command:

.. code-block:: python 

   mato.visualization.create_generation_plot(market_result))

.. raw:: html
   :file: _static/files/market_result_dispatch.html


To include redispatch into the model the options have to be altered and the model re-run. The options 
can be directly changed by editing the *options* and change the redispatch options to include it, 
define which zones should be redispatched and what costs should be associated with redispatch. 

The grid representation has to re-created to apply these changes and the model can be re-run. 
Also it might be a good idea to clear the result dictionary.

.. code-block:: python

   mato.data.results = {}
   mato.options["title"] = "DE: Redispatch"

   mato.options["redispatch"]["include"] = True
   mato.options["redispatch"]["zones"] = ["DE"]
   mato.options["redispatch"]["cost"] = 50
 
   mato.create_grid_representation()
   mato.run_market_model()

After the model is complete :code:`mato.data.results` should contain two results, one with
"_market_result" suffix and one with "_redispatch_DE" suffix. We can identify the two new result 
instances by the two suffixes or just use a predefined method to assign them to two variables. 

We can confirm successful redispatch by looking into overloaded lines in the N-0 case.

.. code-block:: python 

   market_result, redisp_result = mato.data.return_results()

   n0_m, _ = market_result.overloaded_lines_n_0()
   print("Number of N-0 Overloads in the market results: ", len(n0_m))

   n0_r, _  = redisp_result.overloaded_lines_n_0()
   print("Number of N-0 Overloads after redispatch: ", len(n0_r))

While there are overloaded lines in the market results, when redispatched all lines should be within
their capacity. Additionally i might be interested to see which plants are redispatched, this can 
be done in a similar way to the result analysis above:

.. code-block:: python 

   # Merge G market result into plant data
   relevant_cols = ["plant_type", "fuel", "technology", "g_max", "node"]
   gen = pd.merge(market_result.data.plants[relevant_cols],
                  market_result.G, left_index=True, right_on="p")

   # Merge redispatch G
   gen = pd.merge(gen, redisp_result.G, on=["p", "t"], 
                  suffixes=("_market", "_redispatch"))
   # Calculate G_redispatch - G_market as delta 
   gen["delta"] = gen["G_redispatch"] - gen["G_market"]
   gen["delta_abs"] = gen["delta"].abs()
   print("Redispatch per hour: ", gen.delta_abs.sum()/len(gen.t.unique()))

Yielding the power plant schedules for the market result and redispatch. To finalize this analysis 
we can, again generate the geo-plot, including the redispatch locations. 

.. code-block:: python 

   mato.visualization.create_geo_plot(redisp_result, show_redispatch=True)


.. raw:: html
   :file: _static/files/redispatch_geoplot.html


Modeling Flow-based Market Coupling
-----------------------------------

This example follows the modeling approach that we have followed in our publications about the
process. In particular `Uncertainty-Aware Capacity Allocation in Flow-Based Market Coupling
<https://ieeexplore.ieee.org/abstract/document/10094020>`_ that is also available on `arXiv
<https://arxiv.org/abs/2109.04968>`_. 

This step-by-step guide is part of the ``/examples`` folder which can be copied as working folder
and contains the code below as a script as well as the input data. 

.. code-block:: python

   from pomato import POMATO
   from pathlib import Path()
   wdir = Path("<your_path>") 
   mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json")
   mato.load_data('data_input/nrel_118_high_res.zip')

The model is initialized with the lines above. First an instance of pomato is created, then data 
is loaded. The option file, part of the initialization looks like this:

.. literalinclude:: _static/files/ieee_doc.json
  :language: JSON

Note, we use the solver ECOS, to enable the chance constrained formulation in the last step. 

We model the flow-based market coupling process in three steps: 
   - Basecase, a best estimate of the system state is at delivery. 
   - Calculation of the flow-based parameters that represent the capacities for the market coupling. 
   - Correcting the dispatch to be network-feasible with redispatch. 

First we set up the basecase calculation:

.. code-block:: python

   mato.options["title"] = "Basecase"
   mato.options["type"] = "opf"
   mato.create_grid_representation()
   mato.update_market_model_data()
   mato.run_market_model()
   result_name = next(r for r in list(mato.data.results))
   basecase = mato.data.return_results("Basecase")

We use the "opf" type, which means we calculate an N-0 dispatch or optimal power flow. 

.. code-block:: python

   mato.options["fbmc"]["minram"] = 0.4
   mato.options["fbmc"]["frm"] = 0.1
   mato.options["fbmc"]["cne_sensitivity"] = 0.05
   mato.options["fbmc"]["gsk"] = "dynamic"
   mato.options["fbmc"]["reduce"] = True
   fb_parameters = mato.create_flowbased_parameters(basecase)

   # FBMC market clearing
   mato.options["title"] = "FB Market Coupling - 40%"
   mato.options["redispatch"]["include"] = True
   mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
   mato.create_grid_representation(flowbased_parameters=fb_parameters)
   mato.update_market_model_data()
   mato.run_market_model()
   fb_market_result, _ = mato.data.return_results(mato.options["title"])

Second we calculate the flow-based parameters based on the available settings. Here we want to
ensure that at least 40% of physical capacity is available to the market coupling, a security margin
of 10% is used, that only network elements are considered that have at least 5% zon-to-zone PTDF in
the flow-based region and that the GSK is based on the running conventional units. We can use the
options of the main class to set parameters for the calculation. The full list of possible settings
can be found in the Options Section :ref:`sec-options`.

After the flow-based parameters are calculated, the market model is run to calculate the market
result as well as the necessary redispatch to ensure N-0 network constraints. 

Lastly, we can run the same calculation with a chance constrained formulation that determines the
security margin, i.e. FRM, based on the expected deviation from intermittend generators. 

.. code-block:: python

   mato.options["title"] = "FB CC Market Coupling - 40%"
   mato.options["chance_constrained"]["include"] = True
   mato.options["chance_constrained"]["fixed_alpha"] = True
   mato.options["chance_constrained"]["cc_res_mw"] = 0
   mato.options["chance_constrained"]["alpha_plants_mw"] = 30
   mato.options["chance_constrained"]["epsilon"] = 0.05
   mato.options["chance_constrained"]["percent_std"] = 0.05
   mato.options["fbmc"]["frm"] = 0

   mato.update_market_model_data()
   mato.run_market_model()
   fb_cc_market_result, _ = mato.data.return_results(mato.options["title"])

Note, the FRM is set to zero, as the security margin is no longer fixed. The chance constrained
formulation can be configured with additional parameters that are described in
:ref:`Options <sec-options-cc>`. Here we enforce a fixed alpha, meaning all conventional generators cover the
forecast error proportional to the installed capacity (*fixed_alpha*), we consider all renewable
generators as uncertain (*cc_res_mw*), plants with more than 30MW are considered to cover forecast
errors, the risk level is set to 5% and we assume a standard deviation for renewable generators of
5% of the forecasted infeed.    

After the model has run, the results can be analysed. The reserved capacity on flow-based
constraints that is reserved for forecast errors can be obtained from the *CC_LINE_MARGIN* variable.
And the standard pomato visualization features can be used.  

.. code-block:: python

   print("Mean CC Margin:", fb_cc_market_result.CC_LINE_MARGIN["CC_LINE_MARGIN"].mean())
   # Visualization of system costs 
   mato.visualization.create_cost_overview(mato.data.results.values())

Additionally we can visualize the flow-based parameters as a flow-based domain and see the impact of
the chance constrained FRM values. 

.. code-block:: python

   domain_x, domain_y, timestep = ("R2", "R3"), ("R1", "R3"), "t0021"
   fbmc_domains = pomato.visualization.FBDomainPlots(mato.data, fb_parameters)
   fbmc_domains.generate_flowbased_domain(
      domain_x, domain_y, timestep=timestep, 
      shift2MCP=True, result=fb_market_result)
      
   fbmc_domains.generate_flowbased_domain(
      domain_x, domain_y, timestep=timestep, 
      shift2MCP=True, result=fb_cc_market_result)

   for elm in fbmc_domains.fbmc_plots:
      elm.x_max, elm.x_min = 1000, -1100
      elm.y_max, elm.y_min = 500, -400
      fig = mato.visualization.create_fb_domain_plot(elm)

.. raw:: html
   :file: _static/files/fbmc.html

.. raw:: html
   :file: _static/files/fbmc_cc.html


Running a Chance-Constrained OPF
--------------------------------

This example highlights the chance constrained functionality for OPF. In this simple example we use
the NREL 118 Bus network with extended capacities of intermittend renewables. 

The files are included in the ``/examples`` folder which can be copied as working folder
and contains the code below as a script as well as the input data. 

.. code-block:: python

   from pomato import POMATO
   from pathlib import Path()
   wdir = Path("<your_path>") 
   mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json")
   mato.load_data('data_input/nrel_118_high_res.zip')

The model is initialized with the lines above. First an instance of pomato is created, then data 
is loaded. The option file, part of the initialization looks like this:

.. literalinclude:: _static/files/ieee_doc.json
  :language: JSON

Note, we use the solver ECOS, to enable the chance constrained formulation in the last step. 

Fist we calculate the OPF, meaning an economic dispatch that takes into account N-0 power flow
constraints.

.. code-block:: python

   mato.options["title"] = "N-0"
   mato.options["type"] = "opf"
   mato.create_grid_representation()
   mato.update_market_model_data()
   mato.run_market_model()

Second, we parameterise the chance constrained formulation that considers a distribution of forecast
errors for intermittend renewable generators, the generator response and its impact on power flows.
By reserving margin on network elements, the model can guarantee that overloads only occur up to an
accepted risk level.  

The model is configured as follows:

.. code-block:: python

   mato.options["title"] = "N-0 CC"
   mato.options["chance_constrained"]["include"] = True
   mato.options["chance_constrained"]["fixed_alpha"] = True
   mato.options["chance_constrained"]["cc_res_mw"] = 0
   mato.options["chance_constrained"]["alpha_plants_mw"] = 30
   mato.options["chance_constrained"]["epsilon"] = 0.05
   mato.options["chance_constrained"]["percent_std"] = 0.1
   mato.create_grid_representation()
   mato.update_market_model_data()
   mato.run_market_model()

The chance constrained formulation can be configured with additional parameters described in
:ref:`Options <sec-options-cc>`. Here we model a fixed alpha first, meaning all conventional generators cover the
forecast error proportional to the installed capacity (*fixed_alpha*), we consider all renewable
generators as uncertain (*cc_res_mw*), plants with more than 30MW are considered to cover forecast
errors, the risk level is set to 5% and we assume a standard deviation for renewable generators of
10% of the forecasted infeed. 

Third, we can recalculate the model with variable alpha. This allows the model to decide which
generators are reserved to cover imbalances caused by forecast errors. Lastly, we can visualize the
resulting system costs of the three runs. 

.. code-block:: python

   mato.options["title"] = "N-0 CC - Variable Alpha"
   mato.options["chance_constrained"]["fixed_alpha"] = False
   mato.create_grid_representation()
   mato.update_market_model_data()
   mato.run_market_model()

   mato.visualization.create_cost_overview(mato.data.results.values())

.. raw:: html
   :file: _static/files/chance_constrained_costs.html
