
Running the Model
=================

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
   
   mato.options["optimization"]["type"] = "dispatch"
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
   
   mato.options["optimization"]["type"] = "nodal"
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
of contingencies. The ResultProcessing method :code:`overloaded_lines_n_1()` returns the overloaded 
line/contingency pairs. 

To also include contingencies, known as security constrained optimal power flow (SCOPF), the 
optimization type has to be changed to "cbco_nodal". The "cbco_option" defines the reduction algorithm
where the option "full" will perform no reduction and options "clarkson" and "clarkson_base" will 
reduce the power flow constraints to a minimal set. 

Therefore the option "clarkson_base" will invoke the RedundancyRemoval algorithm and yield a set
of 540 constraints that guarantee SCOPF. In comparison, the full PTDF, without RedundancyRemoval or
Impact Screening will have a length of approx. 26.000 lines/outages. Given the small network of 118 
buses and a single timestep, this would still be solvable, but scaling the problem will quickly 
increase complexity prohibitively. 

.. code-block:: python

   mato.options["optimization"]["type"] = "cbco_nodal"
   mato.options["grid"]["cbco_option"] = "clarkson_base"

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

The data for the underlying dataset comes from multiple sources: 
   - Power plant data (conventional and renewable) comes from the 
     `Open Power System Data <https://open-power-system-data.org/>`_ Platform (OPSD).
   - The timeseries are also taken from OPSD, which origin from the `ENTSO-E Transparency Platform 
     <https://transparency.entsoe.eu/>`_, additional data, like timeseries for physical cross-border 
     flows are directly taken from ENTSO-E. 
   - The grid data comes from the `GridKit <https://github.com/bdw/GridKit>`_ project, more specifically 
     `PyPSA/pypsa-eur <https://github.com/PyPSA/pypsa-eur/tree/master/data/entsoegridkit>`_ fork, which 
     contains more updated data. 
   - Generation costs are based on data from `ELMOD-DE <http://www.diw.de/elmod>`_ which is 
     openly available and described in detail in `DIW DataDocumentation 83 
     <https://www.diw.de/documents/publikationen/73/diw_01.c.528927.de/diw_datadoc_2016-083.pdf>`_ 

This step-by-step guide is part of the ``/examples`` folder which can be copied as working folder.  

.. code-block:: python

   from pomato import POMATO
   mato = POMATO(wdir=Path(your_working_directory),
                 options_file='profiles/de.json')
   mato.load_data('data_input/dataset_de.zip')

The model is initialized with the lines above. First an instance of pomato is created, then data 
is loaded. The option file, part of the initialization looks like this:

.. literalinclude:: _static/files/de_doc.json
  :language: JSON

As indicated in section :ref:`sec-options` this will set-up pomato for a model clearing without a network
representation ("copper plate") for the model horizon from hour 0 to 24. Additionally, plants where 
plant type is either "hydro_res" or "hydro_psp" are considered storages and plants of type 
"wind onshore", "wind offshore" or "solar" have an availability timeseries attached to them. 
Infeasibility is allowed on the electricity energy balance with costs and upper bound of 1000. The 
data for "demand_el", "net_export" and "availability" are stored in the excel/zip archive in wide/stacked
format. Therefore the option is set so they are restructured when read in. 

After the initialization and data read is is (successfully) done, all data is available to the used 
through the data class. For example:

.. code-block:: python

   mato.data.plants[["g_max", "fuel"]].groupby("fuel").sum()

will aggregate the installed capacity by fuel. Additionally :meth:`~pomato.data.DataManagement.visualize_inputdata()`
will yield default plots visualizing the input data. 

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

Besides the predefined function in :obj:`~pomato.data.ResultProcessing` the results can be manually
accessed easily, like merging the plant list with the generation variable and subsequent calculation 
of utilization per *plant_type*:

.. code-block:: python

   gen = pd.merge(mato.data.plants, market_result.G, 
                  left_index=True, right_on="p")
   util = gen[["plant_type", "g_max", "G"]].groupby("plant_type").sum()
   # Print Utilization by Plant Type
   print(util.G / util.g_max)

Finally, the visualization functionality of pomato allows to create a comprehensive geo-plot, utilizing
the bokeh package, with the command :code:`mato.create_geo_plot()`. 

.. raw:: html
   :file: _static/files//bokeh_market_result.html

To include redispatch into the model the options have to be altered and the model re-run. The options 
can be directly changed by editing the *options* and change the redispatch options to include it, 
define which zones should be redispatched and what costs should be associated with redispatch. 

The grid representation has to re-created to apply these changes and the model can be re-run. 
Also it might be a good idea to clear the result dictionary.

.. code-block:: python

   mato.data.results = {}
   mato.options["optimization"]["redispatch"]["include"] = True
   mato.options["optimization"]["redispatch"]["zones"] = ["DE"]
   mato.options["optimization"]["redispatch"]["cost"] = 50
 
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
   relevant_cols = ["plant_type", "fuel", "tech", "g_max", "node"]
   gen = pd.merge(market_result.data.plants[relevant_cols],
                  market_result.G, left_index=True, right_on="p")

   # Marge redispatch G
   gen = pd.merge(gen, redisp_result.G, on=["p", "t"], 
                  suffixes=("_market", "_redispatch"))
   # Calculate G_redispatch - G_market as delta 
   gen["delta"] = gen["G_redispatch"] - gen["G_market"]
   gen["delta_abs"] = gen["delta"].abs()
   print("Redispatch per hour: ", gen.delta_abs.sum()/len(gen.t.unique()))

Yielding the power plant schedules for the market result and redispatch. To finalize this analysis 
we can, again generate the geo-plot, including the redispatch locations. 

.. code-block:: python 

   mato.create_geo_plot()


.. raw:: html
   :file: _static/files//bokeh_redispatch.html
