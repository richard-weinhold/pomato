
.. _sec-options:

Options
-------

With a valid data set the model is able to run different configurations of the market model e.g. 
including transmission lines in the market clear with/without redispatch.

All options are collected in the options attribute of pomato and can be initialized from a json file
located in the ``/profiles`` folder.

The option file does not have to include every possible options. Not specified options are set with 
a default value. The default options are available as a method :meth:`~pomato.tools.default_options`
and are structured in a dictionary or a json file when read into from disk.

The options are structured with thematic groups to make it more readable:

- *type* (string): Defines the with what kind of grid representation the market clearing is 
  represented. Options are:

  - *uniform*: Market clearing without any network representation. 
  - *ntc*: Commercial exchange capacities. Not physical network represented.
  - *opf*: Full N-0 transmission network representation. Corresponds to nodal pricing. Implemented
    via standard PTDF formulation. 
  - *scopf*: This option represents the nodal N-1 grid representation and it runs in conjunction
    with the RedundancyRemoval algorithm to obtain the minimal set of critical branches under critical
    outages which guarantee N-1 secure optimal power flow when clearing the market. 
  - *fbmc*: Analogues to the *scopf* option, but based on a zonal N-1 PTDF.  

- *model_horizon* (2-element list): Defines over what (sub)-set of timesteps the market model is run. 

- *plant_types*: The plant type specifies what kind of constraints are attributed the generation 
  of a power plant. The model can accommodate variable renewable in-feed (ts), meaning that a
  timeseries dictates hourly generation availability, electricity and heat storages storages (es/hs) 
  and power to heat units (ph). The plant types that are attributed to these subsets of constraints
  are specified in the plant_type table of the plant data. 

- *timeseries*: The market clearing can be split into multiple sections of predefined 
  length instead of a single large model. This is especially useful when redispatching a market result
  and not inter-temporal constraints are present.

  - *market_horizon* (int): Length of section in market model. The intention is to set this value
    large to clear the market over a long model horizon. 
  - *redispatch_horizon* (int): Length of section in redispatch model. This could be set to 24 
    168 hours to redispatch the system over multiple smaller model horizons.

- *redispatch*: Redispatch is an integral part of pomato. The idea is to clear the market in absence
  of a network representation and then adjust the market result to the network constraints (N-0 
  as a default). Redispatch can be associated with (additional) costs and done over all market 
  zones or for individual separately. 

  - *include* (bool): Include redispatch.
  - *zones* (List of zones indices): Zones which are redispatched.
  - *zonal_redispatch* (bool): If True, each zone will be redispatched separately.  
  - *cost* (float): Redispatch cost.

- *curtailment*: As a default, generation from ts-type plants, i.e. renewable generation is must-take.
  However curtailment can be enabled and associated with a cost. This will add a variable *CURT* for each 
  plant of set *ts*. 

  - *include* (bool): Include curtailment.
  - *cost* (float): Curtailment cost.

.. _sec-options-cc:

- *chance_constrained*: The market model can be run with chance constraints on the line flow constraints
  endogenously co-optimizing uncertainty of renewable in-feed and capacity reserves of conventional 
  generation units. Including these constraints requires MOSEK solver and the resulting problems 
  are hard to solve. Therefore three parameters are included to alleviate the computational burden.

  - *include* (bool): Include chance constraints.
  - *fixed_alpha* (bool): Alpha represents the distribution of reserve capacity to conventional 
    generation within the system and sums up to one. Alpha can be a decision variable, allowing 
    for optimal reaction to uncertainty, or fixed which saves computational effort and is also
    commonly applied in practice. 
  - *cc_res_mw* (float): A minimum capacity in MW from which the generation of renewables is considered
    uncertain. The goal is to save computational effort and only include those generation units 
    that can account for large deviations.
  - *alpha_plants_mw* (float): Analogues to the *cc_res_mw* options, setting a minimum capacity
    for generators to participate in the provision of reserves. 
  - *epsilon* (float): Desired risk level. Ensuring lines are not overloaded with the
    probability of (1-epsilon) given a distribution of forecast errors as well as the generators
    that cover imbalances caused by the forecast errors. 
  - *percent_std* (float): Assumed standard deviation of intermittend renewable generators as a
    percent of the forecasted feed-in. 

- *storages*: Storages can be difficult to model, as they connect all timesteps into a single
  optimization problem rather than many individual ones (which are easier to solve). Therefore,
  pomato allows to run a simplified storage model ex-ante to the main optimization to derive a
  storage regime over the full model horizon and allowing to solve the model in multiple splits
  with start- and endlevels from the ex-ante storage optimization.  

  - *storage_model* (bool): Include simplified storage regime precalculation.
  - *storage_model_resolution* (int): Storage model resolution, i.e. taking every n-th timestamp.  
  - *storage_start* (float): Storage start-level in % of reservoir capacity at the start of the
    full model horizon. 
  - *storage_end* (float): Storage end-level in % of reservoir capacity at the end of the
    full model horizon. 
  - *smooth_storage_level* (bool): Smooth the storage regime resulting from the simplified model.
    This is recommended. 

- *heat*: pomato allows to include heat constraints, when the corresponding data is available.

  - *include* (bool): Include heat model, i.e. energy balance, storages and CHP constraints. 
  - *default_storage_level*: start and end level of heat storages. 
  - *chp_efficiency*: CHP efficiency as described in the model formulation
    (:ref:`sec-heat-constraints`).

- *infeasibility*: The models energy balances include infeasibility variables that allow to conclude
  model runs without infeasibility to to specific load situations. Infeasibility variables are
  positive and can be enabled for any balance included in the model together with costs and upper
  bounds:
  
  - *heat*: Include (bool), cost (float), bound (float).
  - *electricity*: Include (bool), cost (float), bound (float).
  - *storages*: Include (bool), cost (float), bound (float).

- *grid*: The *Grid* options define how the grid representation is generated. The options are mainly 
  relevant for the N-1 grid representation (*scopf* optimization type) and are 
  used to run security constrained optimal power flow (SCOPF). This algorithm is described in great 
  detail in `Weinhold and Mieth (2020), Fast Security-Constrained Optimal Power Flow through 
  Low-Impact and Redundancy Screening <https://ieeexplore.ieee.org/document/9094021>`_.

   - *redundancy_removal_option* (string): Option to specify how/if reduce the PTDF matrix. Options are:
      - *full*: Including all N-1 constraints. The number should correspond to L x L minus lines that 
        are either radial or disconnect the network (indicated by contingency = false) and duplicates
        which are removed by the *preprocess* options. 
      - *redundancy_removal*: Runs the RedundancyRemoval algorithm to find the minimal set of critical branches
        under critical outages to guarantee SCOPF.
      - *conditional_redundancy_removal*: Analog to *redundancy_removal* however including nodal
        injection limits into the algorithm, resulting in a smaller set of network elements that
        guarantee SCOPF under the condition nodal injections do not exceed these limits. 
      - *save*: Saves the necessary data to run the RedundancyRemoval. Used for debugging/testing the
        algorithm itself. 

   - *precalc_filename* (string): Since the RedundancyRemoval algorithm can take substation time to 
     complete it makes sense to reuse previously identified sets of constraints. 
   - *sensitivity* (float): The sensitivity parameter is used in the pre-filtering of the N-1 PTDF 
     that is the input to the RedundancyRemoval algorithm. The idea is that only lines, that in case of 
     an outage, impact line flows above a certain sensitivity are potentially part of the essential 
     set or in short cbco's. See the description of the method 
     :meth:`~pomato.grid.GridTopology.create_filtered_n_1_ptdf` 
     or the Section on `Impact Screening` in the publication for more information. 
   - *include_contingencies_redispatch* (bool): Redispatch to N-1 constraints. 
   - *short_term_rating_factor* (float): Multiplies line capacities by the given value for normal operation (N-0). 
   - *long_term_rating_factor* (float): Multiplies line capacities by the given value for contingency cases.
   - *preprocess* (bool): Preprocessing the N-1 PTDF means removing duplicates. This can be omitted
     to obtain the true full N-1 PTDF. 


- *fbmc*: The FBMC options define how FB-parameters are processed:
  
  - *gsk* (string): Generation Shift Key is a term used in flow based market coupling, describing how nodes
    participate in changes of the net position, representing a linear mapping of zonal net position
    to nodal injection. This can be used to translate a nodal PTDF into a zonal PTDF. Options are
    `gmax`, `flat` or `dynamic`, weighting nodal injection by installed capacity (of conventional
    generators), by generation in the basecase or equally.
  - *minram* (float): Enforcing a minimum capacity on network elements considered in the FB-parameters. 
  - *flowbased_region* (list): Defines for which market areas FB parameters are calculated. Defaults to all.
  - *cne_sensitivity* (float): Defines with which sensitivity critical network elements are selected
    from zone-to-zone PTDF. 
  - *lodf_sensitivity* (float): Defines the sensitivity for which outages
    are selected for the previously found set of CNE. 
  - *frm* (float): percentage of line capacity to reduce RAM by as reliability margin. 
  - *reduce* (bool): Run the RedundancyRemoval on the resulting
    FB-parameters to obtain a non-redundant (presolved) set of constraints. 
  - *enforce_ntc_domain* (bool): Enforces the NTC domain to be included in the FB-parameter feasible
    region. 

- *solver*: Allows to set the used solver as part of MarketModel and RedundancyRemoval. 

  - *name* (string): Solver name. Currently supported are Clp, ECOS and Gurobi. Clp and ECOS are
    installed per default. Gurobi must be installed manually. 
  - *solver_options* (dict): Dictionary of solver options. E.g. {"Method": 3, "Threads": 8} for
    Gurobi. 