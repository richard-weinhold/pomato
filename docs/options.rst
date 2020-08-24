
.. _sec-options:

Options
-------

With a valid data set the model is able to run different configurations of the market model e.g. 
including transmission lines in the market clear with/without redispatch.

All options are collected in the options attribute of pomato and can be initialized from a json file
located in the ``/profiles`` folder.

The option file does not have to include every possible options. Not specified options are set with 
a default value. The default options are available as a method ``pomato.tools.default_options()`` 
and are structured in a dictionary or a json file when read into from disk.

The options are divided into three sections: Optimization, Grid, Data and are as follows:

- Optimization:
   - *type* (string): Defines the with what kind of grid representation the market clearing is 
     represented. Options are:

      - *dispatch*: Market clearing without any network representation. 
      - *ntc*: Commercial exchange capacities. Not physical network represented.
      - *nodal*: Full N-0 transmission network representation. Corresponds to nodal pricing. Implemented
        via standard PTDF formulation. 
      - *zonal*: Using a zonal PTDF to clear the market. The linear mapping between the nodal injections 
        and zonal net position is based on the *gsk* option, which can be *flat* or *gmax*.
      - *cbco_nodal*: This option represents the nodal N-1 grid representation and it runs in conjunction
        with the RedundancyRemoval algorithm to obtain the minimal set of critical branches under critical
        outages which guarantee N-1 secure optimal power flow when clearing the market. 
      - *cbco_zonal*: Analogues to the *cbco_nodal* option, but based on a zonal N-1 PTDF.  

   - *model_horizon* (2-element list): Defines over what (sub)-set of timesteps the market model is run. 
   
   - *plant_types*: The plant type specifies what kind of constraints are attributed the generation 
     of a power plant. The model can accommodate variable renewable in-feed (ts), meaning that a
     timeseries dictates hourly generation availability, electricity and heat storages storages (es/hs) 
     and power to heat units (ph). The plant types that are attributed to these subsets of constraints
     are specified in the plant_type table of the plant data. 
   
   - *split_model_horizon*: The market clearing can be split into multiple sections of predefined 
     length instead of a single large model. This is especially useful when redispatching a market result
     and not inter-temporal constraints are present.

      - *include* (bool): Split market clearing into sections. 
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
     plant of set ts. 
    
      - *include* (bool): Include curtailment.
      - *cost* (float): Curtailment cost.

   - *constrain_nex* (bool): Constrain the net position for each market area. This can be useful when
     modeling Flow Based Market Coupling. Requires net_position data as specified in :ref:`model_data`.
   
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

   - *parameters*: This was supposed to collect various single-use model parameters. Over the 
     development only the storage level at model start remained.

      - *storage_start* (float): Storage start-level in % of reservoir capacity. 
   
   - *infeasibility*: The models energy balances include infeasibility variables that allow to conclude
     model runs without infeasibility to to specific load situations. For both heat and electricity
     and specify whether to include them, costs and upper bounds. 

   - *heat_model* (bool): Include heat model, i.e. energy balance, storages and CHP constraints. 

- *Grid*: The *Grid* options define how the grid representation is generated. The options are mainly 
  relevant for the N-1 grid representation (*cbco_nodal*, *cbco_zonal* optimization type) and are 
  used to run security constrained optimal power flow (SCOPF). This algorithm is described in great 
  detail in `Weinhold and Mieth (2020), Fast Security-Constrained Optimal Power Flow through 
  Low-Impact and Redundancy Screening <https://ieeexplore.ieee.org/document/9094021>`_.

   - *cbco_option* (string): Option to specify how/if reduce the PTDF matrix. Options are:
      - *full*: Including all N-1 constraints. The number should correspond to L x L minus lines that 
        are either radial or disconnect the network (indicated by contingency = false) and duplicates
        which are removed by the *preprocess* options. 
      - *clarkson*: Runs the RedundancyRemoval algorithm to find the minimal set of critical branches
        under critical outages to guarantee SCOPF.
      - *clarkson_base*: Analog to *clarkson* however including nodal injection limits into the 
        algorithm, resulting in a smaller set of cbco's that guarantee SCOPF under the condition 
        nodal injections do not exceed these limits. 
      - *save*: Saves the necessary data to run the RedundancyRemoval. Used for debugging/testing the
        algorithm itself. 

   - *precalc_filename* (string): Since the RedundancyRemoval algorithm can take substation time to 
     complete it makes sense to reuse previously identified sets of constraints. 
   - *sensitivity* (float): The sensitivity parameter is used in the pre-filtering of the N-1 PTDF 
     that is the input to the RedundancyRemoval algorithm. The idea is that only lines, that in case of 
     an outage, impact line flows above a certain sensitivity are potentially part of the essential 
     set or in short cbco's. See the description of the method 
     :meth:`~pomato.grid.GridModel.create_filtered_n_1_ptdf` 
     or the Section on `Impact Screening` in the publication for more information. 
   - *capacity_multiplier* (float): Multiplies line capacities by a factor. 
   - *preprocess* (bool): Preprocessing the N-1 PTDF means removing duplicates. This can be omitted
     to obtain the true full N-1 PTDF. 
   - *gsk*: Generation Shift Key is a term used in flow based market coupling, describing how nodes
     participate in changes of the net position, representing a linear mapping of zonal net position 
     to nodal injection. This can be used to translate a nodal PTDF into a zonal PTDF. Options are 
     `gmax` or `flat`, weighting nodal injection by installed capacity (of conventional generators) 
     or equally.
   - *minram*: This option is only relevant in the FBMC module of pomato. Forcing a minimum capacity 
     on cbco's that make of the Flow Based Domain. 
- Data: The following options relate to the input data. Over the corse of the development of pomato, 
  the rules on input data got more strict, therefore less input data is processed in pomato itself. 
  The following functions remain:

   - *stacked* (list of data): Excel can have problems with long tables. So for example the demand 
     table can be read in a wide format and than stacked to fit the predefined structure. This has to 
     be declared here. 
   - *unique_mc* (bool): Sometimes it can be beneficial to computation time to have unique generation 
     costs. This option add small increments to make all plants have unique marginal costs.  