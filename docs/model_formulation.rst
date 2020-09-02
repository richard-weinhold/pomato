Model Formulation
-----------------

The market model follows the typical structure of an econonomic dispatch, where the generation
schedule of each power plant is determined based on the short run marginal costs and the
availability of renewable energy sources (RES). Therefore, the model represents an optimization
problem minimizing total generation costs subject to constraints regarding generation capacity and
transport limitations. Which constraints are present depends on the options.  

Each power plant is categorized by its plant type. Depending on plant type different subsets of
constraints apply to its generation variables. The model distinguishes between the following plant
types:

  - :math:`\mathcal{P}`: Conventional power plants. 
  - :math:`\mathcal{TS}`: Plants that have a timedependant capacity factor (availability). 
  - :math:`\mathcal{ES} \subset \mathcal{P}` : Electricity storages.
  
  - :math:`\mathcal{HE} \subset \mathcal{P}`: Plants that produce heat.
  - :math:`\mathcal{CHP} \subset \mathcal{P}`: Plants that provide both heat and electricity.
  - :math:`\mathcal{PH} \subset \mathcal{P}`: Power to heat plants.
  - :math:`\mathcal{HS} \subset \mathcal{P}`: Heat Storages. 


The generation of a power plant can be characterized by different decision variables, depending on
plant type, which are as follows:

  - :math:`G`: Electricity generation 
  - :math:`D`: Electricity demand, for plants of type ES and PH.
  - :math:`\mathit{CURT}`: Curtailment variable for plant of type TS. 
  - :math:`L^{es}`: Storage level for electricity storages (plant type ES). 
  - :math:`H`: Heat generation for all plants of type HE.
  - :math:`L^{hs}`: Storage level for heat storages (plant type HS). 

Generation Constraints
**********************

Electricity generation is bound by the installed capacity per plant. For RES the generation is 
determined by the availability minus the curtailment variable. 

.. math::

  0 &\leq G_{p,t} \leq g^{max}_p &\forall p \in \mathcal{P}, t \in \mathcal{T}  \\
  0 &\leq \mathit{CURT}_{ts,t} \leq g^{max}_p \cdot \mathrm{availability}_{ts,t} 
  \quad &\forall ts \in \mathcal{TS}, t \in \mathcal{T} \\
  \\
  L_{es,t} &= L_{es,t-1} - G_{es,t} + \eta \cdot D_{es,t} + \mathrm{inflow}_{es,t} \quad 
      &\forall es \in \mathcal{ES}, t \in \mathcal{T} \\
  0 &\leq L_{es,t} \leq l^{max}_{es} \quad &\forall es \in \mathcal{ES}, t \in \mathcal{T} \\
  0 &\leq D_{es,t} \leq d^{max}_{es} \quad &\forall es \in \mathcal{ES}, t \in \mathcal{T} \\

Storages are represented by a storage level, that redices by generation and increases by
charging/demand. Storage efficiency reduces the electricity stored. 

Electricity generation occurs costs for fuel and possibly curtailment, which will be included in the
objective function:

.. math::

  \mathit{COST\_G} &= \sum_{p \in \mathcal{P}, t \in \mathcal{T}} mc_p \cdot G_{p,t} \\
  \mathit{COST\_CURT} &= \sum_{ts \in \mathcal{TS}, t \in \mathcal{T}} \mathrm{curt\_cost} 
  \cdot \mathit{CURT}_{ts,t} \\


Energy Balance
**************

Due to transport constraints applying to both the zonal energy balance, i.e. the net position (NEX)
represented by the exchange variable EX which represents commercial exchange between two market
areas, the nodal balance, both balances have to be explicitly made and utilize mapping between the
power plant sets and their nodal/zonal affiliation. Generally, the zonal energy balance allows to
constrain commercial exchange while the nodal energy balance set the nodal injection (INJ) and
therefore is constrained by the transmission network or its representation. 

.. math:: 

  \mathit{INJ}_{n,t} &= \sum_{p \in \mathcal{P}_n} G_{p,t} - D_{p,t} 
  + \sum_{ts \in \mathcal{TS}_n} g^{max}_{ts} \cdot \mathrm{availability}_{ts,t} - \mathit{CURT}_{ts,t} \\
  &+ \sum_{dc \in \mathcal{DC}} \mathrm{incidence}_{dc,n} \cdot F^{dc}_{dc, t} \\
  &+ \mathrm{net\_export}_{n,t} - \mathrm{demand}_{n,t} 
  &\forall n \in \mathcal{N}, t \in \mathcal{T}\\
  \\
  \sum_{zz \in \mathcal{Z}} \mathit{EX}_{z,zz,t} -  \mathit{EX}_{zz,z,t} 
  &= \sum_{p \in \mathcal{P}_z} G_{p,t} - D_{p,t} + \sum_{ts \in \mathcal{TS}_z} g^{max}_{ts}
  \cdot \mathrm{availability}_{ts,t} - \mathit{CURT}_{ts,t} \\
  &+ \sum_{n \in \mathcal{N}_z} \mathrm{net\_export}_{n,t} - \mathrm{demand}_{n,t} 
  &\forall z \in \mathcal{Z}, t \in \mathcal{T}\\

Transport Constraints
*********************

Based on the chosen configuration, tranport is constrained either as commercial exchange, i.e.
in between market areas, or based on the nodal injections in a more technically accurate representation
on the transmission system. 

The power flow constraints follow the DCLF implementation using power transfer distribution factors
(PTDF) to map nodal injections to line flows. 

.. math:: 

  F^{+}_t - F^{-}_t &= \mathit{PTDF} \cdot \mathit{INJ}_t \quad &\forall t \in \mathcal{T}\\
  0 \leq F^{+}_t &\leq f^{max}  &\forall t \in \mathcal{T} \\
  0 \leq F^{-}_t &\leq f^{max} &\forall t \in \mathcal{T} \\

Note that the DCLF constraints are intentionally written in matrix form instead of elementwise like
above. Thereby the formulation is more general, which reflects the actual formulation in which the
PTDF can accommodate any power flow configuration POMATO offers. This can be N-0 (nodal pricing,
OPF), N-1 (SCOPF) in a reduced representation or full, including combined contingencies (n-k) or 
specified contingency groups. Assinging the line flows to two positive variables reduces the model 
complexity, as the dense and possibly extremely large PTDF matrix is only used once per timestep. 

An analogue formulation applies for a zonal PTDF, except that the line flows result from the NEX
(exports minus imports). Note that the PTDF is denoted with index t, indicating a potentially 
time dependant PTDF as used in the implementation of Flow Based Market Coupling (FBMC) and the FB Domain. 

The zonal PTDF is computed based on the given configuration and relies on weighting parameters that 
convert the zonal net position NEX into nodal injections. This concept is farmally defined within 
FBMC as a generation shift key (GSK).

.. math:: 

  F^{+}_t - F^{-}_t &= \mathit{PTDF}_t \cdot \mathit{NEX}_t \quad &\forall t \in \mathcal{T} \\
  0 \leq F^{+}_t &\leq f^{max}  &\forall t \in \mathcal{T} \\
  0 \leq F^{-}_t &\leq f^{max} &\forall t \in \mathcal{T} \\
  \\
  \text{with: } \mathit{NEX}_{z,t} &= \sum_{zz \in \mathcal{Z}} \mathit{EX}_{z,zz,t} -  \mathit{EX}_{zz,z,t} 
  \quad &\forall z \in \mathcal{Z}, t \in \mathcal{T}\\


Beside nodal/zonal transmission network representations, tranport constraints can be includes as 
net trans capacities (NTC), that directly constraint the commercial exchange. 

.. math:: 
  \mathit{EX}_{z,zz,t} &\leq \mathit{ntc}_{z,zz} \quad &\forall z \in \mathcal{Z}, t \in \mathcal{T}\\

DC lines are also constrained to upper and lower bounds. DC lines are modeled as part of the market 
result and their power flow is optimized with system cost in mind. 

.. math:: 
  -f^{max}_{dc} &\leq F^{DC}_{dc,t} \leq f^{max}_{dc} \quad &\forall dc \in \mathcal{DC}, t \in \mathcal{T}\\

The flow on a dc line is mapped to the start and endnodes using the :math:`\mathrm{incidence}_{dc,n}`  
parameter and is included in the nodal energy balance. 


Heat-Generation Constraints
***************************

The model can accommodate	generation of heat into the economic dispatch problem. However, the
additional data needed is difficult to come by. The concept is, that heatareas :math:`\mathcal{HA}`
are defined analog to market areas and a heat demand for each heatarea has to be balanced by plants
which are located within. Plants are subject to a maximum generation and co-generation of heat and
electricity is constraints by additional constraints. There can be heat generated by plants of type
:math:`\mathcal{TS}`, but it cannot be curtailed. 

The generation from CHP is modeled with 2-degrees of freedom, where the first constraint represents
the extraction line, and the second constraint the upper-bound for heat and electricity generation. 
Generally, CHP can be modeled with much greater detail, however the heat formulation's purpose is 
to allow to roughly model the adjacent sector and allow for soft must-run constraints. 

.. math:: 
  
  0 &\leq H_{he,t} \leq h^{max}_{he} &\forall he \in \mathcal{HE}, t \in \mathcal{T}  \\
  G_{chp, t} &\geq \dfrac{g^{max}_{chp} \cdot (1-\eta)}{h^{max}_{chp}} \cdot H_{chp, t} 
  &\forall chp \in \mathcal{CHP}, t \in \mathcal{T} \\ 
  G_{chp, t} &\leq g^{max}_{chp} \cdot (1 - \dfrac{\eta \cdot H_{chp, t}}{h^{max}_{chp}}) 
  &\forall chp \in \mathcal{CHP}, t \in \mathcal{T} \\

Plants of type :math:`\mathcal{PH}` convert an electricity demand into heat and heat storages
:math:`\mathcal{HS}` can shift heat generation to later periods. Note that the inclusion of
storages will always greatly increase model complexity. 

.. math::

  D_{ph, t} &= \eta \cdot H_{ph, t} &\forall ph \in \mathcal{ph}, t \in \mathcal{T} \\ 
  \\
  L_{hs,t} &= \eta \cdot L_{hs,t-1} - H_{hs,t} + D_{hs,t}  \quad 
      &\forall hs \in \mathcal{HS}, t \in \mathcal{T} \\
  0 &\leq L_{hs,t} \leq l^{max}_{hs} \quad &\forall hs \in \mathcal{HS}, t \in \mathcal{T} \\
  0 &\leq D_{hs,t} \leq d^{max}_{hs} \quad &\forall hs \in \mathcal{HS}, t \in \mathcal{T} \\

Heat generation and demand have to be balanced and heat generation will occur costs. 

.. math::

  \mathrm{demand}_{ha,t}  &= \sum_{he \in \mathcal{HE}_ha} H_{he,t} - D_{he,t} 
  + \sum_{ts \in \mathcal{TS}_{ha}} h^{max}_{ts} \cdot \mathrm{availability}_{ts,t}
  \quad &\forall ha \in \mathcal{HA}, t \in \mathcal{T}\\
  \\
  \mathit{COST\_H} &= \sum_{he \in \mathcal{HE}, t \in \mathcal{T}} mc^{he}_{he} \cdot H_{he,t} \\


Objective Value
***************

The objective value represents the total system cost and consist of all individual cost components 
and is subject to all constraints layed out above. Note that not all constraints have to be present
each model run, but depend on the individual configuration through the options of each run. 

.. math::

  \min \text{ OBJ} &= \sum \mathit{COST\_G} + \mathit{COST\_H} + \mathit{COST\_CURT} \\
  \text{s.t. }& \\
  & \text{Generation Constraints} \\
  & \text{Heat Constraints} \\
  & \text{Transport Constraints} \\
  & \text{Energy Balances} \\
