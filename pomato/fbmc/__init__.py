"""The Flow Based Module of POMATO.

Flow based market coupling represents a zonal market clearing process in which the 
commercial exchnage capacities are calculated based on a forecasted dispatch (basecase) and  
subset of critical lines under critical outages. 

The Flow Based module generates the flow based paramters based on a supplied basecase and 
utilizes from other POMATO modules :class:`~pomato.data.DataManagement`, 
:class:`~pomato.grid.GridTopology` and :class:`~pomato.grid.GridModel`. 

The main method is :meth:`~pomato.fbmc.create_flowbased_parameters` that will yield the 
the final product that is essentially a time-dependent zonal PTDF that is commonly denoted 
as the flow based domain. 

The generation can be separated into multiple parts that all come with choices regarding the 
input parameters: 

 - The basecase, a market result that ideally represents the forecasted final dispatch, 
   which therefore should represent a redispatched, N-0 or N-1 secure dispatch. 
   This, however is discussed controversially and potentially more relaxed market results 
   including N-0/N-1 overloads are possible. 
 - The network elements used for the flow based domain, i.e. CBCOs.
 - Calculation of the so called reference flows, which represent (purely theoretical) 
   power flows on each CBCO at NEX=0, representing a pre-coupled state. 
 - Generation Shift Keys (GSK), define the participation factor of each node with in a zone
   towards changes in net position (NEX). The options are currently *flat*, meaning all nodes
   participate equally, *gmax* where nodes participate based on the installed capacity of 
   conventional generation and *dynamic* where nodes participate based on the currently running 
   conventional capacity. 


The network elements are calculated when initializing the FB Module based on the commonly 
accepted 5% zone-to-zone threshold, however custom lists are possible (and potentially more useful).

The basecase has to be supplied to :meth:`~pomato.fbmc.create_flowbased_parameters` itself. In our 
understanding this should be a SCOPF for the entire FB region. From the basecase the reference flows 
are obtained which, together with the GSK will yield the FB domain/parameters. The calculation of the 
reference flows is under ongoing research and different modeling approaches exist in the literature. 
The main question is whether reference flows that exceed line capacities are explicitly redispatched 
or relaxed according to minRAM/security/ATC margins. 


"""
from pomato.fbmc.fbmc_module import FBMCModule
