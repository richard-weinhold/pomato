"""The visualization capabilities of POMATO allow to visualize instantiated results. 

There are three visualization features implemented:

    - Visualization of installed capacity, generation schedule and storage usage, i.e. visualization 
      of input data and model results. This functionality is accessible through 
      :class:`~pomato.visualization.Visualization`.
    - Visualization of Flow Based Domain. This extens the functionality of 
      :class:`~pomato.fbmc.FBMCModule` and provides visualization of the flowbased parameters. This 
      functionality is accessible through :class:`~pomato.visualization.FBDomainPlots`.
    - Interactive :class:`~pomato.visualization.Dashboard`. combining many of the visualization into a web app that acts as a result 
      explorer.
"""

from pomato.visualization.visualization import Visualization
from pomato.visualization.fbmc_domain import FBDomain, FBDomainPlots
from pomato.visualization.dashboard import Dashboard
