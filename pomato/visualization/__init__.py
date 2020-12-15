"""The visualisation capabilities of POMATO are grouped into the visualisation sub-modules. 

There are three visualisation features implemented:
    
    - Visualisation of installed capacity, generation schedule and storage usage, i.e. visualization 
      of input data and model results. This functionality is accessible through 
      :class:`~pomato.visualization.Visualization`.
    - Result visualization with geographic information. This is acessible though 
      :class:`~pomato.visualization.GeoPlot`.
    - Visualization of Flow Based Domain. This extens the functionality of 
      :class:`~pomato.fbmc.FBMCModule` and provides visualization of the flowbased parameters. This 
      functionality is accessible through :class:`~pomato.visualization.FBMCDomainPlots`.
"""

from pomato.visualization.visualization import Visualization
from pomato.visualization.geoplot import GeoPlot
from pomato.visualization.fbmc_domain import FBMCDomainPlots, FBMCDomain
