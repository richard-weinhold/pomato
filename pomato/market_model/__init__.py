"""The MarketModel of POMATO, manages data and options to runs the model and its returns results.

The MarketModel interfaces the DataManagement with the MarketModel written in Julia. Its main purpose
is to save the relevant data, run the model and collecting the results, which are then returned to the
DataManagement. 

The MarketModel allows to dynamically change options and re-save data for multiple model runs, making 
sure all results are saved and correctly processed. 
"""

from pomato.market_model.market_model import MarketModel