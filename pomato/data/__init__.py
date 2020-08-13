"""Data Management of POMATO which interfaces all components.

This module is divided into one main and three sub-modules:
    * :obj:`~pomato.data.DataManagement` : The main hub for the data. An
      instance of this class is attached to the POMATO main module to provide
      access to all relevant data. It manages the read in of raw data,
      processing and validating the raw data and the means to store, process,
      analyses and access the results.
This is done within the two sub-modules:
    - :obj:`~pomato.data.DataWorker` : Reading in data from an excel or matpower file.
    - :obj:`~pomato.data.ResultProcessing` : Since the DataManagement module is available to
      the market model the market result is processed alongside the input
      data in the ResultProcessing Module. It collects different methods
      make standard results available to the user in an easy way and is
      meant to simplify result analysis.
    - additionally the *processing* file provides a collection of functions to allow for some 
      input data processing.    

"""
from pomato.data.data import DataManagement
import pomato.data.input_data_processing as processing
from pomato.data.worker import DataWorker
from pomato.data.results import ResultProcessing
