"""Data Management of POMATO, which glues all components together.

This module is divided into one main and three sub-modules:
    * :obj:`~pomato.data.DataManagement` : The main hub for the data. An
      instance of this class is attached to the POMATO main module to provide
      access to all relevant data. It manages the read in of raw data,
      processing and validating the raw data and the means to store, process, 
      analyses and access the results.
This is done within the three sub-modules:
    - :obj:`~pomato.data.DataWorker` : Reading in data from an excel or matpower file.
    - :obj:`~pomato.data.InputProcessing` : Process input data to calculate missing parameters,
      fill missing/default values etc.. essentially bringing the raw data
      into the desired structure defined in the `data_structure`.
    - :obj:`~pomato.data.ResultProcessing` : Since the DataManagement module is available to
      the market model the market result is processed alongside the input
      data in the ResultProcessing Module. It collects different methods
      make standard results available to the user in an easy way and is
      meant to simplify result analysis.

"""
from pomato.data.data import DataManagement
from pomato.data.input import InputProcessing
from pomato.data.worker import DataWorker
from pomato.data.results import ResultProcessing