POMATO - Power Market Tool
============================
[![Documentation Status](https://readthedocs.org/projects/pomato/badge/?version=latest)](https://pomato.readthedocs.io/en/latest/?badge=latest)


Overview
--------

POMATO stands for (POwer MArket TOol) and is an easy to use tool for the comprehensive
analysis of the modern electricity market. It comprises the necessary power
engineering framework to account for power flow physics, thermal transport
constraints and security policies of the underlying transmission
infrastructure, depending on the requirements defined by the user.
POMATO was specifically designed to realistically model Flow-Based
Market-Coupling (FBMC) and is therefore equipped with a fast security
constrained optimal power flow algorithm and allows zonal market clearing
with endogenously generated flow-based parameters, and redispatch.

Model Structure
---------------
The model is structured in three interconnected parts:
    - Data Management: Data input, processing and result analysis.
    - Market Model: Calculation of the economic dispatch based on the
      dataset and chosen grid representation. asd asd
    - Grid Model: Providing grid representation for economic dispatch
      calculation in chosen granularity (N-0, N-1, FBMC, NTC, copperplate)
      and analysis for ex-post analysis of the market result.

Documentation
-------------
Comprehensive documentation is available [here:](https://pomato.readthedocs.io/)

Installation
------------
POMATO is written in python and julia. Python takes care of the data processing
and julia runs the economic dispatch and N-1 redundancy removal algorithm. The folder
``/project_files`` contains environment files for python (3.6, anaconda, pip) and julia (1.3).
Note julia has to be available on the PATH for POMATO to run.

After the python enviroment is set-up the provided julia environment has to be instantiated. 
This can be done by running the following commands from the pomato root folder:

    julia --project=project_files/pomato
    ] instantiate

After this is completed pomato can be imported:

    import sys
    sys.path.append(pomato_path)
    from pomato import POMATO

Examples
--------
This release includes two examples :
    - The IEEE 118 bus network, which contains a singular timestep

          $ python /scripts/run_pomato_ieee.py

    - The DE case study, based on data from DIW DataDoc [insert more description]
      which is more complex and can be run for much longer timeframes

          $ python /scripts/run_pomato_de.py


However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to acces POMATO objects and develop a personal script based
on the provided functionality and its results.

Release Status
--------------

This release covers all features and a big part of the documentation. The FBMCModule is stil 
changing very often and is not documented. The julia code also lacks documentation until we figure
out how to include both julia and python code into one shpinx script. 

POMATO is part of my PhD and actively developed by Robert and myself. WE are notsoftware engineers,
thus the "program" is not written with robustness in mind. Expect errors, bug, funky behavior, 
stupid code structures, hard-coded mess and lack of obvious features.

Related Publications
--------------------

- [Weinhold and Mieth (2019), Fast Security-Constrained Optimal Power Flow through Low-Impact and Redundancy Screening](https://arxiv.org/abs/1910.09034)



