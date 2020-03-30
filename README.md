[![PomatoLogo](https://github.com/korpuskel91/pomato/blob/master/docs/pomatologo_small.png "Pomato Soup")](#) POMATO - Power Market Tool
=====================================================================================================================================
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
The model is structured in three interconnected parts : 
  * Data Management: Data input, processing and result analysis.
  * Market Model: Calculation of the economic dispatch based on the
    dataset and chosen grid representation.
  * Grid Model: Providing grid representation for economic dispatch
    calculation in chosen granularity (N-0, N-1, FBMC, NTC, copperplate)
    and analysis for ex-post analysis of the market result.

Documentation
-------------
Comprehensive documentation is available at [pomato.readthedocs.io](https://pomato.readthedocs.io/).

Installation
------------
POMATO is written in python and julia. Python takes care of the data processing
and julia runs the economic dispatch and N-1 redundancy removal algorithm. 

The recommended way to install POMATO is through *pip* by creating a virtual environment and install pomato into it:

    python -m venv pomato
    .\pomato\Scripts\activate
    pip install git+https://github.com/korpuskel91/pomato.git

After this is completed pomato can be imported in python:

    from pomato import POMATO

Pomato functions from a *working directory*, ideally the project folder including the virtual environment, and creates additional folders for results, temporary data and logs. The way we use pomato is illustrated by the *examples* folder, cloning its contents into the *working directory* allows to run the included examples.

Note: To install pomato in its current state, julia and gurobi must be available on the PATH within the venv/project. See [Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl) for notes on the installation. 

Examples
--------
This release includes two examples in the *examples* folder. Including the contents of this folder into the pomato working directory will allow their execution:

  - The IEEE 118 bus network, which contains a singular timestep. The data is available under 
    open license at [https://power-grid-lib.github.io/](https://power-grid-lib.github.io/) and rehosted in this repository.

        $ python /run_pomato_ieee.py

  - The DE case study, based on data from [ELMOD-DE](http://www.diw.de/elmod) which is openly available and
    described in detail in [DIW DataDocumentation 83](https://www.diw.de/documents/publikationen/73/diw_01.c.528927.de/diw_datadoc_2016-083.pdf) which represents a more complex system and can be run for longer model horizon (although 
    shortened to allow to host this data in this git).

        $ python /run_pomato_de.py


However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to access POMATO objects and develop a personal script based
on the provided functionality and its results.

Release Status
--------------

This release covers all features and a big part of the documentation. The FBMCModule is still
changing very often and is not documented. The julia code also lacks documentation until we figure
out how to include both julia and python code into one sphinx script. 

POMATO is part of my PhD and actively developed by Robert and myself. WE are not software engineers,
thus the "program" is not written with robustness in mind. Expect errors, bug, funky behavior, 
stupid code structures, hard-coded mess and lack of obvious features.

Related Publications
--------------------

- [Weinhold and Mieth (2019), Fast Security-Constrained Optimal Power Flow through 
   Low-Impact and Redundancy Screening](https://arxiv.org/abs/1910.09034)
- [Schönheit, Weinhold, Dierstein (2020), The impact of different strategies for generation shift keys (GSKs) on the flow-based market coupling domain: A model-based analysis of Central Western Europe](https://www.sciencedirect.com/science/article/pii/S0306261919317544)

Acknowledgments
---------------

Richard and Robert would like to aknowledge the support of Reiner Lemoine-Foundation, the Danish Energy Agency and Federal Ministry for 
Economic Affairs and Energy (BMWi).
Robert Mieth is funded by the Reiner Lemoine-Foundation scholarship. Richard Weinhold is funded by the Danish Energy Agency.
The development of POMATO and its applications was funded by BMWi in the project “Long-term Planning and Short-term Optimization of the German Electricity System Within the European Context” (LKD-EU, 03ET4028A).

