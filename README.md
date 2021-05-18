[![PomatoLogo](https://github.com/richard-weinhold/pomato/blob/master/docs/_static/graphics/pomato_logo_small.png "Pomato Soup")](#) POMATO - Power Market Tool
=====================================================================================================================================
Master Branch: ![Python package](https://github.com/richard-weinhold/pomato/workflows/Python%20package/badge.svg?branch=master) 

Construction Branch: ![Python package](https://github.com/richard-weinhold/pomato/workflows/Python%20package/badge.svg?branch=construction)

Documentation Status: [![Documentation Status](https://readthedocs.org/projects/pomato/badge/?version=latest)](https://pomato.readthedocs.io/en/latest/?badge=latest)

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

The recommended way to install POMATO:
  - Install Julia and have it available on the system path
  - Install POMATO through *pip* in python >= 3.6 by creating a virtual environment and install pomato into it

        python -m venv pomato
        .\pomato\Scripts\activate
        pip install git+https://github.com/richard-weinhold/pomato.git

This will not only clone the master branch of this repository into the local python environment, but also pull the master branch of the MarketModel and RedundancyRemoval Julia packages which are required to run POMATO.

After this is completed pomato can be imported in python:

    from pomato import POMATO

POMATO functions from a *working directory*, ideally the project folder includes the virtual environment, and creates additional folders for results, temporary data and logs. The way we use POMATO is illustrated by the *examples* folder, cloning its contents as a *working directory* allows to run the included examples.

Pomato works with open solvers, if Gurobi is available on the PATH within the venv/project it will be used. See [Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl) for notes on the installation. Additionally, the 
Chance-Constrained model formulation requires MOSEK solver which can be installed from within Pomato, 
but requires a licence to use [Mosek.jl](https://github.com/JuliaOpt/Mosek.jl). 


You can also install the latest version available on the construction branch via 

        pip install git+https://github.com/richard-weinhold/pomato.git@construction

This will not only install the construction branch of POMATO but also of the MarketModel, to remain compadible. 

The integration of Julia and Python can be tricky to manage when updating or changing versions. Besides deleting and reinstalling POMATO provides some means to manage the julia environment via a functions available in *pomato.tools.julia_management*. 

  * *pomato.tools.julia_management.instantiate_julia()* will re-install the julia packages MarketModel and RedundancyRemoval from their git repositories. 
  * *pomato.tools.julia_management.instantiate_julia_dev(redundancyremoval_path, marketmodel_path)* will instantiate the julia environment from local repositories. This is useful when actively changing the code.
  * *pomato.tools.julia_management.add_gurobi()* adds the gurobi solver to the julia environment. 
  * *pomato.tools.julia_management.add_mosek()* adds the mosek solver to the julia environment. 

Examples
--------
This release includes two examples in the *examples* folder. Including the contents of this folder into a pomato working directory will allow their execution:

  - The IEEE 118 bus network, which contains a singular timestep. The data is available under 
    open license at [https://power-grid-lib.github.io/](https://power-grid-lib.github.io/) and re-hosted in this repository.

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

POMATO is part of my PhD and actively developed by Robert and myself. This means it will keep 
changing to include new functionality or to improve existing features. The existing examples, which
are also part of the Getting Started guide in the documentation, are part of a testing suite to 
ensure some robustness. However, we are not software engineers, thus the "program" is not written 
with robustness in mind and our experience is limited when it comes to common best practices. 
Expect errors, bug, funky behavior and code structures from the minds of two engineering economists.  

Related Publications
--------------------
- (*preprint*) [Weinhold and Mieth (2020), Power Market Tool (POMATO) for the Analysis of Zonal 
   Electricity Markets](https://arxiv.org/abs/2011.11594)
- [Weinhold and Mieth (2020), Fast Security-Constrained Optimal Power Flow through 
   Low-Impact and Redundancy Screening](https://ieeexplore.ieee.org/document/9094021)
- [Schönheit, Weinhold, Dierstein (2020), The impact of different strategies for generation 
   shift keys (GSKs) on  the flow-based market coupling domain: A model-based analysis of Central Western Europe](https://www.sciencedirect.com/science/article/pii/S0306261919317544)

Acknowledgments
---------------

Richard and Robert would like to acknowledge the support of Reiner Lemoine-Foundation, the Danish Energy Agency and Federal Ministry for 
Economic Affairs and Energy (BMWi).
Robert Mieth is funded by the Reiner Lemoine-Foundation scholarship. Richard Weinhold is funded by the Danish Energy Agency.
The development of POMATO and its applications was funded by BMWi in the project “Long-term Planning and Short-term Optimization of the German Electricity System Within the European Context” (LKD-EU, 03ET4028A).

