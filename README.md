<img  height="24" src="https://raw.githubusercontent.com/richard-weinhold/pomato/main/docs/_static/graphics/pomato_logo_small.png"> POMATO - Power Market Tool <img  height="24" src="https://raw.githubusercontent.com/richard-weinhold/pomato/main/docs/_static/graphics/pomato_logo_small.png">
=========================================================================================================================================================

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

Documentation
-------------

Comprehensive documentation is available at [pomato.readthedocs.io](https://pomato.readthedocs.io/).

Installation
------------

POMATO is written in python and julia. Python takes care of the data processing
and julia runs the economic dispatch and N-1 redundancy removal algorithm. 

The recommended way to install POMATO with python and pip:

    - Install [python](https://www.python.org/downloads/) for your operating system. On linux
      based operating systems python is often already installed and available under the python3
      command. For Windows install python into a folder of your choice. POMATO is written and tested
      in python 3.7 by any version >= 3.6 should be compadible. 
    
    - Install [julia](https://julialang.org/downloads/) for your operating system. POMATO is
      written and tested with 1.5, but the newest version 1.6 works as well, but throws some
      warnings.  

    - Add *python* and *julia* to the system Path, this allows you to start  *python* and *julia*
      directly for the command line without typing out the full path of the installation. PLattform
      specific instructions on how to do this are part of the [julia installation instructions](https://julialang.org/downloads/platform/)and work analogous for the python .  
    
    - Install POMATO through *pip* in python. It is recommended to create a virtual environment and
      install pomato into it, but not necessary:
    
    .. code-block::

        python -m venv pomato
        ./pomato/Scripts/activate
        pip install git+https://github.com/richard-weinhold/pomato.git


This will not only clone the master branch of this repository into the local python environment, but
also pull the master branch of the MarketModel and RedundancyRemoval julia packages which are
required to run POMATO. This process can take a few minutes to complete.

After this is completed pomato can be imported in python:

.. code-block:: python

    from pomato import POMATO


See the [POMATO Documentation](https://pomato.readthedocs.io/en/latest/installation.html) for
further information on the installation process. 


Examples
--------
This release includes two examples in the *examples* folder. Including the contents of this folder into a pomato working directory will allow their execution:

  - The IEEE 118 bus network, which contains a singular timestep. The data is available under 
    open license at [https://power-grid-lib.github.io/](https://power-grid-lib.github.io/) and re-hosted in this repository.

        $ python /run_pomato_ieee.py

  - The DE case study, based on data from openly available data sources. The file can be run via

        $ python /run_pomato_de.py

See more in depth descriptions of this two case studies part of the [POMATO Documentation](file:///C:/Users/riw/Documents/repositories/pomato/docs/_build/html/running_pomato.html).

The *examples* folder also contains the two examples as Jupyter notebooks. Another possibility to
access the functionality of POMATO with an online REPL/Console when running POMATO inside a IDE with
an interactive IPython Console (e.g. Spyder) to access POMATO objects and variables.

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

Richard and Robert would like to acknowledge the support of Reiner Lemoine-Foundation, the Danish
Energy Agency and Federal Ministry for Economic Affairs and Energy (BMWi). Robert Mieth is funded by
the Reiner Lemoine-Foundation scholarship. Richard Weinhold is funded by the Danish Energy Agency.
The development of POMATO and its applications was funded by BMWi in the project “Long-term Planning
and Short-term Optimization of the German Electricity System Within the European Context” (LKD-EU,
03ET4028A).

