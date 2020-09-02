.. _installation:

Installation
------------

POMATO is written in python and julia. Python takes care of the data processing
and julia runs the economic dispatch and N-1 redundancy removal algorithm. 

The recommended way to install POMATO:

    - Install Julia and have it available on the system path 
    - Install POMATO through *pip* in python
      >= 3.6 by creating a virtual environment and install pomato into it:

      .. code-block::

            python -m venv pomato
            ./pomato/Scripts/activate
            pip install git+https://github.com/richard-weinhold/pomato.git


After this is completed pomato can be imported in python:

.. code-block:: python

    from pomato import POMATO

Pomato functions from a *working directory*, ideally the project folder includes the virtual 
environment, and creates additional folders for results, temporary data and logs. The way we use 
pomato is illustrated by the *examples* folder, cloning its contents as a *working directory* 
allows to run the included examples.

Pomato works with open solvers, if Gurobi is available on the PATH during installation tt will 
be used. See `Gurobi.jl <https://github.com/JuliaOpt/Gurobi.jl>`_ for notes on the installation. 
Additionally, the Chance-Constrained model formulation requires MOSEK solver which can be installed
from within Pomato, but requires a licence to use `Mosek.jl <https://github.com/JuliaOpt/Mosek.jl>`_. 

POMATO will create a julia environment in the *pomato/_installation* subfolder of its
*package_directory*. The julia environment can be managed through the pomato package and its 
possible to update, install gurobi and install mosek at any time after the installation, given the 
respective prerequisites for Gurobi and MOSEK are given. 

.. code-block:: python

    import pomato

    # Add MOSEK solver to the julia env
    pomato.tools.julia_management.add_mosek()

    # Add Gurobi solver to the julia env
    pomato.tools.julia_management.add_gurobi()
    
    # Update the installed packages
    # Including MarketModel.jl and RedundancyRemoval.jl
    pomato.tools.julia_management.update_julia()
