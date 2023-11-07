.. _installation:

Installation
------------

POMATO is written in python and julia. Python takes care of the data processing
and julia runs the economic dispatch and N-1 redundancy removal algorithm. 

This structure can take advantage of both: The computational performance ease of defining
optimization problems with the modeling language JuMP in julia. And the well readable and
maintabable object-oriented programming paradigm in python which solves issues typically related to
static scripts. The combination enables a flexible model structure that ensures consitency on the
data side, especially when deriving network representations from dispatch results and continuously
resolving optimization problems. 

However this also means maintaining multiple model parts. To create a reliable and maintainable model, 
the main part of POMATO resides in this python repository and the two julia components reside in separate 
repositories and are pulled when installing POMATO. The advantage is that each repository 
contains solely code of either julia or python and can be tested individually and functionality can be 
ensured. The main python repository actually performs an integration test that pulls and uses the 
julia components. The disadvantage is that it might be difficult to bring everything together. 

However, this can be dealt with by careful instruction, which is the purpose of this page. 

Installing POMATO
-------------------------------------

The recommended installation with python and pip:

    - Install `python <https://www.python.org/downloads/>`_ for your operating system. On linux
      based operating systems python is often already installed and available under the python3
      command. For Windows install python into a folder of your choice. POMATO is written and tested
      in python 3.7 by any version >= 3.6 should be compadible. 
    
    - Install `julia <https://julialang.org/downloads/>`_ for your operating system. POMATO is
      written and tested with 1.5, but the newest version 1.6 works as well, but throws some
      warnings. Note, the order in which you install python and julia itself does not matter. 
    
    - Add *python* and *julia* to the system Path, this allows you to start  *python* and *julia*
      directly for the command line without typing out the full path of the installation. Plattform
      specific instructions on how to do this are part of the `julia installation instructions
      <https://julialang.org/downloads/platform/>`_ and work analogous for the python.   
    
    - When both, *python* and *julia* are available on the user's system the installation of POMATO
      is managed through Python via pip. It is recommended to create a virtual environment and
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

See :doc:`running_pomato` for instruction how to run a model.

Notes on the Installation:
**************************

  - The pip install command will install all python dependencies of POMATO first. If this process
    fails for a specific dependency, it can help to manually install this dependency and re-run this 
    command. The installation procedure will then use the pre-installed package and continue.
  - Anaconda is a widely used python package manager on Windows. Unfortunately, installing it
    alongside a python/pip installation does cause problems. Therefore it is difficult for me to
    provide extensive support for anaconda/pomato. One shortcut that has worked was to use pip/git
    from within anaconda to install Pomato (See `here <https://stackoverflow.com/a/50141879>`_).
  - The *example* folder also constains two jupyter notebooks with the IEEE and DE examples. Note
    that you need to install the requirements to run a jupyter notebook manually to run these
    examples. 


Solvers
*******

The optimization performed as part of the Julia packages MarketModel and RedundancyRemoval utilize 
the optimization library JuMP and can therefore interface with many open and commercial solvers. 

Currently one of three solvers is chosen by POMATO based on availability and model type:

    - `Clp.jl <https://github.com/jump-dev/Clp.jl>`_ the default solver, which is always installed
      with POMATO. 
    - `ECOS <https://github.com/jump-dev/ECOS.jl>`_. Solver available under GNU General Public
      licence, that allows solving conic constraints that are used in the chance constrained
      formulations in POMATO. 
    - `Gurobi.jl <https://github.com/JuliaOpt/Gurobi.jl>`_. Generally more performant than Clp, but
      requires a valid licence and has to be seperately installed. If Gurobi is available it can be
      used with POMATO and will automatically used. If Gurobi is installed on your system you can
      add the Gurobi.jl package to pomato by running
      :code:`pomato.tools.julia_management.add_gurobi()` as described below.  

Using other solvers is possible, however requires implementation through a few lines of code. If
there is need/interest let me know. 

Lastest
*******

You can also install the latest version available on the construction branch via 

.. code-block:: python

        pip install git+https://github.com/richard-weinhold/pomato.git@construction

This will not only install the construction branch of POMATO but also of the MarketModel, to remain
compadible. 


Managing the Julia environment
******************************

The two julia packages are installed together with POMATO. More specifically, within the
``_installation`` folder of the POMATO package directory (meaning the folder where the package
resides  e.g. in the Lib\site-packages folder within the python environment for python installation,
or accessible via :code:`pomato.__path__[0]`.) POMATO will create a julia environment consisting of
MarketModel and RedundancyRemoval, and subsequently all their dependencies. While manually changing
this environment is possible, POMATO provides some means to manage the julia environment via a
functions available in :code:`pomato.tools.julia_management`. 

  - :code:`pomato.tools.julia_management.instantiate_julia(redundancyremoval_branch="master", marketmodel_branch="master")` 
    will re-install the julia packages MarketModel and RedundancyRemoval from their git
    repositories. Allows to specify a specific branch to use for the repsective modules.
  - :code:`pomato.tools.julia_management.instantiate_julia_dev(redundancyremoval_path, marketmodel_path)`
    will instantiate the julia environment from local repositories. This is useful when actively
    changing the code.
  - :code:`pomato.tools.julia_management.add_gurobi()` adds the gurobi solver to the julia environment. 

Developping POMATO
******************

Changing code to implement new features or improving implementation and functionality for POMATO and
its two julia modules MarketModel and RedundancyRemoval requires a suitable installation where local 
changes are immediately used without re-installing or updating the packages. 

For this, a setup where POMATO and its Julia modules MarketModel and RedundancyRemoval are installed
from local repositories in development mode is advisable. 

To set POMATO up in this manner: 

    - Have python and julia installed on your system. Python version has to be >= 3.6, i personally
      use python 3.7, but 3.8 and 3.9 should work as well. For julia version 1.5 is recommended. 
    - Clone the repositories pomato, MarketModel and RedundancyRemoval on you machine. - Install
      pomato into a environment of your choice via the including the -e flag:
      :code:`pip install -e path-to-pomato-repository` 
    - Start a python session and instantiate the julia environment from the local repositories:
    
    .. code-block:: python

        >> from pomato.tools.julia_management import instantiate_julia_dev
        >> instantiate_julia_dev(path-to-RedundancyRemoval-repository, 
                                 path-to-MarketModel-repository)

    - This command instantiates a julia environment within the ``_installation`` subfolder of the
      pomato repository. 
