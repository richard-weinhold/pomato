POMATO - Power Market Tool
============================


Overview
--------

The idea of Pomato is to provide an easy to use tool that inhibits all the power system engineering that is necessary for comprehensive analyses of the modern power market. It is more flexible and powerful than studies in Excel but sets aside the unnecessary detail of engineering software like Integral or PowerFactory. 

Pomato currently includes:

* Data
    * Supports Excel Data
    * Supports MatPowerCase Data

* Grid
    * Fast network representation with recalculated network constraints
    * Including n-1 security constrained dispatch

* Various options for market representation
    * Nodal or Zonal optimization
    * Including Electricity-Heat Coupling

* GUI 
    * Powerful Interactive Plotting
    * Possible Map-Layouts
    * Soon to come: GUI based input data handling

Requirements
------------

Pomato is (until now) a console tool mainly written in python and therefore requires *Python 3*. 
Its optimization kernel uses Julia. Julia needs to be installed and added to your system environment (it needs to be callable via console) manually. Pomato requires *Julia 1.1.0* or higher. Download the Julia Command line version from the [Julia-Website](https://julialang.org/). The folder *project_files* contains env for anaconda (python) and julia make set up easier. 






