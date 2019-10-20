POMATO - Power Market Tool
============================

[![Build Status](https://travis-ci.org/robert-mieth/pomato.svg?branch=master)](https://travis-ci.org/robert-mieth/pomato)
[![Coverage Status](https://coveralls.io/repos/github/robert-mieth/pomato/badge.svg?branch=master)](https://coveralls.io/github/robert-mieth/pomato?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pomato/badge/?version=latest)](http://pomato.readthedocs.io/en/latest/?badge=latest)

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
Its optimization kernel uses Julia. Julia needs to be installed and added to your system environment (it needs to be callable via console) manually. Pomato requires *Julia 0.6.2* or higher. Download the Julia Command line version from the [Julia-Website](https://julialang.org/).


Installation
------------

After you have installed Python and Julia in the correct versions (see above), use the `setup.py` file to install pomato. Example command in console:

```
python setup.py install
```

For a quick start to Pomato check out the [DOCS](http://pomato.readthedocs.io/en/latest/) and the `run_pomato_simple_example.py` script. *Note*: In the first run, Julia will install all the necessary packages. This might take a couple of minutes depending on your machine and your internet connection. 





