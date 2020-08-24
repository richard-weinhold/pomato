Folder Structure
-------------------------

Running pomato requires besides input data, temporary directories to store data and results from the 
market model or redundancy removal. The idea is to have a working folder to run the model including 
all relevant inputs and store outputs and results. The folder structure is defined as follows and will
be automatically generated when running the model: 

::

    working_directory
    ├── profiles/
    ├── logs/
    ├── data_input/
    ├── data_output/
    ├── data_temp/
        ├── bokeh_files/
        └── julia_files/
             ├── cbco_data/
             ├── results/
             └── data/
