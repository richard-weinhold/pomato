
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# %% Init POMATO with the options file and the dataset
import pomato

wdir = Path("C:/Users/riw/tubCloud/Uni/Market_Tool/pomato_studies")
mato = pomato.POMATO(wdir=wdir, options_file="profiles/ieee118.json")
mato.load_data('data_input/pglib_opf_case118_ieee.m')


mato.options["optimization"]["type"] = "zonal"
mato.options["grid"]["cbco_option"] = "clarkson"

mato.create_grid_representation()

mato.grid

t = mato.grid_representation.grid_representation

print("end")

