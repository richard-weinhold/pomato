
import subprocess, json, time, logging
from pathlib import Path
from multiprocessing import Process
import threading, sys
pomato_path = Path.cwd().joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO

# %%


# %%
mato = POMATO(wdir=Path.cwd().parent, options_file="profiles/ieee118.json")
mato.load_data('data_input/pglib_opf_case118_ieee.m')


from pomato.tools import JuliaDeamon

t = JuliaDeamon(mato.logger, mato.wdir, "marketmodel")

t.run()
t.join()

# t.julia_deamon.is_alive()

