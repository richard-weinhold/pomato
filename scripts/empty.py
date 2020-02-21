import sys
from pathlib import Path

import numpy as np
import pandas as pd

pomato_path = Path.cwd().parent.joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO
