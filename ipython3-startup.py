import argparse, pathlib, re, glob
import pandas as pd
import numpy as np
from partitionsets import partition
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import CategoryToNumberAssignment as c2n
import OBapiExtraction as oboA
import OBOModelling as oboM
import OBOPartitioning as oboP
import OBOValidation as oboV