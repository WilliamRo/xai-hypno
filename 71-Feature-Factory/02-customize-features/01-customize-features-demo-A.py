"""This foundation-model benchmark SOP involves:
(1) Loading data: D = {x_i, y_i}_i
(2) Extracting fixed-sized feature vectors from data using `Algorithm`
(3) Testing features using frameworks such as k-fold cross-validation by Omix
"""
import os

import numpy as np

os.environ['BYPASS_TF'] = '1'

from collections import OrderedDict
from freud.benchmarks.sop import BenchmarkSOP
from freud.datasets.sleepedfx_sc import SleepEDFxSC
from pictor.objects import SignalGroup



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
# (1) Data config
DATA_DIR = r'E:\data\sleepedfx-sc'

# (2) Model config
#  note: algorithms should follow the `SignalGroup` protocol
from freud.hypno_tools.ho_alg import Algorithm

# (3) Testbench config
test_bench = 'pipeline'
study_name = f'ff-sc-gender-demo'
study_name = None

overwrite = 1

# -----------------------------------------------------------------------------
#  I Load data
# -----------------------------------------------------------------------------
sc = SleepEDFxSC(DATA_DIR)

# TODO: this is for demo only
sg_file_list = sc.sg_file_list[15:25]
sc.sg_file_list = sg_file_list

# -----------------------------------------------------------------------------
#  II Run algorithm
# -----------------------------------------------------------------------------
# Define a simple algorithm that extracts the mean of each signal
def calc_stats(sg: SignalGroup) -> OrderedDict:
  s: np.ndarray = sg.digital_signals[0].data
  s = s.astype(np.float32)
  mu, sigma = np.mean(s), np.std(s)

  od = OrderedDict()
  od['mu'] = mu
  od['sigma'] = sigma

  return od

alg = Algorithm(dataset=sc)
alg.set_probe('dummy', func=calc_stats, probe_type='III')
alg.overwrite_type_III = 0

# -----------------------------------------------------------------------------
#  III Run omix-based benchmark
# -----------------------------------------------------------------------------
bm = BenchmarkSOP(hypno_data=sc, model=alg, test_bench=test_bench,
                  overwrite=overwrite, study_name=study_name)

omix = bm.generate_omix(target_key='gender', target_labels=('female', 'male'),
                        data_name='SC-Gender')

omix.show_in_explorer()
