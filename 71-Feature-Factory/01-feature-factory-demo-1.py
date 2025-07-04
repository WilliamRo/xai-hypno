"""This foundation-model benchmark SOP involves:
(1) Loading data: D = {x_i, y_i}_i
(2) Extracting fixed-sized feature vectors from data using `Algorithm`
(3) Testing features using frameworks such as k-fold cross-validation by Omix
"""
import os
os.environ['BYPASS_TF'] = '1'

from freud.benchmarks.sop import BenchmarkSOP
from freud.datasets.sleepedfx_sc import SleepEDFxSC



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
# (1) Data config
DATA_DIR = r'E:\data\sleepedfx-sc'

# (2) Model config
#  note: algorithms should follow the `SignalGroup` protocol
from freud.hypno_tools.ho_alg import HOAlgorithm as Algorithm

# (3) Testbench config
test_bench = 'pipeline'
study_name = f'ff-sc-gender-demo'

overwrite = 0

# -----------------------------------------------------------------------------
#  I Load data
# -----------------------------------------------------------------------------
sc = SleepEDFxSC(DATA_DIR)

# -----------------------------------------------------------------------------
#  II Run algorithm
# -----------------------------------------------------------------------------
hoa = Algorithm(dataset=sc,
                probe_config=[])

# -----------------------------------------------------------------------------
#  III Run omix-based benchmark
# -----------------------------------------------------------------------------
bm = BenchmarkSOP(hypno_data=sc, model=hoa, test_bench=test_bench,
                  overwrite=overwrite, study_name=study_name)

omix = bm.generate_omix(target_key='gender', target_labels=('female', 'male'),
                        data_name='SC-Gender')

omix.show_in_explorer()
