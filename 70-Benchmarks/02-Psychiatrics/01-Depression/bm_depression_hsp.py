"""This foundation-model benchmark SOP involves:
(1) Loading data: D = {x_i, y_i}_i
(2) Extracting fixed-sized feature vectors from data using `Algorithm`
(3) Testing features using frameworks such as k-fold cross-validation by Omix
"""
import os
os.environ['BYPASS_TF'] = '1'

from freud.benchmarks.sop import BenchmarkSOP
from freud.datasets.hsp import HSP



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
# (1) Data config
DATA_DIR = r'E:\data\hsp'
DATA_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp'
META_FILE_NAME = r'HSP_depression_150_250701.meta'

# (2) Model config
#  note: algorithms should follow the `SignalGroup` protocol
from freud.hypno_tools.ho_alg import HOAlgorithm as Algorithm

# (3) Testbench config
test_bench = 'pipeline'
study_name = f'hsp_depression_hypnomics_{test_bench}'

overwrite = 0

# -----------------------------------------------------------------------------
#  I Load data
# -----------------------------------------------------------------------------
hsp = HSP(DATA_DIR, meta_file_name=META_FILE_NAME)
meta = hsp.load_meta()

# -----------------------------------------------------------------------------
#  II Run algorithm
# -----------------------------------------------------------------------------
hoa = Algorithm(dataset=hsp)

# -----------------------------------------------------------------------------
#  III Run omix-based benchmark
# -----------------------------------------------------------------------------
bm = BenchmarkSOP(hypno_data=hsp, model=hoa, test_bench=test_bench,
                  overwrite=overwrite, study_name=study_name)


omix = bm.generate_omix(target_key='depression',
                        target_labels=('no-depression', 'depression'),
                        data_name='HSP-Depression')

bm.pipeline_test_bench(
  omix,
  sf_config=[
    ('ucp', {'k': 50, 'threshold': 0.9}),
    ('ucp', {'k': 50, 'threshold': 0.7}),
    ('ucp', {'k': 100, 'threshold': 0.9}),
    ('ucp', {'k': 100, 'threshold': 0.7}),
    ('ucp', {'k': 200, 'threshold': 0.9}),
    ('ucp', {'k': 200, 'threshold': 0.7}),
    ('lasso', {}),
  ],
  ml_config=[
    ('lr', {'n_splits': 2}),
    ('svm', {'n_splits': 2}),
  ],
)
