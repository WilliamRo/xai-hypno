"""This foundation-model benchmark SOP involves:
(1) Loading data: D = {x_i, y_i}_i
(2) Extracting fixed-sized feature vectors from data using `Algorithm`
(3) Testing features using frameworks such as k-fold cross-validation by Omix
"""
# Add path in order to be compatible with Linux
import sys, os
os.environ['BYPASS_TF'] = '1'

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['xai-kit', 'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe',
             'hypnomics']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.benchmarks.sop import BenchmarkSOP
from freud.datasets.srrsh import SRRSH



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
# (1) Data config
DATA_DIR = r'..\..\..\data\srrsh_eds'

META_FILE_NAME = r'eds_250801.meta'
GROUPS = ('NT1', 'NT2', 'IH', 'OSA', 'HC')

# (2) Model config
#  note: algorithms should follow the `SignalGroup` protocol
from freud.hypno_tools.ho_alg import HOAlgorithm as Algorithm

# (3) Testbench config
test_bench = 'pipeline'
study_name = f'srrsh_eds_hypnomics_{test_bench}'

overwrite = 0
# -----------------------------------------------------------------------------
#  I Load data
# -----------------------------------------------------------------------------
SRRSH.abbreviation = 'srrsh_eds'
eds = SRRSH(DATA_DIR, meta_file_name=META_FILE_NAME)

# meta = eds.load_meta()
# -----------------------------------------------------------------------------
#  II Run algorithm
# -----------------------------------------------------------------------------
hoa = Algorithm(dataset=eds)

# -----------------------------------------------------------------------------
#  II Run benchmark
# -----------------------------------------------------------------------------
# (1) Generate Omix
bm = BenchmarkSOP(hypno_data=eds, model=hoa, test_bench=test_bench,
                  overwrite=overwrite, study_name=study_name)

omix = bm.generate_omix(target_key='diagnosis',
                        target_labels=('HC', 'NT1', 'NT2', 'IH', 'OSA'),
                        data_name='SRRSH-EDS')

target_labels = ('NT1', 'IH')
omix = omix.select_samples(indices=None, target_labels=target_labels)

# (2) Run pipeline
hold = 1
if hold:
  omix.report()
  omix.show_in_explorer()
  exit()

bm.pipeline_test_bench(
  omix,
  sf_config=[
    ('ucp', {'k': 50, 'threshold': 0.9, 'm': 1000}),
    ('ucp', {'k': 50, 'threshold': 0.7, 'm': 1000}),
    ('ucp', {'k': 100, 'threshold': 0.9, 'm': 1000}),
    ('ucp', {'k': 100, 'threshold': 0.7, 'm': 1000}),
    ('ucp', {'k': 200, 'threshold': 0.9, 'm': 1000}),
    ('ucp', {'k': 200, 'threshold': 0.7, 'm': 1000}),
    ('lasso', {}),
  ],
  ml_config=[
    ('lr', {'n_splits': 5, 'repeats': 5}),
    ('svm', {'n_splits': 5, 'repeats': 5}),
  ],
)
