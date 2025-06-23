# Add path in order to be compatible with Linux
import sys, os

import numpy as np

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.gui.freud_gui import Freud
from freud.talos_utils.sleep_sets.hsp import HSPAgent, HSPOrganization
from roma import io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SRC_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'

# Number of .sg files to visualize
N = 10

# -----------------------------------------------------------------------------
# (2) Get folder list
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, data_dir=SRC_PATH)

patient_dict = ha.filter_patients_meta(should_have_annotation=1)
folder_list = ha.convert_to_folder_names(patient_dict, local=True)[:N]

sg_path_list = [os.path.join(TGT_PATH, HSPOrganization(p).get_sg_file_name(
  dtype=np.float16, max_sfreq=128)) for p in folder_list]

signal_groups = [io.load_file(p, verbose=True)
                 for p in sg_path_list if os.path.exists(p)]

Freud.visualize_signal_groups(
  signal_groups, title='HSP', default_win_duration=9999999,
)
