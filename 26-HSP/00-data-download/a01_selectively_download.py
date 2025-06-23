# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')
sys.path.append(SOLUTION_DIR)

import os


from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma.console.console import console

# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
ACCESS_POINT_NAME = 's3://arn:aws:s3:us-east-1:184438910517:accesspoint'
DATA_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')

META_TIME_STAMP = '20231101'
META_PATH = os.path.join(
  META_DIR, DATA_DIR, f'bdsp_psg_master_{META_TIME_STAMP}.csv')

# -----------------------------------------------------------------------------
# (2) Select folders
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP, ACCESS_POINT_NAME)

patient_dict = ha.filter_patients_meta(
  should_have_annotation=True,
  should_have_psq=True,              # should have PSQ
)

folder_list = ha.convert_to_folder_names(patient_dict)

console.show_status(f'{len(patient_dict)} patients included.')

# For demo, only download the first folder
ha.download_folders(folder_list[:2])



