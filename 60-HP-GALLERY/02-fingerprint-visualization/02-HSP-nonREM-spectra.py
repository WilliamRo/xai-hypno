# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['32-SC', 'dev/tools',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from spectra_explorer import SpectraExplorer, SignalGroup
from roma import io

import a00_common as hub
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_378

CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']

N = 50

# -----------------------------------------------------------------------------
# (2) Generate sg tuples
# -----------------------------------------------------------------------------
# (2.1) Get patient_dict
sg_groups = []
meta = {}
patient_dict = hub.ha.load_subset_dict(SUBSET_FN, max_subjects=N)

for pid, ses_dict in patient_dict.items():
  group = []
  for ses_id, ses_info in ses_dict.items():
    bids_id = pid.split(ses_info['site_id'])[1]
    lb = f"{bids_id[2:]}-{ses_id.split('-')[1]}"

    # (2.2.1) Load sg
    ho = hub.HSPOrganization(ses_id=ses_id, sub_id=pid, data_dir=hub.DATA_DIR)
    sg_fn = ho.get_sg_file_name(np.float16, 128)
    sg_path = os.path.join(hub.SG_DIR, sg_fn)
    sg: SignalGroup = io.load_file(sg_path, verbose=True)
    sg.label = lb
    group.append(sg)

    # (2.2.2) Set meta
    meta[lb] = {k: ses_info[k] for k in ('age', 'gender')}

  sg_groups.append(group)

# -----------------------------------------------------------------------------
# (3) Visualize
# -----------------------------------------------------------------------------
# Visualize signal groups
ee = SpectraExplorer.explore(sg_groups, channels=CHANNELS, meta=meta,
                             figure_size=(12, 4))

