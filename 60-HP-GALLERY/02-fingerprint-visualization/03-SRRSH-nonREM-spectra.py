# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['31-SRRSH', 'dev/tools',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'[TASK] Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from spectra_explorer import SpectraExplorer, SignalGroup
from roma import io, finder

import numpy as np
import srrsh as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SG_PATTERN = '*.sg'
CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']

# -----------------------------------------------------------------------------
# (2) Get sg list
# -----------------------------------------------------------------------------
path_groups = hub.SRRSHAgent.get_filepath_groups_from_sg_dir(
  hub.PAIR_SG_DIR, return_dict=True)

sg_groups = []
for lb, path_list in path_groups.items():
  sg_group = []
  for sg in [io.load_file(p, True) for p in path_list]:
    stage_anno = sg.annotations['stage Ground-Truth']
    sg_group.append(sg)
  sg_groups.append(sg_group)

# -----------------------------------------------------------------------------
# (3) Visualize
# -----------------------------------------------------------------------------
# Visualize signal groups
ee = SpectraExplorer.explore(sg_groups, channels=CHANNELS, figure_size=(12, 5))

