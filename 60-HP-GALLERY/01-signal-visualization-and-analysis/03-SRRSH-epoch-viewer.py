# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['32-SC', 'dev', 'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'[TASK] Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from ee_walker import EpochExplorer, SignalGroup
from roma import finder
from roma import io

import srrsh as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SG_PATTERN = '*.sg'
CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']

N = 20

# -----------------------------------------------------------------------------
# (2) Select .sg files and visualize
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(hub.PAIR_SG_DIR, pattern=SG_PATTERN)
sg_file_list = sg_file_list[:N]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  sg = sg.extract_channels(CHANNELS)
  signal_groups.append(sg)

# Visualize signal groups
ee = EpochExplorer.explore(signal_groups, plot_wave=True)

