# Add path in order to be compatible with Linux
import sys, os
os.environ['BYPASS_TF'] = '1'

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['32-SC', 'dev', 'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from ee_walker import EpochExplorer, SignalGroup
from roma import finder
from roma import io

import sc as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
PATTERN = f'*(trim1800;128).sg'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

N = 2

# -----------------------------------------------------------------------------
# (2) Select .sg files and visualize
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(hub.SG_DIR, pattern=PATTERN)
sg_file_list = sg_file_list[:N]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  sg = sg.extract_channels(CHANNELS)
  signal_groups.append(sg)

# Visualize signal groups
ee = EpochExplorer.explore(signal_groups, plot_wave=True)

