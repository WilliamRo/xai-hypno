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
from roma import finder
from roma import io

import sc as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
PATTERN = f'*(trim1800;128).sg'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

N = 10

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

sg_label_list = [sg.label for sg in signal_groups]
# Load meta
meta = hub.sc_tools.load_sc_meta(hub.XLSX_PATH, sg_label_list)

lbs_1, lbs_2 = hub.sc_tools.get_paired_sg_labels(
  sg_label_list, return_two_lists=True)

# Sort label list by name
lbs_1 = sorted(lbs_1, key=lambda lb: int(meta[lb]['age']))
lbs_2 = sorted(lbs_2, key=lambda lb: int(meta[lb]['age']))

sg_pairs = []
for lb1, lb2 in zip(lbs_1, lbs_2):
  sg1 = [sg for sg in signal_groups if sg.label == lb1][0]
  sg2 = [sg for sg in signal_groups if sg.label == lb2][0]
  sg_pairs.append((sg1, sg2))

# Visualize signal groups
ee = SpectraExplorer.explore(sg_pairs, channels=CHANNELS, meta=meta,
                             figure_size=(10, 4))

