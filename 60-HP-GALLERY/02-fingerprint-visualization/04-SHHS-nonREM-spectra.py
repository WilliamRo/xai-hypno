# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['35-SHHS', 'dev/tools',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from spectra_explorer import SpectraExplorer, SignalGroup
from roma import io

import shhs as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
CHANNELS = ['EEG C3-A2', 'EEG C4-A1']

N = 100

# -----------------------------------------------------------------------------
# (2) Select .sg files and visualize
# -----------------------------------------------------------------------------
patient_dict = hub.sa.actual_two_visits_dict

sg_pairs = []
for i, pid in enumerate(patient_dict.keys()):
  if i > N: break

  sg_labels = [hub.sa.get_sg_label(pid, sid) for sid in ('1', '2')]
  sg_file_names = [hub.sa.get_sg_file_name(lb) for lb in sg_labels]
  sg_paths = [os.path.join(hub.SG_DIR, fn) for fn in sg_file_names]
  sg_pairs.append([io.load_file(p, True) for p in sg_paths])

meta = {}
for pid in patient_dict.keys():
  for sid in ('1', '2'):
    lb = hub.sa.get_sg_label(pid, sid)
    meta[lb] = {k: patient_dict[pid][sid][k] for k in ('age', 'gender')}

# Visualize signal groups
ee = SpectraExplorer.explore(sg_pairs, channels=CHANNELS, meta=meta,
                             figure_size=(10, 4))

