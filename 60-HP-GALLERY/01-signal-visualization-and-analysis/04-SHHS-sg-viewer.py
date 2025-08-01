# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

print(f'[SHHS] Solution dir = {SOLUTION_DIR}')
sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from freud.gui.freud_gui import Freud, SignalGroup
from freud.datasets.shhs import SHHS

from roma import io



# -----------------------------------------------------------------------------
#  O Configuration
# -----------------------------------------------------------------------------
DATA_DIR = r'E:\data\shhs'
META_FILE_NAME = r'SHHS-CVD-250709.meta'

# -----------------------------------------------------------------------------
#  I Load and visualize data
# -----------------------------------------------------------------------------
# shhs = SHHS(DATA_DIR, meta_file_name=META_FILE_NAME)
# sg_file_path = shhs.sg_file_list[0]

sg_file_path = r"E:\data\shhs\shhs_sg\200077-1(float16,100Hz).sg"
sg: SignalGroup = io.load_file(sg_file_path, True)

Freud.visualize_signal_groups(
  [sg], title='SHHS', default_win_duration=9999999)

