from freud.gui.freud_gui import Freud
from freud.talos_utils.sleep_sets.sleepeason import SleepEason
from fnmatch import fnmatch
from pictor.objects import SignalGroup
from roma import finder
from roma import io



# Set directories
data_dir = r'../../../data/'
data_dir += 'sleepeasonx'

prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
pattern = f'{prefix}*.sg'
# pattern = f'*SC4081*'
# pattern = f'*SC4712*'

# Select .sg files
sg_file_list = finder.walk(data_dir, pattern=pattern)

signal_groups = []
for path in sg_file_list:
  sg = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Visualize signal groups
Freud.visualize_signal_groups(signal_groups, figure_size=(11, 6),
                              default_win_duration=999999)





