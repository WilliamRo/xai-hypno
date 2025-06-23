from freud.gui.data_explorers.epoch_explorer import EpochExplorer
from roma import finder
from roma import io



# Set directories
data_dir = r'../../../data/'
data_dir += 'sleepeason1'

prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
pattern = f'{prefix}*.sg'

# Select .sg files
sg_file_list = finder.walk(data_dir, pattern=pattern)

signal_groups = []
for path in sg_file_list:
  sg = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Visualize signal groups
EpochExplorer.explore(signal_groups)



