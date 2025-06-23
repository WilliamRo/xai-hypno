from tframe import console
from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx, SleepSet
from freud.talos_utils.slp_config import SleepConfig

import numpy as np



console.suppress_logging()
th = SleepConfig(as_global=True)
th.data_config = 'sleepedfx 1,2'

data_dir = r'../../../data/sleepedfx-mini'

preprocess = 'trim;iqr'
ds: SleepSet = SleepEDFx.load_as_sleep_set(
  data_dir, overwrite=0, preprocess=preprocess)

# Configure ds so that tapes can be extracted according to th.data_config
ds.configure()

sg = ds.signal_groups[0]
# data, labels = ds._sample_seqs_from_sg(sg, 0, 60, with_stage=True)

# Report overall stage distribution
console.show_info('Overall stage distribution:')
Ns = [len(ds.epoch_table[sid]) for sid in range(5)]
M = sum(Ns)
for sid in range(5): console.supplement(f'[{sid}] {Ns[sid]/M*100:.1f}%')

# Report stage distribution with different epoch_num
for epoch_num in (1, 5, 10, 20):
  th.epoch_num = epoch_num

  batch_size = 500
  batch = ds._get_sequence_randomly(batch_size)

  # Count stage numbers
  labels = np.argmax(
    np.reshape(batch.targets, [epoch_num * batch_size, -1]), axis=-1)

  N = len(labels)
  console.show_info(f'{epoch_num} epochs per sequence, totally {N} epochs')
  for sid in range(5):
    n = sum(labels == sid)
    console.supplement(f'[{sid}] {n/N*100:.1f}%', level=2)
