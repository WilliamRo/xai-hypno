from typing import List
from freud.talos_utils.slp_config import SleepConfig
from freud.talos_utils.slp_set import SleepSet
from freud.talos_utils.sleep_sets.sleepeason import SleepEason
from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx
from freud.talos_utils.sleep_sets.ucddb import UCDDB
from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1

from tframe import console
from tframe.data.base_classes import DataAgent

import os



class SleepAgent(DataAgent):
  """Load sleep data according to `th.data_config` whose syntax is
      '<data_name> (configs)*'
  Here, built-in <data_name>s include
  (1) 'sleepedfx'; (2) 'ucddb'; (3) 'rrshv1'.
  Note that <data_name> is also the corresponding folder name.

  Customized dataset can be registered into `roster` via
  `SleepAgent.register_dataset` method.
  """

  roster = {'sleepedfx': SleepEDFx, 'ucddb': UCDDB, 'rrshv1': RRSHSCv1,
            'sleepeason1': SleepEason, 'sleepeasonx': SleepEason}


  @classmethod
  def load_data(cls) -> List[SleepSet]:
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    ds = cls.load_as_tframe_data()

    # Put tapes into each group
    ds.configure()

    # TODO: PATCH
    ds.NUM_STAGES = th.num_classes
    SleepSet.NUM_STAGES = th.num_classes

    # Split data set
    if isinstance(ds, SleepEason): return ds.split()

    train_ids = list(range(ds.num_signal_groups))
    val_test_ids = [train_ids, train_ids]

    for i, key in enumerate(['val_ids', 'test_ids']):
      if key in th.data_kwargs:
        val_test_ids[i] = [int(s) for s in th.data_kwargs[key].split(',')]
        train_ids = list(set(train_ids) - set(val_test_ids[i]))

    return [ds.get_subset_by_patient_id(ids, name) for name, ids
            in zip(['train', 'val', 'test'],
                   [train_ids, val_test_ids[0], val_test_ids[1]])]


  @classmethod
  def load_as_tframe_data(cls, **kwargs) -> SleepSet:
    """Load data as tframe dataset. The data loading method must be implemented
    in the corresponding subclass of SleepSet
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    if th.data_name not in cls.roster:
      raise KeyError(f'!! Unknown data set `{th.data_name}`.')

    data_dir = os.path.join(th.data_dir, th.data_name)
    ds = cls.roster[th.data_name].load_as_sleep_set(data_dir)

    return ds


  @classmethod
  def register_dataset(cls, data_name, data_class):
    cls.roster[data_name] = data_class



if __name__ == '__main__':
  from freud.talos_utils.slp_config import SleepConfig as Hub

  th = Hub(as_global=True)

  th.data_config = 'sleepedfx 1,2;3'
  th.data_dir = r'../../data/'

  ds: SleepSet = SleepAgent.load_data()
  # ds = SleepAgent.load_as_tframe_data(th)
  # ds.signal_groups[0].truncate(20000, 60000)
  # sg = ds.signal_groups[0]
  val_set = ds.validation_set
  # ds.show()

