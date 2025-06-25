from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals.signal_group import SignalGroup, Annotation

import matplotlib.pyplot as plt
import numpy as np



class ExplorerBase(Pictor):

  STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')

  class Keys(Pictor.Keys):
    STAGES = 'StAgEs'
    EPOCHS = 'EpOcHs'
    CHANNELS = 'ChAnNeLs'

    STAGE_EPOCH_DICT = 'stage_epoch_dict'
    ANNO_KEY_GT_STAGE = 'stage Ground-Truth'
    MAP_DICT = 'Keys::map_dict'


  @property
  def selected_signal_group(self) -> SignalGroup:
    return self.get_element(self.Keys.OBJECTS)


  @classmethod
  def get_map_dict(cls, sg: SignalGroup):
    """This method unifies the stage labels in the annotation to the following
       five stages: 'W', 'N1', 'N2', 'N3', 'R'.
    """
    anno: Annotation = sg.annotations[cls.Keys.ANNO_KEY_GT_STAGE]

    def _init_map_dict(labels):
      map_dict = {}
      for i, label in enumerate(labels):
        if 'W' in label: j = 0
        elif '1' in label: j = 1
        elif '2' in label: j = 2
        elif '3' in label or '4' in label: j = 3
        elif 'R' in label: j = 4
        else: j = None
        map_dict[i] = j
        # console.supplement(f'{label} maps to {j}', level=2)
      return map_dict

    return sg.get_from_pocket(
      cls.Keys.MAP_DICT, initializer=lambda: _init_map_dict(anno.labels))


  @classmethod
  def get_sg_stage_epoch_dict(cls, sg: SignalGroup):
    def _init_sg_stage_epoch_dict():
      ds = sg.digital_signals[0]
      T = int(ds.sfreq * 30)
      # Get annotation
      anno: Annotation = sg.annotations[cls.Keys.ANNO_KEY_GT_STAGE]
      # Get reshaped tape
      E = ds.data.shape[0] // T
      tape = ds.data[:E * T]
      tape = tape.reshape([E, T, ds.data.shape[-1]])
      # Generate map_dict
      map_dict = cls.get_map_dict(sg)

      se_dict, cursor = {k: [] for k in cls.STAGE_KEYS}, 0
      for interval, anno_id in zip(anno.intervals, anno.annotations):
        n = int((interval[-1] - interval[0]) / 30)
        sid = map_dict[anno_id]
        if sid is not None:
          skey = cls.STAGE_KEYS[map_dict[anno_id]]
          for i in range(cursor, cursor + n):
            if i < len(tape): se_dict[skey].append(tape[i])
        cursor += n

      return se_dict

    return sg.get_from_pocket(
      cls.Keys.STAGE_EPOCH_DICT, initializer=_init_sg_stage_epoch_dict)


