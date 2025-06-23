from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from pictor import Pictor
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console

import os
import numpy as np



class Freud(Pictor, DialogUtilities):

  def __init__(self, title='Freud', figure_size=(12, 7)):
    super(Freud, self).__init__(title, figure_size=figure_size)

    self.monitor = self.add_plotter(SleepMonitor(self))

  # region: Commands

  def open(self, edf_path: str = None, dtype=float, auto_refresh=True):
    """Open an EDF file. If `edf_path` is not provided, an `open_file` dialog
    will be popped up for manually selecting file."""
    if edf_path is None:
      edf_path = self.load_file_dialog('Please select an EDF file')
    if edf_path in ('', ): return

    fn = os.path.basename(edf_path)

    with self.busy(f'Reading data from `{edf_path}` ...', auto_refresh):
      digital_signals = read_digital_signals_mne(edf_path, dtype=dtype)
      sg = SignalGroup(digital_signals, label=f'{fn}')
      self.objects.append(sg)

    # Refresh if necessary
    if auto_refresh: self.refresh()

  def standardize_stage_annotation(self, standard='aasm', auto_refresh=True):
    from freud.talos_utils.slp_set import SleepSet

    assert standard == 'aasm'
    ANNO_KEY = 'stage Ground-Truth'
    for sg in self.objects:
      assert isinstance(sg, SignalGroup)
      if ANNO_KEY not in sg.annotations: continue

      map_dict = SleepSet.get_map_dict(sg)
      for k in list(map_dict.keys()):
        if map_dict[k] is None: map_dict[k] = 6

      anno: Annotation = sg.annotations[ANNO_KEY]
      anno.labels = SleepSet.AASM_LABELS
      anno.annotations = [map_dict[a] for a in anno.annotations]

      console.show_status(f'Stages in `{sg.label}` has been standardized.')

    if auto_refresh: self.refresh()
  ssa = standardize_stage_annotation

  def show_stage_num_histogram(self):
    data = [sg.total_duration // 30  for sg in self.objects]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.hist(data, bins=50)
    plt.title(f'{min(data)}-{max(data)}')
    fig.show()
  ssnh = show_stage_num_histogram

  # endregion: Commands

  # region: APIs

  @staticmethod
  def visualize_signal_groups(signal_groups, title='Freud',
                              figure_size=(12, 8), **kwargs):
    fre = Freud(title, figure_size)
    for k, v in kwargs.items(): fre.monitor.set(k, v, auto_refresh=False)
    fre.objects = signal_groups
    fre.show()

  # endregion: APIs



if __name__ == '__main__':
  freud = Freud()
  freud.show()
