from freud.talos_utils.slp_set import SleepSet
from fnmatch import fnmatch
from pictor.objects.signals.signal_group import Annotation
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from roma import console
from typing import List

import os
import numpy as np



class SleepEDFx(SleepSet):
  """The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep
  recordings, containing EEG, EOG, chin EMG, and event markers. Some records
  also contain respiration and body temperature. Corresponding hypnograms
  (sleep patterns) were manually scored by well-trained technicians according
  to the Rechtschaffen and Kales manual, and are also available.
   The data comes from two studies, ...

  Reference: https://www.physionet.org/content/sleep-edfx/1.0.0/
  """

  CHANNELS = {'1': 'EEG Fpz-Cz',
              '2': 'EEG Pz-Oz',
              '3': 'EOG horizontal',
              '4': 'Resp oro-nasal',
              '5': 'EMG submental',
              '6': 'Temp rectal',
              '7': 'Event marker'}

  ANNO_LABELS = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                 'Movement time', 'Sleep stage ?']

  # region: Data Loading

  @classmethod
  def load_sg_from_raw_files(cls, data_dir, pid, **kwargs):
    hypno_fn = kwargs.pop('hypno_fn')

    # Create sg from raw file
    psg_fn = f'{pid}0-PSG.edf'

    # (1) read psg data as digital signals
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      os.path.join(data_dir, psg_fn), dtype=np.float16)

    # (2) read annotation
    annotation = cls.read_annotations_mne(
      os.path.join(data_dir, hypno_fn), labels=cls.ANNO_LABELS)

    # Wrap data into signal group
    sg = SignalGroup(digital_signals, label=f'{pid}')
    sg.annotations[cls.ANNO_KEY_GT_STAGE] = annotation

    return sg

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of SleepEDFx dataset is as follows:

       sleep-edf-database-expanded-1.0.0
         |- sleep-cassette
            |- SC4001E0-PSG.edf
            |- SC4001EC-Hypnogram.edf
            |- ...
         |- sleep-telemetry
            |- ST7011J0-PSG.edf
            |- ST7011JP-Hypnogram.edf
            |- ...

    However, this method supports loading SleepEDFx data from arbitrary
    folder, given that this folder contains SleepEDFx data.

    Parameters
    ----------
    :param data_dir: a directory contains pairs of *-PSG.edf and *-Hypnogram.edf
    """
    signal_groups: List[SignalGroup] = []

    # Traverse all hypnogram files
    hypno_file_names: List[str] = walk(data_dir, 'file', '*Hypnogram*',
                                       return_basename=True)

    if 'fn_pattern' in kwargs:
      pat = kwargs['fn_pattern']
      hypno_file_names = [fn for fn in hypno_file_names if fnmatch(fn, pat)]

    n_patients = len(hypno_file_names)
    for i, hypno_fn in enumerate(hypno_file_names):
      # Parse patient ID and get find PSG file name
      pid = hypno_fn.split('-')[0][:7]

      load_raw_sg = lambda: cls.load_as_raw_sg(
        data_dir, pid, n_patients=n_patients, i=i, hypno_fn=hypno_fn, **kwargs)

      # Parse pre-process configs
      pp_configs, suffix = cls.parse_preprocess_configs(
        kwargs.get('preprocess', ''))

      if suffix == '':
        signal_groups.append(load_raw_sg())
        continue

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, pid + f'({suffix})' + '.sg')
      pp_kwargs = kwargs.copy()
      if kwargs.get('overwrite_pp', False): pp_kwargs['overwrite'] = True
      if cls.try_to_load_sg_directly(pid, sg_path, n_patients, i,
                                     signal_groups, **pp_kwargs): continue

      # Load raw signal group and preprocess
      sg = cls.preprocess_sg(load_raw_sg(), pp_configs)
      signal_groups.append(sg)

      # Save sg if necessary
      cls.save_sg_file_if_necessary(
        pid, sg_path, n_patients, i, sg, **kwargs)

      # This is for 00-data-conversion scripts
      if kwargs.get('just_conversion', False): signal_groups.clear()

    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups

  @classmethod
  def pp_trim(cls, sg, config):
    """Recommended trim setting: `trim,1800`"""
    trim = 1800 if config == '' else float(config)

    anno: Annotation = sg.annotations[cls.ANNO_KEY_GT_STAGE]
    ds0 = sg.digital_signals[0]

    def pop_last():
      anno.intervals.pop(-1)
      anno.annotations = anno.annotations[:-1]

    # For SleepEDFx data, last interval is usually invalid.
    if anno.intervals[-1][0] >= ds0.ticks[-1]: pop_last()

    T1, T2 = 0, ds0.ticks[-1]

    # `7` represents `Sleep stage ?`
    while anno.annotations[-1] == 7:
      pop_last()
      T2 = anno.intervals[-1][-1]

    # trim start
    if anno.annotations[0] == 0:
      t1, t2 = anno.intervals[0]
      if t2 - t1 > trim:
        T1 = t2 - trim
        anno.intervals[0] = (T1, t2)

    # trim end
    if anno.annotations[-1] == 0:
      t1, t2 = anno.intervals[-1]
      if t2 - t1 > trim:
        T2 = t1 + trim
        anno.intervals[-1] = (t1, T2)

    for i in range(len(sg.digital_signals)):
      sg.digital_signals[i] = sg.digital_signals[i][T1:T2]

  # endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/sleepedfx'
  data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'

  tic = time.time()
  preprocess = ''
  preprocess = 'trim,1800;iqr;128'

  fn_pattern = '*SC*'
  fn_pattern = '*SC4[9]*'
  ds = SleepEDFx.load_as_sleep_set(
    data_dir, overwrite=0, fn_pattern=fn_pattern, preprocess=preprocess,
    overwrite_pp=1)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  # ds.show()


