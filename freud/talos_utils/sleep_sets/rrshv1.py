from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import SignalGroup, DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from roma.spqr.finder import walk
from roma import console
from typing import List

import os
import numpy as np



class RRSHSCv1(SleepSet):
  """This class is for wrapping data exported from Compumedics PSG devices.
  """

  CHANNEL_NAMES = ['E1-M2', 'E2-M2', 'Chin 1-Chin 2', 'F3-M2', 'C3-M2', 'O1-M2',
                   'F4-M1', 'C4-M1', 'O2-M1', 'RIP ECG', 'Pleth',
                   'Nasal Pressure', 'Therm', 'Thor', 'Abdo', 'Sum', 'SpO2',
                   'Snore', 'Leg/L', 'Leg/R', 'PositionSen', 'Pulse']

  ANNO_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Unknown']

  # region: Data Loading

  @classmethod
  def load_sg_from_raw_files(cls, data_dir, pid, **kwargs):
    import xml.dom.minidom as minidom

    edf_fn = kwargs.get('edf_fn')
    max_sfreq = kwargs.get('max_sfreq', 128)

    # (1) read psg data as digital signals
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      os.path.join(data_dir, edf_fn), dtype=np.float16, max_sfreq=max_sfreq)

    # Wrap data into signal group
    sg = SignalGroup(digital_signals, label=f'{pid}')

    # (2) read annotations
    xml_fp = os.path.join(data_dir, f'{pid}.xml')
    xml_root = minidom.parse(xml_fp).documentElement

    # (2.1) set stage annotations
    stage_elements = xml_root.getElementsByTagName('SleepStage')
    stages = np.array([int(se.firstChild.data) for se in stage_elements])
    stages[stages == 5] = 4
    stages[stages == 9] = 5
    sg.set_annotation(cls.ANNO_KEY_GT_STAGE, 30, stages, cls.ANNO_LABELS)

    # (2.2) set events annotations
    events = xml_root.getElementsByTagName('ScoredEvent')
    # event_keys = ['Limb Movement (Left)', 'Limb Movement (Right)']

    anno_dict = {}
    for eve in events:
      nodes = eve.childNodes
      tagNames = [n.tagName for n in nodes]
      if tagNames != ['Name', 'Start', 'Duration', 'Input']: continue

      key = nodes[0].childNodes[0].data
      key = 'event ' + key.replace(' ', '-')
      input_channel = nodes[3].childNodes[0].data
      if key not in anno_dict: anno_dict[key] = Annotation(
        [], labels=input_channel)

      # Append interval
      start, duration = [float(nodes[i].childNodes[0].data) for i in (1, 2)]
      anno_dict[key].intervals.append((start, start + duration))

    sg.annotations.update(anno_dict)

    return sg


  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of RRSHSCv1 dataset is as follows:

       data-root
         |- CYG.edf                # PSG data
         |- CYG.xml                # annotation
         |- ...

    Parameters
    ----------
    :param data_dir: a directory contains pairs of *.edf and *.xml.XML files
    :param max_sfreq: maximum sampling frequency
    """
    signal_groups: List[SignalGroup] = []

    # Traverse all .edf files
    edf_file_names: List[str] = walk(data_dir, 'file', '*.edf',
                                     return_basename=True)
    n_patients = len(edf_file_names)

    for i, edf_fn in enumerate(edf_file_names):
      # Parse patient ID and get find PSG file name
      pid = edf_fn.split('.')[0]

      load_raw_sg = lambda: cls.load_as_raw_sg(
        data_dir, pid, n_patients=n_patients, i=i, edf_fn=edf_fn,
        suffix='(max_sf_128)', **kwargs)

      # Parse pre-process configs
      pp_configs, suffix = cls.parse_preprocess_configs(
        kwargs.get('preprocess', ''))

      if suffix == '':
        signal_groups.append(load_raw_sg())
        continue

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, pid + f'({suffix})' + '.sg')
      if cls.try_to_load_sg_directly(pid, sg_path, n_patients, i,
                                     signal_groups, **kwargs): continue

      # Load raw signal group and preprocess
      sg = cls.preprocess_sg(load_raw_sg(), pp_configs)
      signal_groups.append(sg)

      # Save sg if necessary
      cls.save_sg_file_if_necessary(pid, sg_path, n_patients, i, sg, **kwargs)

    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups


  @classmethod
  def pp_trim(cls, sg: SignalGroup, config):
    assert config == ''

    anno: Annotation = sg.annotations[cls.ANNO_KEY_GT_STAGE]

    # Find T1 and T2 based on annotation curve
    ticks, labels = anno.curve

    UNLABELED = 5
    MAX_IDLE_EPOCHS = 5
    i, T1, T2 = 0, None, None
    while i < len(ticks):
      t1, t2, lb = ticks[i], ticks[i+1], labels[i]

      # If long unlabeled period is detected
      if lb == UNLABELED and t2 - t1 > MAX_IDLE_EPOCHS * 30:
        if T1 is None: T1 = t2
        else:
          T2 = t1
          break

      # Move cursor forward
      i += 2

    assert T1 is not None
    if T2 is None: T2 = ticks[-1]

    # Trim digital-signals in sg
    sg.truncate(start_time=T1, end_time=T2)

  # endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/rrsh-night'
  # data_dir = r'../../../data/rrsh-band'

  tic = time.time()
  preprocess = 'trim;iqr'
  # preprocess = ''
  ds = RRSHSCv1.load_as_sleep_set(data_dir, overwrite=0,
                                  preprocess=preprocess)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  ds.show(default_win_duration=100000)
