from freud.talos_utils.slp_set import SleepSet
from freud.talos_utils.longitudinal_manager import LongitudinalManager
from pictor.objects.signals.signal_group import SignalGroup, DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from roma.spqr.finder import walk
from roma import console, io
from typing import List

import os
import re
import numpy as np



class SRRSH(SleepSet):
  """This class is for wrapping data exported from Compumedics PSG devices.
  """

  CHANNEL_NAMES = ['E1-M2', 'E2-M2', 'Chin 1-Chin 2', 'F3-M2', 'C3-M2', 'O1-M2',
                   'F4-M1', 'C4-M1', 'O2-M1', 'RIP ECG', 'Pleth',
                   'Nasal Pressure', 'Therm', 'Thor', 'Abdo', 'Sum', 'SpO2',
                   'Snore', 'Leg/L', 'Leg/R', 'PositionSen', 'Pulse']
  GROUPS = [('EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
             'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1'),
            ('EOG E1-M2', 'EOG E2-M2')]

  ANNO_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Unknown']

  # region: Data Loading

  @staticmethod
  def channel_map(edf_ck):
    """Map EDF channel names to standard channel names. Used in reading raw data
    """
    # For edf_ck match 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'
    # Using regular expression
    if re.match(r'^[FCO][1234][\-]M[12]$', edf_ck):
      return f'EEG {edf_ck}'

    # In some cases, two EOG channels may use the same reference electrode
    if re.match(r'^E[12][\-]M[12]$', edf_ck):
      return f'EOG {edf_ck}'

    # if re.match(r'^Chin 1-Chin 2', edf_ck): return f'EMG Ch1-Ch2'

    return edf_ck

  @classmethod
  def convert_rawdata_to_signal_groups(
      cls, edf_path_list, tgt_dir, dtype=np.float16, max_sfreq=100, **kwargs):

    # (0) Check target directory
    if not os.path.exists(tgt_dir): os.makedirs(tgt_dir)
    console.show_status(f'Target directory set to `{tgt_dir}` ...')

    # (1) Find file list to convert
    N = len(edf_path_list)
    if not kwargs.get('overwrite', False):
      edf_path_list = [
        edf_path for edf_path in edf_path_list
        if not os.path.exists(SRRSHAgent.edf_path_to_sg_path(tgt_dir, edf_path))
      ]

      console.show_status(
        f'{N - len(edf_path_list)} files already converted.')

    n = len(edf_path_list)
    console.show_status(f'Converting {n}/{N} files ...')

    # (2) Convert files
    for i, edf_path in enumerate(edf_path_list):
      sg_label = SRRSHAgent.edf_path_to_sg_label(edf_path)

      console.show_status(f'Converting {i + 1}/{n} {sg_label} ...')
      console.print_progress(i, n)

      sg: SignalGroup = cls.load_sg_from_raw_files(edf_path, max_sfreq, dtype)

      sg_path = SRRSHAgent.edf_path_to_sg_path(tgt_dir, edf_path)
      io.save_file(sg, sg_path, verbose=True)

    console.show_status(f'Successfully converted {n} files.')


  @classmethod
  def load_sg_from_raw_files(cls, edf_path, max_sfreq=100, dtype=np.float16,
                             **kwargs):
    import xml.dom.minidom as minidom

    N_CHANNELS = sum([len(g) for g in cls.GROUPS])

    # (1) read psg data as digital signals
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      edf_path, dtype=dtype, max_sfreq=max_sfreq,
      chn_map=cls.channel_map, groups=cls.GROUPS, n_channels=N_CHANNELS)

    # Wrap data into signal group
    pid = os.path.basename(edf_path).split('.')[0]
    sg = SignalGroup(digital_signals, label=f'{pid}')

    # (2) read annotations
    xml_fp = edf_path.replace('.edf', '.XML')
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
    """Directory structure of SRRSH dataset is as follows:

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



class SRRSHAgent(LongitudinalManager):

  @classmethod
  def get_sg_file_name(cls, sg_label, dtype=np.float16, max_sfreq=100):
    dtype_str = str(dtype).split('.')[-1].replace('>', '')
    dtype_str = dtype_str.replace("'", '')
    return f'{sg_label}({dtype_str},{max_sfreq}Hz).sg'

  @classmethod
  def edf_path_to_sg_label(cls, edf_path):
    return os.path.basename(edf_path).split('.')[0]

  @classmethod
  def edf_path_to_sg_path(cls, sg_dir, edf_path, dtype=np.float16,
                          max_sfreq=100):
    sg_label = cls.edf_path_to_sg_label(edf_path)
    sg_file_name = cls.get_sg_file_name(sg_label, dtype, max_sfreq)
    return os.path.join(sg_dir, sg_file_name)

  @classmethod
  def get_filepath_groups_from_sg_dir(cls, sg_dir, return_dict=False):
    from roma import finder

    get_label = lambda s: s.split('(')[0].split('/')[-1]
    get_pid = lambda s: get_label(s).split('-')[0]

    sg_file_path_list = finder.walk(sg_dir, 'file', '*.sg')
    sg_labels = [get_label(fn) for fn in sg_file_path_list]

    unique_labels = sorted(list(set([lb.split('-')[0] for lb in sg_labels])))

    groups, group_dict = [], {}
    for lb in unique_labels:
      group = [path for path in sg_file_path_list if get_pid(path) == lb]
      group_dict[lb] = group
      groups.append(group)

    if return_dict: return group_dict
    return groups



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/rrsh-night'
