from collections import OrderedDict
from datetime import datetime
from freud.talos_utils.slp_set import SleepSet
from freud.talos_utils.longitudinal_manager import LongitudinalManager
from roma import console, io, Nomear
from pictor.objects.signals.signal_group import Annotation
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from typing import List

import os
import numpy as np



class SHHSet(SleepSet):
  """Reference: ...

  Dataset Folder Structure
  ------------------------
  ...
  """

  # [Protocol] It is important that the annotation labels follow XAI protocol,
  #   that labels uniquely map to (W, 1, 2, 3, R) according to the key
  #   characters, i.e., 'W', '1', '2', '3', 'R'.
  # Example of a invalid wake-stage label: 'Wake|1' (invalid because it contains
  #   both 'W' and '1')
  ANNO_LABELS = ['Wake|0', 'Stage 1 sleep|1', 'Stage 2 sleep|2',
                 'Stage 3 sleep|3', 'Stage 4 sleep|4',  'REM sleep|5',
                 'Unscored|9', 'Movement|6']

  GROUPS = [('EEG C3-A2', 'EEG C4-A1', 'EMG'), ('EOG Left', 'EOG Right')]

  @staticmethod
  def channel_map(edf_ck):
    """Map EDF channel names to standard channel names. Used in reading raw data
    """
    if edf_ck == 'EEG': return 'EEG C3-A2'
    if edf_ck in ('EEG(SEC)', 'EEG sec', 'EEG2', 'EEG 2', 'EEG(sec)'):
      return 'EEG C4-A1'
    if edf_ck == 'EOG(L)': return 'EOG Left'
    if edf_ck == 'EOG(R)': return 'EOG Right'
    return edf_ck

  # region: Data Conversion

  @classmethod
  def load_sg_from_raw_files(cls, edf_path, anno_path, sg_label,
                             dtype=np.float16, max_sfreq=100, **kwargs):
    """Convert an `.edf` and a '.xml' (annotation) file into a SignalGroup.
    """
    # (0) Sanity check
    assert os.path.exists(edf_path) and os.path.exists(anno_path)

    # (1) Read annotations
    annotation = cls.load_shhs_stage_annotation(anno_path)

    # (2) Read psg data as digital signals
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      edf_path, dtype=dtype, max_sfreq=max_sfreq,
      chn_map=cls.channel_map, groups=cls.GROUPS, n_channels=5)

    # (3) Wrap data into signal group
    sg = SignalGroup(digital_signals, label=sg_label)

    sg.annotations[cls.ANNO_KEY_GT_STAGE] = annotation

    return sg

  @classmethod
  def load_shhs_stage_annotation(cls, anno_path):
    """
    """
    import xml.etree.ElementTree as ET

    # Parse the XML file
    tree = ET.parse(anno_path)
    root = tree.getroot()

    # Sanity check
    assert root.find('EpochLength').text == '30'

    # First find SleepStages element, then get children
    label2int = {lb: i for i, lb in enumerate(cls.ANNO_LABELS)}
    intervals, annotations = [], []
    for elem in root.find('ScoredEvents'):
      if elem.find('EventType').text != 'Stages|Stages': continue

      onset = float(elem.find('Start').text)
      duration = float(elem.find('Duration').text)

      anno_int = label2int[elem.find('EventConcept').text]

      # Append interval and annotation
      intervals.append((onset, onset + duration))
      annotations.append(anno_int)

    return Annotation(intervals, annotations, labels=cls.ANNO_LABELS)


  @classmethod
  def convert_rawdata_to_signal_groups(
      cls, edf_anno_label_tuples, tgt_dir, dtype=np.float16, max_sfreq=100,
      **kwargs):

    # (0) Check target directory
    if not os.path.exists(tgt_dir): os.makedirs(tgt_dir)
    console.show_status(f'Target directory set to `{tgt_dir}` ...')

    # (1) Find file list to convert
    N = len(edf_anno_label_tuples)
    if not kwargs.get('overwrite', False):
      edf_anno_label_tuples = [
        (edf, anno, label) for edf, anno, label in edf_anno_label_tuples
        if not os.path.exists(os.path.join(tgt_dir, SHHSAgent.get_sg_file_name(
          label, dtype, max_sfreq)))]

      console.show_status(
        f'{N - len(edf_anno_label_tuples)} files already converted.')

    n = len(edf_anno_label_tuples)
    console.show_status(f'Converting {n}/{N} files ...')

    # (2) Convert files
    for i, (edf_path, anno_path, sg_label) in enumerate(edf_anno_label_tuples):
      console.show_status(f'Converting {i + 1}/{n} {sg_label} ...')
      console.print_progress(i, n)

      sg: SignalGroup = cls.load_sg_from_raw_files(
        edf_path, anno_path, sg_label, dtype, max_sfreq)

      sg_path = os.path.join(
        tgt_dir, SHHSAgent.get_sg_file_name(sg_label, dtype, max_sfreq))
      io.save_file(sg, sg_path, verbose=True)

    console.show_status(f'Successfully converted {n} files.')

  # endregion: Data Conversion



class SHHSAgent(LongitudinalManager):
  """This class is responsible for
   (1) manage patient information

  According to sun2019:
   We also use a subset of the SHHS data set, which contains repeated EEGs
   from the same participant in two visits about 5 years apart, making it
   possible to evaluate the longitudinal reliability of our model at the
   population level. The participant inclusion criteria are
    (1) having EEGs from both visits;
    (2) chronological age between 40 and 80 years at both visits (minimum age
        in SHHS is 40);
    (3) having EEG and sleep stage scoring of high quality according to
        SHHS specifications (Table S3); and
    (4) having no neurological or cardiovascular disease (Table S3).
   We also exclude EEGs with missing sleep stages. As a result, 987 EEGs from
   SHHS visit 1 and 987 paired EEGs from visit 2 are used.
  """

  def __init__(self, meta_dir, data_dir=None, meta_version='0.21.0'):
    self.meta_dir = meta_dir
    self.meta_version = meta_version
    self.data_dir = data_dir

  # region: Properties

  @property
  def meta_path(self): return os.path.join(
    self.meta_dir, f'shhs-harmonized-dataset-{self.meta_version}.csv')

  @Nomear.property()
  def two_visits_dict(self):
    return {k: v for k, v in self.patient_dict.items() if len(v) == 2}

  @Nomear.property()
  def actual_two_visits_dict(self):
    dict_path = self.meta_path.replace(self.META_EXTENSION, '_actual_2.od')
    if os.path.exists(dict_path) and not self.in_pocket('OVERWRITE_PD'):
      return io.load_file(dict_path, verbose=True)

    od = self.generate_actual_2_dict()
    io.save_file(od, dict_path, verbose=True)
    return od

  # endregion: Properties

  # region: Public Methods

  # region: - Patient Information

  @staticmethod
  def generate_patient_dict(meta_path) -> OrderedDict:
    import pandas as pd

    # (0) Read meta data
    df = pd.read_csv(meta_path)

    # (1) Read patients' info as a list of dictionaries
    n_rows = df.shape[0]

    keys = df.columns.to_list()
    key_map = {k: k for k in keys}
    key_map['nsrrid'] = 'pid'
    key_map['visitnumber'] = 'ses_id'
    key_map['nsrr_age'] = 'age'
    key_map['nsrr_sex'] = 'gender'
    key_map['nsrr_race'] = 'race'
    key_map['nsrr_bmi'] = 'bmi'

    patient_dict = OrderedDict()
    for i, row in df.iterrows():
      # (1.1) Read row
      pid = str(row['nsrrid'])
      ses_id = str(row['visitnumber'])

      # (1.2) Create slot
      if ses_id == '1':
        assert pid not in patient_dict
        patient_dict[pid] = OrderedDict()

      patient_dict[pid][ses_id] = OrderedDict()

      # (1.3) Assign information
      for raw_key in keys:
        patient_dict[pid][ses_id][key_map[raw_key]] = row[raw_key]

    # (2) Report progress and return
    console.show_status(f'Successfully read {len(patient_dict)} patients from'
                        f' {meta_path}, altogether {n_rows} rows.')

    return patient_dict

  def generate_actual_2_dict(self):
    N = len(self.two_visits_dict)

    od = OrderedDict()
    for i, pid in enumerate(self.two_visits_dict.keys()):
      console.print_progress(i, N)

      flag = True
      for sid in ('1', '2'):
        edf_path, anno_path = self.get_edf_anno_by_id(pid, sid)

        if any([not os.path.exists(path) for path in (edf_path, anno_path)]):
          flag = False
          continue

      if flag: od[pid] = self.two_visits_dict[pid]

    console.show_status(f'Actual-2-visits-dict (N={len(od)}) generated.')
    return od

  # endregion: - Patient Information

  # region: - Raw Data Path

  def get_edf_anno_by_id(self, pid: str, sid: str):
    assert pid.startswith('20') and sid in ('1', '2')
    edf_path = os.path.join(
      self.data_dir, f'polysomnography/edfs/shhs{sid}/shhs{sid}-{pid}.edf')
    # anno_path = os.path.join(
    #   self.data_dir, f'polysomnography/annotations-events-profusion/shhs{sid}/shhs{sid}-{pid}-profusion.xml')
    anno_path = os.path.join(
      self.data_dir, f'polysomnography/annotations-events-nsrr/shhs{sid}/shhs{sid}-{pid}-nsrr.xml')
    return edf_path, anno_path

  # endregion: - Raw Data Path

  # region: - Data Conversion

  @classmethod
  def get_sg_label(cls, pid, sid): return f'{pid}-{sid}'

  @classmethod
  def get_sg_file_name(cls, sg_label, dtype=np.float16, max_sfreq=100):
    dtype_str = str(dtype).split('.')[-1].replace('>', '')
    dtype_str = dtype_str.replace("'", '')
    return f'{sg_label}({dtype_str},{max_sfreq}Hz).sg'

  # endregion: - Data Conversion

  # endregion: Public Methods



if __name__ == '__main__':
  import pandas as pd

  OVERWRITE = 1


  print('Done!')
