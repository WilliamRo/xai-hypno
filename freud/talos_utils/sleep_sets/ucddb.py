from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from roma import io
from tframe import console
from typing import List

import os
import numpy as np



class UCDDB(SleepSet):
  """This database contains 25 full overnight polysomnograms with simultaneous
  three-channel Holter ECG, from adult subjects with suspected sleep-disordered
  breathing. A revised version of this database was posted on 1 September 2011.

  Reference: https://physionet.org/content/ucddb/1.0.0/
  """

  CHANNEL_NAMES = ['C3A2', 'C4A1', 'ECG', 'Lefteye', 'RightEye', 'EMG',
                   'SpO2', 'Sound', 'Flow', 'Sum', 'ribcage', 'abdo',
                   'BodyPos', 'Pulse']

  ANNO_LABELS = ['Wake', 'REM', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
                  'Artifact', 'Indeterminate']

  class DetailKeys:
    number = 'Study Number'
    height = 'Height (cm)'
    weight = 'Weight (kg)'
    gender = 'Gender'
    bmi = 'BMI'
    age = 'Age'
    sleepiness_score = 'Epworth Sleepiness Score'
    study_duration = 'Study Duration (hr)'
    sleep_efficiency = 'Sleep Efficiency (%)'
    num_blocks = 'No of data blocks in EDF'

  # region: Data Loading

  @classmethod
  def load_sg_from_raw_files(cls, data_dir, pid, **kwargs):
    df = kwargs.pop('df')
    rec_fp = kwargs.pop('rec_fp')

    # Get detail
    detail_dict = df.loc[df[cls.DetailKeys.number] == pid.upper()].to_dict(
      orient='index').popitem()[1]

    # (1) Read .rec file
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      rec_fp, dtype=np.float16, allow_rename=True)

    # (2) Read ECG data TODO: currently ommited
    fp = os.path.join(data_dir, pid + '_lifecard.edf')
    # ecg_signals: List[DigitalSignal] = cls.read_digital_signals_mne(fp)

    # (3) Read stage labels
    fp = os.path.join(data_dir, pid + '_stage.txt')
    assert os.path.exists(fp)
    with open(fp, 'r') as stage:
      stage_ann = [int(line.strip()) for line in stage.readlines()]
    stages = np.array(stage_ann)
    stages[stages > 7] = 7
    assert max(stages) <= 7

    # Wrap data into signal group
    sg = SignalGroup(digital_signals, label=f'{pid}', **detail_dict)
    sg.set_annotation(cls.ANNO_KEY_GT_STAGE, 30, stages, cls.ANNO_LABELS)

    return sg

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of UCDDB dataset is as follows:

       ucddb
         |- SubjectDetails.xls
         |- ucddb002.rec
         |- ucddb002_lifecard.edf
         |- ucddb002_respevt.txt
         |- ucddb002_stage.txt
         |- ...

    Currently, only <pid>.rec and <pid>_stage.txt files are used.

    Parameters
    ----------
    :param data_dir - root directory of UCDDB dataset
    """
    import pandas as pd

    signal_groups: List[SignalGroup] = []

    # Read SubjectDetails.xls
    xls_path = os.path.join(data_dir, 'SubjectDetails.xls')
    df = pd.read_excel(xls_path)

    # Get all .edf files
    rec_file_list: List[str] = walk(data_dir, 'file', '*.rec*')
    n_patients = len(rec_file_list)

    # Read records in order
    for i, rec_fp in enumerate(rec_file_list):
      # Get id
      pid: str = os.path.split(rec_fp)[-1].split('.r')[0]

      load_raw_sg = lambda: cls.load_as_raw_sg(
        data_dir, pid, n_patients=n_patients, i=i, df=df, rec_fp=rec_fp,
        **kwargs)

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

  # endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/ucddb'

  tic = time.time()
  preprocess = 'iqr;128'
  ds = UCDDB.load_as_sleep_set(data_dir, overwrite=1, preprocess=preprocess)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  ds.show()


