from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from roma import console
from typing import List

import os
import numpy as np



class MASS(SleepSet):
  """The Montreal Archive of Sleep Studies (MASS) is an open-access and
  collaborative database of laboratory-based polysomnography (PSG) recordings.
  Its goal is to provide a standard and easily accessible source of data for
  benchmarking the various systems developed to help the automation of sleep
  analysis. This cohort comprises polysomnograms of 200 complete nights
  recorded (SS1-SS5: 97 males aged 42.9 ± 19.8 years and 103 females aged
  38.3 ± 18.9 years; total sample 40.6 ± 19.4 years, age range: 18–76 years).

       subjects#      age      EEG#    EEG-REF   stage rule
       male+female
  ----------------------------------------------------
  SS1  34+19=53       55-76    17/19   CLE/LER   AASM
  SS2  8+11=19        18-33    19      CLE       R&K
  SS3  29+33=62       20-69    20      LER       AASM
  SS4  14+26=40       18-35    4       CLE       R&K
  SS5  13+13=26       20-59    20      LER       R&K
  ----------------------------------------------------
  """

  CHANNELS = {}

  ANNO_LABELS = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                 'Sleep stage ?']

  # region: Data Loading

  @classmethod
  def load_sg_from_raw_files(cls, data_dir, pid, **kwargs):
    """Load signal group from raw files.

    Args:
      data_dir: str
      pid: str, e.g., '01-0001'
      **kwargs: dict
    """
    assert isinstance(pid, str)
    ss_id = int(pid[:2])

    ss_fn = f'mass{ss_id}'

    edf_path = os.path.join(data_dir, ss_fn, f'01-{pid} PSG.edf')
    anno_path = os.path.join(data_dir, ss_fn, f'01-{pid} Base.edf')

    # (1) read psg data as digital signals
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      edf_path, dtype=np.float16, **kwargs)

    # (2) read annotation
    maverick = ['01-0035']
    if ss_id in (1, 3) and pid not in maverick:
      labels = [
        'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
        'Sleep stage 3', 'Sleep stage R', 'Sleep stage ?']
    else: labels = cls.ANNO_LABELS

    annotation = cls.read_annotations_mne(anno_path, labels=labels)

    # Wrap data into signal group
    sg = SignalGroup(digital_signals, label=f'{pid}')
    sg.annotations[cls.ANNO_KEY_GT_STAGE] = annotation

    return sg


  @classmethod
  def load_as_signal_groups(cls, data_dir, ssid, max_sfreq=128,
                            **kwargs) -> List[SignalGroup]:
    """Directory structure of MASS dataset is as follows:

       MASS_ROOT  := data_dir
         |- mass1
            |- 01-01-0001 Base.edf
            |- 01-01-0001 PSG.edf
            |- 01-01-0002 Base.edf
            |- 01-01-0002 PSG.edf
            |- ...
         |- mass2
            |- 01-02-0001 Base.edf
            |- 01-02-0001 PSG.edf
            |- 01-02-0002 Base.edf
            |- 01-02-0002 PSG.edf
            |- ...
         |- ...

    PID is defined as 0X-00YY, where X represents SSID.

    Parameters
    ----------
    :param data_dir: MASS_ROOT
    """
    # (0) Get configs
    JUST_CONVERSION = kwargs.get('just_conversion', False)
    PREPROCESS = kwargs.get('preprocess', '')

    # (1) Find patient IDs
    ss_path = os.path.join(data_dir, f'mass{ssid}')
    pids = [fn[3:10] for fn in walk(
      ss_path, 'file', '*Base.edf', return_basename=True)]
    n_patients = len(pids)

    # (2) Load signal groups
    signal_groups: List[SignalGroup] = []

    for i, pid in enumerate(pids):
      # (2.1) Create a function to load raw signal group
      load_raw_sg = lambda: cls.load_as_raw_sg(
        data_dir, pid, n_patients=n_patients, i=i,
        max_sfreq=max_sfreq, **kwargs)

      # (2.2) Parse pre-process configs
      pp_configs, suffix = cls.parse_preprocess_configs(PREPROCESS)
      # TODO: currently we don't support suffix
      assert suffix == ''

      # (2.3) Try to load raw signal group
      if suffix == '':
        try:
          sg = load_raw_sg()
        except Exception as e:
          import traceback
          console.warning(f'Failed to load {pid}. Error: {e}')
          traceback.print_exc()
          continue

        if not JUST_CONVERSION: signal_groups.append(sg)
        continue

      # (2.4) TODO


      # This is for 00-data-conversion scripts
      if JUST_CONVERSION: signal_groups.clear()

    # (-1) Show status
    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups

# endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r''

  tic = time.time()

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')


