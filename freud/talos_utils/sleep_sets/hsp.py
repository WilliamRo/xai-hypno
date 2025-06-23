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
import math
import re
import traceback



class HSPSet(SleepSet):
  """Reference: https://bdsp.io/content/hsp/2.0/

  Dataset Folder Structure
  ------------------------
  The folder structure follows the BIDS (Brain Imaging Data Structure)
  specification version 1.7.0 for organizing EEG (electroencephalogram)
  data collected from multiple sites.

  There are four main levels of the folder hierarchy, these are:
  bids -> sub-ID -> ses-ID -> eeg

  Bids-root-folder/
	└── dataset_description.json
	└── participants.json
	└── participants.tsv
	└── README
	└── sub-Id/
		└── ses-01/
			└── sub-SiteIdPatientId_ses-01_scans.tsv
			└── eeg
				└── sub-Id_ses-1_task-psg_annotations.tsv
				└── sub-Id_ses-1_task-psg_channels.tsv
				└── sub-Id_ses-1_task-psg_eeg.edf
				└── sub-Id_ses-1_task-psg_eeg.json
				└── sub-Id_ses-1_task-psg_pre.csv
  """

  ANNO_LABELS = ['Sleep_stage_W', 'Sleep_stage_N1', 'Sleep_stage_N2',
                 'Sleep_stage_N3', 'Sleep_stage_R',  'Sleep_stage_?']

  EEG_EOG = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
             'E1-M[12]', 'E2-M[12]']

  # GROUPS = [('EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
  #            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1'),
  #           ('EOG E1-M2', 'EOG E2-M1',
  #            'EOG E1-M1', 'EOG E2-M2', )]

  GROUPS = [('EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
             'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1')]

  BIPOLAR_GROUPS = [('Fpz', 'Cz', 'Pz', 'Oz')]

  @staticmethod
  def channel_map(edf_ck):
    """Map EDF channel names to standard channel names. Used in reading raw data
    """
    # For edf_ck match 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'
    # Using regular expression
    if re.match(r'^[FCO][1234][\-]M[12]$', edf_ck):
      return f'EEG {edf_ck}'

    # In some cases, two EOG channels may use the same reference electrode
    # e.g., E1-M2, E2-M2 (sub-S0001111190905_ses-1)
    #       E1-M2, E2-M1 (sub-S0001111190905_ses-4)
    if re.match(r'^E[12][\-]M[12]$', edf_ck):
      return f'EOG {edf_ck}'

    return edf_ck

  # region: Data Conversion

  @classmethod
  def load_sg_from_raw_files(cls, ses_dir, dtype=np.float16, max_sfreq=128,
                             bipolar=False, **kwargs):
    """Convert an `.edf` file into a SignalGroup.

    Arg
    ---
    ses_dir: str, session directory
             e.g., ...\hsp_raw\sub-S0001111190905\ses-1
    """

    # (0) Check file completeness
    ho = HSPOrganization(ses_dir)
    if not os.path.exists(ho.edf_path) or not os.path.exists(ho.anno_path):
      raise FileNotFoundError(f'File not found: {ho.edf_path} or {ho.anno_path}')

    # (1) Read annotations
    annotation = cls.load_hsp_annotation(ho.anno_path)

    # (2) Read psg data as digital signals
    if bipolar:
      digital_signals: List[DigitalSignal] = cls.read_bipolar(
        ho.edf_path, dtype=dtype, max_sfreq=max_sfreq)
    else:
      digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
        ho.edf_path, dtype=dtype, max_sfreq=max_sfreq,
        chn_map=cls.channel_map, groups=cls.GROUPS, n_channels=6)

    # if bipolar:
    #   ds: DigitalSignal = digital_signals[0]
    #   fpz_cz, pz_oz = ds['Fpz'] - ds['Cz'], ds['Pz'] - ds['Oz']
    #   data = np.stack([fpz_cz, pz_oz], axis=1)
    #   digital_signals = [DigitalSignal(
    #     data, channel_names=['EEG Fpz-Cz', 'EEG Pz-Oz'],
    #     sfreq=ds.sfreq, label='Fpz-Cz,Pz-Oz')]

    # (3) Wrap data into signal group
    sg = SignalGroup(digital_signals, label=ho.sg_label)
    assert len(sg.channel_names) == (2 if bipolar else 6)

    sg.annotations[cls.ANNO_KEY_GT_STAGE] = annotation

    # (4) Sanity check and return
    record_minus_anno = sg.total_duration - annotation.intervals[-1][1]
    sg.put_into_pocket('edf_duration-anno_duration',
                       record_minus_anno, local=True)
    return sg


  @classmethod
  def read_bipolar(cls, file_path, dtype, max_sfreq):
    import mne.io

    open_file = lambda include=(): mne.io.read_raw_edf(
      file_path, include=include, preload=False, verbose=False)

    with open_file(include=('Fpz', 'Cz', 'Pz', 'Oz')) as file:
      sfreq = file.info['sfreq']

      # Resample to `max_sfreq` if necessary
      if max_sfreq is not None and sfreq > max_sfreq:
        file.resample(max_sfreq)
        sfreq = max_sfreq

      # Read signal, group signals with the same sfreq
      data_tuple = (file.ch_names, file.get_data())
      signal_dict = {ck: s for ck, s in zip(data_tuple[0], data_tuple[1])}

    # Wrap data into DigitalSignals
    digital_signals = []

    data = np.stack(
      [signal_dict['Fpz'] - signal_dict['Cz'],
       signal_dict['Pz'] - signal_dict['Oz']], axis=1).astype(dtype)

    channel_names = ['EEG Fpz-Cz', 'EEG Pz-Oz']

    digital_signals.append(DigitalSignal(
      data, channel_names=channel_names, sfreq=sfreq,
      label=','.join(channel_names)))

    return digital_signals


  @classmethod
  def load_hsp_annotation(cls, anno_path):
    """
    Notes:
    (1) first/last epoch in csv file may not have sleep stage annotation !
    """
    import pandas as pd

    # Read intervals and annotations
    label2int = {lb: i for i, lb in enumerate(cls.ANNO_LABELS)}
    # PITFALL: 'Sleep_stage_R' and 'Sleep_stage_REM' both exist
    label2int['Sleep_stage_REM'] = label2int['Sleep_stage_R']

    # PITFALL: 'Stage - W/R/N1/N2/N3' format exists
    for s in ('W', 'N1', 'N2', 'N3', 'R'):
      label2int[f'Stage - {s}'] = label2int[f'Sleep_stage_{s}']

    intervals, annotations = [], []
    a_epochs = []

    df = pd.read_csv(anno_path)
    time_fmt = '%H:%M:%S'
    global_tic, anno_tic, anno_toc = None, None, None
    # Traverse rows
    for i, row in df.iterrows():
      epoch, duration, evt = row['epoch'], row['duration'], row['event']
      time_stamp = row['time']

      # PITFALL: ' 22:49:31.00' does not match format '%H:%M:%S' (sub-S0001111531526/ses-4)
      if not isinstance(time_stamp, str) and np.isnan(time_stamp): continue

      # global tic !
      assert isinstance(time_stamp, str), f'!! `{time_stamp}` is not a string'

      if '.' in time_stamp: _time_fmt = time_fmt + '.%f'
      else: _time_fmt = time_fmt

      toc = datetime.strptime(time_stamp.strip(), _time_fmt)
      if i == 0: global_tic = toc

      # Calculate onset
      onset = (toc - global_tic).total_seconds()
      if onset < 0: onset += 24 * 3600

      # Only sleep stage event is supported for now
      if evt not in label2int:
        # if duration == 30:
        #   print(evt)   # For debug
        continue

      # Convert epoch and duration from string to int/float
      duration = float(duration)
      assert math.isclose(duration, 30, abs_tol=1e-6)

      # Record first/last time stamp and epoch with stage annotation
      if len(intervals) == 0: anno_tic = toc
      anno_toc = toc

      # Append interval and annotation
      intervals.append((onset, onset + duration))
      annotations.append(label2int[evt])

      a_epochs.append(int(epoch))

    # Sanity check
    # tic_toc_duration = (anno_toc - anno_tic).total_seconds()
    # if tic_toc_duration < 0: tic_toc_duration += 24 * 3600
    anno_duration = sum([itv[1] - itv[0] for itv in intervals])

    # PITFALL: Subject going to bathroom may cause missing epochs
    # if a_epochs[-1] - a_epochs[0] != len(a_epochs) - 1:
    #   missings = [i for i in range(a_epochs[0], a_epochs[-1] + 1)
    #               if i not in a_epochs]
    #   raise ValueError(f'Epochs are not continuous: {missings}')
    #
    # if not math.isclose(anno_duration, tic_toc_duration + 30, abs_tol=1e-6):
    #   raise ValueError(f'anno_duration ({anno_duration}) != tic_toc_duration ({tic_toc_duration})')

    assert len(intervals) > 0, f'No stage annotation found in {anno_path}'

    return Annotation(intervals, annotations, labels=cls.ANNO_LABELS)


  @classmethod
  def convert_rawdata_to_signal_groups(
      cls, ses_folder_list: list, tgt_dir, dtype=np.float16, max_sfreq=128,
      bipolar=False, **kwargs):
    # (0) Check target directory
    if not os.path.exists(tgt_dir): os.makedirs(tgt_dir)
    console.show_status(f'Target directory set to `{tgt_dir}` ...')

    # (0.1) Report number of files to be converted
    ho_list = [HSPOrganization(ses_folder) for ses_folder in ses_folder_list]

    # PITFALL: Some .edf files do not exist (ses folder in online database
    #          is empty)
    ho_list = [ho for ho in ho_list if os.path.exists(ho.edf_path)]

    sg_file_paths = [
      os.path.join(tgt_dir, ho.get_sg_file_name(dtype, max_sfreq, bipolar))
      for ho in ho_list]
    convert_list = [(sg_p, ses_p)
                    for sg_p, ses_p in zip(sg_file_paths, ses_folder_list)
                    if not os.path.exists(sg_p) or kwargs.get('overwrite', False)]
    n_total = len(ses_folder_list)
    n_convert = len(convert_list)

    console.show_status(f'{n_total - n_convert} files already converted.')
    console.show_status(f'Converting {n_convert}/{n_total} files ...')

    # (1) Convert files
    n_success = 0
    sg_job_list = [sg_p for sg_p, _ in convert_list]
    success_sg_path_list = [p for p in sg_file_paths if p not in sg_job_list]
    for i, (sg_path, ses_path) in enumerate(convert_list):
      console.show_status(f'Converting {i + 1}/{n_convert} {ses_path} ...')
      console.print_progress(i, n_convert)

      try:
        sg: SignalGroup = cls.load_sg_from_raw_files(
          ses_dir=ses_path, dtype=dtype, max_sfreq=max_sfreq, bipolar=bipolar)

        # TODO: sg.label does not match sg_path ?????
        io.save_file(sg, sg_path, verbose=True)
        success_sg_path_list.append(sg_path)
        n_success += 1
      except Exception as e:
        # Print error message
        console.warning(
          f"Failed to convert `{ses_path}`. Full error traceback:")
        # Or traceback.print_exc() for direct output
        console.warning(traceback.format_exc())
        continue

    console.show_status(f'Successfully converted {n_success}/{n_convert} files.')

    # Return available sg paths
    return success_sg_path_list


  @classmethod
  def load_as_signal_groups(cls, src_dir, max_sfreq=128,
                            **kwargs) -> List[SignalGroup]:

    # (0) Get configs
    JUST_CONVERSION = kwargs.get('just_conversion', False)
    PREPROCESS = kwargs.get('preprocess', '')

    # TODO: Implement this

    return

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

  # endregion: Data Conversion



class HSPAgent(LongitudinalManager):
  """
      patient_dict[pid][session_id].keys = {
        'site_id', 'bids_folder', 'pre_sleep_questionnaire',
        'has_annotations', 'has_staging', 'study_type',
        'age', 'gender'
      }
  """

  VALID_STUDY_TYPES = ''
  ACQ_TIME_KEY = 'acq_time'

  def __init__(self, meta_dir, data_dir=None, meta_time_stamp='20231101',
               access_point_name=None):
    self.meta_dir = meta_dir
    self.meta_time_stamp = meta_time_stamp
    self.access_point_name = access_point_name

    self.data_dir = data_dir

  # region: Properties

  @property
  def meta_path(self):
    meta_file_name = f'bdsp_psg_master_{self.meta_time_stamp}.csv'
    _meta_path = os.path.join(self.meta_dir, meta_file_name)
    assert os.path.exists(_meta_path), f'Meta data not found: {_meta_path}'
    return _meta_path

  @Nomear.property()
  def pre_sleep_questionnaire_dict(self):
    psq_dict_path = self.meta_path.replace('.csv', '.psq')
    if os.path.exists(psq_dict_path) and not self.in_pocket('OVERWRITE_PSQ'):
      return io.load_file(psq_dict_path, verbose=True)

    od = self.generate_pre_sleep_questionnaire_dict()
    io.save_file(od, psq_dict_path, verbose=True)
    return od

  @Nomear.property()
  def pre_sleep_questionnaire_dataframe(self):
    import pandas as pd

    od = OrderedDict()
    for pid, sess_dict in self.pre_sleep_questionnaire_dict.items():
      for ses_id, psq_dict in sess_dict.items():
        od[f'{pid}-{ses_id}'] = psq_dict

    return pd.DataFrame.from_dict(od, orient="index")

  # endregion: Properties

  # region: Public Methods

  # region: - Match Logic

  @staticmethod
  def get_dual_nebula(nebula, max_age_diff=1):
    night_1, buffer_1 = [], []
    night_2, buffer_2 = [], []

    for label in nebula.labels:
      pid = label.split('-')[1].split('_')[0]
      if pid not in buffer_1:
        buffer_1.append(pid)
        night_1.append(label)
      else:
        if pid in buffer_2: continue
        # assert pid not in buffer_2
        buffer_2.append(pid)
        night_2.append(label)

    # Filter by age diff
    _night_1, _night_2 = [], []
    for lb1, lb2 in zip(night_1, night_2):
      acq_time_1 = datetime.strptime(nebula.meta[lb1][HSPAgent.ACQ_TIME_KEY], '%Y-%m-%d')
      acq_time_2 = datetime.strptime(nebula.meta[lb2][HSPAgent.ACQ_TIME_KEY], '%Y-%m-%d')
      delta = (acq_time_2 - acq_time_1).days / 365.25
      if delta <= 0:
        print(f'Error: {lb1} and {lb2} have negative age difference: {delta}')
        continue
      # assert delta > 0
      if delta <= max_age_diff:
      # if abs(nebula.meta[lb1]['age'] - nebula.meta[lb2]['age']) <= max_age_diff:
        _night_1.append(lb1)
        _night_2.append(lb2)

    console.show_status(f'Found {len(_night_1)} pairs within age_diff={max_age_diff}.')
    return nebula[_night_1], nebula[_night_2]

  # endregion: - Match Logic

  # region: - Data Conversion

  def load_nebula_from_clouds(self, sub_dict, cloud_path, channels,
                              time_resolution, probe_keys):
    from hypnomics.freud.freud import Freud

    ho_list = [HSPOrganization(p)
               for p in self.convert_to_folder_names(sub_dict, local=True)]

    # (1) Get sg_labels
    sg_labels = [ho.sg_label for ho in ho_list]

    freud = Freud(cloud_path)
    nebula = freud.load_nebula(sg_labels=sg_labels,
                               channels=channels,
                               time_resolution=time_resolution,
                               probe_keys=probe_keys, verbose=True)

    # (2) Set metadata
    for ho in ho_list:
      nebula.meta[ho.sg_label] = {}
      nebula.meta[ho.sg_label]['age'] = sub_dict[ho.sub_id][ho.ses_id]['age']
      acq_time = self.get_acq_time(ho.ses_path, return_str=True)
      nebula.meta[ho.sg_label][self.ACQ_TIME_KEY] = acq_time

      nebula.meta[ho.sg_label]['gender'] = sub_dict[ho.sub_id][ho.ses_id]['gender']
      nebula.meta[ho.sg_label]['study_type'] = sub_dict[ho.sub_id][ho.ses_id]['study_type']

    return nebula

  # endregion: - Data Conversion

  # region: - Filters

  @staticmethod
  def filter_by_min_sessions(_patient_dict: dict, min_n_sessions=2):
    filtered_dict = OrderedDict()
    for pid, sess_dict in _patient_dict.items():
      # Check session number
      if len(sess_dict) >= min_n_sessions:
        filtered_dict[pid] = sess_dict
    return filtered_dict

  # TODO: BETA
  def filter_patients_meta(self, min_n_sessions=1, should_have_annotation=True,
                           should_have_psq=False, study_types=None,
                           return_folder_names=False):
    """Filter subjects based on meta description, which might be incorrect.
    E.g., data folder of some sessions listed in meta is empty in AWS
    """
    filtered_dict = OrderedDict()
    for pid, sess_dict in self.patient_dict.items():
      # Check annotation if required
      if should_have_annotation:
        sess_dict = {k: v for k, v in sess_dict.items()
                     if v['has_annotations'] and v['has_staging']}
      # Check study types
      if study_types is not None:
        sess_dict = {k: v for k, v in sess_dict.items()
                     if v['study_type'] in study_types}
      # Check pre-sleep questionnaire
      if should_have_psq:
        sess_dict = {k: v for k, v in sess_dict.items()
                     if v['pre_sleep_questionnaire']}
      # Check session number
      if len(sess_dict) >= min_n_sessions:
        filtered_dict[pid] = sess_dict
    if return_folder_names: return self.convert_to_folder_names(filtered_dict)
    return filtered_dict

  def filter_patients_local(self, patient_dict: dict, min_n_sessions=1,
                            should_have_annotation=False, verbose=False):
    """Filter patients based on AWS database downloaded to local."""
    filtered_dict = OrderedDict()

    if verbose: console.show_status('Scanning local directory ...')
    N = len(patient_dict)
    n_missing_anno = 0
    for i, (pid, sess_dict) in enumerate(patient_dict.items()):
      if verbose and i % 10 == 0: console.print_progress(i, N)

      folder_list = self.convert_to_folder_names({pid: sess_dict}, local=True)
      path_ho_tuples = [(path, HSPOrganization(path)) for path in folder_list]

      # Check edf file
      path_ho_tuples = [(p, ho) for (p, ho) in path_ho_tuples
                        if os.path.exists(ho.edf_path)]

      # Check annotation if required
      if should_have_annotation:
        _path_ho_tuples = [(p, ho) for (p, ho) in path_ho_tuples
                          if os.path.exists(ho.anno_path)]
        n_missing_anno += len(path_ho_tuples) - len(_path_ho_tuples)
        path_ho_tuples = _path_ho_tuples

      # Check session number
      if len(path_ho_tuples) >= min_n_sessions:
        _sess_dict = OrderedDict()
        for (p, ho) in path_ho_tuples:
          _sess_dict[ho.ses_id] = sess_dict[ho.ses_id]
        filtered_dict[pid] = _sess_dict

    if verbose:
      console.show_status('Filtered dict generated.')
      console.show_info('Details:')
      N0, N1 = len(patient_dict), len(filtered_dict)
      console.supplement(f'n_subjects: {N0} -> {N1} (-{N0 - N1})')
      console.supplement(f'Missing annotations: {n_missing_anno}', level=2)

    return filtered_dict

  def filter_patients_sg(self, patient_dict: dict, sg_dir, min_n_sessions=1,
                         verbose=False, dtype=np.float16, max_sfreq=128):
    """Filter patients based on sg with at least 8 channels (6 EEG + 2 EOG)"""
    filtered_dict = OrderedDict()

    if verbose: console.show_status('Scanning SG directory ...')
    N = len(patient_dict)
    n_missing_sg = 0
    for i, (pid, sess_dict) in enumerate(patient_dict.items()):
      if verbose and i % 10 == 0: console.print_progress(i, N)

      folder_list = self.convert_to_folder_names({pid: sess_dict}, local=True)
      path_ho_tuples = [(path, HSPOrganization(path)) for path in folder_list]

      # Check .sg file
      _path_ho_tuples = [
        (p, ho) for (p, ho) in path_ho_tuples if os.path.exists(
          os.path.join(sg_dir, ho.get_sg_file_name(dtype, max_sfreq)))]

      n_missing_sg += len(path_ho_tuples) - len(_path_ho_tuples)
      path_ho_tuples = _path_ho_tuples

      # Check session number
      if len(path_ho_tuples) >= min_n_sessions:
        _sess_dict = OrderedDict()
        for (p, ho) in path_ho_tuples:
          _sess_dict[ho.ses_id] = sess_dict[ho.ses_id]
        filtered_dict[pid] = _sess_dict

    if verbose:
      console.show_status('Filtered dict generated.')
      console.show_info('Details:')
      N0, N1 = len(patient_dict), len(filtered_dict)
      console.supplement(f'n_subjects: {N0} -> {N1} (-{N0 - N1})')
      console.supplement(f'Missing sg files: {n_missing_sg}', level=2)

    return filtered_dict

  def filter_patients_neb(self, patient_dict: dict, neb_dir, min_n_sessions=1,
                          verbose=True, time_resolution=30, pk='AMP-1',
                          ck='EEG C3-M2', min_hours=2):
    """Filter patients based on nebula:
       (1) .clouds file exists
       (2) sleep time >= min_hours (a typical sleep cycle usually contains upto 110 mins)
       (3) have N2 stage
    """
    filtered_dict = OrderedDict()

    if verbose: console.show_status('Scanning nebula directory ...')
    N = len(patient_dict)
    n_invalid_clouds = 0
    for i, (pid, sess_dict) in enumerate(patient_dict.items()):
      if verbose and i % 10 == 0: console.print_progress(i, N)

      folder_list = self.convert_to_folder_names({pid: sess_dict}, local=True)
      path_ho_tuples = [(path, HSPOrganization(path)) for path in folder_list]

      # Check each .cloud file
      _path_ho_tuples = []
      for p, ho in path_ho_tuples:
        # (1) Make sure file exist
        cloud_path = os.path.join(
          neb_dir, ho.sg_label, ck, f'{time_resolution}s', f'{pk}.clouds')

        if not os.path.exists(cloud_path):
          console.warning(f'{cloud_path} not found.')
          continue

        # (2) Make sure file exist
        cloud: dict = io.load_file(cloud_path)
        hours = sum([len(cloud[k]) for k in ['N1', 'N2', 'N3', 'R']]) * time_resolution / 3600

        if hours < min_hours:
          console.warning(f'Total sleep time of {ho.sg_label} = {hours} hours < minimal ({min_hours} hours)')
          continue

        # (3) Make sure N2 exists
        if len(cloud['N2']) == 0:
          console.warning(f'No N2 stage found in {ho.sg_label}')
          continue

        # (-1) append
        _path_ho_tuples.append((p, ho))

      n_invalid_clouds += len(path_ho_tuples) - len(_path_ho_tuples)
      path_ho_tuples = _path_ho_tuples

      # Check session number
      if len(path_ho_tuples) >= min_n_sessions:
        _sess_dict = OrderedDict()
        for (p, ho) in path_ho_tuples:
          _sess_dict[ho.ses_id] = sess_dict[ho.ses_id]
        filtered_dict[pid] = _sess_dict

    if verbose:
      console.show_status('Filtered dict generated.')
      console.show_info('Details:')
      N0, N1 = len(patient_dict), len(filtered_dict)
      console.supplement(f'n_subjects: {N0} -> {N1} (-{N0 - N1})')
      console.supplement(f'Missing sg files: {n_invalid_clouds}', level=2)

    return filtered_dict

  def filter_patients_by_channels(
      self, patient_dict: dict, channels, min_n_sessions=1, verbose=False):
    filtered_dict = OrderedDict()

    if verbose: console.show_status('Examining channels ...')
    N = len(patient_dict)

    for i, (pid, sess_dict) in enumerate(patient_dict.items()):
      if verbose and i % 10 == 0: console.print_progress(i, N)

      for ses_id, infor_dict in sess_dict.items():
        ho = HSPOrganization(self.get_raw_path(pid, ses_id))
        if any([ck not in ho.channel_dict for ck in channels]):
          continue

        if pid not in filtered_dict: filtered_dict[pid] = OrderedDict()
        filtered_dict[pid][ses_id] = infor_dict

    # Filter by min_sessions
    filtered_dict = {k: v for k, v in filtered_dict.items()
                     if len(v) >= min_n_sessions}

    if verbose:
      console.show_status('Filtered dict generated.')
      console.show_info('Details:')
      N0, N1 = len(patient_dict), len(filtered_dict)
      console.supplement(f'n_subjects: {N0} -> {N1} (-{N0 - N1})')

    return filtered_dict

  # endregion: - Filters

  # region: - IO Methods

  def load_subset_dict(self, file_name=None, file_path=None, max_subjects=None,
                       return_ho=False, return_folder_list=False, verbose=True):
    if file_path is None: file_path = os.path.join(self.meta_dir, file_name)
    assert os.path.exists(file_path), f'File not found: {file_path}'
    subset_dict = io.load_file(file_path, verbose=True)

    if verbose:
      n_folders = sum([len(v) for v in subset_dict.values()])
      console.show_status(
        f'Loaded subset containing {len(subset_dict)} subjects ({n_folders} PSGs).',
        prompt='[HSPAgent]')

    if isinstance(max_subjects, int) and max_subjects < len(subset_dict):
      subset_dict = {
        k: subset_dict[k] for k in list(subset_dict.keys())[:max_subjects]}
      if verbose:
        n_folders = sum([len(v) for v in subset_dict.values()])
        console.show_status(
          f'Loaded subsubset containing {len(subset_dict)} subjects ({n_folders} PSGs).',
          prompt='[HSPAgent]')

    if return_ho:
      assert not return_folder_list
      folder_paths = self.convert_to_folder_names(subset_dict, local=True)
      return [HSPOrganization(p) for p in folder_paths]

    if return_folder_list:
      return self.convert_to_folder_names(subset_dict, local=True)

    return subset_dict

  def load_pair_labels(self, ss_file_name):
    ss_file_path = os.path.join(self.meta_dir, ss_file_name)
    assert os.path.exists(ss_file_path), f'File not found: {ss_file_path}'
    subset_dict = io.load_file(ss_file_path, verbose=True)

    nights_1, nights_2 = [], []
    subject_ids = sorted(list(subset_dict.keys()))
    for pid in subject_ids:
      ses_keys = sorted(list(subset_dict[pid].keys()))
      nights_1.append(f'{pid}_{ses_keys[0]}')
      nights_2.append(f'{pid}_{ses_keys[1]}')

    return nights_1, nights_2


  def get_raw_path(self, pid, ses_id):
    return f'{self.data_dir}/{pid}/{ses_id}'


  def convert_to_folder_names(self, patient_dict: OrderedDict, local=False):
    """e.g., <APN>/bdsp-psg-access-point/PSG/bids/sub-S0001111190905/ses-1/"""
    if local: src_path = self.data_dir
    else: src_path = f'{self.access_point_name}/bdsp-psg-access-point/PSG/bids/S0001'

    assert src_path[-1] != '/'

    folder_list = []
    for pid, sess_dict in patient_dict.items():
      for sess_id, sess_info in sess_dict.items():
        folder_name = f'{src_path}/{pid}/{sess_id}'
        folder_list.append(folder_name)
    return folder_list


  def get_longitudinal_pairs(self, patient_dict: dict, return_age_delta=False):
    label_pairs = []
    age_delta_dict = OrderedDict()
    for pid, sess_dict in patient_dict.items():
      sess_keys = sorted(list(sess_dict.keys()))
      if len(sess_keys) < 2: continue
      for i in range(len(sess_keys) - 1):
        for j in range(i + 1, len(sess_keys)):
          si, sj = sess_keys[i], sess_keys[j]
          pair_key = (f'{pid}_{si}', f'{pid}_{sj}')
          label_pairs.append(pair_key)

          if not return_age_delta: continue
          assert self.ACQ_TIME_KEY in sess_dict[si]

          ad = (sess_dict[sj][self.ACQ_TIME_KEY] -
                sess_dict[si][self.ACQ_TIME_KEY]).days / 365.25
          if ad < 0: ad = sess_dict[sj]['age'] - sess_dict[si]['age']

          age_delta_dict[pair_key] = ad

    if return_age_delta: return label_pairs, age_delta_dict
    return label_pairs


  @staticmethod
  def generate_patient_dict(meta_path) -> OrderedDict:
    import pandas as pd

    # (0) Read meta data
    df = pd.read_csv(meta_path)

    # (1) Read patients' info as a list of dictionaries
    to_boolean = {'Y': True, 'N': False}
    n_rows = df.shape[0]
    patient_dict = OrderedDict()
    for i, row in df.iterrows():
      # (1.0) Show progress
      # TODO: actually this is really fast, thus progress bar is not necessary
      if i == 0 or i == n_rows // 100: console.print_progress(i, n_rows)

      # (1.1) Read row
      pid = row['BDSPPatientID']
      session_id = row['SessionID']

      bids_folder = row['BidsFolder']
      site_id = row['SiteID']
      pre_sleep_questionnaire = to_boolean[row['PreSleepQuestionnaire']]
      has_annotations = to_boolean[row['HasAnnotations']]
      has_staging = to_boolean[row['HasStaging']]
      study_type = row['StudyType']
      age = int(row['AgeAtVisit'])
      gender = row['SexDSC']

      # (1.2) Create patient slot if not exists
      _patient_label = f'sub-{site_id}{pid}'
      _session_label = f'ses-{session_id}'

      if _patient_label not in patient_dict:
        patient_dict[_patient_label] = OrderedDict()
      assert _session_label not in patient_dict[_patient_label]
      patient_dict[_patient_label][_session_label] = {
        'site_id': site_id,
        'bids_folder': bids_folder,
        'pre_sleep_questionnaire': pre_sleep_questionnaire,
        'has_annotations': has_annotations,
        'has_staging': has_staging,
        'study_type': study_type,
        'age': age,
        'gender': gender,
      }

    # (2) Report progress and return
    console.show_status(f'Successfully read {len(patient_dict)} patients from'
                        f' {meta_path}')
    return patient_dict

  @staticmethod
  def check_acq_time_in_pd(patient_dict: dict):
    for pid, sess_dict in patient_dict.items():
      for ses_id, sess_info in sess_dict.items():
        if HSPAgent.ACQ_TIME_KEY not in sess_info:
          return False

    return True

  @staticmethod
  def get_acq_time(ses_path, return_str=False):
    import pandas as pd

    ho = HSPOrganization(ses_path)
    if not os.path.exists(ho.tsv_path): return None
    df = pd.read_csv(ho.tsv_path, sep='\t')
    date_str = df['acq_time'].str.split('T').str[0].iloc[0]

    # Check data_str format
    try:
      if ':' in date_str: date_str = date_str.split(' ')[0]
      if return_str: return date_str
      return datetime.strptime(date_str, '%Y-%m-%d')
    except:
      raise AssertionError(f'Invalid date string: {date_str}')

  def get_ses_path(self, pid, sid):
    return f'{self.data_dir}/{pid}/{sid}'

  def generate_pre_sleep_questionnaire_dict(self) -> OrderedDict:
    import pandas as pd

    def _is_float(x):
      try:
        float(x)
        return True
      except:
        return False

    od = OrderedDict()
    N = len(self.patient_dict)
    attr_list = None
    # Traverse all subjects
    suspicious_pids = []
    for i, (pid, ses_dict) in enumerate(self.patient_dict.items()):
      if i % 10 == 0: console.print_progress(i, N)

      # Traverse all sessions of subject `pid`
      for ses_id, ses_info in ses_dict.items():
        if not ses_info['pre_sleep_questionnaire']: continue

        # Get PSQ file path
        # ses_path = f'{self.data_dir}/{pid}/{ses_id}'
        ses_path = self.get_ses_path(pid, ses_id)
        ho = HSPOrganization(ses_path)
        pre_path = ho.pre_path
        if not os.path.exists(pre_path):
          suspicious_pids.append(pid)
          console.warning(f'{pre_path} not found.')
          continue

        # Create slot for pid if not exists
        if pid not in od: od[pid] = OrderedDict()

        # Read PSQ data
        _od = OrderedDict()
        od[pid][ses_id] = _od

        # (1) Insert meta data
        _od.update(ses_info)

        # (2) Insert tsv data
        if os.path.exists(ho.tsv_path):
          # df = pd.read_csv(ho.tsv_path, sep='\t')
          # acq_time = df['acq_time'].str.split('T').str[0].iloc[0]
          # _od['acq_time'] = acq_time
          _od['acq_time'] = self.get_acq_time(ses_path, return_str=True)

        # (3) Insert questionnaire data
        df = pd.read_csv(pre_path)
        if attr_list is None: attr_list = list(df.iloc[:, 0])

        # Make sure all subjects have the same attributes
        assert attr_list == list(df.iloc[:, 0])

        for key in attr_list:
          value = df.loc[df.iloc[:, 0] == key].iloc[0, 1]

          if value == 'missingData':
            value = None
          elif value == '':
            value = None
          elif value in ('0', '1'):
            value = bool(int(value))
          elif _is_float(value):
            value = float(value)
            if np.isnan(value): value = None

          _od[key] = value

    console.show_status(f'PSQ dict with {len(od)} subjects created.',
                        prompt='[HSPAgent]')
    return od

  # endregion: - IO Methods

  # region: - AWS Commands

  def list_folders(self, project_folder, recursive=True):
    import subprocess

    # Define command and arguments
    command = ['aws', 's3', 'ls', project_folder]
    if recursive: command.append('--recursive')

    console.show_status(f'Executing command: {" ".join(command)}')

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Access the output and error
    stdout = result.stdout
    stderr = result.stderr

    # Display the outputs
    if stderr: console.warning(stderr)
    else:
      console.show_status('stdout:')
      console.split()
      print(stdout)
      console.split()

  @staticmethod
  def run_command_realtime(command):
    import subprocess, sys

    # Start the process
    process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
      bufsize=1
    )

    try:
      # Read stdout and stderr in real-time
      while True:
        # Read one line at a time
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
          break
        if output:
          console.clear_line()
          sys.stdout.write(output[:-1])
          sys.stdout.flush()

    except KeyboardInterrupt:
      # If interrupted, terminate the subprocess
      process.terminate()
      print("\nProcess terminated.")
      return None  # or return an appropriate code

    # Get the return code
    return_code = process.poll()

    # Print any remaining stderr
    for line in process.stderr: console.warning(line)

    console.clear_line()

    return return_code

  def check_folder_complete(self, folder_path):
    # Check level 0
    if not os.path.exists(folder_path): return False

    ho = HSPOrganization(folder_path)

    # Check level 1
    paths = [ho.eeg_path, ho.tsv_path]
    if not all([os.path.exists(p) for p in paths]): return False

    # Check level 2
    paths = [ho.channel_path, ho.edf_path]
    if not all([os.path.exists(p) for p in paths]): return False

    return True

  def copy_a_folder(self, project_folder):
    # (0) Check if the folder already exists
    local_path = os.path.join(self.data_dir, project_folder.split('bids/')[-1])
    local_path = os.path.abspath(local_path)

    #  e.g. local_path = 'F:\\data\\hsp\\sub-S0001111190905/ses-1/'
    if self.check_folder_complete(local_path):
      console.show_status(f'Folder already exists: {local_path}')
      return 'exist'

    # (1) Download data
    # Define the command and its arguments
    command = ['aws', 's3', 'cp', project_folder, local_path, '--recursive']

    return_code = self.run_command_realtime(command)

    if return_code != 0:
      console.warning(f"Command failed with return code {return_code}")
      return 'error'
    else:
      console.show_status(f'Downloaded data to: {local_path}')
      return 'success'

  def download_folders(self, folder_list: list):
    N = len(folder_list)
    n_success, n_exist, n_error = 0, 0, 0
    for i, path in enumerate(folder_list):
      console.show_status(f'Progress: {i}/{N} ...')
      ret = self.copy_a_folder(path)

      if ret == 'exist': n_exist += 1
      elif ret == 'success': n_success += 1
      elif ret == 'error': n_error += 1
      else: raise KeyError(f'!! unknown return code `{ret}`')

    console.show_info('Summary:')
    console.supplement(f'{n_exist} folders already exist.', level=2)
    console.supplement(f'Successfully downloaded {n_success} folders.', level=2)
    console.supplement(f'Failed to downloaded {n_error} folders.', level=2)

  def download_metadata(self):
    src_path = f'{self.access_point_name}/bdsp-psg-access-point/PSG/metadata/'

    local_path = os.path.join(self.meta_dir, 'metadata')
    local_path = os.path.abspath(local_path)

    command = ['aws', 's3', 'cp', src_path, local_path, '--recursive']

    return_code = self.run_command_realtime(command)
    if return_code != 0:
      console.warning(f"Command failed with return code {return_code}")
      return 'error'
    else:
      console.show_status(f'Downloaded data to: {local_path}')
      return 'success'

  # endregion: AWS Commands

  # endregion: Public Methods



class HSPOrganization(Nomear):
  """ Bids-root-folder/
      └── dataset_description.json
      └── participants.json
      └── participants.tsv
      └── README
      └── sub-Id/
        └── ses-01/
          └── sub-SiteIdPatientId_ses-01_scans.tsv
          └── eeg
            └── sub-Id_ses-1_task-psg_annotations.tsv
            └── sub-Id_ses-1_task-psg_channels.tsv
            └── sub-Id_ses-1_task-psg_eeg.edf
            └── sub-Id_ses-1_task-psg_eeg.json
            └── sub-Id_ses-1_task-psg_pre.csv

      example session path = '<path>\hsp_raw\sub-S0001118501829\ses-1'
  """

  def __init__(self, ses_path=None, ses_id=None, sub_id=None, data_dir=None):
    if ses_path is None:
      assert os.path.exists(data_dir), f'Data directory not found: {data_dir}'
      ses_path = os.path.join(data_dir, f'{sub_id}/{ses_id}')

    self.ses_path = ses_path

    self.ses_id = os.path.basename(ses_path)
    self.sub_id = os.path.basename(os.path.dirname(ses_path))

    prefix = self.sg_label

    # Level 1
    self.eeg_path = os.path.join(ses_path, 'eeg')
    self.tsv_path = os.path.join(ses_path, f'{prefix}_scans.tsv')

    # Level 2
    prefix = f'{self.sg_label}_task-psg'
    self.anno_path = os.path.join(self.eeg_path, f'{prefix}_annotations.csv')
    self.channel_path = os.path.join(self.eeg_path, f'{prefix}_channels.tsv')
    self.edf_path = os.path.join(self.eeg_path, f'{prefix}_eeg.edf')
    self.json_path = os.path.join(self.eeg_path, f'{prefix}_eeg.json')
    self.pre_path = os.path.join(self.eeg_path, f'{prefix}_pre.csv')

  @property
  def sg_label(self): return f'{self.sub_id}_{self.ses_id}'

  @Nomear.property()
  def channel_dict(self):
    import pandas as pd

    df = pd.read_csv(self.channel_path, sep='\t')
    cd = {row['name']: row.drop('name').to_dict() for _, row in df.iterrows()}
    return cd

  def get_sg_file_name(self, dtype, max_sfreq, bipolar=False):
    dtype_str = str(dtype).split('.')[-1].replace('>', '')
    dtype_str = dtype_str.replace("'", '')
    bipolar_str = ',bipolar' if bipolar else ''
    return f'{self.sg_label}({dtype_str},{max_sfreq}Hz{bipolar_str}).sg'


if __name__ == '__main__':
  import pandas as pd

  OVERWRITE = 1

  data_dir = r'../../../data/hsp'
  meta_path = os.path.join(data_dir, 'bdsp_psg_master_20231101.csv')
  patient_dict_path = os.path.join(data_dir, 'patient_dict_20231101.od')

  if os.path.exists(patient_dict_path) and not OVERWRITE:
    patient_dict = io.load_file(patient_dict_path, verbose=True)
  else:
    patient_list = HSPAgent.generate_patient_dict(meta_path)
    io.save_file(patient_list, patient_dict_path, verbose=True)

  print('Done!')


