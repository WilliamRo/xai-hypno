import os.path

from freud.benchmarks.algorithm import Algorithm
from freud.datasets.dataset_base import HypnoDataset
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.extractor import Extractor
from roma import check_type
from roma import console, io

from .probe_tools import get_extractor_dict, get_probe_keys

import numpy as np



class HOAlgorithm(Algorithm):
    """A hypnomics algorithm used in benchmark SOPs.
       The input of the algorithm should be a SignalGroup.

    IMPORTANT SETTING:
    - A. dataset: HypnoDataset
      - A.1 dataset.signal_group_dir
      - A.2 dataset.sg_file_list
      - A.3 dataset.channels
    - B. algorithm: HOAlgorithm
      - B.1 algorithm.time_resolution

    """

    version = '1.0.0'
    prompt = f'[HOAlgo] >>'

    def __init__(self, dataset: HypnoDataset, probe_config='Ab'):
      super().__init__(dataset, time_resolution=30)

      # Set probe configurations
      self.probe_config = probe_config
      self.probe_arg = probe_config if isinstance(probe_config, str) else 'X'

      # Report the configuration
      console.supplement(f'Probe keys: ')
      for i, key in enumerate(self.probe_keys_for_extracting_features):
        console.supplement(f'[{i + 1}] {key}', level=2)

    @property
    def probe_keys_for_generating_clouds(self):
      return self._get_probe_keys(False)

    @property
    def probe_keys_for_extracting_features(self):
      return self._get_probe_keys(True)

    def _get_probe_keys(self, for_extracting_features):
      if isinstance(self.probe_config, (tuple, list)): return self.probe_config
      assert isinstance(self.probe_config, str), f'Invalid probe configuration: {self.probe_config}'
      return get_probe_keys(self.probe_config, expand_group=for_extracting_features)

    # region: Public Methods

    def extract_features(self, **kwargs):
      """Extract feature vectors from a list of signal group filenames
      """
      show_status = lambda text: console.show_status(text, prompt=self.prompt)

      # (1) Cloud
      show_status('Generating clouds ...')
      self.generate_clouds(self.time_resolution,
                           probe_keys=self.probe_keys_for_generating_clouds,
                           sg_file_list=self.hypno_data.sg_file_list)
      n_sg_files = len(self.hypno_data.sg_file_list)
      show_status(f'Clouds (N={n_sg_files}) generated.')

      # (2) Nebula
      show_status('Loading nebula from clouds ...')
      nebula = self.load_nebula_from_clouds(
        self.time_resolution,
        probe_keys=self.probe_keys_for_extracting_features)
      n_clouds = len(self.hypno_data.sg_labels)
      show_status(f'Nebula (N={n_clouds}) loaded.')

      # (3) Type-I Features
      show_status('Generating features ...')
      extractor_settings = {
        'include_statistical_features': 1,
        'include_inter_stage_features': 1,
        'include_inter_channel_features': 1,

        # Deprecated features
        'include_proportion': False,
        'include_stage_mean': False,
        'include_stage_shift': False,
        'include_stage_wise_covariance': False,
        'include_channel_shift': False,
        'include_all_mean_std': False,
      }
      extractor = Extractor(**extractor_settings)
      feature_dict = extractor.extract(nebula, return_dict=True)
      features = np.stack([np.array(list(v.values()))
                           for v in feature_dict.values()], axis=0)
      feature_names = list(list(feature_dict.values())[0].keys())

      show_status(f'{len(feature_names)} features generated.')

      # (4) Type-III Features
      show_status('Loading macro features (alpha) ...')
      features_III, feature_names_III = self.load_macro_alpha()
      features = np.concatenate([features, features_III], axis=1)
      feature_names = feature_names + feature_names_III

      # (-1) Return
      return features, {'feature_names': feature_names,
                        'nebula': nebula,
                        'extractor_settings': extractor_settings,
                        'sg_labels': self.hypno_data.sg_labels, }


    def generate_clouds(self, time_resolution, probe_keys, overwrite=False,
                        sg_file_list=None):
      # Sanity check
      if not isinstance(time_resolution, (list, tuple)):
        time_resolution = [time_resolution]
      check_type(time_resolution, (list, tuple), int)

      # Generate clouds (Type-I) using hypnomics.Freud
      freud = Freud(self.hypno_data.cloud_dir)
      console.show_status('Reading sampling frequency ...', prompt=self.prompt)
      fs = freud.get_sampling_frequency(self.hypno_data.signal_group_dir,
                                        self.hypno_data.sg_fn_pattern,
                                        self.hypno_data.channels)
      console.show_status(f'Sampling frequency: {fs} Hz', prompt=self.prompt)

      extractor_dict = get_extractor_dict(probe_keys, fs=fs)
      freud.generate_clouds(self.hypno_data.signal_group_dir,
                            pattern=self.hypno_data.sg_fn_pattern,
                            channels=self.hypno_data.channels,
                            time_resolutions=time_resolution,
                            overwrite=overwrite,
                            sg_file_list=sg_file_list,
                            extractor_dict=extractor_dict)

      # Generate macro features (Type-III)
      pass

      # Generate macro features (Type-III)
      freud.generate_macro_features(self.hypno_data.signal_group_dir,
                                    sg_file_list=sg_file_list)


    def load_nebula_from_clouds(self, time_resolution: int,
                                probe_keys=None) -> Nebula:
      # (0) Sanity check
      if probe_keys is None:
        probe_keys = self.probe_keys_for_extracting_features

      # (1) Load nebula
      freud = Freud(self.hypno_data.cloud_dir)
      nebula = freud.load_nebula(sg_labels=self.hypno_data.sg_labels,
                                 channels=self.hypno_data.channels,
                                 time_resolution=time_resolution,
                                 probe_keys=probe_keys)

      # (2) Set metadata
      meta_dict = self.hypno_data.load_meta()

      # Check if metadata matches signal group labels
      if len(meta_dict) != len(self.hypno_data.sg_labels):
        console.show_info(
          'No metadata found or metadata does not match signal group labels.')
        return nebula

      for pid in self.hypno_data.sg_labels:
        nebula.meta[pid] = meta_dict[pid]

      return nebula


    def load_macro_alpha(self, config='alpha'):
      freud = Freud(self.hypno_data.cloud_dir)
      features, feature_names = [], None

      for sg_label in self.hypno_data.sg_labels:
        cloud_path = freud._check_hierarchy(sg_label, create_if_not_exist=False)
        macro_path = os.path.join(cloud_path, f'macro_{config}.od')

        macro_dict: dict = io.load_file(macro_path)
        features.append(list(macro_dict.values()))

        if feature_names is None:
          feature_names = [f'Macro_{n}' for n in macro_dict.keys()]

      features = np.stack(features, axis=0)
      return features, feature_names

    # endregion: Public Methods

    # region: Private Methods

    # endregion: Private Methods
