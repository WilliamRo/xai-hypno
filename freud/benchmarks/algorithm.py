from collections import OrderedDict
from freud.datasets.dataset_base import HypnoDataset
from pictor.objects import SignalGroup
from roma import console, Nomear, io

import numpy as np
import os



class Algorithm(Nomear):
  """A base class of algorithm used in SOPs"""

  prompt = f'[Algo] >>'

  save_type_III_features = True  # Save Type III probes by default

  def __init__(self, dataset: HypnoDataset, time_resolution=30):
    super().__init__()

    self.hypno_data = dataset
    self.time_resolution = time_resolution

    self._type_I_probe_dict = OrderedDict()
    self._type_II_probe_dict = OrderedDict()
    self._type_III_probe_dict = OrderedDict()

    self.overwrite_type_III = False

    # Report the configuration
    console.show_status('Hypnomic pipeline initiated with',
                        prompt=self.prompt)

    console.supplement(f'Data directory: {dataset.data_dir}')
    console.supplement(f'Signal channels: {dataset.channels}')
    console.supplement(f'Time resolution: {self.time_resolution} s')


  def extract_features(self, **kwargs):
    """Extract feature vectors from a list of signal group filenames.

    Logic Description (prompt):

    """
    # (0) Initialize features and feature_names
    pass

    # (III) Gather type_III features
    feature_dict = self.gather_type_III_features()
    feature_names = list(feature_dict.keys())
    features = np.stack(list(feature_dict.values()), axis=1)

    # (-1) Finalize and return
    return features, {'feature_names': feature_names}


  def gather_type_III_features(self) -> OrderedDict:
    N = len(self.hypno_data.sg_labels)
    od = OrderedDict()  # od['<probe_key>'].shape = (N,)

    self.show_status('Gathering type-III features ...')
    for i, (pid, sg_path) in enumerate(zip(self.hypno_data.sg_labels,
                                        self.hypno_data.sg_file_list)):
      console.print_progress(i, N)

      assert pid in sg_path  # Sanity check
      sg = None

      # Gather type-III features from groups
      for group_key, func in self._type_III_probe_dict.items():
        # Try to load group from disk
        group_fn = f'{group_key}.od'
        group_path = os.path.join(self.hypno_data.cloud_dir, pid, group_fn)

        if os.path.exists(group_path) and not self.overwrite_type_III:
          group_dict = io.load_file(group_path, verbose=True)
        else:
          # If not exists, create a new group
          if sg is None: sg: SignalGroup = io.load_file(sg_path)
          group_dict = func(sg)

          # Save if required
          if self.save_type_III_features:
            io.save_file(group_dict, group_path, verbose=True)

        # Generate type_III features for each sg
        for pk, value in group_dict.items():
          if pk not in od: od[pk] = np.zeros(N, dtype=np.float32)
          od[pk][i] = value

    self.show_status(f'Gathered type-III features from {N} signal groups.')
    return od



  def set_probe(self, key, func: callable, probe_type: str):
    # Check probe type and get the corresponding dictionary
    assert probe_type in ('I', 'II', 'III'), f'Invalid probe type: {probe_type}'
    probe_dict = getattr(self, f'_type_{probe_type}_probe_dict')

    # Check if key already exists
    if key in probe_dict:
      raise ValueError(f'Probe key `{key}` already exists. Use a different key or overwrite it.')

    # Check func
    if not callable(func):
      raise ValueError(f'Probe function for key `{key}` must be callable.')

    # Check func signature
    # Type I: func(s: np.ndarray) -> float
    # Type II: func(s_dict: dict) -> float
    # Type III: func(sg: SignalGroup) -> OrderedDict
    assert probe_type == 'III', 'Only Type III probes are supported for now.'

    # Add the probe function to the corresponding dictionary
    probe_dict[key] = func


  def show_status(self, text: str):
    console.show_status(text, prompt=self.prompt)



if __name__ == '__main__':
  pass