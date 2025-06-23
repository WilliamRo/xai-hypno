from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from typing import List

import os
import numpy as np



def read_digital_signals_mne(
    file_path: str,
    groups=None,
    dtype=np.float32,
    max_sfreq=None,
    **kwargs
) -> List[DigitalSignal]:
  """Read .edf file using `mne` package.

  :param groups: A list/tuple of channel names groups by sampling frequency.
         If not provided, data will be read in a channel by channel fashion.
  :param max_sfreq: maximum sampling frequency
  :param allow_rename: option to allow rename file when target extension is
         not .edf.
  """
  import mne.io

  chn_map = kwargs.get('chn_map', None)
  n_channels = kwargs.get('n_channels', None)

  # (1) Read file using mne.io
  # (1.1) Since mne.io only support .edf file, rename file if necessary
  if file_path[-4:] != '.edf' and kwargs.get('allow_rename', False):
    os.rename(file_path, file_path + '.edf')
    file_path += '.edf'

  # (1.2) Define a lambda function to open file
  open_file = lambda include=(): mne.io.read_raw_edf(
    file_path, include=include, preload=False, verbose=False)

  # TODO: Deprecated, the line below is used in early mne version where
  #   `include` argument is not supported.
  # open_file = lambda exclude=(): mne.io.read_raw_edf(
  #   file_path, exclude=exclude, preload=False, verbose=False)

  # (2) Read all channel names, create a reverse map.
  #   This is for handling situations in which channel names are not consistent.
  with open_file() as file: edf_channel_names = file.ch_names
  # (2.1) Create a map from edf channel names to standard channel names
  if callable(chn_map):
    chn_map_dict = {chn: chn_map(chn) for chn in edf_channel_names}
  else:
    chn_map_dict = {chn: chn for chn in edf_channel_names}

  # (2.2) Create a reverse map, this is for excluding channels specified via
  #       groups containing standard channel names.
  chn_rev_map = {v: k for k, v in chn_map_dict.items()}

  # (3) Initialize groups if not provided, otherwise get channel_names from
  #     groups. E.g., groups = [[Fpz-Oz, Pz-Oz], [EOG horizontal]]
  if groups is None:
    # If groups is not provided, read all channels. Putting each channel in a
    #   group is to avoid unnecessary resampling during mne.read_raw_edf.
    groups = [[chn_map_dict[chn]] for chn in edf_channel_names]

  # (4) Generate include lists. Each list contains channels with the same sfreq.
  #     Note that `channel_names` a in standard format.
  # TODO: group can include channels not in edf_channel_names
  include_lists = [[chn_rev_map[chn] for chn in g if chn in chn_rev_map]
                   for g in groups]

  n_include = sum([len(g) for g in include_lists])
  if n_channels is not None and n_include != n_channels:
    raise AssertionError(f'!! n_include = {n_include} != {n_channels}')

  # TODO: Deprecated, the line below is used in early mne version where
  #   `include` argument is not supported.
  # exclude_lists = [[chn_rev_map[chn] for chn in channel_names if chn not in g]
  #                  for g in groups]

  # Read raw data
  signal_dict = {}
  for include_list in include_lists:
    with open_file(include=include_list) as file:
      sfreq = file.info['sfreq']

      # Resample to `max_sfreq` if necessary
      if max_sfreq is not None and sfreq > max_sfreq:
        file.resample(max_sfreq)
        sfreq = max_sfreq

      # Read signal, group signals with the same sfreq
      if sfreq not in signal_dict: signal_dict[sfreq] = []
      signal_dict[sfreq].append((file.ch_names, file.get_data()))

  # Wrap data into DigitalSignals
  digital_signals = []
  for sfreq, signal_lists in signal_dict.items():
    data = np.concatenate([x for _, x in signal_lists], axis=0)
    data = np.transpose(data).astype(dtype)

    channel_names = [chn_map_dict[name] for names, _ in signal_lists
                     for name in names]

    digital_signals.append(DigitalSignal(
      data, channel_names=channel_names, sfreq=sfreq,
      label=','.join(channel_names)))

  return digital_signals


def read_annotations_mne(file_path: str, labels=None) -> Annotation:
  """Read annotations using `mne` package"""
  import mne

  # Read mne.Annotations
  mne_anno: mne.Annotations = mne.read_annotations(file_path)

  # Automatically generate labels if necessary
  if labels is None: labels = list(sorted(set(mne_anno.description)))

  # Read intervals and annotations
  intervals, annotations = [], []
  label2int = {lb: i for i, lb in enumerate(labels)}
  for onset, duration, label in zip(
      mne_anno.onset, mne_anno.duration, mne_anno.description):
    intervals.append((onset, onset + duration))
    annotations.append(label2int[label])

  return Annotation(intervals, annotations, labels=labels)
