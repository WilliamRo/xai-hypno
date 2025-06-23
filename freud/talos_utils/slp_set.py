import numpy as np
import os

from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import io, Nomear, console
from typing import List

if os.environ.get('BYPASS_TF', '0') == '0':
  from freud.talos_utils.slp_config import SleepConfig
  from tframe.data.sequences.seq_set import SequenceSet, DataSet
  from tframe.utils.misc import convert_to_one_hot
  from tframe.layers.common import BatchReshape
  from tframe import pedia
else:
  class DataSet(Nomear):
    def __init__(self, name):
      self.name = name
      self.properties = {}

  SequenceSet = DataSet
  convert_to_one_hot = lambda x, y: x
  BatchReshape = object
  console = io = pedia = object



class SleepSet(DataSet):

  class Keys:
    tapes = 'SleepSet::Keys::tapes'
    map_dict = 'SleepSet::Keys::map_dict'
    epoch_tables = 'SleepSet::Keys::epoch_table'

  ANNO_KEY_GT_STAGE = 'stage Ground-Truth'

  EPOCH_DURATION = 30.0

  CHANNELS = {}
  NUM_STAGES = 5

  AASM_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM', '?']

  # valid data-kwargs in th.data_config
  VALID_KWARGS = ['val_ids',
                  'test_ids',
                  'preprocess',
                  'mad',                  # max absolute deviation (IQR)
                  'sg_preprocess']

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  @signal_groups.setter
  def signal_groups(self, val):
    self.properties['signal_groups'] = val

  @property
  def num_signal_groups(self) -> int: return len(self.signal_groups)

  @SequenceSet.property()
  def validation_set(self) -> DataSet:
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    if th.use_rnn: return self.extract_seq_set(include_targets=True)

    return self.extract_data_set(include_targets=True)

  @SequenceSet.property()
  def epoch_table(self):
    """Epoch table will not be generated before it is called first-time.
    This table contains NUM_STAGES + 1 (typically 6) lists, each of which
    contains a list of tuples comprising information of a period of stage,
    i.e., table[<STAGE_ID>] = [(sg, start_t, duration), ...]. For example,
    table[0] contains all wake stage information from all signal groups.
    """
    # Generate table
    table = [[] for _ in range(self.NUM_STAGES + 1)]
    for sg in self.signal_groups:
      for i in range(self.NUM_STAGES + 1):
        table[i].extend(self.get_sg_epoch_tables(sg)[0][i])
    return table

  # endregion: Properties

  # region: Overwriting

  def _check_data(self): pass

  # region: gen_batches

  # region: Multi-epoch sampling

  def _sample_seqs_from_sg(self, sg, start_time, duration, with_stage=False):
    """Sample a sequence from a signal group.

    :param sg - signal group.
    :param start_time - start time in seconds. Will be randomized if a negative
                        number is provided
    :param duration - signal duration in seconds.

    :return (data, label) if with_stage. Otherwise, return data.
            data.shape = [L, C]; label.shape = [E],
            here E is epoch number, equals to L / fs / 30.
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig) and isinstance(sg, SignalGroup)

    # For now consider only 1 branch TODO
    assert len(th.fusion_channels) == 1

    # Only one tape (branch) is considered for now
    # tape.shape = [L_tape, C]
    tape, fs = sg.get_from_pocket(self.Keys.tapes)[0]

    # Randomize `start_time` if required
    if start_time < 0:
      high = (sg.total_duration - duration) // 30
      if high < 0: raise ValueError('!! `duration` is greater than sg length')
      relative_start_time = np.random.randint(0, high) * 30
      start_time = relative_start_time + sg.dominate_signal.ticks[0]

    # Convert unit to ticks
    start_i, L = int(start_time * fs), int(duration * fs)
    assert L < tape.shape[0]
    assert start_i % 30 == 0   # TODO: this assertion is for sleep staging only

    # Make sure start_i is legal, in SleepEDFx, valid stage length may be
    # shorter than valid tape length
    valid_tape_L = tape.shape[0]
    if with_stage:
      stage_ids = self.get_sg_epoch_tables(sg)[1]
      valid_stage_L = int(len(stage_ids) * fs * self.EPOCH_DURATION)
      valid_tape_L = min(valid_tape_L, valid_stage_L)
    start_i = min(max(0, start_i), valid_tape_L - L)

    # Apply shift window augmentation
    shift = int((np.random.rand() * 2 - 1)
                * fs * self.EPOCH_DURATION * th.epoch_delta)
    i1 = min(max(0, start_i + shift), valid_tape_L - L)
    data = tape[i1 : i1+L]

    # Padding data if required
    if th.epoch_pad > 0:
      assert th.epoch_num == th.eval_epoch_num == 1
      data_list, P = [data], int(th.epoch_pad * fs * self.EPOCH_DURATION)
      # Pad left
      data_list.insert(0, tape[max(i1 - P, 0):i1])
      if i1 - P < 0:
        data_list.insert(0, np.zeros_like(tape[:P - i1], dtype=float))
      # Pad right
      data_list.append(tape[i1 + L:i1 + L + P])
      if i1 + L + P > tape.shape[0]:
        data_list.append(np.zeros_like(tape[:i1 + L + P - tape.shape[0]]))
      data = np.concatenate(data_list, axis=0)

    # Return
    if with_stage:
      start_j, L = int(start_i / fs / 30), int(duration / 30)
      labels = stage_ids[start_j:start_j+L]
      if th.epoch_pad == 0: assert data.shape[0] // 30 // fs == len(labels)
      return data, labels

    return data

  def _get_sequence_randomly_rnn(self, batch_size):
    """This method had been revised to fit `gen_rnn_batches`
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    features, targets, masks = [], [], []

    # Randomly choose <bs> sg to sample, but this is not fair for long sgs
    for sg in np.random.choice(self.signal_groups, size=batch_size):
      # Randomly sample sequences from sg
      data, labels = self._sample_seqs_from_sg(sg, -1, th.epoch_num * 30, True)
      # data.shape = [L_d, C]
      data = np.reshape(data, [th.epoch_num, -1, data.shape[-1]])
      # data.shape = [E, fs*30, C]

      # Check invalid labels
      if th.use_batch_mask:
        labels = [0 if l is None else l for l in labels]
        masks.append([l is not None for l in labels])
      elif None in labels: raise ValueError(
        '!! Invalid labels found while not `use_batch_mask`')

      features.append(data)
      t = convert_to_one_hot(labels, self.NUM_STAGES)
      t = np.reshape(t, [th.epoch_num, self.NUM_STAGES])
      targets.append(t)  # t.shape = [E, 5]

    # Assemble features and targets for DataSet
    features = np.stack(features, axis=0)  # [bs, E, fs*30, C]
    targets = np.stack(targets, axis=0)    # [bs, E, 5]

    data_dict = {}
    if th.use_batch_mask:
      masks = np.stack(masks, axis=0)  # [bs, E]
      data_dict[pedia.batch_mask] = masks
    properties = {}
    properties[BatchReshape.DEFAULT_PLACEHOLDER_KEY] = th.epoch_num
    # This block happens only in training. During validation,
    # tensor_block_size should be specified manually.
    return DataSet(features, targets, data_dict=data_dict,
                   NUM_CLASSES=self.NUM_STAGES, check_data=False,
                   **properties)

  def _get_sequence_randomly_fnn(self, batch_size):
    """Randomly samples a DataSet, whose features.shape = [B, L, C],
       targets.shape = [B, E, 5], here L = E * ticks_per_epoch.
       Note features.shape can be easily reshaped to [B, E, L/E, C].
       Unlike single epoch sampling, class balance is more difficult to
       achieve when sampling a sequence.
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    features, targets, masks = [], [], []

    # epoch_table = [[(sg, start_t, duration), ...], ...]
    for sid in np.random.randint(0, self.NUM_STAGES, batch_size):
      table = self.epoch_table[sid]
      sg, start_t, duration = table[np.random.randint(0, len(table))]

      # Randomly sample sequences from sg
      data, labels = self._sample_seqs_from_sg(
        sg, start_t, th.epoch_num * 30, with_stage=True)

      # Check invalid labels
      if th.use_batch_mask:
        labels = [0 if l is None else l for l in labels]
        masks.extend([l is not None for l in labels])
      elif None in labels: raise ValueError(
        '!! Invalid labels found while not `use_batch_mask`')

      features.append(data)
      t = convert_to_one_hot(labels, self.NUM_STAGES)
      t = np.reshape(t, [th.epoch_num, self.NUM_STAGES])
      targets.append(t)

    # Assemble features and targets for DataSet
    features = np.stack(features, axis=0)
    targets = np.concatenate(targets, axis=0)

    data_dict = {}
    if th.use_batch_mask: data_dict[pedia.batch_mask] = masks
    properties = {}
    properties[BatchReshape.DEFAULT_PLACEHOLDER_KEY] = th.epoch_num
    # This block happens only in training. During validation,
    # tensor_block_size should be specified manually.
    return DataSet(features, targets, data_dict=data_dict,
                   NUM_CLASSES=self.NUM_STAGES, check_data=False,
                   **properties)

  # endregion: Multi-epoch sampling

  # region: Single-epoch sampling

  def _get_branches_randomly(self, batch_size: int):
    """Currently only 1 branch is supported"""
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    if th.use_rnn: raise NotImplementedError(
      '!! RNN inputs are not supported currently')

    # Generate FNN inputs
    branches = [[] for _ in th.fusion_channels]
    assert len(branches) == 1
    stage_ids = []

    # epoch_table = [[(sg, start_t, duration), ...], ...]
    # For each record in this batch, first decide its corresponding stage.
    for sid in np.random.randint(0, self.NUM_STAGES, batch_size):
      table = self.epoch_table[sid]
      # Then choose a random interval
      sg, start_t, duration = table[np.random.randint(0, len(table))]

      # Get tape and fs
      for branch, tape_tuple in zip(
          branches, sg.get_from_pocket(self.Keys.tapes)):
        tape, fs = tape_tuple
        # Sliding-window augmentation, default epoch_delta = 0.2
        start_t += (np.random.rand() * 2 - 1) * duration * th.epoch_delta
        d = int(duration * fs)
        start_i = min(max(int(start_t * fs), 0), len(tape) - d)
        branch.append(tape[start_i:start_i+d])
        stage_ids.append(sid)

    # Generate features and targets
    features = np.stack(branches[0], axis=0)
    targets = convert_to_one_hot(stage_ids, self.NUM_STAGES)

    return DataSet(features, targets, NUM_CLASSES=self.NUM_STAGES)

  # endregion: Single-epoch sampling

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    """Generate FNN batches (xs, ys). Each xs[i] has a shape of [E, L, C].
    Here C represents number of channels, E is the number of epochs,
    and L = fs * 30, where fs is sampling rate and E. Each ys[i] has a shape of
    [E, 5].
    """
    from tframe import hub as th

    # Validate training set
    if not is_training:
      for batch in self.validation_set.gen_batches(batch_size): yield batch
      return
    elif callable(self.data_fetcher): self.data_fetcher(self)

    round_len = self.get_round_length(batch_size, training=is_training)
    assert batch_size != -1

    # Generate batches
    for i in range(round_len):
      if th.epoch_num >= 1:
        data_batch = self._get_sequence_randomly_fnn(batch_size)
      else:
        data_batch = self._get_branches_randomly(batch_size)

      # Preprocess if necessary
      if self.batch_preprocessor is not None:
        data_batch = self.batch_preprocessor(data_batch, is_training)
      # Make sure data_batch is a regular array
      if not data_batch.is_regular_array: data_batch = data_batch.stack
      # Yield data batch
      yield data_batch

    # Clear dynamic_round_len
    self._clear_dynamic_round_len()


  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False,
                      is_training=False, act_lens=None):
    """Yields data of shape [batch_size, num_steps, ...].
    """
    from tframe import hub as th

    if not is_training:
      raise NotImplementedError
    elif callable(self.data_fetcher): self.data_fetcher(self)

    round_len = th.epoch_num // th.num_steps
    if is_training: self._set_dynamic_round_len(round_len)

    # Sample <th.batch_size> sequences of length <th.epoch_num>
    ds = self._get_sequence_randomly_rnn(batch_size)
    ds.is_rnn_input = True

    for batch in ds.gen_rnn_batches(
        batch_size, num_steps, is_training=is_training):
      # TODO: temp workaround for mask issue
      mask_key = 'batch_mask'
      if mask_key in batch.data_dict:
        mask = batch.data_dict[mask_key]
        batch.data_dict[mask_key] = np.ravel(mask)

      yield batch

    # `_clear_dynamic_round_len` will be called in gen_rnn_batches
    self._clear_dynamic_round_len()

  # endregion: gen_batches

  @property
  def size(self): return len(self.validation_set.features)

  # endregion: Overwriting

  # region: Standards

  @classmethod
  def get_map_dict(cls, sg: SignalGroup):
    # TODO: currently only AASM standard is supported
    # assert cls.NUM_STAGES == 5
    anno: Annotation = sg.annotations[cls.ANNO_KEY_GT_STAGE]

    def _init_map_dict(labels):
      map_dict = {}
      # TODO: this will be called for each sg, which is verbosing
      # console.show_status('Creating mapping ...')
      for i, label in enumerate(labels):
        if 'W' in label: j = 0
        elif '1' in label: j = 1
        elif '2' in label: j = 2
        elif '3' in label or '4' in label: j = 3
        elif 'R' in label: j = 4
        else: j = None
        map_dict[i] = j
        # console.supplement(f'{label} maps to {j}', level=2)
      return map_dict

    return sg.get_from_pocket(
      cls.Keys.map_dict, initializer=lambda: _init_map_dict(anno.labels))

  @classmethod
  def get_sg_epoch_tables(cls, sg: SignalGroup):
    """`epoch_tables` will be used in
       (i) SleepSet.epoch_table, where `table_per_class` will be gathered, and
       (ii) SleepSet._sample_seqs_from_sg and extract_data_set,
            where `stage_ids` will be extracted from `table_ids`

    [Rule] 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM, None: Unknown
       (1) table_per_class = [[(sg, start_i, duration), ...], ...]
       (2) table_id = [0, 1, 2, ...], contains stage_id for each epoch in order
    """
    def _init_sg_epoch_tables():
      from tframe import hub as th

      # Get annotation
      anno: Annotation = sg.annotations[cls.ANNO_KEY_GT_STAGE]
      # Generate map_dict (maps raw stage ID to AASM stage ID)
      map_dict = cls.get_map_dict(sg)
      # Generate tgt_map_dict (maps AASM 5 stages to grouped stages)
      tgt_map_dict = th.tgt_map_dict
      tgt_map_dict[None] = None
      # 5 stages + 1 unknown label
      table_per_class = [[] for _ in range(cls.NUM_STAGES + 1)]
      table_id = []

      t0 = anno.intervals[0][0]
      for interval, anno_id in zip(anno.intervals, anno.annotations):
        # RAW anno to AASM(W:0, N1:1, N2:2, N3:3, REM:4, others:None)
        sid = map_dict[anno_id]
        # AASM to grouped (th.tgt_config)
        sid = tgt_map_dict[sid]
        N = (interval[-1] - interval[0]) / cls.EPOCH_DURATION
        # Check N
        assert N == int(N)

        # The 1st element in tuple is for future concatenating
        NUM_STAGES = th.num_classes
        table_per_class[sid if sid is not None else NUM_STAGES].extend([
          (sg, interval[0] + i * cls.EPOCH_DURATION - t0, cls.EPOCH_DURATION)
          for i in range(int(N))])
        table_id.extend([sid] * int(N))
      return table_per_class, table_id

    return sg.get_from_pocket(
      cls.Keys.epoch_tables, initializer=_init_sg_epoch_tables)

  # endregion: Standards

  # region: Methods for configuration

  def configure(self):
    """Configure dataset.
       th.data_config example:
       (1) 'sleepedfx 1,2;3'
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    # Check data-kwargs
    for key in th.data_kwargs:
      if key not in self.VALID_KWARGS: raise KeyError(
        f'!! `{key}` is not a valid data-kwargs')

    # (0) Set class names for ConfusionMatrix
    # self.properties['CLASSES'] = ['Wake', 'N1', 'N2', 'N3', 'REM']
    self.properties['CLASSES'] = [tp[0] for tp in th.tgt_tuples]

    # (1) extract required channels as tapes according to channel selection
    self.extract_sg_tapes()

  def extract_sg_tapes(self):
    """Extract signal tapes from each sg and put them into its pocket.
    Tapes are for efficiently sampling sub-sequences for training and
    evaluation.
    """
    from tframe import hub as th

    assert isinstance(th, SleepConfig)
    console.show_status('Extracting tapes ...')

    for i, sg in enumerate(self.signal_groups):
      console.print_progress(i, len(self.signal_groups))

      if sg.in_pocket(self.Keys.tapes): continue

      tapes = []

      for chn_lst in th.fusion_channels:
        if 'x' not in chn_lst[0]:
          # Case 1: fusion_channels = [['1', '2'], ['3']]
          # tape.shape = [L, C], TODO: fusion channels should have the same sfreq
          tape = np.stack([sg[self.CHANNELS[key]] for key in chn_lst], axis=-1)
          sfreq = sg.channel_signal_dict[self.CHANNELS[chn_lst[0]]].sfreq
        else:
          # Case 2： fusion_channels = [['EEGx2', 'EOGx1'], ['EMGx1']]
          tape_stack = []
          for chn_str in chn_lst:
            assert isinstance(chn_str, str)
            chn_str_parts = chn_str.split('x')
            key, num = chn_str_parts[0], int(chn_str_parts[1])

            candidates = [d for n, (_, d) in sg.name_tick_data_dict.items()
                          if n.startswith(key)]

            # TODO (beta) Append wake calibration if required
            if 'wake' in key.lower():
              assert len(candidates) == 0
              eeg = [d for n, (_, d) in sg.name_tick_data_dict.items()
                     if n.startswith('EEG')][0]
              T = 30 * 128
              for i in range(num):
                wake_let = eeg[i * T : (i + 1) * T]
                L = eeg.shape[0]
                N = int(np.ceil(L / T))
                wake = np.concatenate([wake_let] * N)[:L]
                candidates.append(wake)

            if len(candidates) < num: raise AssertionError(
              f'!! Not enough `{key}` channels to extract')

            # Currently, only first `num` channels will be extracted
            for i in range(num):
              tp = self.pre_process_tapes(key, candidates[i])
              assert isinstance(tp, list)
              tape_stack.extend(tp)

          tape = np.stack(tape_stack, axis=-1)
          # In case 2, all DigitalSignals in sg has a same sfreq
          sfreq = sg.digital_signals[0].sfreq

        # Get corresponding sfreq
        tapes.append((tape, sfreq))

      sg.put_into_pocket(self.Keys.tapes, tapes)

  @staticmethod
  def pre_process_tapes(key, x: np.ndarray):
    from tframe import hub as th
    assert isinstance(th, SleepConfig)
    assert len(x.shape) == 1

    if 'eeg' not in key.lower() or not th.pp_config: return [x]
    configs = th.pp_config.split(':')
    assert len(configs) == 2
    assert configs[0].startswith('alpha') and configs[0][-1] in ('1', '2')

    tape_list = []
    ks = int(configs[1])
    red_line = np.convolve(x, [1/ks] * ks, 'same')

    include_red = configs[0][-1] == '2'
    if include_red: tape_list.append(red_line)
    tape_list.append(x - red_line)

    return tape_list


  # endregion: Methods for configuration

  # region: Public Methods

  def get_subset_by_patient_id(self, indices, name_suffix='subset'):
    return self.__class__(
      name=f'{self.name}-{name_suffix}',
      signal_groups=[self.signal_groups[i] for i in indices],
      CLASSES=self['CLASSES'], NUM_CLASSES=self.NUM_STAGES)

  def extract_data_set(self, include_targets=False):
    """Extract talos.DataSet from self.signal_groups based on th.
       Note that self.configure method should be called beforehand. """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    # Generate FNN inputs
    branches = [[] for _ in th.fusion_channels]
    stage_ids = []
    N = th.eval_epoch_num
    for sg in self.signal_groups:
      tapes = sg.get_from_pocket(self.Keys.tapes)
      for branch, tape_tuple in zip(branches, tapes):
        tape, sfreq = tape_tuple
        assert isinstance(tape, np.ndarray)

        # tape.shape = [L, C], default epoch_num is 1
        ticks_per_seq = int(self.EPOCH_DURATION * sfreq) * N

        # Make sure all epochs in tape has annotation if `include_targets`
        if include_targets:
          anno_len = len(self.get_sg_epoch_tables(sg)[1])
          valid_L = int(anno_len * sfreq * self.EPOCH_DURATION)
          if tape.shape[0] > valid_L: tape = tape[:valid_L]

        # Truncate tape if necessary
        L = tape.shape[0] // ticks_per_seq * ticks_per_seq
        x = tape[:L].reshape([-1, ticks_per_seq, tape.shape[-1]])

        # Pad x if required
        if th.epoch_pad > 0:
          assert th.epoch_num == th.eval_epoch_num == 1
          T = ticks_per_seq
          for _ in range(th.epoch_pad):
            x_left = np.pad(x[:-1, :T], ((1, 0), (0, 0), (0, 0)), 'constant')
            x_right = np.pad(x[1:, -T:], ((0, 1), (0, 0), (0, 0)), 'constant')
            x = np.concatenate([x_left, x, x_right], axis=1)

        branch.append(x)
      # Find stage_ids if necessary
      if include_targets:
        sids = self.get_sg_epoch_tables(sg)[1][:x.shape[0]*N]
        stage_ids.extend(sids)

    # TODO currently only single branch is supported
    assert len(branches) == 1
    features = np.concatenate(branches[0], axis=0)

    # TODO: here len(ids) may > len(branches[i]), e.g., in SleepEDFx data
    data_dict = {}
    if include_targets:
      # Set mask
      assert len(stage_ids) == len(features) * N
      mask = [si is not None for si in stage_ids]
      data_dict[pedia.batch_mask] = np.stack(mask, axis=-1).reshape([-1, N, 1])

      # Set targets and dense-labels
      dense_labels = [0 if si is None else si for si in stage_ids]
      targets = convert_to_one_hot(dense_labels, self.NUM_STAGES)
      targets = targets.reshape([-1, N, targets.shape[-1]])
    else: targets = None

    # NUM_CLASSES and CLASSES properties are for confusion matrix label
    # features.shape = [S, N * Ls, C]
    # targets.shape = [S, N, NUM_STAGES]
    ds = DataSet(features, targets, data_dict, name=f'{self.name}-eva',
                 NUM_CLASSES=self.NUM_STAGES, CLASSES=self['CLASSES'])

    def batch_preprocessor(ds: DataSet, _):
      """This is for batch-evaluation"""
      from tframe.layers.common import BatchReshape

      # Set tensor_block_size for potential reshape
      key = BatchReshape.DEFAULT_PLACEHOLDER_KEY
      ds.properties[key] = N

      if ds.targets is not None:
        ds.targets = np.reshape(ds.targets, [-1, ds.targets.shape[-1]])
        ds.data_dict[pedia.batch_mask] = np.reshape(
          ds.data_dict[pedia.batch_mask], [-1])
      return ds

    ds.batch_preprocessor = batch_preprocessor
    ds.properties['signal_groups'] = self.signal_groups
    return ds

  def extract_seq_set(self, include_targets=False):
    """Extract talos.SequenceSet from self.signal_groups based on th.
       Note that self.configure method should be called beforehand. """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    sequences, stage_ids, masks = [], [], []
    for sg in self.signal_groups:
      # (1) prepare tape
      tape, fs = sg.get_from_pocket(self.Keys.tapes)[0]
      L, C = tape.shape
      ticks_per_epoch = int(self.EPOCH_DURATION * fs)
      E = L // ticks_per_epoch

      # (2) append targets and masks
      if include_targets:
        stages = self.get_sg_epoch_tables(sg)[1]
        E = min(len(stages), E)

        # Truncate stages
        stages = stages[:E]

        mask = [si is not None for si in stages]
        masks.append(mask)

        stages = [0 if si is None else si for si in stages]
        stages = convert_to_one_hot(stages, self.NUM_STAGES)
        stage_ids.append(stages)

      # (*) append truncated features
      L = E * ticks_per_epoch
      tape = np.reshape(tape[:L], [E, ticks_per_epoch, C])
      sequences.append(tape)

    # Put data into a SequenceSet
    data_dict = {}
    data_dict[pedia.features] = sequences
    if include_targets:
      data_dict[pedia.targets] = [np.array(ids) for ids in stage_ids]
      data_dict[pedia.batch_mask] = [np.array(m) for m in masks]
    ss = SequenceSet(data_dict=data_dict, name=f'{self.name}-eva',
                     NUM_CLASSES=self.NUM_STAGES, CLASSES=self['CLASSES'])

    ss.properties['signal_groups'] = self.signal_groups
    return ss

  # endregion: Public Methods

  # region: Abstract Methods (for reading data)

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    raise NotImplementedError

  @classmethod
  def load_as_sleep_set(cls, data_dir, **kwargs) -> SequenceSet:
    try:
      from tframe import hub as th
      kwargs['preprocess'] = th.data_kwargs.get('sg_preprocess', '')
    except: pass

    sg = cls.load_as_signal_groups(data_dir, **kwargs)
    return cls(name=cls.__name__, signal_groups=sg, NUM_CLASSES=cls.NUM_STAGES)

  @classmethod
  def load_as_raw_sg(cls, data_dir, pid, n_patients, i, **kwargs):
    # If the corresponding .sg file exists, read it directly
    suffix = kwargs.get('suffix', '(raw)')
    raw_sg_path = os.path.join(data_dir, pid + suffix + '.sg')

    bucket = []
    if cls.try_to_load_sg_directly(
        pid, raw_sg_path, n_patients, i, bucket, **kwargs):
      return bucket[0]

    sg = cls.load_sg_from_raw_files(data_dir, pid, **kwargs)

    # Save sg if necessary
    cls.save_sg_file_if_necessary(
      pid, raw_sg_path, n_patients, i, sg, **kwargs)
    return sg

  @classmethod
  def load_sg_from_raw_files(cls, data_dir, pid, **kwargs):
    raise NotImplementedError

  # endregion: Abstract Methods

  # region: Data Reading

  @staticmethod
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
    from freud import read_digital_signals_mne
    return read_digital_signals_mne(
      file_path, groups, dtype, max_sfreq, **kwargs)

  @staticmethod
  def read_annotations_mne(file_path: str, labels=None) -> Annotation:
    """Read annotations using `mne` package"""
    from freud import read_annotations_mne
    return read_annotations_mne(file_path, labels)

  # region: Common Utilities

  @staticmethod
  def try_to_load_sg_directly(
      pid, sg_path, n_patients, i, signal_groups, **kwargs):
    console_symbol = f'[{i + 1}/{n_patients}]'
    if os.path.exists(sg_path) and not kwargs.get('overwrite', False):
      console.show_status(
        f'Loading `{pid}` data from `{sg_path}` ...', symbol=console_symbol)
      console.print_progress(i, n_patients)
      sg = io.load_file(sg_path)
      signal_groups.append(sg)
      return True

    # Otherwise, create sg from raw file
    console.show_status(f'Reading `{pid}` data ...', symbol=console_symbol)
    console.print_progress(i, n_patients)
    return False

  @staticmethod
  def save_sg_file_if_necessary(pid, sg_path, n_patients, i, sg, **kwargs):
    if kwargs.get('save_sg', True):
      console.show_status(f'Saving `{pid}` data ...')
      console.print_progress(i, n_patients)
      io.save_file(sg, sg_path)
      console.show_status(f'Data saved to `{sg_path}`.')

  @staticmethod
  def parse_preprocess_configs(cfg_str: str):
    configs, suffix_list = {}, []

    for config in cfg_str.split(';'):
      mass = config.split(',')
      if 'trim' in mass[0]:
        trim = '' if len(mass) == 1 else mass[1]
        configs['trim'] = trim
        suffix_list.append(f'trim{trim}')
      elif 'iqr' == mass[0]:
        norm = ('iqr', '1' if len(mass) < 2 else mass[1],
                '20' if len(mass) < 3 else mass[2])
        configs['norm'] = norm
        suffix_list.append(f"{','.join(norm)}")
      elif '128' == mass[0]:
        configs['sfreq'] = 128
        suffix_list.append('128')
      elif mass[0] == '': continue
      else: raise KeyError(f'!! Unknown config option `{mass[0]}`')

    return configs, ';'.join(suffix_list)

  @staticmethod
  def pp_resample(sg: SignalGroup, fs):
    from scipy import signal

    ds0: DigitalSignal = sg.digital_signals[0]

    # TODO: 64 for ucddb EMG channels, 100 for sleepedfx EEG channels
    #       for ucddb, ds0 happens to be the ds needed for resampling
    assert ds0.sfreq in (64, 100)

    N = int(ds0.length // ds0.sfreq * fs)
    data_new = np.stack([
      signal.resample(ds0.data[:, i], num=N)
      for i in range(ds0.num_channels)], axis=-1)

    data_new = data_new.astype(ds0.data.dtype)

    sg.digital_signals[0] = DigitalSignal(
      data_new, fs, channel_names=ds0.channels_names, label=ds0.label)

  @classmethod
  def pp_trim(cls, sg, config):
    raise NotImplementedError

  @staticmethod
  def pp_normalize(sg: SignalGroup, config):
    norm = config
    if norm[0] == 'iqr':
      # Rescale data so that median value is 0, put 25 and 75 percentile to
      # [-0.5, 0.5], clip values out of max deviation
      iqr, mad = int(norm[1]), int(norm[2])
      for ds in sg.digital_signals:
        dtype = ds.data.dtype
        ds.data = DigitalSignal.preprocess_iqr(
          ds.data, iqr=iqr, max_abs_deviation=mad, labels=ds.channels_names)
        ds.data = ds.data.astype(dtype)
    else:
      raise KeyError(f'!! unknown normalization method {norm[0]}')

  @classmethod
  def preprocess_sg(cls, sg: SignalGroup, configs: dict):
    # (i) up-sampling if necessary
    key = 'sfreq'
    if key in configs: cls.pp_resample(sg, configs[key])

    # (ii) trim wake if required
    key = 'trim'
    if key in configs: cls.pp_trim(sg, configs[key])

    # (iii) normalize 1st DigitalSignal if required
    key = 'norm'
    if key in configs: cls.pp_normalize(sg, configs[key])

    return sg

  # endregion: Common Utilities

  # endregion: Data Reading

  # region: Visualization

  def show(self, *funcs, **kwargs):
    from freud.gui.freud_gui import Freud

    # Initialize pictor and set objects
    freud = Freud(title=str(self.__class__.__name__))
    freud.objects = self.signal_groups

    for func in [func for func in funcs if callable(func)]: func(freud.monitor)

    if kwargs.get('return_freud', False): return freud

    for k, v in kwargs.items(): freud.monitor.set(k, v, auto_refresh=False)

    freud.show()

  # endregion: Visualization



if __name__ == '__main__':
  data_root = r'E:\xai-sleep\data'

  edf_path = [
    r'sleepedf\SC4001E0-PSG.edf',
    r'ucddb\ucddb002.rec.edf',
    r'rrsh\JJF.edf',
  ][0]

  file_path = os.path.join(data_root, edf_path)

  for ds in SleepSet.read_digital_signals_mne(file_path): print(ds)
