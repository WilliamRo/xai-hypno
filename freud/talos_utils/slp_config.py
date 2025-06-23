from collections import OrderedDict
from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class SleepConfig(SmartTrainerHub):
  """
  th.data_config format:
  <data-name> <channels> [other-setting]
  """

  # region: SleepSet.gen_batches setting

  epoch_delta = Flag.float(
    0.0, 'Delta for generating batches, should be in [0, 1)', is_key=None)
  epoch_num = Flag.integer(1, 'Number of epochs in one batch', is_key=None)
  eval_epoch_num = Flag.integer(1, 'Number of epochs during evaluation',
                                is_key=None)
  sg_buffer_size = Flag.integer(10, 'Number of signal-groups loaded per round',
                                is_key=None)
  epoch_pad = Flag.integer(0, 'Padding num when epoch_num is 1.', is_key=None)

  # endregion: SleepSet.gen_batches setting

  # region: Data Setting

  pp_config = Flag.string(None, 'Preprocess arguments', is_key=None)
  tgt_config = Flag.string('Wake:0;N1:1;N2:2;N3:3;REM:4',
                           'Target configs', is_key=None)

  # endregion: Data Setting

  # region: Model Setting

  zoom_in_factor = Flag.integer(1, 'Zoom in factor', is_key=None)

  dtp_en_filters = Flag.integer(64, 'Filters in DTP encoder', is_key=None)
  dtp_en_ks = Flag.integer(16, 'Kernel size in DTP encoder', is_key=None)
  dtpM = Flag.integer(3, 'M in DTP', is_key=None)
  dtpR = Flag.integer(2, 'R in DTP', is_key=None)

  # endregion: Model Setting

  # region: Deprecated

  balance_training_stages = Flag.boolean(
    True, 'Whether to balance training stages', is_key=None)
  use_gen_batches_buffer = Flag.boolean(
    False, 'Whether to use gen_batches buffer')

  # endregion: Deprecated

  # region: Properties

  @property
  def data_name(self):
    return Arguments.parse(self.data_config).func_name

  @property
  def data_args(self):
    return Arguments.parse(self.data_config).arg_list

  @property
  def data_kwargs(self) -> dict:
    return Arguments.parse(self.data_config).arg_dict

  @property
  def fusion_channels(self):
    return [s.split(',') for s in self.data_args[0].split(';')]

  @property
  def input_channels(self):
    # Currently only 1 branch is supported
    assert len(self.fusion_channels) == 1
    # Case 1: fusion_channels = [['1', '2'], ['3']]
    if 'x' not in self.fusion_channels[0][0]:
      return len(self.fusion_channels[0])
    # Case 2： fusion_channels = [['EEGx2', 'EOGx1'], ['EMGx1']]
    return sum([int(chn_str[-1]) for chn_str in self.fusion_channels[0]])

  @property
  def tgt_tuples(self):
    """Wake:0,N1:1,N2:2,N3:3,REM:4"""
    kvs = [s.split(':') for s in self.tgt_config.split(';')]
    return [(k, [int(i) for i in v.split(',')]) for k, v in kvs]

  @property
  def tgt_map_dict(self):
    od = OrderedDict()
    for i, (_, js) in enumerate(self.tgt_tuples):
      for j in js: od[j] = i
    return od

  # endregion: Properties

  def smooth_out_conflicts(self):
    super().smooth_out_conflicts()

    if self.epoch_num > 1:
      msg_tail = ' if th.epoch_num > 1.'
      if self.epoch_delta != 0:
        raise AssertionError(f'!! th.epoch_delta should be 0' + msg_tail)


# New hub class inherited from SmartTrainerHub must be registered
SleepConfig.register()
