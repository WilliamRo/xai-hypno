from roma import console, Nomear

import os



class HypnoDataset(Nomear):
  """This data file organization is for hypnomic algorithms"""

  abbreviation = None
  default_sg_fn_pattern = None
  default_channels = None

  def __init__(self, data_dir):
    assert os.path.exists(data_dir), f'Data directory does not exist: `{data_dir}`'
    self.data_dir = data_dir


  @property
  def raw_dir(self): return self._get_path('raw')

  @raw_dir.setter
  def raw_dir(self, value): self._set_path('raw', value)

  @property
  def signal_group_dir(self): return self._get_path('sg')

  @signal_group_dir.setter
  def signal_group_dir(self, value): self._set_path('sg', value)

  @property
  def cloud_dir(self): return self._get_path('clouds')

  @cloud_dir.setter
  def cloud_dir(self, value): self._set_path('clouds', value)

  @property
  def nebula_dir(self): return self._get_path('nebula')

  @nebula_dir.setter
  def nebula_dir(self, value): self._set_path('nebula', value)

  @property
  def omix_dir(self): return self._get_path('omix')

  @omix_dir.setter
  def omix_dir(self, value): self._set_path('omix', value)

  @property
  def sg_fn_pattern(self): return self._get_config('sg_fn_pattern')

  @sg_fn_pattern.setter
  def sg_fn_pattern(self, value):
    self._set_config('sg_fn_pattern', value)

  @property
  def channels(self): return self._get_config('channels')

  @channels.setter
  def channels(self, value): self._set_config('channels', value)

  @property
  def sg_file_list(self):
    return self.get_from_pocket('sg_file_list', default=None)

  @sg_file_list.setter
  def sg_file_list(self, value):
    self.put_into_pocket('sg_file_list', value, exclusive=False)

  @property
  def sg_labels(self):
    # TODO: sg file name should strictly obey the pattern `<sg_label>(...).sg`
    return [os.path.basename(path).split('(')[0] for path in self.sg_file_list]


  def _get_config(self, config_key):
    key = f'config::{config_key}'
    if not self.in_pocket(key):
      value = getattr(self, f'default_{config_key}', None)
      if value is None:
        raise ValueError(f'Default config key `{config_key}` is not defined')
      self._set_config(config_key, value)

    return self.get_from_pocket(key)


  def _set_config(self, config_key, value):
    key = f'config::{config_key}'
    self.put_into_pocket(key, value, exclusive=False)


  def _get_path(self, path_key):
    key = f'path::{path_key}'
    if not self.in_pocket(key):
      if self.abbreviation is not None:
        dir_name = f'{self.abbreviation}_{path_key}'
      else:
        dir_name = path_key

      self._set_path(path_key, os.path.join(self.data_dir, dir_name))

    return self.get_from_pocket(key)


  def _set_path(self, path_key, value):
    if not os.path.exists(value): os.mkdir(value)
    key = f'path::{path_key}'
    self.put_into_pocket(key, value, exclusive=False)


  def load_meta(self, **kwargs) -> dict: return {}





