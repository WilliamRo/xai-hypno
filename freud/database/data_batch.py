from collections import OrderedDict
from freud.database.rule import Rule
from freud.database.record import Record
from roma import Nomear, io, console

import os
import pandas as pd



class DataBatch(Nomear):

  prompt = '[Data Batch] >>'

  def __init__(self, data_path: str, primary_key: str = None):
    self.file_name = os.path.basename(data_path)
    self.raw_data: pd.DataFrame = self.read_data(data_path)

    self.primary_key = primary_key
    if primary_key is not None and primary_key not in self.raw_data.columns:
      raise ValueError(f'Primary key `{primary_key}` not found in data columns:'
                       f' {self.raw_data.columns.tolist()}')

    self.med_base = None

  # region: Properties

  @property
  def n_records(self): return len(self.raw_data)


  @property
  def columns(self): return self.raw_data.columns.tolist()


  @Nomear.property(local=True)
  def data_hash(self):
    row_hashes = pd.util.hash_pandas_object(self.raw_data, index=True).values
    # To get a single hash for the DataFrame:
    return hash(tuple(row_hashes))


  @Nomear.property(local=True)
  def registered_record_dict(self) -> OrderedDict:
    """Keys: primary key; Values: a list of pd rows"""
    return OrderedDict()


  @Nomear.property(local=True)
  def pending_data(self) -> list:
    """List of pd rows w/o primary key"""
    return []


  @property
  def total_registered_records(self) -> int:
    """Total number of registered records."""
    return sum(len(records)
               for records in self.registered_record_dict.values())

  # endregion: Properties

  # region: Public Methods

  def parse(self, rule: Rule, overwrite=False, **kwargs):
    # Check if the data batch is empty
    assert self.n_records > 0, 'No records found in the data batch.'

    # Check if the data batch has been parsed before
    if overwrite:
      self.registered_record_dict.clear()
      self.pending_data.clear()
    else:
      if len(self.registered_record_dict) > 0 or len(self.pending_data) > 0:
        console.warning('Data batch already parsed. Use `overwrite=True` to '
                        're-parse the data.')
        return

    # Iterate through the raw data
    for _, row in self.raw_data.iterrows():
      record = Record(row, batch=self)

      if self.primary_key is not None:
        key = row[self.primary_key]
        key = str(key)  # Restrict key to string type
        if rule.is_primary_key(key):
          rule.register(key)  # Register the primary key

          # Initialize the registered data if not exists
          if key not in self.registered_record_dict: self.registered_record_dict[key] = []

          self.registered_record_dict[key].append(record)

        else: self.pending_data.append(record)
      else: self.pending_data.append(record)


  def report(self, level=1, number=None):
    # Define handy supplement function
    sup = lambda text, lv=level: console.supplement(text, level=lv)

    # Report the basic information
    n_reg, n_pending = self.total_registered_records, len(self.pending_data)
    text = f'`{self.file_name}`: {n_reg} rows registered, {n_pending} pending.'
    if number is not None: text = f'[{number}] {text}'
    sup(text)

  # endregion: Public Methods

  # region: Private Methods

  def read_data(self, data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
      raise FileNotFoundError(f'`{data_path}` does not exist.')

    return pd.read_excel(data_path)

  def show_status(self, text): console.show_status(text, prompt=self.prompt)

  # endregion: Private Methods
