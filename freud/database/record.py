from collections import OrderedDict
from freud.database.structure import DBStructure
from roma import Nomear, io, console

import pandas as pd



class Record(Nomear):
  """A class representing a record in the database."""

  def __init__(self, row, batch):
    assert isinstance(row, pd.Series), 'Row must be a pandas Series.'
    self.raw_data: pd.Series = row
    self.batch = batch


  @property
  def structure(self) -> DBStructure: return self.batch.med_base.structure


  @property
  def row_dict(self) -> dict: return self.raw_data.to_dict()


  @property
  def group_dict(self) -> OrderedDict:
    """group_dict examples:
    {
      'root': {'primary_key': 'PID000001', 'name': 'John Doe', ...},
      'diagnosis': {'date': '2025-07-17', 'diagnosis1': 'T1N'},
      'lab': {'date': '2025-07-17', 'orexin': 71}
    }
    """
    od = OrderedDict()

    try:
      od['root'] = self.structure.root_group.extract(self.row_dict)
      for leaf_group in self.structure.leaf_groups:
        extracted = leaf_group.extract(self.row_dict)
        if extracted is not None: od[leaf_group.name] = extracted
    except Exception as e:
      console.warning(f'Error in extracting group_dict {self.raw_data} '
                      f'from `{self.batch.file_name}`')
      raise e

    return od


  def __getitem__(self, item):
    """Get item from the record."""
    return self.raw_data[item]
