from collections import OrderedDict
from roma import Nomear, io, console

import pandas as pd



class Record(Nomear):
  """A class representing a record in the database."""

  def __init__(self, row, master):
    assert isinstance(row, pd.Series), 'Row must be a pandas Series.'
    self.raw_data: pd.Series = row
    self.master = master


  def __getitem__(self, item):
    """Get item from the record."""
    return self.raw_data[item]
