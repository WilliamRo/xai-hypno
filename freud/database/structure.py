from collections import OrderedDict
from roma import Nomear, io, console

import pandas as pd



class DBStructure(Nomear):
  """A DataBase Structure for MedBase consists of
  (1) One root group with name 'root'.
      - Must contain an Attribute with name 'primary_key'.
  (2) Several leaf groups with customized names.
      - Must contain an Attribute with name 'timestamp'.
  (3) One pending group with name 'pending'.

  Each group contains a set of Attributes.
  """

  prompt = '[DBStructure] >>'

  def __init__(self, med_base):
    self.med_base = med_base

  # region: Properties

  @Nomear.property()
  def attributes(self) -> list['Attribute']:
    return [Attribute(name='primary_key', group='root')]


  @property
  def col2attribute(self) -> OrderedDict:
    """Mapping of column names to attributes."""
    od = OrderedDict()
    for attr in self.attributes:
      assert attr.name not in od, f'Attribute name `{attr.name}` is not unique.'
      od[attr.name] = attr
      for a in attr.alias:
        assert a not in od, f'Alias `{a}` is not unique.'
        od[a] = attr
    return od


  @property
  def root_group(self) -> 'Group':
    """Root group of the database structure."""
    g = Group(name='root')
    g.attributes.extend([a for a in self.attributes if a.group == 'root'])
    return g


  @Nomear.property()
  def pending_group(self) -> 'Group':
    g = Group(name='pending')
    g.attributes.extend([a for a in self.attributes if a.group == 'pending'])
    return g


  @Nomear.property()
  def leaf_groups(self) -> list['Group']:
    od = OrderedDict()
    for a in self.attributes:
      if a.group in ('root', 'pending'): continue
      if a.group not in od: od[a.group] = Group(name=a.group)
      od[a.group].attributes.append(a)
    return list(od.values())

  # endregion: Properties

  # region: Public Methods

  # region: Communication

  def import_structure(self):
    pass


  def export_structure(self):
    pass


  def update(self):
    """Update structure by scanning whole MedBase. New attributes will be added.
    Registered attributes will be updated.
    """
    from freud.database.data_batch import DataBatch
    from freud.database.med_base import MedBase

    med_base: MedBase = self.med_base

    # Go through all batches
    n_new = 0
    for batch in med_base.batch_dict.values():
      assert isinstance(batch, DataBatch)
      for col in batch.columns:
        if col not in self.col2attribute:
          # Create a new attribute
          attr = Attribute(name=col, group='pending')
          self.attributes.append(attr)
          n_new += 1

    self.show_status(f'{n_new} new attributes registered.')

  # endregion: Communication

  # region: MISC

  def report(self, level=1):
    # Define handy supplement function
    sup = lambda text, lv=level: console.supplement(text, level=lv)
    sup('Database Structure:')
    self.root_group.report(level=level + 1)
    for group in self.leaf_groups:
      group.report(level=level + 1)

    if len(self.pending_group) > 0: self.pending_group.report(level=level + 1)

  # endregion: MISC

  # endregion: Public Methods

  # region: Private Methods

  def show_status(self, text): console.show_status(text, prompt=self.prompt)

  # endregion: Private Methods



class Group(Nomear):

  def __init__(self, name):
    self.name = name
    self.attributes: list[Attribute] = []


  def report(self, level=2):
    # Define handy supplement function
    sup = lambda text, lv=level: console.supplement(text, level=lv)
    sup(f'Group `{self.name}`: {len(self.attributes)} attributes')


  def __len__(self): return len(self.attributes)



class Attribute(Nomear):

  def __init__(self, name, group='pending'):
    self.name = name
    self.alias = []
    self.group = group
