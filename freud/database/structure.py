from ._apis.logger import Logger
from ._apis.structure_related import AttributeContainer
from .parser import Parser
from collections import OrderedDict
from roma import Nomear, io, console

import pandas as pd
import os



class DBStructure(Nomear, AttributeContainer, Logger):
  """A DataBase Structure for MedBase consists of
  (1) One root group with name 'root'.
      - Must contain an Attribute with name 'primary_key'.
  (2) Several leaf groups with customized names.
      - Must contain an Attribute with name 'timestamp'.
  (3) One pending group with name 'pending'.
  (4) One shared group with name 'shared'.
  (5) One dropped group with name 'dropped'.

  Each group contains a set of Attributes.
  """

  prompt = '[DBStructure] >>'

  def __init__(self, med_base):
    self.med_base = med_base

  # region: Properties

  @Nomear.property(local=True)
  def attributes(self) -> list['Attribute']:
    return self._gen_init_attributes(return_dict=False)

  # BUILTIN_GROUPS
  @property
  def root_group(self) -> 'Group': return self._get_group('root')


  @property
  def shared_group(self) -> 'Group': return self._get_group('shared')


  @property
  def pending_group(self) -> 'Group': return self._get_group('pending')


  @property
  def dropped_group(self) -> 'Group': return self._get_group('dropped')


  @property
  def leaf_groups(self) -> list['Group']:
    od = OrderedDict()
    for a in self.attributes:
      # BUILTIN_GROUPS
      if a.group in ('root', 'pending', 'shared', 'dropped'): continue
      if a.group not in od: od[a.group] = Group(a.group, self)
      od[a.group].attributes.append(a)
    return list(od.values())

  # endregion: Properties

  # region: Public Methods

  # region: Communication

  def import_structure(self):
    # (1) Read all sheets from the Excel file
    save_fn = f'{self.med_base.db_name}_structure.xlsx'
    save_path = os.path.join(self.med_base.root_path, save_fn)

    if not os.path.exists(save_path):
      console.warning(f'Structure file `{save_path}` not found. ')
      return

    self.show_status(f'Importing structure from `{save_fn}` ...')
    od = OrderedDict()
    with pd.ExcelFile(save_path) as xls:
      # Read all sheets into a dictionary of DataFrames
      sheet_dict = pd.read_excel(xls, sheet_name=None, dtype=str,
                                 keep_default_na=False)

      # (1.1) Gather all attributes from all sheets
      for df in sheet_dict.values():
        for row in df.itertuples(index=False):
          assert row.column not in od
          # Compatibility patch
          cname = (row.canonical_name if hasattr(row, 'canonical_name')
                   else row.alias)
          remark = row.remark if hasattr(row, 'remark') else ''
          od[row.column] = {'cname': cname, 'dtype': row.dtype,
                            'group': row.group, 'remark': remark}

    # (1.2) Sanity check
    # for col in self.col2attribute.keys():
    #   if col not in od: raise ValueError(
    #     f'Column `{col}` not found in structure file `{save_fn}`.')

    self.show_status(f'{len(od)} columns found. Generating attributes ...')

    # (2) Generate new attribute list
    a_dict = self._gen_init_attributes(return_dict=True)
    for col, att_dict in od.items():
      # Get master name if exists
      cname = att_dict['cname']
      if cname != '':
        if cname not in a_dict:
          # Create a new attribute with alias if not exists
          a_dict[cname] = Attribute(cname, att_dict['group'], att_dict['dtype'],
                                    remark=att_dict['remark'])
        # Append column name to master's alias list. Note that 'col' may have
        #   been registered in built-in attributes !!
        if col not in a_dict[cname].alias: a_dict[cname].alias.append(col)
        if att_dict['group'] != a_dict[cname].group: raise AssertionError(
          f'Canonical attribute `{cname}` has different group (`{a_dict[cname].group})` '
          f'with alias `{col}` (`{att_dict["group"]}`).')
      elif col not in a_dict:
        # Create a new attribute without alias. Note that 'col' may have been
        #   registered as an alias of another attribute !!
        a_dict[col] = Attribute(col, att_dict['group'], att_dict['dtype'],
                                remark=att_dict['remark'])

    # (3) Update attributes list
    self.put_into_pocket('attributes', list(a_dict.values()),
                         exclusive=False, local=True)
    # Make sure
    for batch in self.med_base.batch_dict.values():
      for col in batch.columns:
        # Skip unnamed columns
        if 'Unnamed: ' in col: continue
        if col not in self.col2attribute: raise AssertionError(
          f'Column `{col}` from `{batch.file_name}` not found in attributes.')

    self.show_status(f'{len(a_dict)} attributes imported from `{save_fn}`.')


  def export_structure(self, verbose=False):
    """Export structure to a file."""

    # (-1) Save dataframes to Excel file
    save_fn = f'{self.med_base.db_name}_structure.xlsx'
    save_path = os.path.join(self.med_base.root_path, save_fn)

    if verbose: self.show_status(f'Exporting structure to `{save_path}` ...')
    with pd.ExcelWriter(save_path) as writer:
      def to_excel(g: 'Group'):
        if len(g) == 0: return
        g.data_frame.to_excel( writer, sheet_name=g.name, index=False)

      # Write groups to Excel
      # BUILTIN_GROUPS
      to_excel(self.pending_group)
      to_excel(self.root_group)
      for g in self.leaf_groups: to_excel(g)
      to_excel(self.shared_group)
      to_excel(self.dropped_group)

    if verbose: self.show_status(f'Structure exported to `{save_path}`.')


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
        # Skip unnamed columns
        if 'Unnamed: ' in col: continue
        if col not in self.col2attribute:
          # Create a new attribute
          attr = Attribute(name=col, group='pending')
          if col == batch.primary_key:
            assert self.attributes[0].name == 'primary_key'
            if col not in self.attributes[0].alias:
              self.attributes[0].alias.append(col)
          else:
            self.attributes.append(attr)
            n_new += 1

    self.show_status(f'{n_new} new attributes registered.')

  # endregion: Communication

  # region: MISC

  def report(self, level=1):
    # Define handy supplement function
    sup = lambda text, lv=level: console.supplement(text, level=lv)
    n_attributes = len(self.attributes)
    n_columns = len(self.col2attribute)

    sup(f'Database Structure ({n_columns} column names, {n_attributes} attributes):')
    # BUILTIN_GROUPS
    self.root_group.report(level=level + 1, prefix='Builtin-')
    self.shared_group.report(level=level + 1, prefix='Builtin-')
    for group in self.leaf_groups:
      group.report(level=level + 1)

    self.pending_group.report(level=level + 1, prefix='Builtin-')
    self.dropped_group.report(level=level + 1, prefix='Builtin-')


  def gen_empty_row_dict(self, groups: list[str]) -> OrderedDict:
    """Generate an empty row dictionary with specified groups.
    This is for exporting data to Excel or other formats.
    """
    leaf_groups = self.leaf_groups
    # (1) Share common attributes among leaf groups
    if any([group_name in leaf_groups for group_name in groups]):
      groups = ['shared'] + list(groups)

    # (2) Generate empty row dictionary with primary key
    od = OrderedDict()
    od['primary_key'] = None

    # (3) Add specified groups' attributes
    for group_name in groups:
      g: Group = self._get_group(group_name)
      for attr in g.attributes: od[attr.name] = None

    return od

  # endregion: MISC

  # endregion: Public Methods

  # region: Private Methods

  def _get_group(self, name):
    g = Group(name=name, structure=self)
    g.attributes.extend([a for a in self.attributes if a.group == name])
    return g


  def _gen_init_attributes(self, return_dict=False):
    init_attributes = [
      Attribute(name='primary_key', group='root', alias=['病历号', 'ID']),
      Attribute(name='name', group='root', alias=['姓名']),
      Attribute(name='gender', group='root', alias=['性别']),
      Attribute(name='age', group='shared', alias=['年龄']),
      Attribute(name='date', group='shared', dtype='date', alias=['日期']),
    ]
    if not return_dict: return init_attributes
    return OrderedDict((a.name, a) for a in init_attributes)

  # endregion: Private Methods



class Group(Nomear, AttributeContainer):

  def __init__(self, name, structure: DBStructure = None):
    self.name = name
    self.attributes: list[Attribute] = []
    self.structure = structure


  @property
  def data_frame(self) -> pd.DataFrame:
    col_names = ('column', 'canonical_name', 'dtype', 'group', 'remark')

    # Initialize an empty DataFrame with specified columns
    od = OrderedDict()
    for cn in col_names: od[cn] = []

    # Scan col2att dict
    for name, attr in self.col2attribute.items():
      assert isinstance(attr, Attribute)
      od['column'].append(name)
      od['canonical_name'].append('' if name == attr.name else attr.name)
      od['dtype'].append(attr.dtype)
      od['group'].append(self.name)
      od['remark'].append(attr.remark)

    # (-1) Return DataFrame
    return pd.DataFrame(od, dtype=str)


  def report(self, level=2, prefix=''):
    # Define handy supplement function
    sup = lambda text, lv=level: console.supplement(text, level=lv)
    sup(f'{prefix}Group `{self.name}`: {len(self.attributes)} attributes')


  def extract(self, row_dict: dict) -> OrderedDict:
    """Format: <primary_key>, <shared_keys>, <keys from leaves>
    TODO: Note that in MedBase.export, slots for <primary_key> and <shared_keys>
          will be created and the od extracted by this function is well-matched
          to its corresponding <primary_key>.
          Thus it works fine to omit <primary_key> and <shared_keys> here in
          this function.
    """
    assert self.name not in ('pending', 'shared', 'dropped')

    od = OrderedDict()
    if self.name != 'root':
      # (1) Extract primary key
      od['primary_key'] = self.structure.col2attribute['primary_key'].extract(
        row_dict)

      # (2) Extract shared keys
      for attr in self.structure.shared_group.attributes:
        od[attr.name] = attr.extract(row_dict)

    # (3) Extract self.keys
    flag = False
    for attr in self.attributes:
      od[attr.name] = attr.extract(row_dict)
      flag = flag or od[attr.name] is not None

    if flag: return od
    return None


  def __len__(self): return len(self.attributes)



class Attribute(Nomear):

  TYPES = ('str', 'int', 'float', 'date')

  def __init__(self, name, group='pending', dtype='str', alias=(), remark=''):
    self.name = name
    self.alias = list(alias)
    self.group = group
    self.dtype = dtype
    self.remark = remark


  def extract(self, row_dict: dict):
    """Extract value from a row dictionary."""
    if self.name in row_dict: return self.parse(row_dict[self.name])
    for alias in self.alias:
      if alias in row_dict: return self.parse(row_dict[alias])
    return None


  def parse(self, value):
    return Parser.parse(value, self)
