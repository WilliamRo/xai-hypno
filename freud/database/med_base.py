from collections import OrderedDict
from freud.database.data_batch import DataBatch
from freud.database.record import Record
from freud.database.rule import Rule
from freud.database.patient import Patient
from freud.database.structure import DBStructure
from roma import Nomear, io, console

import os
import re
import pandas as pd



class MedBase(Nomear):

  prompt = '[MedBase] >>'

  def __init__(self, root_path: str, db_name='medical_db'):
    assert os.path.exists(root_path), f'!! Root path not found: `{root_path}`'
    self.root_path = root_path

    self.db_name = db_name

    # Initialize rule & structure
    self.rule = Rule()
    self.structure = DBStructure(med_base=self)

  # region: Properties

  @Nomear.property(local=True)
  def batch_dict(self) -> OrderedDict:
    """key: excel file name; value: pandas dataframe"""
    return OrderedDict()


  @Nomear.property(local=True)
  def patient_dict(self) -> OrderedDict:
    """key: patient ID; value: Patient"""
    od = OrderedDict()
    for key, record_list in self.registered_record_dict.items():
      od[key] = Patient(key, med_base=self)
      od[key].records.extend(record_list)
    return od


  @property
  def registered_record_dict(self) -> OrderedDict:
    """Registered data gathered from all DataBatches."""
    od = OrderedDict()
    for batch in self.batch_dict.values():
      for key, records in batch.registered_record_dict.items():
        if key not in od: od[key] = []
        od[key].extend(records)
    return od


  @property
  def pending_data(self) -> list:
    """Pending data gathered from all DataBatches."""
    pending = []
    for batch in self.batch_dict.values():
      pending.extend(batch.pending_data)
    return pending


  @property
  def total_registered_records(self) -> int:
    """Total number of registered records across all DataBatches."""
    return sum(len(records) for records in self.registered_record_dict.values())


  @property
  def all_records(self) -> list[Record]:
    records = []
    for r_list in self.registered_record_dict.values(): records.extend(r_list)
    records.extend(self.pending_data)
    return records

  # endregion: Properties

  # region: Public Methods

  # region: IO

  @staticmethod
  def load_db(db_path: str, verbose=True) -> 'MedBase':
    """Load MedBase from file"""
    # Check extension
    assert db_path.endswith('.mdb')
    # Load HypnoDB from file
    hdb: MedBase = io.load_file(db_path, verbose=verbose)

    # Set root path, check db_name
    dir_path, db_fn = io.dir_and_fn(db_path)
    hdb.root_path = dir_path
    if hdb.db_name not in db_fn: console.warning(
      f'!! DB name `{hdb.db_name}` not found in file name `{db_fn}`')

    # Import structure
    hdb.structure.import_structure()

    return hdb


  def save_db(self, db_path: str = None, verbose=True, export_structure=True):
    """Save MedBase to file"""
    if db_path is None:
      db_path = os.path.join(self.root_path, f'{self.db_name}.mdb')
    else:
      assert db_path.endswith('.mdb'), 'Database file must have `.mdb` extension'

    # Save HypnoDB to file
    io.save_file(self, db_path, verbose=verbose)

    # Save structure to file
    if export_structure: self.structure.export_structure(verbose=verbose)


  def read_raw_data(self, data_path: str, primary_key: str,
                    auto_register=True, verbose=True) -> DataBatch:
    """Read raw excel/csv data from file. Procedures are:
       (1) Read file, check whether this file already exists in the database.
         (1.1) If it exists, return the existing DataBatch.
         (1.2) If it does not exist, create a new DataBatch and add it to the database.

       (2) If `auto_register` is True, register the DataBatch to the database.

       (3) Refresh the structure (new attributes will be registered).
    """
    if verbose: self.show_status(f'Reading raw data from `{data_path}` ...')

    # (0) Wrap the data in a DataBatch
    batch = DataBatch(data_path, primary_key=primary_key)
    batch.med_base = self

    # (1) Check if the file already exists in the database
    for fn, old_batch in self.batch_dict.items():
      if batch.data_hash == old_batch.data_hash:
        console.warning(f'`{batch.file_name}` already exists (as `{old_batch.file_name}`).')
        # (1.1) If it exists, return the existing DataBatch
        return old_batch

    # (1.2) If it does not exist, add it to the database
    self.batch_dict[batch.file_name] = batch
    if verbose: self.show_status(
      f'Added `{batch.file_name}` '
      f'({batch.raw_data.shape[0]} rows) to data dict.')

    # (2) If `auto_register` is True, register the DataBatch to the database
    #     Actually, this step is for registering the primary keys
    if auto_register: self.register_batch(batch, verbose=verbose)

    # (3) Refresh the structure (new attributes will be registered).
    self.structure.update()

    return batch

  # endregion: IO

  # region: Data Processing

  def register_batch(self, batch: DataBatch, verbose=True):
    """Register a DataBatch to the database"""
    if verbose: self.show_status(f'Registering batch `{batch.file_name}` ...')

    # (1) Parse the batch
    batch.parse(rule=self.rule, overwrite=False)

  # endregion: Data Processing

  # region: Queries

  def _mask_df(self, df: pd.DataFrame):
    if 'primary_key' in df.columns:
      df['primary_key'] = df['primary_key'].apply(
        lambda x: self.rule.primary_key_dict.get(x, ''))

    # Change name to internal key
    if 'name' in df.columns:
      df['name'] = df['primary_key'].apply(lambda x: f'Name_{int(x[3:])}')
    # if 'name' in df.columns: df.drop(columns=['name'], inplace=True)


  def export_all(self, save_to_file=False, groups='*', mask=True,
                 format='hs', col_rename=None, group_rename=None):
    """Export all records from the database from specified groups."""
    if format == 'hs':
      # Using Huashun format
      col_rename = {'primary_key': '病历号', 'age': '年龄', 'date': '检查时间',
                    'name': '姓名', 'gender': '性别', 'orexin': '食欲素',}

      group_rename = {'root': 'info', 'scale': 'naire',}

    # (1) Initialize group_dict, whose keys are group names and values are lists
    #   of record dicts.
    group_dict = OrderedDict()
    group_dict['root'] = []  # Root group is always present

    # Get leaf groups from the structure
    leaf_groups = self.structure.leaf_groups
    if groups == '*':
      for g in leaf_groups: group_dict[g.name] = []
    else:
      assert isinstance(groups, (list, tuple))
      for gn in groups:
        assert gn in [g.name for g in leaf_groups]
        group_dict[gn] = []

    # (2) Fill data for each group
    for pid, patient in self.patient_dict.items():
      # record_list contains all records of the same patient with 'pid'

      # (2.1) Get patient info from root group
      group_dict['root'].append(patient.root_dict)

      # (2.2) Initialize candidate group dict:
      #       {'group_1': [rec_dict_1_1, rec_dict_1_2, ...],
      #        'group_2': [rec_dict_2_1]}
      dict_of_rec_lists: OrderedDict = patient.get_dict_of_rec_lists(
        list(group_dict.keys()))

      for group_name, rec_list in dict_of_rec_lists.items():
        assert group_name in group_dict
        group_dict[group_name].extend(rec_list)

    # (-1) Save to file and return
    if save_to_file:
      from tkinter import filedialog
      save_path = filedialog.asksaveasfilename()

      # Check if the file has a valid extension
      fmt = 'xlsx'
      if re.match('.*\.{}$'.format(fmt), save_path) is None:
        save_path += '.{}'.format(fmt)

      with pd.ExcelWriter(save_path) as writer:
        for gn, rec_list in group_dict.items():
          df = pd.DataFrame(rec_list)
          df.dropna(axis=1, how='all', inplace=True)

          # Mask the data if required
          if mask: self._mask_df(df)

          # Rename columns if rename_dict is provided
          if col_rename is not None:
            df.rename(columns=col_rename, inplace=True)

          # Rename groups if group_rename is provided
          if group_rename is not None and gn in group_rename:
            gn = group_rename[gn]

          # Write to Excel sheet
          df.to_excel(writer, sheet_name=gn, index=False)

      self.show_status(f'Exported data saved to `{save_path}`.')


  def export(self, selector='*', groups=('root',),
             merge_radius=0, save_to_file=False, mask=True,
             include_internal_key=False):
    """Export a dataframe from the database.

    :param selector:
    :param groups:
    :param merge_radius: Radius (in days) for merging related records.
    :param save_to_file: If True, save the exported data to a file.
    :param mask: If True, mask the data (e.g., remove sensitive information).

    !! Exceptions:
    (1) Ambiguity caused by large `merge_radius`
    """
    # (1) Select records to export
    # TODO: restrict selector to a specific format for now
    assert selector == '*', 'Currently only `*` selector is supported.'

    # (2) Initialize a list of row dict, each row dict shares the same keys
    row_dict_list = []

    # For each patient:
    for pid, patient in self.patient_dict.items():
      # record_list contains all records of the same patient with 'pid'

      # (2.1) Get patient info from root group
      patient_info = patient.root_dict
      if include_internal_key:
        patient_info['internal_key'] = patient.internal_key

      # (2.2) Initialize candidate group dict:
      #       {'group_1': [rec_dict_1_1, rec_dict_1_2, ...],
      #        'group_2': [rec_dict_2_1]}
      dict_of_rec_lists: OrderedDict = patient.get_dict_of_rec_lists(groups)

      # (2.3) Iteratively scan and reduce dict_of_rec_lists until all records
      #       are merged
      if len(dict_of_rec_lists) == 0:
        # Handle situation that only root group is required to be exported
        assert 'root' in groups
        row_dict = self.structure.gen_empty_row_dict(groups)
        row_dict.update(patient_info)
        row_dict_list.append(row_dict)
        continue

      while len(dict_of_rec_lists) > 0:
        # (2.3.1) Initialize an empty row dict for the current record
        row_dict = self.structure.gen_empty_row_dict(groups)

        # (2.3.2) Update row_dict with patient info if required
        if 'root' in groups: row_dict.update(patient_info)

        # (2.3.3) Get the first key and its record list, update row_dict
        remaining_group_names = list(dict_of_rec_lists.keys())
        main_group_name = remaining_group_names[0]
        rec_list: list = dict_of_rec_lists[main_group_name]
        rec = rec_list.pop(0)
        assert isinstance(rec, dict) and 'date' in rec
        anchor_date = rec['date']
        row_dict.update(rec)

        # (2.3.4) Match record from other remaining groups
        for group_name in remaining_group_names[1:]:
          rec_list: list = dict_of_rec_lists[group_name]
          for rec in rec_list:
            # TODO: use exact match for now
            if rec['date'] == anchor_date:
              row_dict.update(rec)
              rec_list.remove(rec)
              # safe as long as bread after removal
              break

        # (2.3.5) Append the row_dict to the row_dict_list
        row_dict_list.append(row_dict)

        # (2.3.-1)
        for key in list(dict_of_rec_lists.keys()):
          if len(dict_of_rec_lists[key]) == 0: dict_of_rec_lists.pop(key)

    # (2.4) Convert row_dict_list to a pandas DataFrame, drop empty column
    df = pd.DataFrame(row_dict_list)
    df.dropna(axis=1, how='all', inplace=True)

    # (3) Data masking
    if mask: self._mask_df(df)

    # (-1) Save to file and return
    if save_to_file:
      from tkinter import filedialog
      save_path = filedialog.asksaveasfilename()

      # Check if the file has a valid extension
      fmt = 'xlsx'
      if re.match('.*\.{}$'.format(fmt), save_path) is None:
        save_path += '.{}'.format(fmt)

      df.to_excel(save_path, index=False)

      self.show_status(f'Exported data saved to `{save_path}`.')

    return df


  def report(self):
    # Define handy supplement function
    sup = lambda text, lv=1: console.supplement(text, level=lv)

    console.section(f'Details of `{self.db_name}` database')
    sup(f'Patient #: {len(self.patient_dict)}')
    sup(f'Registered primary key #: {len(self.rule.primary_key_dict)}')
    sup(f'Registered record #: {self.total_registered_records}')
    sup(f'Pending record #: {len(self.pending_data)}')

    self.structure.report()

    sup(f'DataBatch #: {len(self.batch_dict)}')
    for i, batch in enumerate(self.batch_dict.values()):
      batch.report(level=2, number=i+1)

  # endregion: Queries

  # endregion: Public Methods

  # region: Private Methods

  def show_status(self, text): console.show_status(text, prompt=self.prompt)

  # endregion: Private Methods



if __name__ == '__main__':
  # Config
  overwrite = 0

  root_path = r'E:\data\srrsh'
  db_name = 'hypno_db'

  db_path = os.path.join(root_path, f'{db_name}.mdb')

  # Create or load
  if overwrite or not os.path.exists(db_path):
    db = MedBase(root_path, db_name=db_name)
    db.save_db()
  else:
    db = MedBase.load_db(db_path)

  # Read data
  batch_path_1 = r"E:\data\srrsh\srrsh_raw\250605-compumedics-to-250120.xlsx"
  batch_path_2 = r"E:\data\srrsh\srrsh_raw\250714-orexin-20230207.xlsx"

  batch = db.read_raw_data(batch_path_1)
