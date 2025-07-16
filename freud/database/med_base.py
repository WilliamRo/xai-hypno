from collections import OrderedDict
from freud.database.data_batch import DataBatch
from freud.database.record import Record
from freud.database.rule import Rule
from freud.database.structure import DBStructure
from roma import Nomear, io, console

import os



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
    """key: patient ID; value: ???"""
    return OrderedDict()


  @property
  def registered_data(self) -> OrderedDict:
    """Registered data gathered from all DataBatches."""
    od = OrderedDict()
    for batch in self.batch_dict.values():
      for key, records in batch.registered_data.items():
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
    return sum(len(records) for records in self.registered_data.values())


  @property
  def all_records(self) -> list[Record]:
    records = []
    for r_list in self.registered_data.values(): records.extend(r_list)
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
    return hdb


  def save_db(self, db_path: str = None, verbose=True):
    """Save MedBase to file"""
    if db_path is None:
      db_path = os.path.join(self.root_path, f'{self.db_name}.mdb')
    else:
      assert db_path.endswith('.mdb'), 'Database file must have `.mdb` extension'
    # Save HypnoDB to file
    io.save_file(self, db_path, verbose=verbose)


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
