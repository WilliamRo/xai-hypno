from collections import OrderedDict
from freud.database.record import Record
from roma import Nomear, io, console



class Patient(Nomear):
  """A class representing a patient in the medical database."""

  def __init__(self, primary_key: str, med_base):
    self.primary_key = primary_key
    self.med_base = med_base
    self.records: list[Record] = []


  @property
  def n_records(self) -> int:
    """Number of records associated with this patient."""
    return len(self.records)


  @property
  def internal_key(self) -> str:
    """Internal key for the patient."""
    return self.med_base.rule.primary_key_dict[self.primary_key]


  @property
  def root_dict(self) -> OrderedDict:
    # Create an empty row dictionary with the primary key
    od: OrderedDict = self.med_base.structure.gen_empty_row_dict(('root',))
    od['primary_key'] = self.primary_key

    # Scan through records and fill the root_dict
    for record in self.records:
      root_group = record.group_dict.get('root', None)
      if root_group is None: continue
      for key in od.keys():
        if od[key] is None: od[key] = root_group[key]
        elif root_group[key] is not None and od[key] != root_group[key]:
          raise AssertionError(
            f'!! Ambiguous patient {key} (ID={self.primary_key}):' 
            f' {od[key]} != {root_group[key]}.')

    return od


  def __repr__(self):
    return f'Patient ({self.primary_key}, {self.n_records} records)'


  def get_dict_of_rec_lists(self, groups: list):
    """Get a dictionary of records grouped by specified groups.
    Only leaf groups are considered.
    """
    leaf_groups = self.med_base.structure.leaf_groups

    # Initialize an ordered dictionary for the groups
    od = OrderedDict()
    for group_name in groups:
      if group_name not in leaf_groups: continue
      od[group_name] = []

    # Scan through records and fill the dictionary
    for record in self.records:
      rec_grp_dict = record.group_dict

      for group_name in groups:
        if group_name not in leaf_groups: continue
        if group_name in rec_grp_dict: od[group_name].append(record)

    return od
