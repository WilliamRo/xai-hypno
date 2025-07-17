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
    leaf_groups = [g.name for g in self.med_base.structure.leaf_groups]

    # Initialize an ordered dictionary for the groups
    od = OrderedDict()

    # Scan through records and fill the dictionary
    for record in self.records:
      rec_grp_dict = record.group_dict

      for group_name in groups:
        if group_name not in leaf_groups: continue
        if group_name in rec_grp_dict:
          if group_name not in od: od[group_name] = []
          # TODO: merge same record !!
          od[group_name].append(rec_grp_dict[group_name])

    # Merge same records in each group
    for group_name in od.keys():
      merged_list = []
      for this_rec in od[group_name]:
        found_in_merged_list = False
        # Try to find rec in merged_list with a same date
        for that_rec in merged_list:
          if this_rec['date'] == that_rec['date'] != None:
            # If found, merge the records
            for key in this_rec.keys():
              if that_rec[key] is None: that_rec[key] = this_rec[key]
              elif this_rec[key] is not None and that_rec[key] != this_rec[key]:

                # TODO: patch here
                if key == 'BMI' and abs(that_rec[key] - this_rec[key]) < 1.:
                  continue

                raise AssertionError(
                  f'!! Ambiguous record {key} in patient (ID={self.primary_key}): '
                  f'{that_rec[key]} != {this_rec[key]}.')

            found_in_merged_list = True

        if not found_in_merged_list: merged_list.append(this_rec)

      od[group_name] = merged_list

    return od
