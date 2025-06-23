from collections import OrderedDict
from roma import console, io, Nomear

import os



class LongitudinalManager(Nomear):

  META_EXTENSION = '.csv'

  @property
  def meta_path(self): raise NotImplementedError

  @Nomear.property()
  def patient_dict(self):
    patient_dict_path = self.meta_path.replace(self.META_EXTENSION, '.od')
    if os.path.exists(patient_dict_path) and not self.in_pocket('OVERWRITE_PD'):
      return io.load_file(patient_dict_path, verbose=True)

    od = self.generate_patient_dict(self.meta_path)
    io.save_file(od, patient_dict_path, verbose=True)
    return od

  @staticmethod
  def generate_patient_dict(meta_path) -> OrderedDict: raise NotImplementedError
