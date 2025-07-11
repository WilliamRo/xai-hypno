from freud.datasets.dataset_base import HypnoDataset, os
from roma import finder



class SleepEDFxSC(HypnoDataset):

   abbreviation = 'sc'
   default_sg_fn_pattern = '*(trim1800;128).sg'
   default_channels = ('EEG Fpz-Cz', 'EEG Pz-Oz')

   MAX_SUBJECTS = 999

   @property
   def sc_xls_path(self):
      return os.path.join(self.data_dir, 'SC-subjects.xls')

   @property
   def sg_file_list(self):
      if not self.in_pocket('sg_file_list'):
         return finder.walk(self.signal_group_dir, pattern=self.sg_fn_pattern)[:self.MAX_SUBJECTS]
      return self.get_from_pocket('sg_file_list', default=None)

   @sg_file_list.setter
   def sg_file_list(self, value):
      self.put_into_pocket('sg_file_list', value, exclusive=False)

   def load_meta(self, **kwargs) -> dict:
      import pandas as pd

      meta_dict = {}
      df = pd.read_excel(self.sc_xls_path)
      for pid in self.sg_labels:
         index = df['subject'] == int(pid[3:5])
         age = int(df.loc[index, 'age'].values[0])
         gender = 'female' if df.loc[index, 'sex (F=1)'].values[
                                 0] == 1 else 'male'
         meta_dict[pid] = {'age': age, 'gender': gender}
      return meta_dict
