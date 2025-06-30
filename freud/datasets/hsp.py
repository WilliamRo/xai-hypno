from freud.datasets.dataset_base import HypnoDataset, os
from roma import finder



class HSP(HypnoDataset):

   abbreviation = 'hsp'
   default_sg_fn_pattern = '*(float16,128Hz).sg'
   default_channels = ('F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1')

   @property
   def sg_file_list(self):
      if not self.in_pocket('sg_file_list'):
         return finder.walk(self.signal_group_dir, pattern=self.sg_fn_pattern)
      return self.get_from_pocket('sg_file_list', default=None)

   def load_meta(self, **kwargs) -> dict:
      return {}
