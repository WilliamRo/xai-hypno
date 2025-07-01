from freud.datasets.dataset_base import HypnoDataset, os
from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma import finder



class HSP(HypnoDataset):

   abbreviation = 'hsp'
   default_sg_fn_pattern = '*(float16,128Hz).sg'
   default_channels = ('F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1')


   @HypnoDataset.property()
   def agent(self): return HSPAgent(self.data_dir)

