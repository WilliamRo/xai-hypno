from freud.datasets.dataset_base import HypnoDataset, os
from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma import finder



class HSP(HypnoDataset):

   abbreviation = 'hsp'
   default_sg_fn_pattern = '*(float16,128Hz).sg'
   default_channels = ('EEG F3-M2', 'EEG F4-M1', 'EEG C3-M2',
                       'EEG C4-M1', 'EEG O1-M2', 'EEG O2-M1')


   @HypnoDataset.property()
   def agent(self): return HSPAgent(self.data_dir)

