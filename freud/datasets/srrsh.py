from freud.datasets.dataset_base import HypnoDataset, os
from freud.talos_utils.sleep_sets.rrshv2 import SRRSHAgent
from roma import finder



class SRRSH(HypnoDataset):

   abbreviation = 'srrsh'
   default_sg_fn_pattern = '*-*(float16,100Hz).sg'
   default_channels = ('EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
                       'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1')


   @HypnoDataset.property()
   def agent(self): return SRRSHAgent(self.data_dir)

