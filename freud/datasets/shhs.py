from freud.datasets.dataset_base import HypnoDataset, os
from freud.talos_utils.sleep_sets.shhs import SHHSAgent
from roma import finder



class SHHS(HypnoDataset):

   abbreviation = 'shhs'
   default_sg_fn_pattern = '*-*(float16,100Hz).sg'
   default_channels = ('EEG C4-A1', 'EEG C3-A2')


   @HypnoDataset.property()
   def agent(self): return SHHSAgent(self.data_dir)

