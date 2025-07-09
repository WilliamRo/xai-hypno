from freud.datasets.dataset_base import HypnoDataset
from roma import console, Nomear



class Algorithm(Nomear):
  """A base class of algorithm used in SOPs"""

  prompt = f'[Algo] >>'

  def __init__(self, dataset: HypnoDataset, time_resolution=30):
    super().__init__()

    self.hypno_data = dataset
    self.time_resolution = time_resolution

    # Report the configuration
    console.show_status('Hypnomic pipeline initiated with',
                        prompt=self.prompt)

    console.supplement(f'Data directory: {dataset.data_dir}')
    console.supplement(f'Signal channels: {dataset.channels}')
    console.supplement(f'Time resolution: {self.time_resolution} s')


  def extract_features(self, **kwargs):
    show_status = lambda text: console.show_status(text, prompt=self.prompt)

    # (1)



    raise NotImplementedError



if __name__ == '__main__':
  pass