from roma import console, Nomear



class Algorithm(Nomear):
  """A base class of algorithm used in SOPs"""

  def __init__(self):
    pass


  def extract_features(self, **kwargs):
    raise NotImplementedError



if __name__ == '__main__':
  pass