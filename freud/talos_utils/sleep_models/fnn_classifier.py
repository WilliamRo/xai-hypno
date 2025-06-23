from tframe import mu



class FNNClassifier(mu.Classifier):
  """Auto-staging using feed-forward neural network.

  t-file check list:
    (1) th.updates_per_round
    (2) th.epoch_num (default=1)
    (*) th.data_config
  """

  # region: Properties



  # endregion: Properties

  def configure(self):
    """Configure necessary settings for this model to interact with signal
    groups"""
    pass

  def probe(self):
    pass

  def evaluate_slp_set(self, ds, **kwargs):
    """ds.configure method should be called beforehand since tapes will be used
    in this method.
    """
    from freud.talos_utils.slp_set import SleepSet
    assert isinstance(ds, SleepSet)
