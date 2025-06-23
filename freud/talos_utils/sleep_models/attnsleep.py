from freud.talos_utils.sleep_models.dsn import DeepSleepNet

from tframe.core.nomear import Nomear
from tframe import mu
from tframe import tf
from tframe.nets.classic.conv_nets.conv_net import ConvNet

from tframe.layers.pooling import ReduceMean



class AttnSleep(ConvNet):

  def __init__(self, fs, N1, TCE_repeat=2):
    """
    fs: sampling frequency
    N1: channel in DSN-part1's first layer
    """
    # (1) MRCNN configs
    self.fs = fs
    self.N1 = N1

    # (2) TCE configs
    self.TCE_repeat = TCE_repeat


  def _get_layers(self):
    layers = []

    # (1) Add MRCNN part
    #  1.1 Add DSN head
    layers.append(self.get_dsn_head(self.fs, self.N1))
    #  1.2 Add AFR block
    layers.append(self.get_adaptive_feature_recalibration())

    # (2) Add TCE part
    for _ in range(self.TCE_repeat):
      layers.append(self.get_temporal_context_encoder())

    return layers


  @classmethod
  def get_dsn_head(cls, fs, N=64, dp_rate=0.5):
    conv = lambda ks, c, s=1: mu.HyperConv1D(
      filters=c, kernel_size=ks, strides=s,
      use_batchnorm=True, activation='ReLU')
    pool = lambda ps, st: mu.MaxPool1D(pool_size=ps, strides=st)
    dp = lambda: mu.Dropout(dp_rate)

    vertices = [[conv(fs // 2, N, fs // 16), pool(8, 2), dp(),
                 conv(8, 2 * N), conv(8, 2 * N), pool(4, 4)],
                [conv(fs * 4, N, fs // 2), pool(4, 2), dp(),
                 conv(7, 2 * N), conv(7, 2 * N), pool(2, 2)],
                [mu.Merge(mu.Merge.CONCAT, axis=1), dp()]]
    return mu.ForkMergeDAG(vertices, edges='1;10;011')


  @classmethod
  def get_adaptive_feature_recalibration(
      cls, reduced_filter_num=30, reduction=16):
    N = reduced_filter_num
    vertices = [
      [mu.Conv1D(N, 1, use_batchnorm=True, activation='relu'),
       mu.Conv1D(N, 1, use_batchnorm=True)],
      [ReduceMean(axis=1),
       mu.Dense(N // reduction, activation='relu', use_bias=False),
       mu.Dense(N, activation='sigmoid', use_bias=False),
       mu.Reshape(shape=[1, N])],
      mu.Merge(mu.Merge.PROD),
      mu.Conv1D(N, 1, use_batchnorm=True),
      [mu.Merge(mu.Merge.SUM), mu.Activation.ReLU()],
    ]
    edges = '1;01;011;1000;00011'

    return mu.ForkMergeDAG(vertices, edges=edges)


  @classmethod
  def get_temporal_context_encoder(cls):
    vertices = []
    edges = ''
    return mu.ForkMergeDAG(vertices, edges)





