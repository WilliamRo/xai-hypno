from tframe.core.nomear import Nomear
from tframe import mu



class DeepSleepNet(Nomear):

  @staticmethod
  def get_fm_part_1(fs, N, use_bn=True, dp_keep_rate=0.5,
                        activation='relu'):
    conv = lambda ks, c, s=1: mu.HyperConv1D(
      filters=c, kernel_size=ks, strides=s, use_batchnorm=use_bn,
      activation=activation)
    pool = lambda k: mu.MaxPool1D(pool_size=k, strides=k)
    dp = lambda: mu.Dropout(dp_keep_rate)

    vertices = [[conv(fs // 2, N, fs // 16), pool(8), dp(),
                 conv(8, 2 * N), conv(8, 2 * N), conv(8, 2 * N), pool(4)],
                [conv(fs * 4, N, fs // 2), pool(4), dp(),
                 conv(6, 2 * N), conv(6, 2 * N), conv(6, 2 * N), pool(2)],
                [mu.Merge(mu.Merge.CONCAT, axis=1), dp()]]
    fm = mu.ForkMergeDAG(vertices, edges='1;10;011')

    return fm


  @staticmethod
  def get_layers_part_2(): raise NotImplemented




