from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np

from tframe import tf
from tframe.layers.advanced import Dense
from tframe.layers.convolutional import Conv2D
from tframe.layers.merge import Merge
from tframe.layers.convolutional import Conv1D
from tframe.layers.pooling import GlobalAveragePooling2D
from tframe.layers.pooling import AveragePooling1D
from tframe.layers.pooling import MaxPool1D
from tframe.layers.common import Flatten, Dropout
from tframe.layers.layer import Layer
from tframe.activations import sigmoid, relu
from tframe.layers.hyper.conv import Conv1D as HyperConv1D
from tframe.layers.hyper.conv import Deconv1D
from tframe.layers.normalization import BatchNormalization

from tframe.nets.classic.conv_nets.conv_net import ConvNet
from tframe.nets.forkmerge import ForkMergeDAG
from tframe.nets.net import Net
from tframe.layers.pooling import GlobalAveragePooling1D
from tframe.layers.common import Activation 
from tframe.layers.layer import LayerWithNeurons, Layer, single_input
from tframe import context



class AttnSleep(ConvNet):

  def __init__(self):
    super(AttnSleep, self).__init__()

    N = 2  # number of TCE clones
    self.d_model = 112  # set to be 100 for SHHS dataset
    self.d_ff = 120   # dimension of feed forward
    # h = 5  # number of attention heads
    self.dropout = 0.1
    self.attn_heads = 4 # number of attention heads
    self.num_classes = 5
    self.afr_reduced_cnn_size = 30


  def _get_layers(self):
    layers = []
    # (1) Define Encoder
    ## [?, 3000, 1] -->> [?, 80, 128]
    layers.append(MRCNN())
    ## [?, 80, 128] -->> [?, 80, 30]
    layers.append(AFR(self.afr_reduced_cnn_size))

    # (2) Define Temporal Context Encoder
    layers.append(EncoderLayer(
        self.d_model, self.attn_heads, self.d_ff, self.afr_reduced_cnn_size, 2,
        dropout=self.dropout))
    
    return layers
 

class MRCNN(LayerWithNeurons):
    full_name = 'MRCNN' 
    abbreviation = 'MRCNN'

    def __init__(self,
                activation=None,
                use_bias=False,
                weight_initializer='xavier_normal',
                bias_initializer='zeros',
                prune_frac=0,
                **kwargs):
        # Call parent's constructor
        LayerWithNeurons.__init__(
            self, activation, weight_initializer, use_bias, bias_initializer,
            prune_frac=prune_frac, **kwargs)
        self.drate = 0.5
        self.dropout = Dropout(1 - self.drate)
        self.maxpool11 = MaxPool1D(8, 2)
        self.maxpool12 = MaxPool1D(5, 5)
        self.maxpool21 = MaxPool1D(4, 2)
        self.maxpool22 = MaxPool1D(3, 4)

    @property
    def structure_tail(self):
        return ''

    @single_input
    def _link(self, x: tf.Tensor, **kwargs):
        ## feature 1
        ## x Shape [?, 3000, 1]
        # xo1 = tf.keras.layers.Conv1D(64, kernel_size = 50, strides=5, padding='same')(x)     #[?, 600, 64]
        xo1 = tf.keras.layers.Conv1D(64, kernel_size = 50, strides=4, padding='same')(x)     #[?, 600, 64]
        xo1 = tf.layers.batch_normalization(xo1)
        xo1 = tf.keras.layers.ReLU()(xo1)
        xo1 = self.maxpool11(xo1)             ## [?, 300, 64] 
        xo1 = self.dropout(xo1)

        xo1 = tf.keras.layers.Conv1D(128, kernel_size = 8, strides=1, padding='same')(xo1)     #[?, 600, 64]
        xo1 = tf.layers.batch_normalization(xo1)
        xo1 = tf.keras.layers.ReLU()(xo1)
        xo1 = tf.keras.layers.Conv1D(128, kernel_size = 8, strides=1, padding='same')(xo1)     #[?, 600, 64]
        xo1 = tf.layers.batch_normalization(xo1)
        xo1 = tf.keras.layers.ReLU()(xo1)
        xo1 = self.maxpool12(xo1)         ## shape [?, 60, 128]
        # D = 96

         ## feature 2
        # xo2 = tf.keras.layers.Conv1D(64, kernel_size = 400, strides=25, padding='same')(x)     #[?, 600, 64]
        xo2 = tf.keras.layers.Conv1D(64, kernel_size = 400, strides=30, padding='same')(x)     #[?, 600, 64]
        xo2 = tf.layers.batch_normalization(xo2)
        xo2 = tf.keras.layers.ReLU()(xo2)
        xo2 = self.maxpool21(xo2)     ##[?, 60, 64] 
        xo2 = self.dropout(xo2)

        xo2 = tf.keras.layers.Conv1D(128, kernel_size = 7, strides=1, padding='same')(xo2)     #[?, 600, 64]
        xo2 = tf.layers.batch_normalization(xo2)
        xo2 = tf.keras.layers.ReLU()(xo2)
        xo2 = tf.keras.layers.Conv1D(128, kernel_size = 7, strides=1, padding='same')(xo2)     #[?, 600, 64]
        xo2 = tf.layers.batch_normalization(xo2)
        xo2 = tf.keras.layers.ReLU()(xo2)
        xo2 = self.maxpool22(xo2)         ## shape [?, 20, 128]
        # D =

        ## Shape [?, 80, 128]
        x_concat = tf.concat((xo1, xo2), 1)
        x_concat = self.dropout(x_concat)
        return x_concat
 

class AFR(LayerWithNeurons):
    full_name = 'AFR' 
    abbreviation = 'AFR'

    def __init__(self,
                afr_reduced_cnn_size,
                reduction = 16, 
                stride = 1, 
                activation=None,
                use_bias=False,
                weight_initializer='xavier_normal',
                bias_initializer='zeros',
                prune_frac=0,
                **kwargs):
        # Call parent's constructor
        LayerWithNeurons.__init__(
            self, activation, weight_initializer, use_bias, bias_initializer,
            prune_frac=prune_frac, **kwargs)
        self.avg_pool = GlobalAveragePooling1D()
        self.inplanes = 128
        self.reduction = reduction
        self.stride = stride
        self.planes = afr_reduced_cnn_size
        self.blocks = 1
        self.expansion = 1

    @property
    def structure_tail(self):
        return ''

    @single_input
    def _link(self, x: tf.Tensor, **kwargs):
        ## shape [?, 80, 128]
        features = x
        for _ in range(self.blocks):
            out = self.conv1d(features, self.planes, filter_size =1, 
                              strides = 1, padding = 'SAME', scope='afr_0')
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.ReLU()(out)

            out = self.conv1d(out, self.planes, filter_size =1, 
                              strides = 1, padding = 'SAME', scope='afr_1')
            out = tf.keras.layers.BatchNormalization()(out)
            ## se layer
            ## input shape [?, 80, 30]
            y = self.avg_pool(out)            ## shape [?, 30]
            y = tf.keras.layers.Dense(self.planes // self.reduction, use_bias= False)(y)
            y = tf.keras.layers.ReLU()(y)
            y = tf.keras.layers.Dense(self.planes, use_bias= False)(y)
            y = sigmoid(y)
            y = tf.expand_dims(y, 1)        ## shape [?, 1, 30] 
            multi = out * y
            ## Downsample
            if self.stride != 1 or self.inplanes != self.planes * self.expansion:
                downsample = self.conv1d(features, self.planes * self.expansion,
                                         filter_size=1, strides= self.stride, padding = 'SAME', scope='afr_2')
                downsample = tf.keras.layers.BatchNormalization()(downsample)
            else:
                downsample = features
            multi += downsample
            out = tf.keras.layers.ReLU()(multi)
            self.inplanes = self.planes * self.expansion

        return out


class CausualConv1d(tf.layers.Conv1D): 
  full_name = 'causual_conv1d'
  abbreviation = 'causual_conv1d'

  def __init__(self, filters,
               kernel_size,
               strides=1,
               padding='same',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):

        super(CausualConv1d, self).__init__(
               filters,
               kernel_size,
               strides=strides,
               padding=padding,
               data_format=data_format,
               dilation_rate=dilation_rate,
               activation=activation,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint,
               bias_constraint=bias_constraint,
               trainable=trainable,
               name=name,
               **kwargs
        )
    
  def _link(self, x: tf.Tensor, **kwargs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(x, tf.constant([(0, 0), (1, 0), (0, 0)]) * padding)
        results = super(CausualConv1d, self)._link(inputs)
        if padding != 0:
            return results[:, :-padding, :]
        return results


class MultiHeadAttention(LayerWithNeurons):
    full_name = 'MultiHeadAttention' 
    abbreviation = 'MultiHeadAttention'

    def __init__(self,
                 h,
                 d_model,
                 afr_reduced_cnn_size,
                 dropout = 0.1, 
                 activation=None,
                 use_bias=False,
                 weight_initializer='xavier_normal',
                 bias_initializer='zeros',
                 prune_frac=0,
                 **kwargs):
        # Call parent's constructor
        LayerWithNeurons.__init__(
            self, activation, weight_initializer, use_bias, bias_initializer,
            prune_frac=prune_frac, **kwargs)
        assert d_model % h ==0
        self.d_k = d_model // h
        self.h = h
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = Dropout(1 - dropout)
        self.afr_reduced_cnn_size = afr_reduced_cnn_size

    @property
    def structure_tail(self):
        return 'MultiHeadAttention'


    def attention(self, query, key, value, dropout = None):
        import math

        ## Scaled dot product attention
        d_k = query.shape.as_list()[-1]
        key = tf.transpose(key, perm=[0, 1, 3, 2])      ## shape [?, 5, 16, 30]
        scores = tf.matmul(query, key) / math.sqrt(d_k) 

        p_attn = tf.nn.softmax(scores)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return tf.matmul(p_attn, value), p_attn


    def _link(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs):
        ## shape [?, 80, 30]
        chnnum = query.shape.as_list()[-1] 

        ## shape [?, 5, 16, 30]
        query = tf.reshape(query, [-1, self.h, self.d_k, chnnum])
        ## shape [?, 5, 30, 16]
        query = tf.transpose(query, perm=[0, 1, 3, 2])

        # key = CausualConv1d(self.afr_reduced_cnn_size, 7, 1, name = 'key_causual_cnn')(key)        ## shape [?, 80, 30]
        key = tf.reshape(key, [-1, self.h, self.d_k, chnnum])   ## shape [?, 5, 16, 30]
        key = tf.transpose(key, perm=[0, 1, 3, 2])       ## shape [?, 5, 30, 16]

        # value = CausualConv1d(self.afr_reduced_cnn_size, 7, 1, name = 'value_causual_cnn')(value)        ## shape [?, 80, 30]
        value = tf.reshape(value, [-1, self.h, self.d_k, chnnum])   ## shape [?, 5, 16, 30]
        value = tf.transpose(value, perm=[0, 1, 3, 2])       ## shape [?, 5, 30, 16]

        ## x shape [?, 5, 30, 16]
        x, self.attn = self.attention(query, key, value, dropout = self.dropout)
        x = tf.transpose(x, perm=[0, 1, 3, 2])      ## shape [?, 5, 16, 30]
        x = tf.reshape(x, [-1, self.h * self.d_k, chnnum])      ## shape [?, 80, 30]

        x = tf.transpose(x, perm = [0, 2, 1])        ## [?, 30, 80]
        x = self.dense(x)       ##[?, 30, 80]
        x = tf.transpose(x, perm = [0, 2, 1])       ##[?, 80, 30]

        return x

        
class PositionwiseFeedForward(LayerWithNeurons):
    full_name = 'PositionwiseFeedForward' 
    abbreviation = 'PositionwiseFeedForward'

    def __init__(self,
                 d_model,
                 d_ff,
                 dropout = 0.1, 
                 activation=None,
                 use_bias=False,
                 weight_initializer='xavier_normal',
                 bias_initializer='zeros',
                 prune_frac=0,
                 **kwargs):
        # Call parent's constructor
        LayerWithNeurons.__init__(
            self, activation, weight_initializer, use_bias, bias_initializer,
            prune_frac=prune_frac, **kwargs)
        self.w_1 = tf.keras.layers.Dense(d_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.dropout = Dropout(1 - dropout)
 

    @property
    def structure_tail(self):
        return 'PositionwiseFeedForward'


    @single_input
    def _link(self, x: tf.Tensor, **kwargs):
        x = tf.transpose(x, perm=[0, 2, 1])      ## shape [?, 30, 80]
        x = self.w_2(self.dropout(tf.keras.layers.ReLU()(self.w_1(x))))
        x = tf.transpose(x, perm=[0, 2, 1])      ## shape [?, 80, 30]
        return x


class EncoderLayer(LayerWithNeurons):
    full_name = 'EncoderLayer' 
    abbreviation = 'EncoderLayer'

    def __init__(self,
                 size,
                 attn_heads,
                 d_ff,
                 afr_reduced_cnn_size,
                 N, 
                 dropout = 0.1, 
                 activation=None,
                 use_bias=False,
                 weight_initializer='xavier_normal',
                 bias_initializer='zeros',
                 prune_frac=0,
                 **kwargs):
        # Call parent's constructor
        LayerWithNeurons.__init__(
            self, activation, weight_initializer, use_bias, bias_initializer,
            prune_frac=prune_frac, **kwargs)
        self.attn_heads = attn_heads
        self.norm = LayerNorm(size)
        self.dropout = dropout
        self.size = size
        self.d_ff = d_ff
        self.repeat_num = N
        self.afr_reduced_cnn_size = afr_reduced_cnn_size


    @property
    def structure_tail(self):
        return ''


    @single_input
    def _link(self, x: tf.Tensor, **kwargs):
        for _ in range(self.repeat_num):
            query = CausualConv1d(self.afr_reduced_cnn_size, kernel_size=7, strides=1, dilation_rate=1)(x)
            ## Sublayer 0 
            inputs = query
            dropout = Dropout(1 - self.dropout)
            self_attn = MultiHeadAttention(self.attn_heads, self.size, self.afr_reduced_cnn_size)
            key = CausualConv1d(self.afr_reduced_cnn_size, 7, 1, name = 'key_causual_cnn')(self.norm(inputs))        ## shape [?, 80, 30]
            value = CausualConv1d(self.afr_reduced_cnn_size, 7, 1, name = 'value_causual_cnn')(self.norm(inputs))        ## shape [?, 80, 30]
            # x = SublayerOutput(self.size, self.dropout)(query, 
                                            #   lambda x: self_attn(query, x, x))   ## Encoder self-attention
            x = inputs + dropout(self_attn(query, key, value))

            ## Sublayer 1
            ff = PositionwiseFeedForward(self.size, self.d_ff, self.dropout)
            x = SublayerOutput(self.size, self.dropout)(x, ff)

        self.norm = LayerNorm(x.shape.as_list()[1])
        return self.norm(x)

 
class SublayerOutput(Layer):
    abbreviation = 'SublayerOutput'
    full_name = abbreviation
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = Dropout(1 - dropout)


    def _link(self, inputs, sublayer, **kwargs):
        ## Apply residual connections to any sublayer with the same size
        return inputs + self.dropout(sublayer(self.norm(inputs))) 


class LayerNorm(Layer):
    abbreviation = 'LayerNorm'
    full_name = abbreviation
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = tf.constant(np.ones((features, 1)), dtype = tf.float32)
        self.b_2 = tf.constant(np.zeros((features, 1)), dtype = tf.float32)
        self.eps = eps
    

    @single_input
    def _link(self, inputs, **kwargs):
        mean = tf.reduce_mean(inputs, 1, keep_dims=True)        ## [?, 1, C]
        std = tf.math.reduce_std(inputs, 1, keepdims=True)        ## [?, 1, C]
        return self.a_2 * (inputs - mean) / (std + self.eps) + self.b_2


