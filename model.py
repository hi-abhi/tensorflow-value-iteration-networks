import numpy as np
import tensorflow as tf
from utils import *

def conv2d_flipkernel(x, k, name=None):
  return tf.nn.conv2d(x, flipkernel(k), name=name,
                      strides=(1,1,1,1), padding='SAME')

def VI_Block(X, S1, S2, weights, biases, state_batch_size=1, k=0):
  # initial conv layer over image+reward prior
  h = conv2d_flipkernel(X, weights['w0'], name="h0") + biases['bias']

  r = conv2d_flipkernel(h, weights['w1'], name="r")
  q = conv2d_flipkernel(r, weights['w'], name="q")
  v = tf.reduce_max(q, reduction_indices=3, keep_dims=True, name="v")

  for i in range(0, k-1):
    rv = tf.concat(3, [r, v])
    wwfb = tf.concat(2, [weights['w'], weights['w_fb']])

    q = conv2d_flipkernel(rv, wwfb, name="q")
    v = tf.reduce_max(q, reduction_indices=3, keep_dims=True, name="v")

  # do one last convolution
  q = conv2d_flipkernel(tf.concat(3, [r, v]),
                        tf.concat(2, [weights['w'], weights['w_fb']]), name="q")

  # CHANGE TO THEANO ORDERING
  # Since we are selecting over channels, it becomes easier to work with
  # the tensor when it is in NCHW format vs NHWC
  q = tf.transpose(q, perm=[0, 3, 1, 2])

  # Select the conv-net channels at the state position (S1,S2).
  # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
  # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
  # TODO: performance can be impraved here by substituting expensive
  #       transpose calls with better indexing for gather_nd
  bs = tf.shape(q)[0]
  rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
  ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
  ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
  idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
  q_out = tf.gather_nd(tf.transpose(q, [2,3,0,1]), idx_in, name="q_out")

  # softmax output weights
  output = tf.nn.softmax(tf.matmul(q_out, weights['w_o']), name="output")
  return output
