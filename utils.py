import numpy as np
import tensorflow as tf

# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
  if isinstance(x, np.ndarray):
    assert x.ndim==0
    x = x.item()
  if isinstance(x, float): rep = "%g"%x
  else: rep = str(x)
  return " "*(l - len(rep)) + rep

def fmt_row(width, row):
  out = " | ".join(fmt_item(x, width) for x in row)
  return out

def flipkernel(kern):
  return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]

def theano_to_tf(tensor):
  # NCHW -> NHWC
  return tf.transpose(tensor, [0, 2, 3, 1])

def tf_to_theano(tensor):
  # NHWC -> NCHW
  return tf.transpose(tensor, [0, 3, 1, 2])
