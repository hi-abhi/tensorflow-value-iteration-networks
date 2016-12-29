import time
import numpy as np
import tensorflow as tf
from data  import *
from model import *
from utils import *

np.random.seed(0)

# Data
tf.app.flags.DEFINE_string('input',           'data/gridworld_8.mat', 'Path to data')
tf.app.flags.DEFINE_integer('imsize',         8,                      'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              10,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,                     'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False,                  'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            False,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

# symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32,   name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32,   name="S2", shape=[None, config.statebatchsize])
y  = tf.placeholder(tf.int32,   name="y",  shape=[None])

# Construct model (Value Iteration Network)
if (config.untied_weights):
  nn = VI_Untied_Block(X, S1, S2, config)
else:
  nn = VI_Block(X, S1, S2, config)

# Define loss and optimizer
dim = tf.shape(y)[0]
cost_idx = tf.concat(1, [tf.reshape(tf.range(dim), [dim,1]), tf.reshape(y, [dim,1])])
cost = -tf.reduce_mean(tf.gather_nd(tf.log(nn), [cost_idx]))
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input, imsize=config.imsize)

# Launch the graph
with tf.Session() as sess:
  if config.log:
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
  sess.run(init)

  batch_size    = config.batchsize
  print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
  for epoch in range(int(config.epochs)):
    tstart = time.time()
    avg_err, avg_cost = 0.0, 0.0
    num_batches = int(Xtrain.shape[0]/batch_size)
    # Loop over all batches
    for i in range(0, Xtrain.shape[0], batch_size):
      j = i + batch_size
      if j <= Xtrain.shape[0]:
        # Run optimization op (backprop) and cost op (to get loss value)
        fd = {X: Xtrain[i:j], S1: S1train[i:j], S2: S2train[i:j],
              y: ytrain[i * config.statebatchsize:j * config.statebatchsize]}
        _, e_, c_ = sess.run([optimizer, err, cost], feed_dict=fd)
        avg_err += e_
        avg_cost += c_
    # Display logs per epoch step
    if epoch % config.display_step == 0:
      elapsed = time.time() - tstart
      print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
    if config.log:
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
      summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
      summary_writer.add_summary(summary, epoch)
  print("Finished training!")

  # Test model
  correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, y), dtype=tf.float32))
  acc = accuracy.eval({X: Xtest, S1: S1test, S2: S2test, y: ytest})
  print("Accuracy: {}%".format(100 * (1 - acc)))
