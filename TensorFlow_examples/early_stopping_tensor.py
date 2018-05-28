import tensorflow as tf
import numpy as np
import os
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
from datetime import datetime
from __future__ import division, print_function, unicode_literals

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
LOGDIR = './graphs'


if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(
        X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = tf.layers.dense(
        hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='output')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)
    trainig_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# defining tensorboard log directory


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("mnist_dnn")

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

X_valid = mnist.validation.images
y_valid = mnist.validation.labels

X_train = mnist.train.images
X_test = mnist.test.images
y_test = mnist.test.labels
m, n = X_train.shape

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = '/home/light/tmp/my_deep_mnist.ckpt'
checkpoint_epoch_path = checkpoint_path + '.epoch'
final_model_path = './my_deep_mnist'

best_loss = np.infty
epochs_without_progress = 0
max_epoch_without_progress = 50

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, 'rb') as f:
            start_epoch = int(f.read())
        print('Training was interrupted . continuing epoch at', start_epoch)
        saver.restore(sess, checkpoint_epoch_path)

    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(trainig_op, feed_dict={X: X_batch, y: y_batch})

        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)

        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation Accuracy :{:.3f}%".format(accuracy_val * 100),
                  "\tLoss:{:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, 'wb') as f:
                f.write(b'%d' % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epoch_without_progress:
                    print('early stopping at epoch', epoch)
                    break

os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

accuracy_val
