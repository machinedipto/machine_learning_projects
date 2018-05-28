import tensorflow as tf
import numpy as np
import os
import math
from datetime import datetime
from functools import partial

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=None, name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

batch_norm_momentum = 0.9

with tf.name_scope('dnn'):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    my_batch_norm_layer = partial(
        tf.layers.batch_normalization,
        training=training,
        momentum=batch_norm_momentum
    )

    my_dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init
    )

    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name='ooutputs')
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    loss_summary = tf.summary.scalar('log_loss', loss)


learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    trainig_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def log_dir(prefix=''):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = './tf_logs'
    if prefix:
        prefix += '-'
    name = prefix + 'run-' + now
    return '{}/{}/'.format(root_logdir, name)


logdir = log_dir('mnist_selu')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

X_valid = mnist.validation.images
y_valid = mnist.validation.labels

X_train = mnist.train.images
X_test = mnist.test.images
y_test = mnist.test.labels
m, n = X_train.shape

checkpoint_path = '/home/light/tmp/my_deep_batch_norm_mnist.ckpt'
checkpoint_epoch_path = checkpoint_path + '.epoch'
final_model_path = './my_deep_batch_norm_mnist'

best_loss = np.infty
epochs_without_progress = 0
max_epoch_without_progress = 50

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

extra_update_pos = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, 'rb') as f:
            start_epoch = int(f.read())
        print('Training was interrupted at :', start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([trainig_op, extra_update_pos],
                     feed_dict={X: X_batch, y: y_batch})

        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)

        if epoch % 5 == 0:
            print('epoch', epoch,
                  '\tValidation Accuracy {:.3f}'.format(accuracy_val * 100),
                  '\t Loss {:.5f}'.format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, 'wb') as f:
                f.write(b'%d' % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epoch_without_progress:
                    print('early stopping at ', epoch)
                    break

os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

accuracy_val
