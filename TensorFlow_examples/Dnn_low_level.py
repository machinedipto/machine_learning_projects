import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
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
y = tf.placeholder(tf.int32, shape=(None), name='y')


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    tf.summary.scalar('loss', loss)

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 40
batch_size = 50


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"))
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "test"))
    summary_op = tf.summary.merge_all()
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        # acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        summary_result, acc_train = sess.run(
            [summary_op, accuracy], feed_dict={X: X_batch, y: y_batch})
        train_writer.add_summary(summary_result, epoch)
        train_writer.add_run_metadata(
            tf.RunMetadata(), 'epoch:{}'.format(epoch))
        # acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
        #                                     y: mnist.validation.labels})
        summary_result, acc_val = sess.run([summary_op, accuracy], feed_dict={
                                           X: mnist.validation.images, y: mnist.validation.labels})
        test_writer.add_summary(summary_result, epoch)
        test_writer.add_run_metadata(
            tf.RunMetadata(), 'epoch:{}'.format(epoch))
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

    save_path = saver.save(sess, './my_model_final.ckpt')

train_writer.close()
test_writer.close()

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = mnist.test.images[:20]
    z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(z, axis=1)

print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.test.labels[:20])
