import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def print_results(mode, epoch_number, error, batch_xs, batch_ys):
    print mode, " epoch #:", epoch_number, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

y = tf.nn.softmax(tf.matmul(x, W1) + b1)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 1
maximum_validation_errors = 6
validation_error_counter = 0
training_errors = []
validation_errors = []
test_errors = []
last_validation_error = 1
before_validation_error = 1
epoch = 0
validation_error = 0.1
diferrence = 100

while (validation_error <= last_validation_error and diferrence > 0.001):
    epoch += 1
    for jj in xrange(len(train_x) / batch_size):
        batch_training_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_training_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_training_xs, y_: batch_training_ys})

    for kk in xrange(len(valid_x) / batch_size):
        batch_validation_xs = valid_x[kk * batch_size: kk * batch_size + batch_size]
        batch_validation_ys = valid_y[kk * batch_size: kk * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_validation_xs, y_: batch_validation_ys})

    training_error = sess.run(loss, feed_dict={x: batch_training_xs, y_: batch_training_ys})
    training_errors.append(training_error)

    validation_error = sess.run(loss, feed_dict={x: batch_validation_xs, y_: batch_validation_ys})
    validation_errors.append(validation_error)
    if (epoch > 1):
        diferrence = validation_errors[-2] - validation_error
    last_validation_error = validation_errors[-1]

    print_results(mode="Training", epoch_number=epoch, error=training_error,
                  batch_xs=batch_training_xs, batch_ys=batch_training_ys)

    print_results(mode="Validation", epoch_number=epoch, error=validation_error,
                  batch_xs=batch_validation_xs, batch_ys=batch_validation_ys)
# ---------------- Visualizing some element of the MNIST dataset --------------
plt.ylabel('Errors')
plt.xlabel('Epochs')
# test_line, = plt.plot(test_errors)
training_line, = plt.plot(training_errors)
validation_line, = plt.plot(validation_errors)
plt.legend(handles=[training_line, validation_line],
           labels=["Training errors", "Validation errors"])
plt.savefig('mn.png')

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print train_y[57]
#plt.savefig('number.png')


# TODO: the neural net!!
