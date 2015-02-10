import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import load


# load data
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX)
labels_test = np.argmax(t_test, axis=1)


# define symbolic Theano variables
x = T.matrix()
t = T.matrix()


# define model: neural network
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def sgd(cost, params, learning_rate):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * learning_rate])
    return updates

def model(x, w_h, w_o):
    h = T.maximum(0, T.dot(x, w_h))
    p_y_given_x = T.nnet.softmax(T.dot(h, w_o))
    return p_y_given_x

w_h = init_weights((32 * 32, 100))
w_o = init_weights((100, 10))

p_y_given_x = model(x, w_h, w_o)
y = T.argmax(p_y_given_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))
params = [w_h, w_o]
updates = sgd(cost, params, learning_rate=0.01)


# compile theano functions
train = theano.function([x, t], cost, updates=updates)
predict = theano.function([x], y)


# train model
batch_size = 50
for i in range(50):
    print "iteration %d" % (i + 1)
    for start in range(0, len(x_train), batch_size):
        x_batch = x_train[start:start + batch_size]
        t_batch = t_train[start:start + batch_size]
        cost = train(x_batch, t_batch)

    predictions_test = predict(x_test)
    accuracy = np.mean(predictions_test == labels_test)
    print "accuracy: %.5f" % accuracy
    print

