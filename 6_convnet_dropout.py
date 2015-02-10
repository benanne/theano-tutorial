import theano
import theano.tensor as T
import numpy as np

import load

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()


# load data
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX)
labels_test = np.argmax(t_test, axis=1)


# reshape data
x_train = x_train.reshape((x_train.shape[0], 1, 32, 32))
x_test = x_test.reshape((x_test.shape[0], 1, 32, 32))


# define symbolic Theano variables
x = T.tensor4()
t = T.matrix()


# define model: neural network
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def momentum(cost, params, learning_rate, momentum):
    grads = theano.grad(cost, params)
    updates = []
    
    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))

    return updates

def dropout(x, p=0.):
    if p > 0:
        retain_prob = 1 - p
        x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX)
        x /= retain_prob
    return x

def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o, p=0.0):
    c1 = T.maximum(0, conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x'))
    p1 = max_pool_2d(c1, (3, 3))

    c2 = T.maximum(0, conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    p2 = max_pool_2d(c2, (2, 2))

    p2_flat = p2.flatten(2)
    p2_flat = dropout(p2_flat, p=p)
    h3 = T.maximum(0, T.dot(p2_flat, w_h3) + b_h3)
    p_y_given_x = T.nnet.softmax(T.dot(h3, w_o) + b_o)
    return p_y_given_x

w_c1 = init_weights((4, 1, 3, 3))
b_c1 = init_weights((4,))
w_c2 = init_weights((8, 4, 3, 3))
b_c2 = init_weights((8,))
w_h3 = init_weights((8 * 4 * 4, 100))
b_h3 = init_weights((100,))
w_o = init_weights((100, 10))
b_o = init_weights((10,))

params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

p_y_given_x_train = model(x, *params, p=0.5)
p_y_given_x_test = model(x, *params, p=0.0)
y_train = T.argmax(p_y_given_x_train, axis=1)
y_test = T.argmax(p_y_given_x_test, axis=1)

cost_train = T.mean(T.nnet.categorical_crossentropy(p_y_given_x_train, t))

updates = momentum(cost_train, params, learning_rate=0.01, momentum=0.9)


# compile theano functions
train = theano.function([x, t], cost_train, updates=updates)
predict = theano.function([x], y_test)


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

