import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import load


# load data
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX)
labels_test = np.argmax(t_test, axis=1)


# visualize data
plt.imshow(x_train[0].reshape(32, 32), cmap=plt.cm.gray)


# define symbolic Theano variables
x = T.matrix()
t = T.matrix()


# define model: logistic regression
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def model(x, w):
    return T.nnet.softmax(T.dot(x, w))

w = init_weights((32 * 32, 10))

p_y_given_x = model(x, w)
y = T.argmax(p_y_given_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))
g = T.grad(cost, w)
updates = [(w, w - g * 0.001)]


# compile theano functions
train = theano.function([x, t], cost, updates=updates)
predict = theano.function([x], y)


# train model
batch_size = 50
for i in range(100):
    print "iteration %d" % (i + 1)
    for start in range(0, len(x_train), batch_size):
        x_batch = x_train[start:start + batch_size]
        t_batch = t_train[start:start + batch_size]
        cost = train(x_batch, t_batch)

    predictions_test = predict(x_test)
    accuracy = np.mean(predictions_test == labels_test)
    print "accuracy: %.5f" % accuracy
    print