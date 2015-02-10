import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import load

import lasagne as nn

# load data
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX)
labels_test = np.argmax(t_test, axis=1)


# reshape data
x_train = x_train.reshape((x_train.shape[0], 1, 32, 32))
x_test = x_test.reshape((x_test.shape[0], 1, 32, 32))


# define model: neural network
l_in = nn.layers.InputLayer((None, 1, 32, 32))

l_conv1 = nn.layers.Conv2DLayer(l_in, num_filters=4, filter_size=(3, 3))
l_pool1 = nn.layers.MaxPool2DLayer(l_conv1, ds=(3, 3))

l_conv2 = nn.layers.Conv2DLayer(l_pool1, num_filters=8, filter_size=(3, 3))
l_pool2 = nn.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

l3 = nn.layers.DenseLayer(nn.layers.dropout(l_pool2, p=0.5), num_units=100)

l_out = nn.layers.DenseLayer(l3, num_units=10, nonlinearity=T.nnet.softmax)

objective = nn.objectives.Objective(l_out, loss_function=nn.objectives.multinomial_nll)

cost_train = objective.get_loss()
p_y_given_x = l_out.get_output(deterministic=True)
y = T.argmax(p_y_given_x, axis=1)


params = nn.layers.get_all_params(l_out)
updates = nn.updates.momentum(cost_train, params, learning_rate=0.01, momentum=0.9)


# compile theano functions
train = theano.function([l_in.input_var, objective.target_var], cost_train, updates=updates)
predict = theano.function([l_in.input_var], y)


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

