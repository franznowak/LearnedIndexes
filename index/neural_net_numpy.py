# --------------------------------------------------------------------
# neural_net_numpy.py - neural network class using numpy
# December 2018 - May 2019 Franz Nowak, based on exercise from andrew ng's
# coursera course neural network.
# --------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

LAYER_DIMENSIONS = [1, 32, 32, 1]  # 3-layer model
EPOCHS = 2500


def main():
    train_x, train_y, test_x, test_y = load_data()
    parameters = l_layer_model(train_x, train_y, LAYER_DIMENSIONS,
                               num_iterations=EPOCHS, print_cost=True)
    predictions_y = predict(train_x, parameters)
    plt.plot(train_y, predictions_y)
    
    
def load_data():
    train_x = np.add(np.random.randint(100, size=(1, 100)),1)
    train_x.sort()
    train_y = np.asarray([i for i in range(1, train_x.size+1)])
    train_y = train_y.reshape((1, train_x.size))
    test_x = train_x[:30]
    test_y = train_y[:30]
    return train_x, train_y, test_x, test_y


def l_layer_model(x, y, layers_dims, learning_rate=0.0075, num_iterations=3000,
                  print_cost=False):  # lr was 0.009
    """
    Implements a l-layer neural network: [LINEAR->RELU]*(l-1)->LINEAR->SIGMOID.

    :param x: data, numpy array of shape (number of examples, num_px * 
        num_px * 3)
    :param y:true "label" vector,
        of shape (1, number of examples)
    :param layers_dims: list containing the input size and each layer size, 
        of length (number of layers + 1).
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps

    return: parameters learnt by the model. They can then be used to predict.
    
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(l-1) -> LINEAR -> LINEAR.
        al, caches = l_model_forward(x, parameters)

        cost = compute_cost(al, y)

        grads = l_model_backward(al, y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(x, parameters):
    """
    Using the learned parameters, predicts the index for each key in x
    
    :param x: input data of size (n_x, m)
    :param parameters: python dict containing the parameters
    
    :return: vector of predictions of the model
    
    """

    a2, cache = l_model_forward(x, parameters)
    predictions = a2

    return predictions


def linear(z):
    return 1*z, z


def sigmoid(z):
    return 1/(np.exp(-z)+1.0), z


def relu(z):
    return np.vectorize(lambda x: max(0.0, x))(z), z


def lin_backward(da, _):
    return da * 1


def sigmoid_backward(da, z):
    sigma, _ = sigmoid(z)
    return da * sigma * (1.0 - sigma)


def relu_backward(da, z):
    return da * np.vectorize(lambda x: max(0.0, np.heaviside(x, 1.0)))(z)


def initialise_parameters(n_x, n_h, n_y):
    """
    Initialises the parameters.

    :param n_x: size of the input layer
    :param n_h:  size of the hidden layer
    :param n_y: size of the output layer

    :return: python dictionary containing your parameters:
                    w1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    w2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    
    """

    np.random.seed(1)

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims: python array (list) containing the dimensions of
        each layer in our network

    :return: python dictionary containing your parameters "w1", "b1", ..., 
    "wl", "bl":
                    wl -- weight matrix of shape (layer_dims[l], layer_dims[
                    l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    
    """

    np.random.seed(3)
    parameters = {}
    n_layers = len(layer_dims)  # number of layers in the network

    for l in range(1, n_layers):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['w' + str(l)].shape == 
                (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(a, w, b):
    """
    Implement the linear part of a layer's forward propagation.

    :param a: activations from previous layer (or input data): (size of 
        previous layer, number of examples)
    :param w: weights matrix: numpy array of shape (size of current layer, 
        size of previous layer)
    :param b: -- bias vector, numpy array of shape (size of the current 
        layer, 1)

    :returns z, cache: 
        z -- the input of the activation function, 
            also called pre-activation parameter
        cache -- a python dictionary containing "a", "w" and "b"; 
            stored for computing the backward pass efficiently
    
    """

    z = np.dot(w, a) + b

    assert (z.shape == (w.shape[0], a.shape[1]))
    cache = (a, w, b)

    return z, cache


def linear_activation_forward(a_prev, w, b, activation):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer.

    :param a_prev: activations from previous layer (or input data): (size of 
        previous layer, number of examples)
    :param w: weights matrix: numpy array of shape (size of current layer, 
        size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)
    :param activation: the activation to be used in this layer, stored as a 
        text string: "sigmoid" or "relu"

    :returns a, cache: 
    a -- the output of the activation function, also called the 
        post-activation value
    cache -- a python dictionary containing "linear_cache" and 
        "activation_cache"; stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "a_prev, w, b". Outputs: "a, activation_cache".
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    elif activation == "relu":
        # Inputs: "a_prev, w, b". Outputs: "a, activation_cache".
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    elif activation == "linear":
        # Inputs: "a_prev, w, b". Outputs: "a, activation_cache".
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = linear(z)

    else:
        raise Exception("No activation specified")

    assert (a.shape == (w.shape[0], a_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return a, cache


def l_model_forward(x, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(l-1)->LINEAR->SIGMOID
        computation.

    :param x: data, numpy array of shape (input size, number of examples)
    :param parameters: output of initialize_parameters_deep()

    :returns al, caches:
    al -- last post-activation value
    caches -- list of caches containing:
        every cache of linear_activation_forward()
        (there are l-1 of them, indexed from 0 to l-1)

    """

    caches = []
    a = x
    n_layers = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(l-1). Add "cache" to the "caches" list.
    for l in range(1, n_layers):
        a_prev = a
        a, cache = linear_activation_forward(a_prev, parameters['w' + str(l)],
                                             parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)

    # Implement LINEAR -> LINEAR. Add "cache" to the "caches" list.
    al, cache = linear_activation_forward(a, parameters['w' + str(n_layers)],
                                          parameters['b' + str(n_layers)],
                                          activation="linear")
    caches.append(cache)

    assert (al.shape == (1, x.shape[1]))

    return al, caches


def compute_cost(al, y):
    """
    Implement the cost function defined by equation (7).

    :param al: probability vector corresponding to your label predictions,
        shape (1, number of examples)
    :param y: true "label" vector, shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost

    """

    m = y.shape[1]
    prediction = al * m

    # Compute loss from aL and y.

    a = y*np.log(prediction)
    b = (1-y) * np.log(1-prediction)
    c = np.nansum(a+b, axis=1)
    d = -(1/m)
    cost = d*c
    # cost = -(1/m) * \
    #     np.sum(y*np.log(prediction) + (1-y) * np.log(1-prediction), axis=1)
    
    # cost = -(1 / m) * np.sum(np.square(y-al), axis=1)

    # To make sure the cost's shape is what we expect
    # (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return float(cost)


def linear_backward(dz, cache):
    """
    Implement the linear portion of backward propagation for a single layer
        (layer l)

    :param dz: Gradient of the cost with respect to the linear output
        (of current layer l)
    :param cache: tuple of values (a_prev, w, b) coming from the forward
        propagation in the current layer

    :returns:
    da_prev -- Gradient of the cost with respect to the activation
        (of the previous layer l-1), same shape as a_prev
    dw -- Gradient of the cost with respect to w (current layer l),
        same shape as w
    db -- Gradient of the cost with respect to b (current layer l),
        same shape as b

    """
    a_prev, w, b = cache
    m = a_prev.shape[1]

    dw = (1 / m) * np.dot(dz, a_prev.T)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(w.T, dz)

    assert (da_prev.shape == a_prev.shape)
    assert (dw.shape == w.shape)
    assert (db.shape == b.shape)

    return da_prev, dw, db


def linear_activation_backward(da, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    :param da: post-activation gradient for current layer l
    :param cache: tuple of values (linear_cache, activation_cache) we store
        for computing backward propagation efficiently
    :param activation: the activation to be used in this layer, stored as a
        text string: "sigmoid" or "relu"

    :returns:
    da_prev -- Gradient of the cost with respect to the activation
        (of the previous layer l-1), same shape as a_prev
    dw -- Gradient of the cost with respect to w (current layer l),
        same shape as w
    db -- Gradient of the cost with respect to b (current layer l),
        same shape as b

    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dz = relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    elif activation == "sigmoid":
        dz = sigmoid_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    elif activation == "linear":
        dz = lin_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    else:
        raise Exception("No activation specified")

    return da_prev, dw, db


def l_model_backward(al, y, caches):
    """
    Implement the backward propagation for the 
        [LINEAR->RELU] * (l-1) -> LINEAR -> SIGMOID group.

    :param al: probability vector, output of the forward propagation
        (L_model_forward())
    :param y: true "label" vector
    :param caches: list of caches containing:
        every cache of linear_activation_forward() with "relu" (it's
        caches[l], for l in range(n_layersl-1) i.e l = 0...n_layers-2)
        the cache of linear_activation_forward() with "sigmoid"
        (it's caches[n_layers-1])

    :return: a dictionary with the gradients
             grads["da" + str(l)] = ...
             grads["dw" + str(l)] = ...
             grads["db" + str(l)] = ...

    """
    grads = {}
    n_layers = len(caches)  # the number of layers
    m = al.shape[1]
    y = y.reshape(al.shape)  # after this line, y is the same shape as al
    prediction = al * m

    # Initializing the backpropagation
    dal = - (np.divide(y, prediction) - np.divide(1 - y, 1 - prediction))
    # dal = -2*(y-al)*al

    # Lth layer (LINEAR -> LINEAR) gradients. Inputs: "dal, current_cache".
    # Outputs: "grads["dal-1"], grads["dwL"], grads["dbL"]
    current_cache = caches[n_layers - 1]
    grads["da" + str(n_layers - 1)], grads["dw" + str(n_layers)], grads[
        "db" + str(n_layers)] = linear_activation_backward(dal, current_cache,
                                                           activation="sigmoid")

    # Loop from l=l-2 to l=0
    for l in reversed(range(n_layers - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["da" + str(l + 1)], current_cache".
        # Outputs: "grads["da" + str(l)], grads["dw" + str(l + 1)],
        #   grads["db" + str(l + 1)]
        current_cache = caches[l]
        da_prev_temp, dw_temp, db_temp = linear_activation_backward(
            grads["da" + str(l + 1)], current_cache, activation="relu")
        grads["da" + str(l)] = da_prev_temp
        grads["dw" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.

    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients, output of
        l_model_backward
    :param learning_rate:
        learning rate of the gradient descent

    :returns: python dictionary containing your updated parameters
                  parameters["w" + str(l)] = ...
                  parameters["b" + str(l)] = ...

    """

    n_layers = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for n_layers in range(n_layers):
        parameters["w" + str(n_layers+1)] = parameters[
            "w"+str(n_layers+1)] - learning_rate * grads["dw"+str(n_layers+1)]
        parameters["b" + str(n_layers+1)] = parameters[
            "b"+str(n_layers+1)] - learning_rate * grads["db"+str(n_layers+1)]
    return parameters


if __name__ == "__main__":
    main()
