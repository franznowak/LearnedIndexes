def relu(z):
    return np.vectorize(lambda x: max(0.0, x))(z)

layers = [model.model.get_layer(index=i).get_weights() for i in range(len(model.complexity)+1)]
for layer in layers:
	layer[0]=layer[0].transpose()
	layer[1]=layer[1].transpose()

def predict_fast(key):
    out = model._norm(key)
    n_layers = len(model.complexity)+1
    for i in range(n_layers-1):
        out = relu(np.dot(layers[i][0],out) + layers[i][1])
    return int(np.dot(layers[n_layers-1][0],out) + layers[n_layers-1][1])