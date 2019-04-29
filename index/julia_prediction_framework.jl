import Pkg;
Pkg.add("HDF5")
using HDF5
using CSV
using DelimitedFiles

function load_model(filename)
    model_data = readdlm(filename)
    n_layers = trunc(Int,model_data[1])
    m = model_data[2]
    s = model_data[3]
    return n_layers, m, s
end

function load_weights(filename, layer_offset, n_layers)
    layers =  Dict(i => h5read(filename,"dense_$(i+layer_offset)")["dense_$(i+layer_offset)"] for i = 1:n_layers)
    return layers
end

function load_data(filename)
    data = readdlm(filename,',', Int64)
end

function relu(x)
    return x .* (x .> 0)
end

function predict(key, layers, n_layers, m, s)
    out = (key - m)/s
    for i = 1:n_layers-1
        out = relu(layers[i]["kernel:0"]*out + layers[i]["bias:0"])
    end
    return trunc(Int,(layers[n_layers]["kernel:0"]*out + layers[n_layers]["bias:0"])[1])
end

function exponential_search(data, key, prediction)
    data_size = size(data)[1]
    index = max(1, min(data_size, prediction))
    value = data[index]
    jump = 1
    if value < key
        while value < key && jump < data_size
            jump *= 2
            value = data[min(index + jump, data_size)]
        end
        return binary_search(data, key, index+trunc(Int,jump/2), index+jump)
    elseif value > key
        while value > key && jump < data_size
            jump *= 2
            value = data[max(index - jump, 1)]
        end
        return binary_search(data, key, index-jump, index-trunc(Int,jump/2))
    else
        return index
    end
end

function binary_search(data, key, left, right)
    left = max(1, left)
    right = min(right, size(data)[1])
    while true
        if right < left
            throw(KeyError("key not found"))
        end
        index = trunc(Int, (right + left) / 2)
        value = data[index]
        if value < key
            left = index + 1
        elseif value > key
            right = index - 1
        else
            return index
        end
    end
end

struct Model
    layers
    n_layers::Int
    m::Float64
    s::Float64
end

function load_stats(filename)
    model_data = CSV.read(filename, header=false, delim=',')
    m = parse(Float64,model_data[2,3])
    s = parse(Float64,model_data[2,4])
    return m, s
end

function load_index(weights_path, layer_offset)
    index = []
    for i=1:n_stages
        push!(index,[])
        for j=1:stages[i]
            layers = load_weights("$(weights_path)weights$(i-1)_$(j-1).h5", layer_offset, nn_layers[i])
            m, s  = load_stats("$(weights_path)weights$(i-1)_$(j-1).h5_stats.csv")
            model = Model(layers, nn_layers[i], m, s)
            push!(index[i],model)
            layer_offset+=nn_layers[i]
        end
    end
    return index
end

function get_next_model_index(current_stage, current_prediction)
    p = trunc(Int, current_prediction * stages[current_stage + 1] / data_size + 1)
    return min(stages[current_stage + 1], max(1, p))
end

function recursive_predict(key)
    j = 1
    for i=1:n_stages-1
        model = index[i][j]
        j = get_next_model_index(i, predict(key, model.layers, model.n_layers, model.m, model.s))
        println("selecting model $(j)")
    end
    model = index[n_stages][j]
    return predict(key, model.layers, model.n_layers, model.m, model.s)
end

function main()
    n_stages = 2
    stages = [1, 10]
    nn_layers = [3, 3]
    data_file = "/Users/franz/PycharmProjects/LearnedIndexes/data/datasets/Integers_100x10x100k/run0inter9"
    all_data = load_data(data_file)
    data_size = size(all_data)[1]
    weights_path = "/Users/franz/PycharmProjects/LearnedIndexes/data/indexes/recursive_learned_index/Integers_100x10x100k/run0inter9/"
    index = load_index(weights_path,264)
end

main()
