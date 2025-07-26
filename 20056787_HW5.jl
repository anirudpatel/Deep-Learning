using Flux, MLDatasets, Statistics, Random, Plots, Images

import Pkg
Pkg.add("Flux")
Pkg.add("MLDatasets")
Pkg.add("Plots")
Pkg.add("Images")

function get_cifar10_data()
    train_x, train_y = CIFAR10.traindata()
    test_x, test_y = CIFAR10.testdata()

    # Normalize and reshape
    train_x = Float32.(permutedims(train_x, (4, 3, 2, 1))) ./ 255
    test_x = Float32.(permutedims(test_x, (4, 3, 2, 1))) ./ 255

    train_y = Flux.onehotbatch(train_y .+ 1, 1:10)
    test_y = Flux.onehotbatch(test_y .+ 1, 1:10)

    return (train_x, train_y), (test_x, test_y)
end

function LeNet5()
    return Chain(
        Conv((5, 5), 3 => 6, relu), MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)),
        flatten,
        Dense(400, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax
    )
end

function train_model(model, train_x, train_y, test_x, test_y, epochs, batch_size)
    opt = ADAM()
    loss(x, y) = Flux.crossentropy(model(x), y)

    data = Flux.DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)

    for epoch in 1:epochs
        for (x, y) in data
            gs = gradient(Flux.params(model)) do
                loss(x, y)
            end
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        println("Epoch $epoch | Test Acc: ", accuracy(model, test_x, test_y))
    end
    return model
end

function accuracy(model, x, y)
    preds = model(x)
    mean(Flux.onecold(preds) .== Flux.onecold(y))
end

function experiment_varying_examples()
    (train_x, train_y), (test_x, test_y) = get_cifar10_data()
    sizes = [10000, 20000, 30000]
    epochs = [6, 3, 2]
    test_accs = Float64[]

    for (sz, ep) in zip(sizes, epochs)
        model = LeNet5()
        idx = randperm(size(train_x, 4))[1:sz]
        x_subset = train_x[:, :, :, idx]
        y_subset = train_y[:, idx]
        model = train_model(model, x_subset, y_subset, test_x, test_y, ep, 128)
        push!(test_accs, accuracy(model, test_x, test_y))
    end

    plot(sizes, test_accs, xlabel="Training Examples", ylabel="Test Accuracy", title="Effect of More Unique Training Examples", marker=:circle)
end

function LeNet(filter_size)
    return Chain(
        Conv((filter_size, filter_size), 3 => 6, relu), MaxPool((2, 2)),
        Conv((filter_size, filter_size), 6 => 16, relu), MaxPool((2, 2)),
        flatten,
        Dense(400, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax
    )
end

function experiment_filter_sizes()
    (train_x, train_y), (test_x, test_y) = get_cifar10_data()
    filters = [3, 5, 7]
    accs = Float64[]

    for f in filters
        model = LeNet(f)
        println("Training LeNet$f...")
        train_model(model, train_x, train_y, test_x, test_y, 3, 128)
        push!(accs, accuracy(model, test_x, test_y))
    end

    bar(string.("LeNet", filters), accs, xlabel="Architecture", ylabel="Test Accuracy", title="Effect of Filter Size")
end

function visualize_features(model, img)
    img = reshape(Float32.(img) ./ 255, (32, 32, 3, 1))
    conv1 = model[1](img)
    pool1 = model[2](conv1)
    conv2 = model[3](pool1)
    pool2 = model[4](conv2)

    heatmap(Gray.(dropdims(mean(conv1, dims=3)[:, :, 1, :], dims=4)), title="Conv1 Output")
    heatmap(Gray.(dropdims(mean(conv2, dims=3)[:, :, 1, :], dims=4)), title="Conv2 Output")
end

function sample_and_visualize()
    (train_x, train_y), _ = get_cifar10_data()
    model = LeNet(3)
    train_model(model, train_x, train_y, train_x, train_y, 1, 128)

    for i in 1:3
        img = train_x[:, :, :, i]
        display(heatmap(Gray.(img[:, :, 1])))
        visualize_features(model, img)
    end
end
