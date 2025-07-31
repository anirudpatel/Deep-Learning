# using Pkg
# Pkg.activate(".")
# Pkg.add(["CSV", "DataFrames", "Flux", "Statistics", "MLDataPattern", "Plots", "MLUtils", "Dates"])

using CSV
using DataFrames
using Statistics
using Flux
using MLDataPattern
using Plots
using Dates
using Random
using MLUtils: shuffleobs
using Flux: LSTMCell, Recurrence, Chain, Dense, Conv, vec, Recur  # <-- Added Recur here

# Set seed for reproducibility
Random.seed!(123)

# --- Load and preprocess data ---
df = CSV.read("data/s&p500_stock_data.csv", DataFrame)
df[!, :Date] = Date.(df[!, :Date], dateformat"dd-mm-yyyy")
df = df[:, [:Date, :Open, :High, :Low, :Close, Symbol("Adj Close"), :Volume]]
dropmissing!(df)

# Convert numeric columns to Float64 and normalize
for col in names(df)[2:end]
    df[!, col] = Float64.(df[!, col])
    col_min = minimum(df[!, col])
    col_max = maximum(df[!, col])
    df[!, col] = (df[!, col] .- col_min) ./ (col_max - col_min)
end

# --- Prepare data sequences ---
features = Matrix(df[:, Not(:Date)])
window_size = 10
X_seq = [features[i:i+window_size-1, :]' for i in 1:(size(features, 1) - window_size)]
X = cat(X_seq..., dims=3)
y = features[window_size+1:end, 5]  # predicting normalized Close price

# --- Split data ---
n_samples = size(X, 3)
split1 = Int(floor(0.7 * n_samples))
split2 = Int(floor(0.85 * n_samples))

X_train = Float32.(X[:, :, 1:split1]); y_train = Float32.(y[1:split1])
X_val   = Float32.(X[:, :, split1+1:split2]); y_val = Float32.(y[split1+1:split2])
X_test  = Float32.(X[:, :, split2+1:end]); y_test  = Float32.(y[split2+1:end])

input_dim = size(X, 1)

# --- Define Models ---

# LSTM Model
lstm_layer = Recurrence(LSTMCell(input_dim, 10))
lstm_model = Chain(
    lstm_layer,
    x -> x[:, end, :],  # take last time step output
    Dense(10, 1),
    vec
)

# CNN Model
cnn_model = Chain(
    x -> reshape(x, (input_dim, window_size, 1, size(x, 3))),
    Conv((3, 1), 1 => 4, relu),
    x -> reshape(x, :, size(x, 4)),
    Dense((input_dim - 2) * 4, 1),
    vec
)

# Transfer Learning Model
base = Chain(Dense(input_dim, 6, relu))
transfer_model = Chain(
    x -> base.(eachslice(x, dims=3)),
    x -> hcat(x...),
    x -> reshape(x, (6, window_size, size(x, 2) ÷ window_size)),
    Recurrence(LSTMCell(6, 8)),
    x -> x[:, end, :],
    Dense(8, 1),
    vec
)

# --- Loss Functions ---
loss_lstm(x, y) = Flux.Losses.mse(lstm_model(x), y)
loss_cnn(x, y) = Flux.Losses.mse(cnn_model(x), y)
loss_transfer(x, y) = Flux.Losses.mse(transfer_model(x), y)

# --- Training Utility ---
function get_batches(X, y, batch_size)
    n = size(X, 3)
    idx = shuffleobs(1:n)
    [idx[i:min(i + batch_size - 1, n)] for i in 1:batch_size:n]
end

function train!(model, loss_func, X_train, y_train, X_val, y_val;
                epochs=10, batch_size=16, lr=0.001)
    ps = Flux.params(model)
    opt = Flux.setup(Flux.Adam(lr), ps)
    train_losses, val_losses = Float32[], Float32[]

    for epoch in 1:epochs
        # Reset hidden states of recurrent layers
        for layer in model
            if layer isa Recur
                Flux.reset!(layer)
            end
        end

        epoch_loss = Float32[]
        for batch in get_batches(X_train, y_train, batch_size)
            xb, yb = X_train[:, :, batch], y_train[batch]
            gs = gradient(ps) do
                loss_func(xb, yb)
            end
            Flux.update!(opt, ps, gs)
            push!(epoch_loss, loss_func(xb, yb))
        end
        push!(train_losses, mean(epoch_loss))
        push!(val_losses, loss_func(X_val, y_val))
        println("Epoch $epoch | Train Loss: $(round(train_losses[end], digits=5)) | Val Loss: $(round(val_losses[end], digits=5))")
    end
    return train_losses, val_losses
end

# --- Evaluation ---
function evaluate_model(model, X, y_true)
    y_pred = model(X)
    mse = mean((y_pred .- y_true).^2)
    rmse = sqrt(mse)
    mae = mean(abs.(y_pred .- y_true))
    r2 = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)
    return mse, rmse, mae, r2, y_pred
end

# --- Training ---
println("\nTraining LSTM...")
train_lstm, val_lstm = train!(lstm_model, loss_lstm, X_train, y_train, X_val, y_val)

println("\nTraining CNN...")
train_cnn, val_cnn = train!(cnn_model, loss_cnn, X_train, y_train, X_val, y_val)

println("\nTraining Transfer Model...")
train_transfer, val_transfer = train!(transfer_model, loss_transfer, X_train, y_train, X_val, y_val)

# --- Evaluate ---
metrics_lstm = evaluate_model(lstm_model, X_test, y_test)
metrics_cnn = evaluate_model(cnn_model, X_test, y_test)
metrics_transfer = evaluate_model(transfer_model, X_test, y_test)

println("\n--- Test Metrics ---")
println("LSTM     → MSE=$(metrics_lstm[1]) RMSE=$(metrics_lstm[2]) MAE=$(metrics_lstm[3]) R²=$(metrics_lstm[4])")
println("CNN      → MSE=$(metrics_cnn[1]) RMSE=$(metrics_cnn[2]) MAE=$(metrics_cnn[3]) R²=$(metrics_cnn[4])")
println("Transfer → MSE=$(metrics_transfer[1]) RMSE=$(metrics_transfer[2]) MAE=$(metrics_transfer[3]) R²=$(metrics_transfer[4])")

# --- Plot Predictions ---
plot(y_test, label="Actual", lw=2)
plot!(metrics_lstm[5], label="LSTM", lw=2)
plot!(metrics_cnn[5], label="CNN", lw=2)
plot!(metrics_transfer[5], label="Transfer", lw=2)
title!("Actual vs Predicted Adj Close")
xlabel!("Index")
ylabel!("Normalized Price")
savefig("predictions_plot.png")

# --- Export to CSV ---
results = DataFrame(
    Actual = y_test,
    LSTM = metrics_lstm[5],
    CNN = metrics_cnn[5],
    Transfer = metrics_transfer[5]
)
CSV.write("results_predictions.csv", results)

# --- Plot Loss Curves ---
plot(1:10, train_lstm, label="LSTM Train")
plot!(1:10, val_lstm, label="LSTM Val")
plot!(1:10, train_cnn, label="CNN Train")
plot!(1:10, val_cnn, label="CNN Val")
plot!(1:10, train_transfer, label="Transfer Train")
plot!(1:10, val_transfer, label="Transfer Val")
title!("Loss Curve")
xlabel!("Epoch")
ylabel!("Loss")
savefig("loss_plot.png")
