using Pkg
Pkg.activate(".")
Pkg.add(["CSV", "DataFrames", "Flux", "Statistics", "MLDataPattern", "Plots"])

using CSV
using DataFrames
using Flux
using Statistics
using MLDataPattern  # For windowing sequences
using Plots


df = CSV.read("data/s&p500_stock_data.csv", DataFrame)

# Preview
first(df, 5)


# --- Select all columns except 'Date' ---

# --- Select relevant columns ---
df = df[:, [:Date, :Open, :High, :Low, :Close, Symbol("Adj Close"), :Volume]]

# --- Drop missing values ---
dropmissing!(df)

# --- Convert all numerical columns to Float64 ---
for col in names(df)[2:end]
    df[!, col] = Float64.(df[!, col])
end

# --- Normalize all features except Date ---
for col in names(df)[2:end]
    col_min = minimum(df[!, col])
    col_max = maximum(df[!, col])
    df[!, col] = (df[!, col] .- col_min) ./ (col_max - col_min)
end

# --- Create windowed sequences for multivariate LSTM ---
features = Matrix(df[:, Not("Date")])  # shape: (rows, features)
window_size = 30  # 30 days

# Create input sequences (X) and target (y = next-day Adj Close)
X_seq = [features[i:i+window_size-1, :]' for i in 1:(size(features, 1) - window_size)]
X = hcat(X_seq...)  # shape: (features, time_steps, samples)

# y is next day's Adjusted Close price
y = features[window_size+1:end, 5]  # 5th col = "Adj Close"

# Confirm shapes
println("X shape: ", size(X))  # (features, time_steps, samples)
println("y shape: ", size(y))  # (samples,)

import Flux: params
using Flux
using Statistics
# --- Split into train and test ---
n_samples = size(X, 3)
train_ratio = 0.8
split_idx = Int(floor(train_ratio * n_samples))

X_train = X[:, :, 1:split_idx]
y_train = y[1:split_idx]

X_test = X[:, :, split_idx+1:end]
y_test = y[split_idx+1:end]

# --- Define model ---
input_dim = size(X, 1)  # 6 features
hidden_units = 8
output_dim = 1

model = Chain(
    LSTM(input_dim => hidden_units),
    Dense(hidden_units, output_dim),
    x -> Flux.flatten(x)  # To get a 1D output
)


# --- Define loss ---
loss(x, y) = Flux.Losses.mse(model(x), y)

# --- Optimizer ---
opt = ADAM(0.001)

# --- Training loop ---
epochs = 20

println("Training model...")

# --- Explicit training loop (new style) ---
for epoch in 1:epochs
    grads = gradient(model) do m
        loss(X_train, y_train)
    end
    Flux.Optimise.update!(opt, Flux.trainable(model), grads)
    
    train_loss = loss(X_train, y_train)
    println("Epoch $epoch - Train Loss: $(round(train_loss, digits=6))")
end


# --- Evaluate on test data ---
test_loss = loss(X_test, y_test)
println("✅ Final Test Loss: ", round(test_loss, digits=6))