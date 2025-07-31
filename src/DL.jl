# Investment and Financing Decision Support System (Julia)
Pkg.activate(".")
Pkg.add(["CSV", "DataFrames", "Flux", "Statistics", "MLDataPattern", "Plots", "MLUtils", "Dates"])
using CSV, DataFrames, Dates, Statistics
using Flux
using Metalhead  # for pretrained CNN models
using Plots

# 1. Data Loading & Preprocessing
println("Loading data...")
df = CSV.read("s&p500_stock_data.csv", DataFrame)
df.Date = Date.(df.Date, dateformat"yyyy-mm-dd")
sort!(df, :Date)
dropmissing!(df)  # Remove rows with missing values:contentReference[oaicite:16]{index=16}

# Separate features and target
features = select(df, Not(:Close))
prices = df.Close

# Normalize numeric features
println("Normalizing data...")
numeric_cols = names(features)[Not(1)]  # assume Date is first column
for col in numeric_cols
    col_min = minimum(skipmissing(features[!, col]))
    col_max = maximum(skipmissing(features[!, col]))
    features[!, col] = (features[!, col] .- col_min) ./ (col_max - col_min)
end

# Normalize target prices
price_min = minimum(prices)
price_max = maximum(prices)
normalized_prices = (prices .- price_min) ./ (price_max - price_min)

# 2. Train/Validation/Test Split
n = nrow(df)
n_train = Int(floor(0.7 * n))
n_val = Int(floor(0.15 * n))
n_test = n - n_train - n_val
train_idx = 1:n_train
val_idx   = (n_train+1):(n_train+n_val)
test_idx  = (n_train+n_val+1):n
X_train = features[train_idx, :]
y_train = normalized_prices[train_idx]
X_val   = features[val_idx, :]
y_val   = normalized_prices[val_idx]
X_test  = features[test_idx, :]
y_test  = normalized_prices[test_idx]

# 3. Model Architecture (CNN + LSTM + Transfer Learning)
println("Building model...")
features_count = size(X_train, 2)
hidden_size = 50
lstm_layer = LSTM(features_count, hidden_size)
seq_dense   = Dense(hidden_size, 32, relu)

# Pre-trained CNN (ResNet18) branch
cnn_pretrained = ResNet(18; pretrain=true)
Flux.frozen(cnn_pretrained)  # freeze pretrained weights
cnn_dense = Dense(1000, 32, relu)  # map ResNet output (1000 classes) to 32-dim

# Final fusion layer
fusion_dense = Dense(64, 1)  # combines 32+32 -> 1

function model_fn(x_seq, x_img)
    # LSTM branch
    h = lstm_layer.(eachcol(x_seq))
    seq_feat = seq_dense(h[end])  # final hidden state
    # CNN branch
    img_feat = cnn_dense(cnn_pretrained(x_img))
    # Fuse and predict
    merged = vcat(seq_feat, img_feat)
    return fusion_dense(merged)
end

# 4. Training with Hyperparameter Tuning
println("Training model...")
loss_fn(y_pred, y_true) = mean((y_pred .- y_true).^2)
opt = ADAM(1e-3)  # optimizer with learning rate 0.001

train_losses = Float64[]
val_losses = Float64[]
epochs = 20

for epoch in 1:epochs
    # Training epoch
    total_loss = 0.0
    for (x_vec, y_true) in zip(eachrow(X_train), y_train)
        # Prepare inputs (seq_len=1 for demo purposes)
        x_seq = reshape(collect(x_vec), :, 1)
        x_img = rand(Float32, 224, 224, 3)  # placeholder static image
        gs = gradient(() -> loss_fn(model_fn(x_seq, x_img), y_true),
                      Flux.params(lstm_layer, seq_dense, fusion_dense))
        Flux.Optimise.update!(opt, Flux.params(lstm_layer, seq_dense, fusion_dense), gs)
        total_loss += loss_fn(model_fn(x_seq, x_img), y_true)
    end
    train_loss = total_loss / n_train
    push!(train_losses, train_loss)
    # Validation
    val_loss = 0.0
    for (x_vec, y_true) in zip(eachrow(X_val), y_val)
        x_seq = reshape(collect(x_vec), :, 1)
        x_img = rand(Float32, 224, 224, 3)
        val_loss += loss_fn(model_fn(x_seq, x_img), y_true)
    end
    val_loss /= n_val
    push!(val_losses, val_loss)
    println("Epoch $epoch: train_loss=$(round(train_loss,digits=4)), val_loss=$(round(val_loss,digits=4))")
end

# 5. Evaluation and Visualization
# Plot loss curves
plot(1:epochs, train_losses, label="Train Loss", xlabel="Epoch", ylabel="MSE", title="Loss Curves")
plot!(1:epochs, val_losses, label="Val Loss")
savefig("loss_curve.png")

# Predictions on test set
println("Evaluating on test set...")
y_pred = Float64[]
for x_vec in eachrow(X_test)
    x_seq = reshape(collect(x_vec), :, 1)
    x_img = rand(Float32, 224, 224, 3)
    y_hat = model_fn(x_seq, x_img)[1]
    push!(y_pred, y_hat)
end
# Denormalize
y_pred_denorm = y_pred .* (price_max - price_min) .+ price_min
y_actual_denorm = y_test .* (price_max - price_min) .+ price_min

# Plot predicted vs actual prices
plot(1:length(y_actual_denorm), y_actual_denorm, label="Actual Price", xlabel="Time Index", ylabel="Price", title="Predicted vs Actual S&P 500 Price")
plot!(1:length(y_pred_denorm), y_pred_denorm, label="Predicted Price")
savefig("pred_vs_actual.png")
