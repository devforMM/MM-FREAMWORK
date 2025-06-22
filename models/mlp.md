# 🧠 Multi-Layer Perceptron (MLP) from Scratch with PyTorch

This project implements a fully custom **Multi-Layer Perceptron (MLP)** framework from scratch using **PyTorch autograd**. It supports multiple gradient descent variants and activation functions.

## 📦 Features

- Custom-defined `Layer` and `MLP` classes
- Supports **ReLU**, **Sigmoid**, **Tanh**, and **Softmax**
- Mean Squared Error (MSE) and Cross-Entropy Loss with optional **L2 regularization**
- Supports:
  - **Batch Gradient Descent**
  - **Mini-Batch Stochastic Gradient Descent**
  - **Stochastic Gradient Descent**
- Training and validation loss visualization
- Cross-Validation and data split utilities
- Accuracy and RMSE evaluation metrics

## 📈 Example Usage

```python
model = MLP(x, y, learning_rate=0.01)
model.add_layer(input_dim, 64, activation="Relu")
model.add_layer(64, 32, activation="tanh")
model.add_layer(32, num_classes)

# Choose a loss function
model.loss_function = model.Logcrossentropy

# Split data
x_train, y_train, x_val, y_val = model.split_data()

# Train the model
losses, val_losses = model.batch_gd_train(epochs=100, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

# Visualize loss curves
model.plot_loss(losses, val_losses, epochs=100)
