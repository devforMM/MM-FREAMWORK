# 📊 Linear Regression from Scratch with PyTorch

This project provides a clean **from-scratch implementation** of a simple linear regression model using **Mean Squared Error (MSE)** loss and **PyTorch autograd**. It supports several gradient descent optimization techniques:

- **Batch Gradient Descent**
- **Mini-Batch Stochastic Gradient Descent**
- **Stochastic Gradient Descent**
- With optional **L2 Regularization**

## 📦 Features

- Linear forward pass
- Mean Squared Error (MSE) Loss
- L2 Regularization
- Backpropagation and weight updates using PyTorch autograd
- Data splitting and k-fold Cross-Validation
- RMSE evaluation metric
- Training and validation loss visualization

## 📈 Example Usage

```python
# Example training
model = LinearRegression(x, w, y, learning_rate=0.01)
x_train, y_train, x_val, y_val = model.split_data()
losses, val_losses = model.batch_gd(epochs=100, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
model.plot(losses, val_losses, epochs=100)
