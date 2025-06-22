# 📊 Linear Classification from Scratch with PyTorch

This project provides a **from-scratch implementation** of a multi-class linear classifier using **Softmax + Cross-Entropy**, with several optimization strategies:

- **Batch Gradient Descent**
- **Mini-Batch Stochastic Gradient Descent**
- **Stochastic Gradient Descent**
- With optional **L2 Regularization**

## 📦 Features

- Forward pass computation (matrix multiplication)
- Softmax activation
- Cross-Entropy Loss
- L2 Regularization
- Backpropagation and weight updates
- Data splitting and Cross-Validation
- Accuracy evaluation
- Loss curve visualization

## 📈 Example Usage

```python
# Example training
model = LinearClassification(x, w, y, learning_rate=0.01)
x_train, y_train, x_val, y_val = model.split_data()
losses, val_losses = model.batch_gd(epochs=100, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
model.plot(losses, val_losses, epochs=100)
