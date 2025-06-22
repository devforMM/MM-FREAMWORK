
# SimpleDLFramework

A lightweight deep learning framework built from scratch in Python to help understand and experiment with core deep learning techniques without relying on high-level libraries like PyTorch or TensorFlow.

This framework implements key deep learning components such as custom regularization, optimization utilities, learning rate schedulers, model architectures, and more — with clean, modular code.

---

## 📁 Project Structure

```
core/
│
├── dropout.py                 # Dropout implementation
├── dropout.md                 # Theory and explanation of dropout
├── layres.py                  # Definition of neural network layers
├── regularisation.py          # Regularization functions
├── weight_decay.py            # Weight decay implementation
├── weight_decay.md            # Explanation of weight decay

models/
│
├── linear_classification.ipynb  # Notebook for linear classification experiments
├── linear_classification.md     # Notes and theory for linear classification
├── linear_model.ipynb           # Notebook for basic linear models
├── linear_model.md              # Documentation on linear model concepts
├── mlp_model.ipynb              # Multilayer perceptron implementation and test
├── mlp.md                       # Theory and notes for MLP models

utils/
│
├── early_stopping.py           # Early stopping implementation
├── early_stopping.md           # Explanation and theory of early stopping
├── gradient_clipping.py        # Gradient clipping implementation
├── gradient_clipping.md        # Theory and use-cases for gradient clipping
├── initializers.py             # Weight initialization techniques
├── initializers.md             # Documentation for initializers
├── learning_rate.py            # Learning rate scheduling strategies
├── learning_rate.md            # Notes and theory for learning rate schedules
```

---

## 📚 Features

- **Dropout Regularization**  
  Custom implementation of dropout to prevent overfitting during training.

- **Weight Decay (L2 Regularization)**  
  Adds a penalty to the loss function based on the magnitude of weights to improve generalization.

- **Gradient Clipping**  
  Prevents exploding gradients by limiting the gradient values during backpropagation.

- **Early Stopping**  
  Stops training when the model performance on a validation set stops improving.

- **Custom Initializers**  
  Xavier, He, and uniform/normal initialization strategies for neural network weights.

- **Learning Rate Scheduling**  
  Includes step decay, factor decay, cosine annealing, and warmup scheduling methods.

- **Simple Model Architectures**  
  Linear models and multilayer perceptrons (MLPs) implemented and tested via notebooks.

---

## 📖 Goal

This framework is intended for learning and experimentation purposes — allowing developers, students, and researchers to:

- Understand core deep learning concepts by implementing them manually.
- Customize and visualize every part of the training pipeline.
- Experiment with optimization and regularization techniques on real datasets.
- Build intuition on model training dynamics without relying on abstracted high-level APIs.

---

## 🚀 Usage

Clone this repository and start experimenting:

```bash
git clone https://github.com/your-username/SimpleDLFramework.git
cd SimpleDLFramework
```

You can then run the provided Jupyter notebooks inside the `models/` directory to test various models and techniques.

---

## 📌 Requirements

- Python 3.x
- Numpy
- Torch (for tensor operations only — optional)

---

## 📄 License

This project is open-source and available for educational and personal use.

---

## 📣 Author

Created by **[Your Name]** as a personal deep learning framework for understanding, testing, and optimizing machine learning models from scratch.
