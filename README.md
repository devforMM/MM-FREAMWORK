🧠 SimpleDLFramework
A lightweight deep learning framework built from scratch in Python, designed to help understand, build, and experiment with core deep learning components without relying on high-level libraries like PyTorch or TensorFlow.

This framework includes essential deep learning building blocks like custom regularization, optimizers, learning rate schedulers, CNN and MLP architectures, and training strategies — organized into clean, modular, and well-documented code.
📁 Project Structure
graphql
Copier
Modifier
core/
│
├── Droupout_layer.py          # Dropout implementation
├── MLp_initializers.py        # MLP weight initializers (Xavier, He, etc.)
├── MLp_layer.py               # MLP fully connected layers
├── losses.py                  # Loss functions (MSE, MAE, CrossEntropy…)
├── metrics.py                 # Evaluation metrics (accuracy, etc.)
├── model_structure.py         # Deep Learning model class (forward, backward, training loops)
├── optimizers.py              # Gradient Descent, Momentum, Adam, etc.

CNN/
│
├── Loop_based_cnn/
│   ├── Cnn_initializers.py    # CNN weight initializers
│   ├── Cnn_layers.py          # CNN convolution/maxpool/batchnorm layers (loop-based)
│   ├── Cnn_operations.py      # Convolution and pooling operations (loop-based)
│
├── Vectorised_Cnn_operations/
│   ├── Vectorised_Cnn_operations.py  # Vectorized Conv2D, MaxPooling, etc.
│   ├── Vec_cnn_Layers.py             # Vectorized CNN layers (Conv, MaxPool)

utils/
│
├── activations.py             # Activation functions (ReLU, Sigmoid, Softmax…)
├── batch_normalization_Layer.py  # Custom BatchNorm implementation
├── data_manipulation.py       # Data splitting and k-fold cross-validation
├── dropuout_Layer.py          # Dropout implementation (alternative)
├── initializers.py            # Initializers utilities
├── learning_rate.py           # Learning rate scheduling methods
├── weight_decay.py            # Weight decay (L2 regularization)

data/
│
├── MNIST/
│   └── raw/                   # MNIST dataset files (.gz and extracted)

notebooks/
│
├── regression_MLP.ipynb       # Regression with MLP notebook
├── Multi_classification_MLP.ipynb  # Multi-class MLP classification notebook
├── single_perceptron.ipynb    # Simple perceptron test
├── Iris.ipynb                 # Iris dataset classification
├── California_housing.ipynb   # Regression on housing dataset
├── loop_based_mnist.ipynb     # MNIST classification with Loop-based CNN
├── Vec_Cnn_mnist.ipynb        # MNIST classification with vectorized CNN

utils/
├── *.md                       # Theory explanations (dropout, weight decay, learning rate…)

.git/                          # Git version control files
📚 Features
✅ MLP & CNN architectures from scratch

✅ Loop-based and vectorized CNN implementations

✅ Dropout & Batch Normalization

✅ Weight Decay (L2 Regularization)

✅ Gradient Descent, Momentum, Adagrad, RMSProp, Adam optimizers

✅ Learning Rate Schedulers (step decay, factor decay…)

✅ Early Stopping

✅ Custom loss functions (MSE, MAE, CrossEntropy…)

✅ Data split & Cross-validation utilities

✅ Clean modular code, fully documented

📖 Goal
This framework is intended for learning, experimentation, and educational purposes — allowing developers, students, and researchers to :

Understand and implement deep learning concepts by coding them from scratch.

Customize every part of the training process.

Experiment with optimization, regularization, and model architectures on real datasets.

Visualize and debug the training pipeline without the abstraction of high-level APIs.

📌 Requirements
Python 3.x

Numpy

Torch (for tensor operations only — no nn.Module used)

📄 License
This project is open-source and available for educational and personal use.

📣 Author
Created by Hebou Abdelraouf as a personal deep learning framework for building, testing, and optimizing AI models from scratch.



