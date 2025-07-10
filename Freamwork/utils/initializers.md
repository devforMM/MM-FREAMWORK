
🎛️ Weight Initializers for Deep Learning
This module provides clean implementations of the most commonly used weight initialization strategies in deep learning, directly using PyTorch.
Proper weight initialization helps improve the stability, speed, and convergence of neural networks during training.

📚 Included Initializers
Initializer	Description
Xavier Normal	Normal distribution with variance based on n_in and n_out
He Normal	Normal distribution optimized for ReLU-based networks
Xavier Uniform	Uniform distribution with bounds based on n_in and n_out
He Uniform	Uniform distribution optimized for ReLU-based networks
🎯 Purpose
Prevents vanishing and exploding gradients at the start of training

Helps achieve faster convergence

Provides tailored initialization for networks using ReLU, Sigmoid, Tanh, etc.

