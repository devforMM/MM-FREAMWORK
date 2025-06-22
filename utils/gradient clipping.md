📄 README.md
✂️ Gradient Clipping Utility
This module implements a simple Gradient Clipping technique, commonly used in deep learning to prevent the explosion of gradients during backpropagation — especially in recurrent neural networks (RNNs) or deep models.

📚 How It Works
Computes the L2-norm of the gradient.

If the norm exceeds a specified threshold (or borne), rescales the gradient to keep it within that threshold.

Helps stabilize training and avoid NaN or diverging loss values.
🎯 Purpose
Prevents gradient explosion

Stabilizes the learning process

Ensures training remains numerically safe and effective