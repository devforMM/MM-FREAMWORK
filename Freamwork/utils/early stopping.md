🛑 Early Stopping Utility for Deep Learning
This module provides a clean and customizable implementation of Early Stopping in PyTorch or any training loop using Python functions.
It helps stop the training process early when no improvement is observed on the validation loss for a defined number of consecutive epochs — preventing overfitting and saving time.

📚 How It Works
Tracks validation loss at each epoch.

If no improvement is seen over a given patience (number of epochs), training stops.

Keeps a record of training and validation losses over the epochs.
🎯 Purpose
Avoid overfitting by stopping training early

Save computational resources

Improve model generalization by preventing excessive training

