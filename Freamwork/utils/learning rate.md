# 📉 Learning Rate Scheduler

A lightweight Python module to manage different learning rate scheduling strategies during deep learning model training.

## 📦 Overview

The `learning_rate_scheduler.py` file contains a single function `learning_rate_scheduler` that dynamically adjusts the learning rate based on the selected strategy and current epoch.

## 🚀 Features

The function supports several scheduling strategies:

- **Factor Decay**: Multiplies the current learning rate by 0.9 at each call.
- **Multistep Scheduler**: Divides the learning rate by a factor (default is 2).
- **Cosine Annealing**: Smoothly decreases the learning rate following a cosine curve toward a final value.
- **Warmup**: Linearly increases the learning rate over a few initial epochs before stabilizing.
- **Warmup + Cosine**: Combines linear warmup with Cosine Annealing scheduling.


