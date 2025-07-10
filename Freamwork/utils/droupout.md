# 🗑️ Custom Dropout Layer (PyTorch)

A lightweight implementation of the **Dropout** regularization technique for neural networks, built using PyTorch operations.

## 📦 Overview

The `dropout` function randomly zeroes a fraction of the elements of the input tensor during training to prevent overfitting. The remaining active units are scaled by `1 / (1 - p)` to maintain the expected sum of the outputs.

## 🚀 Features

- Pure PyTorch implementation
- Adjustable dropout probability `p`
- Can be integrated into any custom model or training loop
- Keeps output values scaled properly during training

## 📑 Usage Example


