# EMNIST Inference Benchmark Comparison

A comprehensive benchmarking suite designed to evaluate and compare the inference performance (latency and throughput) of a standard Convolutional Neural Network (CNN) across multiple leading deep learning frameworks. The models are trained and evaluated on the EMNIST dataset.

## 📊 Overview

When deploying deep learning models, choosing the right framework can drastically impact production performance. This repository provides a head-to-head inference benchmark of a 2D CNN architecture implemented in:
*   **PyTorch** (Ecosystem & dynamic graph)
*   **PyTorch Lightning** (Clean production code)
*   **TensorFlow / Keras** (Beginner-friendly API)
*   **JAX / Flax** (XLA-compiled functional speed)
*   **ONNX Runtime** (Graph-optimized CPU/GPU inference)

## 📁 Repository Structure

*   `24AI028_lab_1.ipynb`: The primary Jupyter Notebook containing the environment setup, model definitions across all five frameworks, ONNX export logic, and the benchmarking execution loops.

## ⚙️ Prerequisites and Installation

The code is optimized for execution in a Google Colab environment with a T4 GPU, but it can be run locally. Ensure you have Python 3 installed along with the respective framework libraries. 

To install the missing frameworks and dependencies, run:

```bash
pip install lightning flax onnx onnxruntime tabulate jax jaxlib tensorflow torch pandas numpy