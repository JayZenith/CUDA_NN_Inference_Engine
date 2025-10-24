# CUDA Neural Network Inference Engine

This project implements a simple feedforward neural network on the GPU using CUDA, demonstrating high-performance matrix operations and GPU acceleration.

## Features
- Forward pass of a 2-layer neural network (input → hidden → output)
- Tiled matrix multiplication using shared memory for GPU optimization
- Custom activation kernels: ReLU and Softmax
- GPU vs CPU performance comparison (>16× speedup)

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++17 compiler

## Build & Run
```bash
nvcc src/main.cu -o nn_inference
./nn_inference

