# CUDA Neural Network Inference Engine

A minimal GPU implementation of a 2-layer feedforward neural network using CUDA.
Demonstrates matrix multiplication with shared memory tiling and GPU vs CPU performance.

## Features
- Forward pass: input → hidden → output
- ReLU and Softmax activation kernels
- Tiled matrix multiplication (`16×16` blocks)
- GPU vs CPU timing comparison

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++17 compiler

## Build & Run
```bash
nvcc -arch=sm_75 -O3 src/main.cu -o build/main
./build/main
```
## Sample Output
```bash
GPU forward pass time: 0.31 ms
probability of class 0: 0.00
CPU forward pass time: 4.19 ms
```

Tested with CUDA 12.4 and C++17.
Adjust -arch to match your GPU’s compute capability.


