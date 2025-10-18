#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// Matrix size (small for testing)
#define N 1024

// CUDA kernel: each thread computes one element of C
__global__ void matmul_naive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if(row < n && col < n) {
        float sum = 0.0f;
        for(int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CPU version for verification
void matmul_cpu(float *A, float *B, float *C, int n) {
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            float sum = 0;
            for(int k=0; k<n; k++){
                sum += A[i*n+k]*B[k*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

int main() {
    int n = N;
    size_t bytes = n * n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);

    // Initialize matrices with some values
    for(int i=0;i<n*n;i++){
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1)/threadsPerBlock.y);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_time = end - start;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Compute CPU result
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_time = end_cpu - start_cpu;

    // Verify correctness
    int correct = 1;
    for(int i=0;i<n*n;i++){
        if(abs(h_C[i] - h_C_cpu[i]) > 1e-5){
            correct = 0;
            break;
        }
    }

    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    printf("GPU Time: %.3f ms\n", gpu_time.count());
    printf("CPU Time: %.3f ms\n", cpu_time.count());

    // Free memory
    free(h_A); free(h_B); free(h_C); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
