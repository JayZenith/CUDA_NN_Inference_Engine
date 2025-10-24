#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <cmath>

// Matrix size
#define N 1024
#define TILE_SIZE 16

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int n) {
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            float sum = 0.0f;
            for(int k=0;k<n;k++){
                sum += A[i*n+k] * B[k*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

// Naive GPU kernel
__global__ void matmul_naive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < n && col < n) {
        float sum = 0.0f;
        for(int k=0;k<n;k++)
            sum += A[row*n+k]*B[k*n+col];
        C[row*n+col] = sum;
    }
}

// Shared memory GPU kernel

__global__ void matmul_shared(float *A, float *B, float *C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < n/TILE_SIZE; t++) {
        // Load tiles with bounds check
        tile_A[threadIdx.y][threadIdx.x] = (row < n && t*TILE_SIZE + threadIdx.x < n) ?
                                           A[row*n + t*TILE_SIZE + threadIdx.x] : 0.0f;

        tile_B[threadIdx.y][threadIdx.x] = (t*TILE_SIZE + threadIdx.y < n && col < n) ?
                                           B[(t*TILE_SIZE + threadIdx.y)*n + col] : 0.0f;

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row*n + col] = sum;
}




int main() {
    int n = N;
    size_t bytes = n*n*sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_shared = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);

    // Initialize
    for(int i=0;i<n*n;i++){
        h_A[i] = rand()%10;
        h_B[i] = rand()%10;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1)/TILE_SIZE, (n + TILE_SIZE - 1)/TILE_SIZE);

    // --- Naive GPU ---
    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost);
    std::chrono::duration<float, std::milli> naive_time = end - start;

    // --- Shared memory GPU ---
    start = std::chrono::high_resolution_clock::now();
    matmul_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C_shared, d_C, bytes, cudaMemcpyDeviceToHost);
    std::chrono::duration<float, std::milli> shared_time = end - start;

    // --- CPU ---
    start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_time = end - start;

    // Verification using relative error
    float eps = 1e-3f; // 0.1% tolerance
    int correct_naive = 1, correct_shared = 1;

    for(int i=0; i<n*n; i++) {
        float diff_naive = fabs(h_C_naive[i] - h_C_cpu[i]);
        float diff_shared = fabs(h_C_shared[i] - h_C_cpu[i]);

        if(diff_naive / (fabs(h_C_cpu[i]) + 1e-6f) > eps) correct_naive = 0;
        if(diff_shared / (fabs(h_C_cpu[i]) + 1e-6f) > eps) correct_shared = 0;
    }


    printf("Verification Naive GPU: %s\n", correct_naive ? "PASSED":"FAILED");
    printf("Verification Shared GPU: %s\n", correct_shared ? "PASSED":"FAILED");
    printf("CPU Time: %.3f ms\n", cpu_time.count());
    printf("Naive GPU Time: %.3f ms\n", naive_time.count());
    printf("Shared GPU Time: %.3f ms\n", shared_time.count());

    // Free memory
    free(h_A); free(h_B); free(h_C_naive); free(h_C_shared); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
