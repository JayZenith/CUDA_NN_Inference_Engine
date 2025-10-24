
#include <stdio.h>
#include <cuda.h>

#define N 16        // Array size
#define BLOCK_SIZE 8 // Threads per block

// CUDA kernel for parallel reduction (sum)
__global__ void reduce_sum(int *input, int *output, int n) {
    __shared__ int sdata[BLOCK_SIZE]; // Shared memory for this block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int h_input[N];
    for (int i = 0; i < N; i++) h_input[i] = i + 1; // 1,2,...,16

    int *d_input, *d_output;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int h_output[numBlocks];

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, numBlocks * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    reduce_sum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final sum on CPU (combine block sums)
    int total = 0;
    for (int i = 0; i < numBlocks; i++) total += h_output[i];

    printf("Sum of array = %d\n", total);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
