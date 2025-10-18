#include <stdio.h>
#include <cuda_runtime.h>

//A,B = input arrays, C = output array, N = number of elements
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    //Each GPU thread computes which elem index it should work on
    // threadIdx.x = this thread's index within its block
    // blockIdx.x = which block we're in
    // blockDim.x = number of threads per block
    // formula gives every thread a unique global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //Each thread
    if (i < N) {  //only take first 10 threads
        C[i] = A[i] + B[i];
    }

    //Each thread does one element of vector addition
}

int main() {
    int N = 10;
    size_t size = N * sizeof(float); //total # bytes in each array

    float h_A[N], h_B[N], h_C[N]; //host arrays (live in CPU memory)

    //Fill arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    //A = [0,1,2,3,...]
    //B = [0,2,4,6,...]

    float *d_A, *d_B, *d_C; //Pointers to arrays on GPU

    // Allocate space for A,B, C on GPU
    // malloc but on GPU memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from CPU -> GPU memory
    // Now GPU has its own copies of arrays A and B
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //Configure how many threads and blocks we want to launch
    // choose up to 256 threads per block (typical GPU size)
    // then compute how many blocks needed to cover all N elements
    // -> For N = 10, that gives 1 block
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch GPU kernel
    // Each thread handles one element of the arrays
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // CPU waits for GPU

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result C = ");
    for (int i = 0; i < N; i++) {
        printf("%0.1f ", h_C[i]);
    }
    printf("\n");

    // Free GPU memory and exit
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
