#include <cuda.h>
#include <stdio.h>

#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

__global__ void relu_kernel(float* x, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) x[i] = fmaxf(0.0f, x[i]);
}

__global__ void softmax_kernel(float* x, int size){
    float max_val = -1e20;
    for(int i=0;i<size;i++) max_val = fmaxf(max_val, x[i]);

    float sum = 0.0f;
    for(int i=0;i<size;i++){
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for(int i=0;i<size;i++) x[i] /= sum;
}

__global__void softmax_kernel2(float* x, int size){
    extern __shared__ float sdata[]; //dynamically allocate shared memory
    int tid = threadIdx.x;

    //load data into shared memory
    float val = (tid < size) ? x[tid] : -1e20f; //for max_val
    sdata[tid] = val;

    //wait for all threads to finish loading
    __syncthreads();

    //1. Parallel max reduction
    for(int s = blockDim.x/2; s>0; s>>=1){
        if(tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    //2. compute exp(x-max_val)
    float exp_val = (tid < size) ? expf(x[tid] - max_val) : 0.0f;
    sdata[tid] = exp_val;
    __syncthreads();

    //3. Parallel sum reduction
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float sum = sdata[0];
    __syncthreads();

    // Step 4: Normalize
    if(tid < size) x[tid] = exp_val / sum;
}

