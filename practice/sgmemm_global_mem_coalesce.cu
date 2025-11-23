#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M,int N,int K,float alpha,
                                          const float* A,const float* B,
                                          float beta,float* C){
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float tmp = 0.f;
        for (int i = 0; i < K; ++i)
            tmp += A[cRow*K + i] * B[i*N + cCol];

        C[cRow*N + cCol] = alpha * tmp + beta * C[cRow*N + cCol];
    }
}


void sgemm_global_mem_coalesce_launch(int M,int N,int K,float alpha,
                                      const float* A,const float* B,
                                      float beta,float* C){
    const int BS = 32;
    dim3 block(BS*BS);
    dim3 grid((M+BS-1)/BS,(N+BS-1)/BS);
    sgemm_global_mem_coalesce<BS><<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
}
