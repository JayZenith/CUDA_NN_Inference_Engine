#include <cuda_runtime.h>

__global__ void sgemm_naive_kernel(int M,int N,int K,float alpha,const float* A,const float* B,float beta,float* C){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x<M && y<N){
        float tmp=0.f;
        for(int i=0;i<K;i++) tmp += A[x*K+i]*B[i*N+y];
        C[x*N+y] = alpha*tmp + beta*C[x*N+y];
    }
}

void sgemm_naive(int M,int N,int K,float alpha,const float* d_A,const float* d_B,float beta,float* d_C){
    dim3 block(32,32);
    dim3 grid((M+31)/32,(N+31)/32);
    sgemm_naive_kernel<<<grid,block>>>(M,N,K,alpha,d_A,d_B,beta,d_C);
}
