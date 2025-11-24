#include <cublas_v2.h>

void sgemm_cublas(int M,int N,int K,float alpha,const float* d_A,const float* d_B,float beta,float* d_C){
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,d_B,N,d_A,K,&beta,d_C,N);
    cublasDestroy(handle);
}


