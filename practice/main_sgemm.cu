#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>

void sgemm_naive(int,int,int,float,const float*,const float*,float,float*);
// add future kernels like tiled here:
void sgemm_global_mem_coalesce_launch(int,int,int,float,const float*,const float*,float,float*);
// void sgemm_tiled(int,int,int,float,const float*,const float*,float,float*);
void sgemm_cublas(int,int,int,float,const float*,const float*,float,float*);

enum KernelType { NAIVE, GLOBAL_COALESCE, /*TILED,*/ CUBLAS }; // list your custom kernels here

void checkCuda(cudaError_t err,const char* msg){
    if(err!=cudaSuccess){std::cerr<<msg<<": "<<cudaGetErrorString(err)<<std::endl; exit(1);}
}

void sgemm_cpu(int M,int N,int K,float alpha,const float* A,const float* B,float beta,float* C){
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){
        float tmp=0.f;
        for(int k=0;k<K;k++) tmp+=A[i*K+k]*B[k*N+j];
        C[i*N+j]=alpha*tmp+beta*C[i*N+j];
    }
}

const char* kernel_name(KernelType k) {
    switch(k){
        case NAIVE: return "NAIVE";
        case GLOBAL_COALESCE: return "GLOBAL_COALESCE";
        // case TILED: return "TILED";
        case CUBLAS: return "CUBLAS";
        default: return "UNKNOWN";
    }
}


int main(){
    const int M=4096,N=4096,K=4096;
    const float alpha=1.f,beta=0.f;
    KernelType kernel=GLOBAL_COALESCE; // select your custom kernel here
    //KernelType kernel=NAIVE; // select your custom kernel here


    // Host arrays
    float *h_A=new float[M*K];
    float *h_B=new float[K*N];
    float *h_C_custom=new float[M*N];
    float *h_C_cublas=new float[M*N];
    float *h_C_cpu=new float[M*N]; // optional CPU reference

    for(int i=0;i<M*K;i++) h_A[i]=rand()/(float)RAND_MAX;
    for(int i=0;i<K*N;i++) h_B[i]=rand()/(float)RAND_MAX;
    for(int i=0;i<M*N;i++) h_C_custom[i]=h_C_cublas[i]=h_C_cpu[i]=0.f;

    // Device arrays
    float *d_A,*d_B,*d_C_custom,*d_C_cublas;
    checkCuda(cudaMalloc(&d_A,M*K*sizeof(float)),"Malloc d_A failed");
    checkCuda(cudaMalloc(&d_B,K*N*sizeof(float)),"Malloc d_B failed");
    checkCuda(cudaMalloc(&d_C_custom,M*N*sizeof(float)),"Malloc d_C_custom failed");
    checkCuda(cudaMalloc(&d_C_cublas,M*N*sizeof(float)),"Malloc d_C_cublas failed");

    checkCuda(cudaMemcpy(d_A,h_A,M*K*sizeof(float),cudaMemcpyHostToDevice),"Memcpy A failed");
    checkCuda(cudaMemcpy(d_B,h_B,K*N*sizeof(float),cudaMemcpyHostToDevice),"Memcpy B failed");
    checkCuda(cudaMemcpy(d_C_custom,h_C_custom,M*N*sizeof(float),cudaMemcpyHostToDevice),"Memcpy C_custom failed");
    checkCuda(cudaMemcpy(d_C_cublas,h_C_cublas,M*N*sizeof(float),cudaMemcpyHostToDevice),"Memcpy C_cublas failed");

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Run selected custom kernel
    cudaEventRecord(start);
    switch(kernel){
        case NAIVE: sgemm_naive(M,N,K,alpha,d_A,d_B,beta,d_C_custom); break;
        // case TILED: sgemm_tiled(M,N,K,alpha,d_A,d_B,beta,d_C_custom); break;
        case GLOBAL_COALESCE: sgemm_global_mem_coalesce_launch(M,N,K,alpha,d_A,d_B,beta,d_C_custom); break;
        default: std::cerr<<"Unknown kernel selected"<<std::endl; return 1;
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_custom; cudaEventElapsedTime(&ms_custom,start,stop);
    std::cout << kernel_name(kernel) << " kernel time: " << ms_custom << " ms\n";

    // Always run cuBLAS
    cudaEventRecord(start);
    sgemm_cublas(M,N,K,alpha,d_A,d_B,beta,d_C_cublas);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_cublas; cudaEventElapsedTime(&ms_cublas,start,stop);
    std::cout<<"cuBLAS GPU time: "<<ms_cublas<<" ms\n";

    // Copy back results for comparison
    checkCuda(cudaMemcpy(h_C_custom,d_C_custom,M*N*sizeof(float),cudaMemcpyDeviceToHost),"Memcpy back custom failed");
    checkCuda(cudaMemcpy(h_C_cublas,d_C_cublas,M*N*sizeof(float),cudaMemcpyDeviceToHost),"Memcpy back cuBLAS failed");

    // Compute max error vs cuBLAS
    float max_err_gpu=0.f;
    for(int i=0;i<M*N;i++)
        max_err_gpu=std::max(max_err_gpu,std::abs(h_C_custom[i]-h_C_cublas[i]));
    std::cout<<"Max error vs cuBLAS: "<<max_err_gpu<<"\n";

    // Optional CPU reference
    /*
    auto t1=std::chrono::high_resolution_clock::now();
    sgemm_cpu(M,N,K,alpha,h_A,h_B,beta,h_C_cpu);
    auto t2=std::chrono::high_resolution_clock::now();
    std::cout<<"CPU time: "<<std::chrono::duration<double,std::milli>(t2-t1).count()<<" ms\n";

    float max_err_cpu=0.f;
    for(int i=0;i<M*N;i++)
        max_err_cpu=std::max(max_err_cpu,std::abs(h_C_cpu[i]-h_C_cublas[i]));
    std::cout<<"Max error CPU vs cuBLAS: "<<max_err_cpu<<"\n";
    */

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_custom); cudaFree(d_C_cublas);
    delete[] h_A; delete[] h_B; delete[] h_C_custom; delete[] h_C_cublas; delete[] h_C_cpu;
    return 0;
}


