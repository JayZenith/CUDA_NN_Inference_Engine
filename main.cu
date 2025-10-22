#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "src/kernels.cu"

void forward_gpu(float* d_input, float* d_W1, float* d_b1,
                 float* d_W2, float* d_b2, float* d_output,
                 int input_size, int hidden_size, int output_size) {
    float* d_hidden;
    cudaMalloc(&d_hidden, hidden_size * sizeof(float));

    // Layer1: input * W1
    dim3 threads(16,16);
    dim3 blocks((hidden_size+15)/16, (1+15)/16);
    matmul_tiled<<<blocks, threads>>>(d_input, d_W1, d_hidden, 1, input_size, hidden_size);
    cudaDeviceSynchronize();

    // Add bias (simple kernel)
    // ReLU
    relu_kernel<<<(hidden_size+255)/256,256>>>(d_hidden, hidden_size);

    // Layer2: hidden * W2
    matmul_tiled<<<dim3((output_size+15)/16,(1+15)/16), dim3(16,16)>>>(d_hidden, d_W2, d_output, 1, hidden_size, output_size);
    cudaDeviceSynchronize();

    // Add bias (simple kernel)
    softmax_kernel<<<1,1>>>(d_output, output_size);

    cudaFree(d_hidden);
}

int main() {
    int input_size = 1024;
    int hidden_size = 512;
    int output_size = 10;

    // Host allocation
    float *h_input = new float[input_size];
    float *h_W1 = new float[input_size*hidden_size];
    float *h_b1 = new float[hidden_size];
    float *h_W2 = new float[hidden_size*output_size];
    float *h_b2 = new float[output_size];
    float *h_output = new float[output_size];

    // Fill random data
    for(int i=0;i<input_size;i++) h_input[i] = rand()/(float)RAND_MAX;
    for(int i=0;i<input_size*hidden_size;i++) h_W1[i] = rand()/(float)RAND_MAX;
    for(int i=0;i<hidden_size;i++) h_b1[i] = rand()/(float)RAND_MAX;
    for(int i=0;i<hidden_size*output_size;i++) h_W2[i] = rand()/(float)RAND_MAX;
    for(int i=0;i<output_size;i++) h_b2[i] = rand()/(float)RAND_MAX;

    // Device allocation
    float *d_input, *d_W1, *d_b1, *d_W2, *d_b2, *d_output;
    cudaMalloc(&d_input, input_size*sizeof(float));
    cudaMalloc(&d_W1, input_size*hidden_size*sizeof(float));
    cudaMalloc(&d_b1, hidden_size*sizeof(float));
    cudaMalloc(&d_W2, hidden_size*output_size*sizeof(float));
    cudaMalloc(&d_b2, output_size*sizeof(float));
    cudaMalloc(&d_output, output_size*sizeof(float));

    // Copy to GPU
    cudaMemcpy(d_input, h_input, input_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, input_size*hidden_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, hidden_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, hidden_size*output_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, output_size*sizeof(float), cudaMemcpyHostToDevice);

    // GPU forward pass
    auto start = std::chrono::high_resolution_clock::now();
    forward_gpu(d_input, d_W1, d_b1, d_W2, d_b2, d_output, input_size, hidden_size, output_size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_output, d_output, output_size*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU forward pass time: " 
              << std::chrono::duration<float, std::milli>(end-start).count() << " ms\n";

    std::cout << "Output[0]: " << h_output[0] << std::endl;

    // CPU forward pass (naive)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float hidden[hidden_size];
    for(int i=0;i<hidden_size;i++){
        hidden[i] = h_b1[i];
        for(int j=0;j<input_size;j++) hidden[i] += h_input[j]*h_W1[j*hidden_size + i];
        hidden[i] = fmaxf(0.0f, hidden[i]);
    }
    float output[output_size];
    for(int i=0;i<output_size;i++){
        output[i] = h_b2[i];
        for(int j=0;j<hidden_size;j++) output[i] += hidden[j]*h_W2[j*output_size + i];
    }
    // Softmax
    float max_val = -1e20, sum=0.0f;
    for(int i=0;i<output_size;i++) if(output[i]>max_val) max_val=output[i];
    for(int i=0;i<output_size;i++){
        output[i] = expf(output[i]-max_val);
        sum += output[i];
    }
    for(int i=0;i<output_size;i++) output[i]/=sum;
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU forward pass time: "
              << std::chrono::duration<float, std::milli>(cpu_end-cpu_start).count() << " ms\n";

    // Free
    cudaFree(d_input); cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_output);
    delete[] h_input; delete[] h_W1; delete[] h_b1;
    delete[] h_W2; delete[] h_b2; delete[] h_output;

    return 0;
}
