#include <iostream>

//__global__ is a CUDA keyword telling compiler this function
// runs on GPU but called from CPU (host)
__global__ void helloFromGPU() {
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 5>>>();  // 1 block, 5 threads

    //CPU tells GPU to start kernel but launch is asynch.
    // so CPU runs while GPU working.
    // forces CPU to wait
    cudaDeviceSynchronize();
    return 0;
}


//Compile and execute
!nvcc hello.cu -o hello
!./hello

//Expected output
//Hello from GPU! Thread 0
//Hello from GPU! Thread 1
//Hello from GPU! Thread 2
//Hello from GPU! Thread 3
//Hello from GPU! Thread 4
