#include<iostream>
#include<cuda_runtime.h>

#define THREADPERBLOCK 256



__global__ void dotProductKernel(int* A, int* B, int* C, int N) {
    __shared__ int sdata[THREADPERBLOCK];


    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = )


}


void runDotProduct(int N) {
    float * h_a, *h_b, *h_c; 
    float *d_a, *d_b, *d_c; z

    size_t size = N * sizeof(float);

    cudaMallocHost((void**)&h_a, size);
    cudaMallocHost((void**)&h_b, size);
    cudaMallocHost((void**)&h_c, size);


    cudaHostGetDevicePointer((void**)&d_a, h_a, 0);
    cudaHostGetDevicePointer((void**)&d_b, h_b, 0);
    cudaHostGetDevicePointer((void**)&d_c, h_c, 0);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }


    int BLOCKPERGRID = (N + THREADPERBLOCK - 1) / THREADPERBLOCK;
    dotProductKernel<<<BLOCKPERGRID, THREADPERBLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    


}


int main() {
    int N = 1024;
    runDotProduct(N);
    return 0;
}