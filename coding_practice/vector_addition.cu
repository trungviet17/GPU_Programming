#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#define N 10 

/*
Vector addition example in CUDA 
Input : Two vectors A and B of size N
Output: Vector C of size N where C[i] = A[i] + B[i]
*/

__global__ void vector_add(int* a, int* b, int* c) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) { 
        c[i] = a[i] + b[i]; 
    }

}


int main() {


    int* h_a, *h_b, *h_c; // Host vectors
    int* d_a, *d_b, *d_c; // Device vectors

    size_t size = N * sizeof(int);
    
    cudaMallocHost((void**)&h_a, size);
    cudaMallocHost((void**)&h_b, size);
    cudaMallocHost((void**)&h_c, size);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1; 
        h_b[i] = 1;
    }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);


    int threadPerBlock = 256; 
    int blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    vector_add<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize(); 

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for(int i = 0; i < N; i++) {
        cout << h_c[i] << " "; 
    }
    cout << endl;
}

