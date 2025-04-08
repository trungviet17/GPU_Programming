#include<iostream>
#include<cuda_runtime.h>



__global__ void dotProductKernel(int* A, int* B, int* C, int N) {

    __shared__ int cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int temp = 0; 
    if (tid < N) {
        temp = A[i] * B[i];
    }

    cache[tid] = temp; 
    __syncthreads();


    for (int stride = blockDim.x /2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(C, cache[0]);
    }
}


void runDotProduct(int N) {

    int * h_A, * h_B;
    int * d_A, * d_B, *d_C;
    size_t size = N * sizeof(int);

    h_A = new int[N];
    h_B = new int[N];
    int h_C = 0; 

    for(int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, sizeof(int));
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, &h_C, sizeof(int), cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Dot product result: " << h_C << std::endl;
    delete[] h_A;
    delete[] h_B;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    std::cout << "CUDA device reset." << std::endl;
}


int main() {
    int N = 1024;
    runDotProduct(N);
    return 0;
}