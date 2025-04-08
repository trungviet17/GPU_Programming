#include <cuda_runtime.h>
#include <iostream> 


__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }

}



void runVectorAdd(int N) {

    std::cout << "Running vector addition with N = " << N << std::endl;

    size_t size = N * sizeof(int);
    int* h_A, *h_B, *h_C;
    int* d_A, *d_B, *d_C;
    

    // 
    cudaMallocHost((void**)&h_A, size, cudaHostAllocMapped);
    cudaMallocHost((void**)&h_B, size, cudaHostAllocMapped);
    cudaMallocHost((void**)&h_C, size, cudaHostAllocMapped);


    cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0);
    cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0);
    cudaHostGetDevicePointer((void**)&d_C, (void*)h_C, 0);


    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
    }
    std::cout << "Vector addition completed successfully!" << std::endl;
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    std::cout << "CUDA device reset." << std::endl;

}

int main() {
    int N = 1024 * 1024; 
    runVectorAdd(N);
    return 0;
}
