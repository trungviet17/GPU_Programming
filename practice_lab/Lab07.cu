#include <cuda_runtime.h>
#include <iostream> 


__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) { 
        C[i] = A[i] + B[i]; 
    }

}



void runVectorAdd(int N) {

    size_t size = N * sizeof(int); 
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;


    cudaMallocHost((void**)&h_a, size);
    cudaMallocHost((void**)&h_b, size);
    cudaMallocHost((void**)&h_c, size);

    cudaHostGetDevicePointer((void**)&d_a, h_a, 0);
    cudaHostGetDevicePointer((void**)&d_b, h_b, 0);
    cudaHostGetDevicePointer((void**)&d_c, h_c, 0);


    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for (int i = 0; i < N; i++ ) {
        h_a[i] = i;
        h_b[i] = i;
    }


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();


    
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << std::endl;
            break;
        }
    }
    std::cout << "Vector addition completed successfully." << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

}

int main() {
    int N = 10 * 10; 
    runVectorAdd(N);
    return 0;
}
