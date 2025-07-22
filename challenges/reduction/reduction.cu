#include <cuda_runtime.h>

__global__ void block_sum(float* input, float* output, int N) { 
    extern __shared__ float partial_sum[]; 

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int tid = threadIdx.x; 

    partial_sum[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads(); 


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = partial_sum[0]; 
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 1024; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    size_t sharedMemBytes = threadsPerBlock * sizeof(float);

    float *input_d; 
    cudaMalloc((void**)&input_d, sizeof(float) * N); 
    cudaMemcpy(input_d, input, sizeof(float) * N, cudaMemcpyHostToDevice); 

    float *temp_output_h = (float*)malloc(sizeof(float) * blocksPerGrid); 
    float *temp_output_d; 
    cudaMalloc((void**)&temp_output_d, sizeof(float) * blocksPerGrid); 

    block_sum<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(input_d, temp_output_d, N);

    cudaDeviceSynchronize(); 
    cudaMemcpy(temp_output_h, temp_output_d, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost); 

    float sum = 0.0f; 
    for (int i = 0; i < blocksPerGrid; ++i) sum += temp_output_h[i]; 

    cudaMemcpy(output, &sum, sizeof(float), cudaMemcpyHostToDevice); 

    cudaFree(temp_output_d); 
    free(temp_output_h); 
}