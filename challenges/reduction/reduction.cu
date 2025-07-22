#include <cuda_runtime.h>
#include <iostream>
#include <vector>


#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void ReduceKernel(const float *input, float *output, int N)
{
    __shared__ float Shared[BLOCK_DIM];

    const int Tid = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    const int Tx = threadIdx.x;

    Shared[Tx] = 0.0f;
    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        if (Tid + blockDim.x * i < N)
        {
            Shared[Tx] += input[Tid + blockDim.x * i];
        }
    }
    __syncthreads();

    for (int Stride = blockDim.x / 2; Stride > 0; Stride /= 2)
    {
        if (Tx < Stride)
        {
            Shared[Tx] += Shared[Tx + Stride];
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        atomicAdd(output, Shared[0]);
    }
}


extern "C" void solve(const float* input, float* output, int N) {  

    const int BlockDim = BLOCK_DIM;
    const int GridDim = (N + BlockDim - 1) / BlockDim;
    ReduceKernel<<<GridDim, BlockDim>>>(input, output, N);
}