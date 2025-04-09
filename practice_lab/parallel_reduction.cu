#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>

#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void reduce_sum_kernel(float* a, float* sum, const int N)
{
  printf("Kernel\n");
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf(" tid: %d \n", tid);
  if (tid == 0) {
    float tmp_sum = 0;
    for (int i=0;i<N;++i)
      tmp_sum += a[i];
    sum[0] = tmp_sum;
  }
}

int main()
{
  const int N = 1000000;
  std::vector<float> a(N);
  for (auto &v : a)
    // v = 1.0 * (rand() % 10);
    v = 1;

  auto start = std::chrono::high_resolution_clock::now();
  float sum = std::accumulate(a.begin(), a.end(), 0);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "CPU compute time: " << duration_cpu.count() << " sum: " << sum << std::endl;

  float *a_gpu_ptr, *sum_gpu_ptr;
  float sum_gpu;
  CHECK_CUDA(cudaMalloc(&a_gpu_ptr, a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&sum_gpu_ptr, sizeof(float)));

  start = std::chrono::high_resolution_clock::now();
  CHECK_CUDA(cudaMemcpy(a_gpu_ptr, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block_size(64, 1, 1);
  dim3 grid_size(1, 1, 1);
  std::cout << "\nLaunch kernel" << std::endl;
  reduce_sum_kernel<<<grid_size, block_size>>>(a_gpu_ptr, sum_gpu_ptr, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(&sum_gpu, sum_gpu_ptr, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());
  stop = std::chrono::high_resolution_clock::now();
  auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "GPU compute time: " << duration_gpu.count() << " sum: " << sum_gpu << std::endl;
}
