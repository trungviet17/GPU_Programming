#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
      }
      
    }
  }
  printf("Correct!");
}

// bM = 16, bN = 16, bK = 16 - non coalesced
// bM = 32, bN = 32, bK = 32 - coalesced
template<typename T, const size_t bM, const size_t bN, const size_t bK>
__global__ void gemm_kernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {


  // move blocktile to beginning of A's row and B's column
  const size_t cRow = blockIdx.y;
  const size_t cCol = blockIdx.x;

  d_a += cRow * bM * K;
  d_b += cCol * bN;
  d_c += cRow * bM * N + cCol * bN;

  // The total shared memory used is (bM * bK * 4 (bytes) + bK * bN * 4 (bytes))
  __shared__ T As[bM * bK];
  __shared__ T Bs[bK * bN];

  // At thread level
  const size_t threadCol = threadIdx.x % bN;
  const size_t threadRow = threadIdx.x / bN;

  T tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += bK) {
    // luu + sinh thread -> can  phai dong bo để  lưu dữ liệu
    As[threadRow * bK + threadCol] = d_a[threadRow * K + threadCol]; // is this coalesced? 32=yes, 16=no
    Bs[threadRow * bN + threadCol] = d_b[threadRow * N + threadCol];

    __syncthreads();

    d_a += bK;
    d_b += bK * N;

    for (size_t dotIdx = 0; dotIdx < bK; dotIdx++) {
      // load data từ share memory
      tmp += As[threadRow * bK + dotIdx] * Bs[dotIdx * bN + threadCol];
    }
    __syncthreads(); // barrier synchronization ~ synchronize threads within a block
    //printf("tmp: %f\n", tmp);
  }

  d_c[threadRow * N + threadCol] = tmp;
  //printf("d_c[%d * N + %d]: %f\n", threadRow, threadCol, d_c[threadRow * N + threadCol]);
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N , size_t K) {
  size_t a_bytes = sizeof(T) * M * K;
  size_t b_bytes = sizeof(T) * K * N;
  cudaError_t err = cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy h_a to d_a (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy h_b to d_b (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T, const size_t bM, const size_t bN, const size_t bK>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block(bM * bK, 1, 1);
  dim3 grid((M + bM - 1) / bM, (N + bN - 1) / bN, 1);
  // Ways to affect occupancy
  // 1. Changing template parameter affects the shared memory size (bM, bN, bK)
  // 2. Changing block size
  gemm_kernel<T, bM, bN, bK><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  cudaDeviceSynchronize();
}

template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  size_t c_bytes = sizeof(T) * M * N;
  cudaError_t err = cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy from d_c to h_c (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  cudaError_t err = cudaFree(d_a);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to deallocate d_a (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_b);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to deallocate d_b (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_c);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to deallocate d_c (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void cleanUpDevice() {
  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to clean up device (error code: %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    std::string value(argv[i+1]);
    i++;
    if (option.compare("-m") == 0) {
      M = std::stoi(value);
    }
    else if (option.compare("-n") == 0) {
      N = std::stoi(value);
    }
    else if (option.compare("-k") == 0) {
      K = std::stoi(value);
    }
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  std::tuple<int, int, int>parsedCmdLineArgsTuple = parseCmdLineArgs(argc, argv);
  int M = std::get<0>(parsedCmdLineArgsTuple);
  int N = std::get<1>(parsedCmdLineArgsTuple);
  int K = std::get<2>(parsedCmdLineArgsTuple);

  float* h_a = (float*)malloc(M * K * sizeof(float));
  float* h_b = (float*)malloc(K * N * sizeof(float));
  float* h_c = (float*)malloc(M * N * sizeof(float));

  // initialize
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      h_a[i * K + j] = rand() % 10;
    }
  }

  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      h_b[i * N + j] = rand() % 10;
    }
  }

  // allocate memory on device side
  float *d_a, *d_b, *d_c;
  cudaMalloc((float **)&d_a, M * K * sizeof(float));
  cudaMalloc((float **)&d_b, K * N * sizeof(float));
  cudaMalloc((float **)&d_c, M * N * sizeof(float));

  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );
  executeKernel<float, 32, 32, 32>(d_a, d_b, d_c, M, N, K);

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  cudaEventElapsedTime( &time, start, stop );
  printf("Time taken for GEMM: %f ms\n", time);
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  std::cout << "Performance: " << 2LL*M*N*K/(time * 1e-3 * 1e9) << " GFLOP/s\n";

  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}