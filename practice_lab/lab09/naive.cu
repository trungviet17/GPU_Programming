#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>

using namespace std;

template<typename T>
__global__ void matmul_kernel(const T *a, const T *b, T *c, int M, int N, int K) {
  int col = blockIdx.x * 32 + (threadIdx.x % 32);
  int row = blockIdx.y * 32 + (threadIdx.x / 32);
  if (row < M && col < N) {
    c[row * N + col] = 0;
    for (int k = 0; k < K; ++k) {
      c[row * N + col] += a[row * K + k] * b[k * N + col]; // each thread accesses () global memory 2N times
    }
  }
}

template<typename T>
__host__ void verifyResult(T *a, T *b, T *c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += a[i * K + k] * b[k * N + j];
      }
      //printf("sum: %d, c[%d * N + %d]: %d\n", sum, i, j, c[i * N + j]);
      
    }
  }
  cout << "Result is correct!\n";
}

template<typename T>
__host__ void copyFromHostToDevice(T *h_a, T *h_b, T *d_a, T *d_b, int M, int N, int K) {
  size_t a_bytes = M * K * sizeof(T);
  size_t b_bytes = K * N * sizeof(T);
  cudaError_t err = cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
  if (err !=  cudaSuccess) {
    fprintf(stderr, "Failed to copy h_a to d_a (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);
  if (err !=  cudaSuccess) {
    fprintf(stderr, "Failed to copy h_b to d_b (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void executeKernel(T *d_a, T *d_b, T *d_c, int M, int N, int K) {
  // block size is the multiple of 32
  int block_dim = 32;
  dim3 block(block_dim * block_dim);
  dim3 grid((M + block_dim - 1) / block_dim, (N + block_dim - 1) / block_dim);
  matmul_kernel<T><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void copyFromDeviceToHost(T *d_c, T *h_c, int M, int N) {
  size_t bytes = M * N * sizeof(T);
  cudaError_t err = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy d_c to h_c (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void deallocateMemory(T *d_a, T *d_b, T *d_c) {
  cudaError_t err = cudaFree(d_a);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free d_a (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaFree(d_b);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free d_b (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaFree(d_c);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free d_c (error code %s)", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void cleanUpDevice() {
  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to clean up device (error code %s)", cudaGetErrorString(err));
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

  // allocate memory on host side
  int *h_a = (int *)malloc(M * K * sizeof(int));
  int *h_b = (int *)malloc(K * N * sizeof(int));
  int *h_c = (int *)malloc(M * N * sizeof(int));

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
  int *d_a, *d_b, *d_c;
  cudaMalloc((int **)&d_a, M * K * sizeof(int));
  cudaMalloc((int **)&d_b, K * N * sizeof(int));
  cudaMalloc((int **)&d_c, M * N* sizeof(int));

  copyFromHostToDevice<int>(h_a, h_b, d_a, d_b, M, N, K);

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );

  executeKernel<int>(d_a, d_b, d_c, M, N, K);

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  cudaEventElapsedTime( &time, start, stop );
  printf("Time taken for GEMM: %f ms\n", time);
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  std::cout << "Performance: " << 2LL*N*N*N/(time * 1e-3 * 1e9) << " GFLOP/s\n";

  copyFromDeviceToHost<int>(d_c, h_c, M, N);
  verifyResult<int>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<int>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}