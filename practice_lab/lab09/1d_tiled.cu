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
      //if (i == 0 && j == 0) {
      //  printf("sum: %f, h_c[%d * K + %d]: %f\n", sum, i, j, h_c[i * K + j]);
      //}
      assert(h_c[i * N + j] == sum);
    }
  }
  printf("Correct!\n");
}

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
  }
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

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / TM);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  gemm_kernel<T, BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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

  executeKernel<float, 64, 64, 8, 8>(d_a, d_b, d_c, M, N, K);

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