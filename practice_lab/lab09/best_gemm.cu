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
      //printf("sum: %f, h_c[%d * K + %d]: %f\n", sum, i, j, h_c[i * K + j]);
      assert(h_c[i * N + j] == sum);
    }
  }
  printf("Correct!\n");
}

// BM = 128, BN = 128, BK = 8, TM = 8, TN = 8

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // Calculate thread local index within a block with respect to matrix C
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem - double buffers
  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // calculating the local indices w.r.t matrix A that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};
  
  // Initialize ping-pong buffer index
  uint bufferIdx = 0;
  
  // Pre-load the first tile before entering the main loop
  if (K >= BK) {
    reinterpret_cast<float4 *>(&As[bufferIdx][innerRowA * BK + innerColA * 4])[0] =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];

    reinterpret_cast<float4 *>(&Bs[bufferIdx][innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
  }
  
  // Advance pointers for next tile
  A += BK;     // move BK columns to right
  B += BK * N; // move BK rows down
  
  __syncthreads();

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Calculate next buffer index for double buffering
    uint nextBufferIdx = 1 - bufferIdx;
    
    // Pre-load next tile if available (except for the last iteration)
    if (bkIdx + BK < K) {
      reinterpret_cast<float4 *>(&As[nextBufferIdx][innerRowA * BK + innerColA * 4])[0] =
          reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];

      reinterpret_cast<float4 *>(&Bs[nextBufferIdx][innerRowB * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    }
    
    // Compute using current buffer while loading into the other buffer
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[bufferIdx][(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[bufferIdx][dotIdx * BN + threadCol * TN + i];
      }
      
      #pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll  
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    
    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    
    // Switch buffers
    bufferIdx = nextBufferIdx;
    
    // Synchronize threads before next iteration
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM+=1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN+=4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = threadResults[resIdxM * TN + resIdxN];
      tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
      tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
      tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
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

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
  gemm_kernel<T, BM, BN, BK, TM, TN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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

  executeKernel<float, 128, 128, 8, 8, 8>(d_a, d_b, d_c, M, N, K);

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