#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Define error checking macro
#define CHECK_LAST_CUDA_ERROR() { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(-1); \
    } \
}

template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
            cout << "Thread (" << C_row_idx << ", " << C_col_idx
                 << ") computes: " << A[C_row_idx * lda + k_idx] << " * "
                 << B[k_idx * ldb + C_col_idx] << " = " << sum << endl;
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

int main() {
    // Define matrix dimensions
    const size_t m = 4; // Rows of A and C
    const size_t n = 4; // Columns of B and C
    const size_t k = 4; // Columns of A and rows of B
    
    // Leading dimensions
    const size_t lda = k;
    const size_t ldb = n;
    const size_t ldc = n;
    
    // Host matrices
    float *h_A = new float[m * k];
    float *h_B = new float[k * n];
    float *h_C = new float[m * n];
    
    // Initialize matrices with simple values for verification
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            h_A[i * lda + j] = static_cast<float>(i + j + 1);
        }
    }
    
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            h_B[i * ldb + j] = static_cast<float>(i + j + 1);
        }
    }
    
    // Initialize C with zeros
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            h_C[i * ldc + j] = 0.0f;
        }
    }
    
    // Print input matrices
    std::cout << "Matrix A:" << std::endl;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            std::cout << h_A[i * lda + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nMatrix B:" << std::endl;
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << h_B[i * ldb + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Set alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    float *d_alpha, *d_beta;
    
    // Allocate device memory
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMalloc(&d_alpha, sizeof(float));
    cudaMalloc(&d_beta, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, &beta, sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    launch_gemm_kernel_v00<float>(m, n, k, d_alpha, d_A, lda, d_B, ldb, d_beta, d_C, ldc, stream);
    cudaEventRecord(stop, stream);
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the result matrix
    std::cout << "\nResult Matrix C:" << std::endl;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << h_C[i * ldc + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nKernel execution time: " << milliseconds << " ms" << std::endl;
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    
    return 0;
}




