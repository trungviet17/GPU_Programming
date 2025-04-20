#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Warm-up kernel
__global__ void warmupKernel() {
    // No operation, just for initialization
}

// TODO: Fuse kernels
__global__ void fusedKernel(uchar3* input, unsigned char* output, int width, int height,
                          float* sobelXKernel, float* sobelYKernel) {
    // Kernel fusion logic goes here


    
}







// RGB to Grayscale conversion kernel
__global__ void rgbToGrayscaleKernel(uchar3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = input[idx];
        output[idx] = static_cast<unsigned char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    }
}

// Convolution kernel
__global__ void convolutionKernel(unsigned char* input, unsigned char* output,
                                int width, int height, float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;
    float result = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int px = x + kx;
            int py = y + ky;

            px = max(0, min(px, width - 1));
            py = max(0, min(py, height - 1));

            float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
            result += static_cast<float>(input[py * width + px]) * kernelValue;
        }
    }

    output[y * width + x] = static_cast<unsigned char>(min(255.0f, max(0.0f, abs(result))));
}

// Edge combination kernel
__global__ void combineEdgesKernel(unsigned char* horizontal, unsigned char* vertical,
                                 unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float combined = sqrtf(static_cast<float>(horizontal[idx]) * horizontal[idx] +
                        static_cast<float>(vertical[idx]) * vertical[idx]);
        output[idx] = static_cast<unsigned char>(min(255.0f, max(0.0f, combined)));
    }
}
// TODO-END

float measureKernels(cv::Mat& inputImage, cv::Mat& outputImage, int niter=1000) {
    // Convert to 3-channel if needed
    cv::Mat input3Channel;
    if (inputImage.channels() == 4) {
        cvtColor(inputImage, input3Channel, cv::COLOR_BGRA2BGR);
    } else if (inputImage.channels() == 1) {
        cvtColor(inputImage, input3Channel, cv::COLOR_GRAY2BGR);
    } else {
        input3Channel = inputImage.clone();
    }

    const int width = input3Channel.cols;
    const int height = input3Channel.rows;
    const int imageSize = width * height * sizeof(unsigned char);

    std::cout << "Edge detection on image " << width << "x" << height << std::endl;

    // Allocate device memory
    uchar3* d_input = nullptr;
    unsigned char* d_gray = nullptr;
    unsigned char* d_sobelX = nullptr;
    unsigned char* d_sobelY = nullptr;
    unsigned char* d_edges = nullptr;
    float* d_sobelXKernel = nullptr;
    float* d_sobelYKernel = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, width * height * sizeof(uchar3)));
    CHECK_CUDA(cudaMalloc(&d_gray, imageSize));
    CHECK_CUDA(cudaMalloc(&d_sobelX, imageSize));
    CHECK_CUDA(cudaMalloc(&d_sobelY, imageSize));
    CHECK_CUDA(cudaMalloc(&d_edges, imageSize));
    CHECK_CUDA(cudaMalloc(&d_sobelXKernel, 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sobelYKernel, 9 * sizeof(float)));

    // Copy input image to device
    CHECK_CUDA(cudaMemcpy(d_input, input3Channel.data, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Define Sobel kernels
    float sobelXKernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobelYKernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    CHECK_CUDA(cudaMemcpy(d_sobelXKernel, sobelXKernel, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sobelYKernel, sobelYKernel, 9 * sizeof(float), cudaMemcpyHostToDevice));

    // Set up block and grid dimensions
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y);

    // Warm-up the GPU
    warmupKernel<<<1, 1>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> all_ms;
    for (int _=0;_<niter;++_) {
      float milliseconds = 0.0f;

      // Record start time
      CHECK_CUDA(cudaEventRecord(start));

      // TODO execute fused kernel only
      // Execute kernels
      rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_input, d_gray, width, height);
      convolutionKernel<<<gridSize, blockSize>>>(d_gray, d_sobelX, width, height, d_sobelXKernel, 3);
      convolutionKernel<<<gridSize, blockSize>>>(d_gray, d_sobelY, width, height, d_sobelYKernel, 3);
      combineEdgesKernel<<<gridSize, blockSize>>>(d_sobelX, d_sobelY, d_edges, width, height);
      // TODO-END

      // Record stop time and synchronize
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

      // Check for kernel errors
      CHECK_CUDA(cudaGetLastError());

      all_ms.push_back(milliseconds);
    }

    // Allocate host memory for result
    std::vector<unsigned char> edgeImage(width * height);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(edgeImage.data(), d_edges, imageSize, cudaMemcpyDeviceToHost));

    // Create output image
    outputImage.create(height, width, CV_8UC1);
    memcpy(outputImage.data, edgeImage.data(), imageSize);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_gray));
    CHECK_CUDA(cudaFree(d_sobelX));
    CHECK_CUDA(cudaFree(d_sobelY));
    CHECK_CUDA(cudaFree(d_edges));
    CHECK_CUDA(cudaFree(d_sobelXKernel));
    CHECK_CUDA(cudaFree(d_sobelYKernel));

    std::sort(all_ms.begin(), all_ms.end());
    return all_ms[niter / 2];
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load input image
    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Could not open or find the image: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat edgeImage;
    float kernelTime = measureKernels(inputImage, edgeImage);

    std::cout << "Median kernel execution time: " << kernelTime << " ms" << std::endl;

    if (!cv::imwrite(argv[2], edgeImage)) {
        std::cerr << "Failed to save image: " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Edge detection completed successfully. Output saved to: " << argv[2] << std::endl;

    return EXIT_SUCCESS;
}