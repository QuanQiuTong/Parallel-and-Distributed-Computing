#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#define TILE_WIDTH 16

// 错误检查宏
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ================= Kernel 1: Naive =================
// 使用 __restrict__ 告诉编译器指针不重叠，允许更多优化
__global__ void MatrixMulNaive(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ================= Kernel 2: Tiled =================
__global__ void MatrixMulTiled(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    
    // 循环遍历所有 Tile
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        // 协作加载数据到 Shared Memory
        int tiledCol = t * TILE_WIDTH + tx;
        int tiledRow = t * TILE_WIDTH + ty;

        // 边界检查与 Zero Padding
        if (row < N && tiledCol < N)
            As[ty][tx] = A[row * N + tiledCol];
        else
            As[ty][tx] = 0.0f;

        if (col < N && tiledRow < N)
            Bs[ty][tx] = B[tiledRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads(); // 等待所有线程加载完毕

        // 计算当前 Tile 的点积
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads(); // 等待计算完毕，准备加载下一个 Tile
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ================= Helper: CPU Verification =================
// 简单验证逻辑：随机抽取几行进行 CPU 计算对比
void verify_result(float* h_A, float* h_B, float* h_C_GPU, int N) {
    float max_diff = 0.0f;
    int check_rows = 5; // 只检查前5行，节省时间
    
    for (int i = 0; i < check_rows; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            float diff = fabs(h_C_GPU[i * N + j] - sum);
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    if (max_diff < 1e-3) 
        std::cout << "[PASS]";
    else 
        std::cout << "[FAIL] Max Diff: " << max_diff;
}

// ================= Benchmark Helper =================
template <typename Kernel>
float benchmark(Kernel kernel, dim3 grid, dim3 block,
                float* d_A, float* d_B, float* d_C,
                int N, int repeat = 10) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

int main() {
    std::vector<int> sizes = {512, 1024, 2048, 4096};

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MatrixMul CUDA Benchmark (RTX 3070 Ti)\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "N\tNaive(ms)\tTiled(ms)\tSpeedup\tTiled(GFLOPS)\tCheck\n";

    for (int N : sizes) {
        size_t size = N * N * sizeof(float);

        // Host Memory
        float* h_A = (float*)malloc(size);
        float* h_B = (float*)malloc(size);
        float* h_C = (float*)malloc(size); // 用于回传结果校验

        // Initialize
        for (int i = 0; i < N * N; i++) {
            h_A[i] = 1.0f; // 简化为全1，方便肉眼看
            h_B[i] = 0.01f; 
        }

        // Device Memory
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc(&d_A, size));
        CHECK(cudaMalloc(&d_B, size));
        CHECK(cudaMalloc(&d_C, size));

        CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        // Config
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                  (N + TILE_WIDTH - 1) / TILE_WIDTH);

        // Run Benchmark
        float tNaive = benchmark(MatrixMulNaive, grid, block, d_A, d_B, d_C, N);
        float tTiled = benchmark(MatrixMulTiled, grid, block, d_A, d_B, d_C, N);

        // Copy back for verification
        CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

        double flops = 2.0 * N * N * N;
        double gflops = flops / (tTiled * 1e6);

        std::cout << N << "\t"
                  << tNaive << "\t\t"
                  << tTiled << "\t\t"
                  << tNaive / tTiled << "\t"
                  << gflops << "\t\t";
        
        verify_result(h_A, h_B, h_C, N);
        std::cout << "\n";

        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C);
    }

    return 0;
}