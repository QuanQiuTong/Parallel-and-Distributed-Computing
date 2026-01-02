#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define TILE_WIDTH 16

// ================= Kernel 1: Naive =================
__global__ void MatrixMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ================= Kernel 2: Tiled =================
__global__ void MatrixMulTiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (col < N && tiledRow < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// ================= Benchmark Helper =================
template <typename Kernel>
float benchmark(Kernel kernel,
                dim3 grid, dim3 block,
                float* d_A, float* d_B, float* d_C,
                int N, int repeat = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

int main() {
    std::vector<int> sizes = {512, 1024, 2048, 4096};

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "MatrixMul CUDA Benchmark (RTX 3070 Ti)\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "N\tNaive(ms)\tTiled(ms)\tSpeedup\tTiled(GFLOPS)\n";

    for (int N : sizes) {
        size_t size = N * N * sizeof(float);

        // Host
        float* h_A = (float*)malloc(size);
        float* h_B = (float*)malloc(size);
        for (int i = 0; i < N * N; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 1.0f;
        }

        // Device
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                  (N + TILE_WIDTH - 1) / TILE_WIDTH);

        float tNaive = benchmark(MatrixMulNaive, grid, block, d_A, d_B, d_C, N);
        float tTiled = benchmark(MatrixMulTiled, grid, block, d_A, d_B, d_C, N);

        double flops = 2.0 * N * N * N;
        double gflops = flops / (tTiled * 1e6);

        std::cout << N << "\t"
                  << tNaive << "\t\t"
                  << tTiled << "\t\t"
                  << tNaive / tTiled << "\t"
                  << gflops << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
    }

    return 0;
}
