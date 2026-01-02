#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

template<int TILE>
__global__ void MatrixMulTiled(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE + tx;
        int tiledRow = t * TILE + ty;

        As[ty][tx] = (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0f;
        Bs[ty][tx] = (col < N && tiledRow < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

template <typename Kernel>
float benchmark(Kernel kernel, dim3 grid, dim3 block,
                float* d_A, float* d_B, float* d_C, int N) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Warm-up
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    for (int i = 0; i < 10; i++)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms / 10;
}

int main() {
    int N = 2048;
    size_t size = N * N * sizeof(float);

    float* h = (float*)malloc(size);
    for (int i = 0; i < N * N; i++) h[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h, size, cudaMemcpyHostToDevice);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Sweep TILE_WIDTH (N=" << N << ")\n";
    std::cout << "---------------------------------\n";
    std::cout << "TileWidth\tTime(ms)\tGFLOPS\n";

    // Block size 8x8
    {
        dim3 block(8, 8);
        dim3 grid((N + 7) / 8, (N + 7) / 8);
        float t = benchmark(MatrixMulTiled<8>, grid, block, d_A, d_B, d_C, N);
        std::cout << "8\t\t" << t << "\t\t" << (2.0 * N * N * N) / (t * 1e6) << "\n";
    }
    // Block size 16x16
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        float t = benchmark(MatrixMulTiled<16>, grid, block, d_A, d_B, d_C, N);
        std::cout << "16\t\t" << t << "\t\t" << (2.0 * N * N * N) / (t * 1e6) << "\n";
    }
    // Block size 32x32
    {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (N + 31) / 32);
        float t = benchmark(MatrixMulTiled<32>, grid, block, d_A, d_B, d_C, N);
        std::cout << "32\t\t" << t << "\t\t" << (2.0 * N * N * N) / (t * 1e6) << "\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h);
    return 0;
}