#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>

// ==========================================
// 全局设置
// ==========================================

// 数据规模阈值：小于该值时改用串行快排
constexpr std::size_t CUTOFF = 2000;

// ==========================================
// 1. 快速排序核心算法
// ==========================================

std::size_t partition(int* data, std::size_t size) {
    int pivot = data[size - 1];
    std::size_t i = 0;

    for (std::size_t j = 0; j + 1 < size; ++j) {
        if (data[j] < pivot) {
            std::swap(data[i], data[j]);
            ++i;
        }
    }

    std::swap(data[i], data[size - 1]);
    return i;
}

void quickSortSerial(int* data, std::size_t size) {
    if (size <= 1) {
        return;
    }

    std::size_t pivot_pos = partition(data, size);
    quickSortSerial(data, pivot_pos);
    quickSortSerial(data + pivot_pos + 1, size - pivot_pos - 1);
}

void quickSortParallel(int* data, std::size_t size) {
    if (size <= 1) {
        return;
    }

    if (size < CUTOFF) {
        quickSortSerial(data, size);
        return;
    }

    std::size_t pivot_pos = partition(data, size);

    int* left_data  = data;
    std::size_t left_size = pivot_pos;

    int* right_data = data + pivot_pos + 1;
    std::size_t right_size = size - pivot_pos - 1;

    #pragma omp task shared(left_data, left_size)
    {
        quickSortParallel(left_data, left_size);
    }

    #pragma omp task shared(right_data, right_size)
    {
        quickSortParallel(right_data, right_size);
    }

    #pragma omp taskwait
}

// ==========================================
// 2. 辅助工具
// ==========================================

// 生成随机数据
std::vector<int> generateData(std::size_t size) {
    std::vector<int> data(size);

    std::mt19937 gen(42);  // 固定种子，保证可重复性
    std::uniform_int_distribution<> dis(1, 10'000'000);

    for (auto& x : data) {
        x = dis(gen);
    }
    return data;
}

// 运行一次测试并返回耗时（毫秒）
double runTest(std::vector<int> data, int threads) {
    omp_set_num_threads(threads);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quickSortParallel(data.data(), data.size());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ==========================================
// 3. 主函数（测试驱动）
// ==========================================

int main() {
    // 测试规模
    const std::vector<std::size_t> data_sizes = {
        1'000, 5'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000
    };

    // 线程数配置（i9-12900H）
    const std::vector<int> thread_counts = {
        1, 2, 4, 8, 12, 16, 20
    };

    std::cout << "=== OpenMP Parallel QuickSort Benchmark ===\n";
    std::cout << "CPU: i9-12900H (14C / 20T)\n";
    std::cout << "Cutoff Threshold: " << CUTOFF << "\n";
    std::cout << "-------------------------------------------\n";

    // 表头
    std::cout << std::left << std::setw(12) << "DataSize";
    for (int t : thread_counts) {
        std::cout << std::setw(12) << (std::to_string(t) + "T(ms)");
    }
    std::cout << '\n';

    // 运行测试
    for (auto n : data_sizes) {
        auto original_data = generateData(n);

        std::cout << std::left << std::setw(12) << n;

        for (int t : thread_counts) {
            double time_ms = runTest(original_data, t);
            std::cout << std::setw(12)
                      << std::fixed << std::setprecision(3)
                      << time_ms;
        }
        std::cout << '\n';
    }

    std::cout << "\nBenchmark Finished.\n";
    return 0;
}
