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
// 阈值：低于此大小时不再创建新线程，改为串行
// 在 i9-12900H 上，建议 1000-5000 之间
const int CUTOFF = 2000; 

// ==========================================
// 1. 快速排序核心算法
// ==========================================

// 标准 Partition (分区) 函数
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high]; 
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// 串行快速排序 (用于小数据量递归终点)
void quickSortSerial(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSortSerial(arr, low, pi - 1);
        quickSortSerial(arr, pi + 1, high);
    }
}

// OpenMP 并行快速排序
void quickSortParallel(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        // 优化：如果数据块很小，直接串行处理，避免任务调度开销
        if (high - low < CUTOFF) {
            quickSortSerial(arr, low, high);
            return;
        }

        int pi = partition(arr, low, high);

        // 创建并行任务
        #pragma omp task shared(arr) firstprivate(low, pi)
        {
            quickSortParallel(arr, low, pi - 1);
        }

        #pragma omp task shared(arr) firstprivate(high, pi)
        {
            quickSortParallel(arr, pi + 1, high);
        }
        
        // 等待子任务完成
        #pragma omp taskwait
    }
}

// ==========================================
// 2. 辅助工具
// ==========================================

// 生成随机数据
std::vector<int> generateData(size_t size) {
    std::vector<int> data(size);
    // 使用固定种子以确保每次测试的数据是一样的，保证公平性
    std::mt19937 gen(42); 
    std::uniform_int_distribution<> dis(1, 10000000);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// 运行测试并返回耗时(ms)
double runTest(std::vector<int> data, int threads) {
    omp_set_num_threads(threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 开启并行区域
    #pragma omp parallel
    {
        // 只由一个主线程发起第一次调用
        #pragma omp single nowait
        {
            quickSortParallel(data, 0, data.size() - 1);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // 简单的正确性校验
    /* if (!std::is_sorted(data.begin(), data.end())) {
        std::cerr << "Error: Sort failed!" << std::endl;
    } */
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ==========================================
// 3. 主函数 (测试驱动)
// ==========================================
int main() {
    // 1. 定义测试规模 (题目要求: 1K, 5K, 10K, 100K)
    auto data_sizes = {1'000, 5'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000};
    
    // 2. 定义线程数 (适配 i9-12900H: 1-20线程)
    auto thread_counts = {1, 2, 4, 8, 12, 16, 20};

    std::cout << "=== OpenMP Parallel QuickSort Benchmark ===" << std::endl;
    std::cout << "CPU: i9-12900H (14 Cores / 20 Threads)" << std::endl;
    std::cout << "Cutoff Threshold: " << CUTOFF << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // 打印表头
    std::cout << std::left << std::setw(10) << "DataSize";
    for (int t : thread_counts) {
        std::string h = std::to_string(t) + "T(ms)";
        std::cout << std::setw(12) << h;
    }
    std::cout << std::endl;

    // 开始循环测试
    for (size_t n : data_sizes) {
        std::vector<int> original_data = generateData(n);
        
        std::cout << std::left << std::setw(10) << n;
        
        // 遍历不同线程数
        for (int t : thread_counts) {
            // 复制数据，避免排序对下一次测试造成影响
            double time_ms = runTest(original_data, t);
            std::cout << std::setw(12) << std::fixed << std::setprecision(3) << time_ms;
        }
        std::cout << std::endl;
    }

    std::cout << "\nBenchmark Finished." << std::endl;
    return 0;
}