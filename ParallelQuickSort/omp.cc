#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
using namespace std;

// 快速排序核心
void quicksort(vector<int>& arr, int left, int right, int depth)
{
    if (left >= right) return;

    int i = left, j = right;
    int pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            swap(arr[i], arr[j]);
            i++; j--;
        }
    }

    // 使用 OpenMP 任务并行
    if (right - left > 1000) {  // 小区间直接串行，避免线程开销
        #pragma omp task shared(arr) if(depth < 4)
        quicksort(arr, left, j, depth + 1);

        #pragma omp task shared(arr) if(depth < 4)
        quicksort(arr, i, right, depth + 1);
    } else {
        quicksort(arr, left, j, depth + 1);
        quicksort(arr, i, right, depth + 1);
    }
}

// 封装接口
void parallel_quicksort(vector<int>& arr)
{
    #pragma omp parallel
    {
        #pragma omp single nowait
        quicksort(arr, 0, arr.size() - 1, 0);
    }
}

void ssort(vector<int>& arr, int left, int right)
{
    if (left >= right) return;

    int i = left, j = right;
    int pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            swap(arr[i], arr[j]);
            i++; j--;
        }
    }

    ssort(arr, left, j);
    ssort(arr, i, right);
}

// 随机生成数据
void generate(vector<int>& arr, int maxVal = 1000000)
{
    for (auto& x : arr)
        x = rand() % maxVal;
}

// 性能测试
void test(int N)
{
    vector<int> data(N);
    generate(data);
    vector<int> sdata = data;

    double start = omp_get_wtime();
    parallel_quicksort(data);
    double end = omp_get_wtime();
    cout << "N=" << N << ", time=" << (end - start) * 1000 << " ms" << endl;

    double sstart = omp_get_wtime();
    ssort(sdata, 0, sdata.size() - 1);
    double send = omp_get_wtime();
    cout << "N=" << N << ", serial time=" << (send - sstart) * 1000 << " ms" << endl;
    
    cout << "Speedup: " << (send - sstart) / (end - start) << "x" << endl;

    if (!is_sorted(data.begin(), data.end()))
        cerr << "❌ Sorting failed!\n";
}

int main()
{
    srand(time(0));

    omp_set_num_threads(omp_get_max_threads()); // 自动使用所有核

    cout << "Threads: " << omp_get_max_threads() << endl;
    for (int N : {1000, 5000, 10000, 100000, 1000000, 10000000, 100000000})
        test(N);

    return 0;
}
