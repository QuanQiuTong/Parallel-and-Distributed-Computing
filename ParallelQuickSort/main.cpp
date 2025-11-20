#include <QtConcurrent/QtConcurrent>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QVector>
#include <QRandomGenerator>
#include <iostream>
#include <algorithm>


// 并行深度控制，防止线程爆炸
const int MAX_DEPTH = 3;  // 可根据CPU核数调整，例如 log2(QThread::idealThreadCount())

// 快速排序核心
void parallelQuickSort(QVector<int>& arr, int left, int right, int depth = 0)
{
    if (left >= right) return;

    int i = left, j = right;
    int pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (arr[i] < pivot) ++i;
        while (arr[j] > pivot) --j;
        if (i <= j) {
            qSwap(arr[i], arr[j]);
            ++i; --j;
        }
    }

    if (depth < MAX_DEPTH) {
        // 并行执行左右两部分
        QFuture<void> leftFuture  = QtConcurrent::run(parallelQuickSort, std::ref(arr), left, j, depth + 1);
        QFuture<void> rightFuture = QtConcurrent::run(parallelQuickSort, std::ref(arr), i, right, depth + 1);
        leftFuture.waitForFinished();
        rightFuture.waitForFinished();
    } else {
        // 递归到一定深度后改为串行
        parallelQuickSort(arr, left, j, depth + 1);
        parallelQuickSort(arr, i, right, depth + 1);
    }
}

void QuickSort(QVector<int>& arr, int left, int right)
{
    if (left >= right) return;

    int i = left, j = right;
    int pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (arr[i] < pivot) ++i;
        while (arr[j] > pivot) --j;
        if (i <= j) {
            qSwap(arr[i], arr[j]);
            ++i; --j;
        }
    }

    QuickSort(arr, left, j);
    QuickSort(arr, i, right);
}

// 测试函数
void testQuickSort(int N)
{
    QVector<int> data;
    data.reserve(N);
    for (int i = 0; i < N; ++i)
        data.append(QRandomGenerator::global()->bounded(1000000));

    QElapsedTimer timer;
    timer.start();

    parallelQuickSort(data, 0, N - 1);

    qint64 ms = timer.elapsed();
    std::cout << "N=" << N << " sorted in " << ms << " ms" << std::endl;

    timer.restart();

    QuickSort(data, 0, N - 1);

    qint64 serialMs = timer.elapsed();
    std::cout << "N=" << N << " serially sorted in " << serialMs << " ms" << std::endl;

    std::cout << "Speedup: " << static_cast<double>(serialMs) / ms << "x" << std::endl;

    // 校验结果正确性
    if (!std::is_sorted(data.begin(), data.end()))
        std::cerr << "❌ Sorting failed!" << std::endl;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    QList<int> testSizes = {1000, 5000, 10000, 100000, 1000000, 10000000};

    for (int N : testSizes)
        testQuickSort(N);

    return 0;
}
