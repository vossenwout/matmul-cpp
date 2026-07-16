#include "matmul/matrix.h"
#include <chrono>
#include <iostream>
#include <limits>

struct BenchmarkResults
{
    double best_run_time;
    double gflops;
};

const BenchmarkResults benchmark(const Matrix &a, const Matrix &b, std::size_t best_of)
{
    using Clock = std::chrono::steady_clock;
    double best_run_time = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < best_of; i++)
    {
        const auto start = Clock::now();
        Matrix c = a.matmul_ikj(b);
        // Matrix c = a.matmul_parallel_blocked_ikj(b, 512);
        const auto end = Clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (elapsed.count() < best_run_time)
        {
            best_run_time = elapsed.count();
        }
    }
    float gflops = (2 * a.shape()[0] * a.shape()[1] * b.shape()[1]) / 1e9;
    return BenchmarkResults({best_run_time, gflops});
}

void report_results(BenchmarkResults results)
{
    std::cout << "--- Results --- \n";
    std::cout << results.best_run_time << " s\n";
    std::cout << results.gflops << " gflops\n";
    double gflops_s = results.gflops / results.best_run_time;
    std::cout << std::to_string(gflops_s) + " glops/s\n";

    // cpu windows laptop
    // 96 Gflops per s/core
    const double cpu_gflops_s_core = 96;
    const double single_core_peak_p = 100 * (gflops_s / cpu_gflops_s_core);
    std::cout << std::to_string(single_core_peak_p) + "% single c peak";
    std::cout << "\n--- End Results --- \n";
}

int main()
{
    std::cout << "--- Execution started ---\n";

    // init matrices
    // Matrix a = Matrix(2, 2, {2, 2, 2, 2});
    // Matrix b = Matrix(2, 2, {2, 2, 2, 2});
    const std::size_t M = 1024;
    const std::size_t K = 1024;
    const std::size_t N = 1024;
    Matrix a = Matrix(M, K);
    Matrix b = Matrix(K, N);

    // benchmark matmul
    const BenchmarkResults results = benchmark(a, b, 3);
    report_results(results);
}
