// reduction_demo.cpp
#include <omp.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    // 反復回数（十分大きい方が差が出やすい）
    // 例: ./reduction_demo 50000000
    long long N = 50'000'000LL;
    if (argc >= 2) {
        N = std::atoll(argv[1]);
        if (N <= 0) {
            std::cerr << "N must be positive.\n";
            return 1;
        }
    }

    // OpenMPのスレッド数（環境変数 OMP_NUM_THREADS が優先されるのが普通）
    int max_threads = omp_get_max_threads();

    std::cout << "N = " << N << "\n";
    std::cout << "omp_get_max_threads() = " << max_threads << "\n";

    // ---- (A) reduction なし：データ競合が起きる ----
    long long sum_race = 0;
    double t0 = omp_get_wtime();

#pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        // 競合しやすい典型：共有変数に +=
        sum_race += 1;
    }

    double t1 = omp_get_wtime();

    // ---- (B) reduction あり：正しい ----
    long long sum_reduction = 0;
    double t2 = omp_get_wtime();

#pragma omp parallel for reduction(+:sum_reduction)
    for (long long i = 0; i < N; ++i) {
        sum_reduction += 1;
    }

    double t3 = omp_get_wtime();

    // ---- (C) 参考：atomic（正しいが遅くなりがち）----
    long long sum_atomic = 0;
    double t4 = omp_get_wtime();

#pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
#pragma omp atomic
        sum_atomic += 1;
    }

    double t5 = omp_get_wtime();

    // 期待値（逐次なら必ず N）
    const long long expected = N;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n[No reduction (race)]\n";
    std::cout << "  sum = " << sum_race << " (expected " << expected << ")\n";
    std::cout << "  error = " << (expected - sum_race) << "\n";
    std::cout << "  time = " << (t1 - t0) << " sec\n";

    std::cout << "\n[Reduction]\n";
    std::cout << "  sum = " << sum_reduction << " (expected " << expected << ")\n";
    std::cout << "  error = " << (expected - sum_reduction) << "\n";
    std::cout << "  time = " << (t3 - t2) << " sec\n";

    std::cout << "\n[Atomic (reference)]\n";
    std::cout << "  sum = " << sum_atomic << " (expected " << expected << ")\n";
    std::cout << "  error = " << (expected - sum_atomic) << "\n";
    std::cout << "  time = " << (t5 - t4) << " sec\n";

    std::cout << "\nTip: run multiple times; the 'race' result often changes.\n";
    return 0;
}
