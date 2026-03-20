#include <iostream>
#include <omp.h>

int main() {
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("thread %d / %d\n", tid, nthreads);
    }
    return 0;
}