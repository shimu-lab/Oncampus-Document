#include <iostream>
#include <omp.h>

int main() {
    int x = 0;

#pragma omp parallel
    {
        x += 1;
    }

    std::cout << x << std::endl;
}
