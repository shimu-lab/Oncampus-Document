#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <limits>

using cd = std::complex<double>;

extern "C" {

void dsyevd_(
    char* jobz, char* uplo,
    int* n,
    double* a, int* lda,
    double* w,
    double* work, int* lwork,
    int* iwork, int* liwork,
    int* info
);

void zheevd_(
    char* jobz, char* uplo,
    int* n,
    cd* a, int* lda,
    double* w,
    cd* work, int* lwork,
    double* rwork, int* lrwork,
    int* iwork, int* liwork,
    int* info
);

}

template <class T>
inline T& A(std::vector<T>& a, int i, int j, int n) {
    return a[i + n * j];
}
template <class T>
inline const T& A(const std::vector<T>& a, int i, int j, int n) {
    return a[i + n * j];
}

double norm2(const std::vector<double>& x) {
    long double s = 0.0L;
    for (double v : x) s += (long double)v * (long double)v;
    return std::sqrt((double)s);
}
double norm2(const std::vector<cd>& x) {
    long double s = 0.0L;
    for (cd v : x) s += (long double)std::norm(v);
    return std::sqrt((double)s);
}

template <class T>
double residual_norm2(const std::vector<T>& Aorig,
                      const std::vector<T>& V,
                      const std::vector<double>& w,
                      int n, int j) {
    std::vector<T> r(n, T{});
    for (int i = 0; i < n; ++i) {
        T sum = T{};
        for (int k = 0; k < n; ++k) {
            sum += A(Aorig, i, k, n) * A(V, k, j, n);
        }
        r[i] = sum - (T)w[j] * A(V, i, j, n);
    }
    return norm2(r);
}

void diagonalize_hermitian(std::vector<cd>& a, int n, std::vector<double>& w) {
    w.assign(n, 0.0);

    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int info = 0;

    // Workspace query
    int lwork = -1, lrwork = -1, liwork = -1;
    cd work_query;
    double rwork_query;
    int iwork_query;

    zheevd_(&jobz, &uplo, &n,
            a.data(), &lda,
            w.data(),
            &work_query, &lwork,
            &rwork_query, &lrwork,
            &iwork_query, &liwork,
            &info);

    if (info != 0) {
        throw std::runtime_error("zheevd workspace query failed, info=" + std::to_string(info));
    }

    lwork  = (int)std::max(1.0, std::floor(work_query.real()));
    lrwork = (int)std::max(1.0, std::floor(rwork_query));
    liwork = std::max(1, iwork_query);

    std::vector<cd> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);

    // Actual computation
    zheevd_(&jobz, &uplo, &n,
            a.data(), &lda,
            w.data(),
            work.data(), &lwork,
            rwork.data(), &lrwork,
            iwork.data(), &liwork,
            &info);

    if (info != 0) {
        throw std::runtime_error("zheevd failed, info=" + std::to_string(info));
    }
}

std::vector<cd> build_example_hermitian_3x3(int& n_out) {
    const int n = 3;
    n_out = n;
    std::vector<cd> H(n * n, cd{0.0, 0.0});

    // H =
    // [ 1.0, 0.2+0.1i, 0.0
    //   0.2-0.1i, 2.0, 0.3
    //   0.0, 0.3, 3.0 ]
    A(H, 0, 0, n) = 1.0;
    A(H, 1, 1, n) = 2.0;
    A(H, 2, 2, n) = 3.0;

    cd x = cd(0.2, 0.1);
    A(H, 0, 1, n) = x;
    A(H, 1, 0, n) = std::conj(x);

    A(H, 1, 2, n) = 0.3;
    A(H, 2, 1, n) = 0.3;

    // (0,2) = 0
    return H;
}

int main() {
    try {
        int n = 0;
        std::vector<cd> H = build_example_hermitian_3x3(n);

        std::vector<cd> Horig = H;

        std::vector<double> w;
        diagonalize_hermitian(H, n, w);

        std::cout << "Eigenvalues (ascending):\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "  w[" << i << "] = " << w[i] << "\n";
        }

        std::cout << "\nEigenvectors (columns):\n";
        for (int j = 0; j < n; ++j) {
            std::cout << "v[" << j << "] = ( ";
            for (int i = 0; i < n; ++i) {
                std::cout << A(H, i, j, n) << " ";
            }
            std::cout << ")\n";
        }

        double max_res = 0.0;
        for (int j = 0; j < n; ++j) {
            double rj = residual_norm2(Horig, H, w, n, j);
            max_res = std::max(max_res, rj);
        }
        std::cout << "\nMax residual norm2: " << max_res << "\n";

        {
            std::ofstream ofs("eigvals.dat");
            ofs << "# i  w[i]\n";
            for (int i = 0; i < n; ++i) {
                ofs << i << " " << w[i] << "\n";
            }
        }

        {
            std::ofstream ofs("eigvecs.dat");
            ofs << "# columns are eigenvectors v_j; row i is component i\n";
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    cd v = A(H, i, j, n);
                    ofs << v.real() << " " << v.imag();
                    if (j + 1 < n) ofs << "   ";
                }
                ofs << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}