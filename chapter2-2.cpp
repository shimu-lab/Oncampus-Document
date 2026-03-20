#include <fftw3.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

static inline int idx(int ix, int iy, int Ny) { return ix * Ny + iy; }

int main(int argc, char** argv) {
    int Nx = 128;
    int Ny = 128;
    double gamma = 6.0;

    if (argc >= 3) {
        Nx = std::atoi(argv[1]);
        Ny = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        gamma = std::atof(argv[3]);
    }
    if (Nx <= 0 || Ny <= 0) {
        std::cerr << "Nx, Ny must be positive.\n";
        return 1;
    }
    if (gamma <= 0.0) {
        std::cerr << "gamma must be positive.\n";
        return 1;
    }

    fftw_complex* in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    if (!in || !out) {
        std::cerr << "fftw_malloc failed.\n";
        fftw_free(in);
        fftw_free(out);
        return 1;
    }

    for (int ix = 0; ix < Nx; ++ix) {
        const double x = ix - Nx / 2.0;  // 座標 : [-Nx/2, ..., Nx/2)
        for (int iy = 0; iy < Ny; ++iy) {
            const double y = iy - Ny / 2.0;

            const double val = (gamma * gamma) / (x * x + y * y + gamma * gamma);

            // シフトして(0,0)に中心を置く
            const int ix0 = (ix + Nx / 2) % Nx;
            const int iy0 = (iy + Ny / 2) % Ny;

            in[idx(ix0, iy0, Ny)][0] = val;  // 実部
            in[idx(ix0, iy0, Ny)][1] = 0.0;  // 虚部
        }
    }

    // FFTW plan の作成及び実行
    fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) {
        std::cerr << "fftw_plan_dft_2d failed\n";
        fftw_free(in);
        fftw_free(out);
        return 1;
    }
    fftw_execute(plan);

    {
        std::ofstream ofs("lorentzian_real.dat");
        ofs << "# x y f(x,y)\n";
        for (int x = -Nx / 2; x < Nx / 2; ++x) {
            // x に対応する元の ix
            const int ix = x + Nx / 2;
            for (int y = -Ny / 2; y < Ny / 2; ++y) {
                const int iy = y + Ny / 2;

                // in[] は前向きにシフトして格納したので、逆シフトで読む
                const int ix0 = (ix + Nx / 2) % Nx;
                const int iy0 = (iy + Ny / 2) % Ny;

                ofs << x << " " << y << " " << in[idx(ix0, iy0, Ny)][0] << "\n";
            }
            ofs << "\n";
        }
    }

    // ------------------------------------------------------------
    // 3) k空間: FFTの結果と、連続変換の厳密解を「同じ格子点」で比較して出力
    //
    // FFTWの周波数:
    //   kx = 2π * mx / Nx,   mx = -Nx/2..Nx/2-1（表示用）
    //   ky = 2π * my / Ny
    //
    // 連続変換の厳密解（無限平面）:
    //   f(r) = gamma^2/(r^2+gamma^2)
    //   F(k) = 2π * gamma^2 * K0(gamma*|k|)
    //
    // 注意: k=0 は対数発散（無限大）なので比較不能。
    //       ここでは NaN を出しておく（gnuplotで弾ける）
    // ------------------------------------------------------------
    {
        std::ofstream ofs("lorentzian_k_fft_exact.dat");
        ofs << "# mx my kx ky |F_fft| Re(F_fft) Im(F_fft) F_exact\n";

        const double pi = M_PI;

        for (int sx = 0; sx < Nx; ++sx) {
            const int mx = sx - Nx / 2;                 // 表示用（fftshifted）
            const int ix = (sx + Nx / 2) % Nx;          // 実際のFFTW index
            const double kx = 2.0 * pi * mx / Nx;       // 物理波数（dx=1）

            for (int sy = 0; sy < Ny; ++sy) {
                const int my = sy - Ny / 2;
                const int iy = (sy + Ny / 2) % Ny;
                const double ky = 2.0 * pi * my / Ny;

                const double re = out[idx(ix, iy, Ny)][0];
                const double im = out[idx(ix, iy, Ny)][1];
                const double mag = std::sqrt(re * re + im * im);

                const double kabs = std::sqrt(kx * kx + ky * ky);

                double exact;
                if (kabs < 1e-14) {
                    exact = std::numeric_limits<double>::quiet_NaN();  // 発散点はNaN
                } else {
                    exact = 2.0 * pi * gamma * gamma * std::cyl_bessel_k(0, gamma * kabs);
                }

                ofs << mx << " " << my << " "
                    << kx << " " << ky << " "
                    << mag << " " << re << " " << im << " "
                    << exact << "\n";
            }
            ofs << "\n";
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    std::cout << "Done.\n"
              << "Outputs:\n"
              << "  - lorentzian_real.dat\n"
              << "  - lorentzian_k_fft_exact.dat\n";
    return 0;
}
