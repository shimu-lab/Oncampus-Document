#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

static constexpr double PI = 3.1415926535897932384626433832795;

struct KPoint { double kx, ky; };

// ---------- model dispersion (without mu) ----------
static inline double eps_k(double kx, double ky, double t, double tp){
    return -2.0*t*(std::cos(kx) + std::cos(ky))
           -4.0*tp*(std::cos(kx)*std::cos(ky));
}

// ---------- stable fermi function f(x)=1/(e^x+1) ----------
static inline double fermi(double x){
    if (x >  50.0) return 0.0;
    if (x < -50.0) return 1.0;
    return 1.0 / (std::exp(x) + 1.0);
}

// centered k in (-pi, pi)
static inline double k_centered(int i, int Nk){
    return 2.0*PI*( (i + 0.5)/double(Nk) - 0.5 );
}

// ---------- estimate band edges (parallel) ----------
static void estimate_band_edges(double t, double tp, int Nk_edge, double &emin, double &emax){
    double loc_min = +1e300;
    double loc_max = -1e300;

#pragma omp parallel
    {
        double tmin = +1e300;
        double tmax = -1e300;

#pragma omp for collapse(2) nowait
        for(int iy=0; iy<Nk_edge; ++iy){
            for(int ix=0; ix<Nk_edge; ++ix){
                const double ky = k_centered(iy, Nk_edge);
                const double kx = k_centered(ix, Nk_edge);
                const double e  = eps_k(kx, ky, t, tp);
                if(e < tmin) tmin = e;
                if(e > tmax) tmax = e;
            }
        }

#pragma omp critical
        {
            if(tmin < loc_min) loc_min = tmin;
            if(tmax > loc_max) loc_max = tmax;
        }
    }

    emin = loc_min;
    emax = loc_max;
}

// ---------- filling n(mu) per site (spinful g=2) ----------
static double filling_n(double mu, double t, double tp, double T, int Nk, int g=2){
    const double beta = (T > 0.0) ? (1.0/T) : std::numeric_limits<double>::infinity();
    double sum = 0.0;

#pragma omp parallel for collapse(2) reduction(+:sum)
    for(int iy=0; iy<Nk; ++iy){
        for(int ix=0; ix<Nk; ++ix){
            const double ky = k_centered(iy, Nk);
            const double kx = k_centered(ix, Nk);
            const double e  = eps_k(kx, ky, t, tp);
            if(T > 0.0) sum += fermi(beta*(e - mu));
            else        sum += (e < mu) ? 1.0 : 0.0;
        }
    }

    return double(g) * sum / (double(Nk)*double(Nk));
}

// ---------- solve mu for target filling by bisection ----------
static double solve_mu_bisect(double n_target, double t, double tp, double T, int Nk,
                              int g=2, double tol_n=1e-10, int max_iter=200){
    if(n_target < 0.0 || n_target > double(g)){
        throw std::runtime_error("n_target out of range. For spinful g=2: n in [0,2].");
    }

    double emin, emax;
    estimate_band_edges(t, tp, 700, emin, emax);

    const double pad = (T > 0.0) ? (20.0*T) : 5.0;
    double mu_lo = emin - pad;
    double mu_hi = emax + pad;

    double n_lo = filling_n(mu_lo, t, tp, T, Nk, g);
    double n_hi = filling_n(mu_hi, t, tp, T, Nk, g);

    int widen = 0;
    while((n_lo > n_target || n_hi < n_target) && widen < 25){
        mu_lo -= 2.0*pad;
        mu_hi += 2.0*pad;
        n_lo = filling_n(mu_lo, t, tp, T, Nk, g);
        n_hi = filling_n(mu_hi, t, tp, T, Nk, g);
        widen++;
    }
    if(n_lo > n_target || n_hi < n_target){
        throw std::runtime_error("Failed to bracket mu. Try larger Nk or T>0.");
    }

    double mu_mid = 0.0;
    for(int it=0; it<max_iter; ++it){
        mu_mid = 0.5*(mu_lo + mu_hi);
        const double n_mid = filling_n(mu_mid, t, tp, T, Nk, g);
        const double err = n_mid - n_target;
        if(std::abs(err) < tol_n) return mu_mid;

        if(n_mid < n_target) mu_lo = mu_mid;
        else                 mu_hi = mu_mid;
    }
    return mu_mid;
}

// ---------- high-symmetry path Γ->X->M->Γ ----------
static std::vector<KPoint> make_path(int nseg){
    std::vector<KPoint> pts;
    pts.reserve(3*nseg + 1);
    auto add_seg = [&](KPoint a, KPoint b){
        for(int i=0;i<nseg;i++){
            const double s = double(i)/double(nseg);
            pts.push_back({a.kx*(1.0-s)+b.kx*s, a.ky*(1.0-s)+b.ky*s});
        }
    };
    const KPoint G{0.0,0.0}, X{PI,0.0}, M{PI,PI};
    add_seg(G,X);
    add_seg(X,M);
    add_seg(M,G);
    pts.push_back(G);
    return pts;
}

static inline double lerp_zero(double a, double b){
    const double denom = (b-a);
    if(std::abs(denom) < 1e-14) return 0.5;
    return -a/denom;
}

static std::string tag_num(double x){
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(4)<<x;
    std::string s=ss.str();
    for(char& c: s){
        if(c=='.') c='p';
        if(c=='-') c='m';
    }
    return s;
}

// ---------- DOS with Gaussian broadening (parallel, thread-local accumulate) ----------
static void compute_dos_gaussian(double t, double tp, int Nk, int g,
                                 int nE, double eta,
                                 const std::string& dos_dat){
    double emin, emax;
    estimate_band_edges(t, tp, 900, emin, emax);

    const double Epad = 6.0*eta;
    const double E0 = emin - Epad;
    const double E1 = emax + Epad;
    const double dE = (E1 - E0) / double(nE-1);

    std::vector<double> dos(nE, 0.0);
    const double norm_gauss = 1.0 / (std::sqrt(2.0*PI) * eta);

    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<double>> dos_local(nthreads, std::vector<double>(nE, 0.0));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto &buf = dos_local[tid];

#pragma omp for collapse(2) nowait
        for(int iy=0; iy<Nk; ++iy){
            for(int ix=0; ix<Nk; ++ix){
                const double ky = k_centered(iy, Nk);
                const double kx = k_centered(ix, Nk);
                const double e  = eps_k(kx, ky, t, tp);

                // accumulate contribution of this k-point to all energies
                for(int iE=0; iE<nE; ++iE){
                    const double E = E0 + dE*double(iE);
                    const double x = (E - e)/eta;
                    buf[iE] += norm_gauss * std::exp(-0.5*x*x);
                }
            }
        }
    }

    // reduce buffers
    for(int tid=0; tid<nthreads; ++tid){
        for(int iE=0; iE<nE; ++iE){
            dos[iE] += dos_local[tid][iE];
        }
    }

    const double pref = double(g) / (double(Nk)*double(Nk));
    std::ofstream ofs(dos_dat);
    ofs << std::setprecision(16);
    ofs << "# E  DOS(E)   (Gaussian eta="<<eta<<")\n";
    for(int iE=0; iE<nE; ++iE){
        const double E = E0 + dE*double(iE);
        ofs << E << " " << pref * dos[iE] << "\n";
    }
}

int main(int argc, char** argv){
    // ---- defaults ----
    double t  = 1.0;
    double tp = -0.2;

    double n_target = 1.0; // spinful: [0,2]
    double T = 0.02;
    int Nk_mu = 900;
    int Nk_fs = 700;
    int nseg  = 260;

    int nE = 1200;
    double eta = 0.02;

    // Usage:
    // ./tbpipe [n_target] [T] [Nk_mu] [Nk_fs] [eta] [t] [tp]
    if(argc >= 2) n_target = std::atof(argv[1]);
    if(argc >= 3) T        = std::atof(argv[2]);
    if(argc >= 4) Nk_mu    = std::atoi(argv[3]);
    if(argc >= 5) Nk_fs    = std::atoi(argv[4]);
    if(argc >= 6) eta      = std::atof(argv[5]);
    if(argc >= 7) t        = std::atof(argv[6]);
    if(argc >= 8) tp       = std::atof(argv[7]);

    if(Nk_mu < 80) Nk_mu = 80;
    if(Nk_fs < 80) Nk_fs = 80;
    if(nseg  < 50) nseg  = 50;
    if(nE    < 200) nE   = 200;
    if(eta <= 0.0) throw std::runtime_error("eta must be > 0.");
    if(T < 0.0)    throw std::runtime_error("T must be >= 0.");

    const int g = 2;
    const double tol_n = 1e-10;

    std::cout << std::setprecision(16);
    std::cout << "OpenMP threads = " << omp_get_max_threads() << "\n";
    std::cout << "params: t="<<t<<" tp="<<tp<<"  n_target="<<n_target
              << "  T="<<T<<"  Nk_mu="<<Nk_mu<<" Nk_fs="<<Nk_fs
              << "  eta="<<eta<<"\n";

    // 1) solve mu
    const double mu = solve_mu_bisect(n_target, t, tp, T, Nk_mu, g, tol_n, 200);
    const double n_check = filling_n(mu, t, tp, T, Nk_mu, g);

    std::cout << "mu_solved = " << mu << "\n";
    std::cout << "n(mu_solved) = " << n_check << "  err=" << (n_check - n_target) << "\n";

    // tags
    const std::string tag = "n"+tag_num(n_target)+"_T"+tag_num(T)+"_t"+tag_num(t)+"_tp"+tag_num(tp);
    const std::string mu_tag = "mu"+tag_num(mu);

    const std::string band_dat = "band_"+tag+"_"+mu_tag+".dat";
    const std::string fs_dat   = "fs_"+tag+"_"+mu_tag+".dat";
    const std::string dos_dat  = "dos_"+tag+"_eta"+tag_num(eta)+".dat";

    const std::string band_plt = "plot_band_"+tag+"_"+mu_tag+".plt";
    const std::string fs_plt   = "plot_fs_"+tag+"_"+mu_tag+".plt";
    const std::string dos_plt  = "plot_dos_"+tag+"_eta"+tag_num(eta)+".plt";

    // 2) band
    {
        std::ofstream ofs(band_dat);
        ofs<<std::setprecision(16);
        ofs<<"# s  E(k)=eps-mu  kx  ky\n";

        auto path = make_path(nseg);
        double sdist = 0.0;
        for(size_t i=0;i<path.size();++i){
            if(i>0){
                const double dx = path[i].kx - path[i-1].kx;
                const double dy = path[i].ky - path[i-1].ky;
                sdist += std::sqrt(dx*dx + dy*dy);
            }
            const double e = eps_k(path[i].kx, path[i].ky, t, tp) - mu;
            ofs<<sdist<<" "<<e<<" "<<path[i].kx<<" "<<path[i].ky<<"\n";
        }
    }
    // gnuplot for band
    {
        const double dGX = PI;
        const double dXM = PI;
        const double dMG = std::sqrt(2.0)*PI;
        const double sG0 = 0.0;
        const double sX  = sG0 + dGX;
        const double sM  = sX + dXM;
        const double sG1 = sM + dMG;

        std::ofstream ofs(band_plt);
        ofs<<"set terminal pngcairo size 1000,700\n";
        ofs<<"set output 'band_"<<tag<<"_"<<mu_tag<<".png'\n";
        ofs<<"set xlabel 'k-path'\n";
        ofs<<"set ylabel 'E(k) = eps(k) - mu'\n";
        ofs<<"set grid\n";
        ofs<<"set title 'Band (OpenMP): t-t\\047 square, "<<tag<<", "<<mu_tag<<"'\n";
        ofs<<"set arrow from "<<sX<<", graph 0 to "<<sX<<", graph 1 nohead dt 2\n";
        ofs<<"set arrow from "<<sM<<", graph 0 to "<<sM<<", graph 1 nohead dt 2\n";
        ofs<<"set xtics ('{/Symbol G}' "<<sG0<<", 'X' "<<sX<<", 'M' "<<sM<<", '{/Symbol G}' "<<sG1<<")\n";
        ofs<<"plot '"<<band_dat<<"' using 1:2 with lines title 'E(k)'\n";
    }

    // 3) FS extraction (parallel over cells; thread-local vectors)
    {
        const double dk = 2.0*PI/double(Nk_fs);
        auto idx = [&](int ix, int iy){ return ix + (Nk_fs+1)*iy; };
        std::vector<double> E((Nk_fs+1)*(Nk_fs+1), 0.0);
        const double eps0 = 1e-3;

        // build grid E(k)=eps-mu
#pragma omp parallel for collapse(2)
        for(int iy=0; iy<=Nk_fs; ++iy){
            for(int ix=0; ix<=Nk_fs; ++ix){
                const double ky = -PI + dk*double(iy);
                const double kx = -PI + dk*double(ix);
                E[idx(ix,iy)] = eps_k(kx,ky,t,tp) - mu;
            }
        }

        // per-thread buffers for FS points
        const int nthreads = omp_get_max_threads();
        std::vector<std::vector<KPoint>> fs_local(nthreads);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto &buf = fs_local[tid];
            buf.reserve(size_t(Nk_fs)*size_t(Nk_fs)/nthreads);

            auto add_edge = [&](double eA, double eB, double kxA, double kyA, double kxB, double kyB){
                if(std::abs(eA) < eps0){ buf.push_back({kxA,kyA}); return; }
                if(std::abs(eB) < eps0){ buf.push_back({kxB,kyB}); return; }
                if(eA*eB < 0.0){
                    const double s = lerp_zero(eA,eB);
                    buf.push_back({kxA + s*(kxB-kxA), kyA + s*(kyB-kyA)});
                }
            };

#pragma omp for collapse(2) nowait
            for(int iy=0; iy<Nk_fs; ++iy){
                for(int ix=0; ix<Nk_fs; ++ix){
                    const double ky0 = -PI + dk*double(iy);
                    const double ky1 = ky0 + dk;
                    const double kx0 = -PI + dk*double(ix);
                    const double kx1 = kx0 + dk;

                    const double e00 = E[idx(ix,  iy  )];
                    const double e10 = E[idx(ix+1,iy  )];
                    const double e01 = E[idx(ix,  iy+1)];
                    const double e11 = E[idx(ix+1,iy+1)];

                    add_edge(e00,e10,kx0,ky0,kx1,ky0);
                    add_edge(e01,e11,kx0,ky1,kx1,ky1);
                    add_edge(e00,e01,kx0,ky0,kx0,ky1);
                    add_edge(e10,e11,kx1,ky0,kx1,ky1);
                }
            }
        }

        // merge
        std::vector<KPoint> fs;
        size_t total = 0;
        for(auto &v: fs_local) total += v.size();
        fs.reserve(total);
        for(auto &v: fs_local){
            fs.insert(fs.end(), v.begin(), v.end());
        }

        std::ofstream ofs(fs_dat);
        ofs<<std::setprecision(16);
        ofs<<"# kx ky  (FS points: E(k)=0)\n";
        for(const auto& p: fs) ofs<<p.kx<<" "<<p.ky<<"\n";
    }

    // gnuplot for FS
    {
        std::ofstream ofs(fs_plt);
        ofs<<"set terminal pngcairo size 850,850\n";
        ofs<<"set output 'fs_"<<tag<<"_"<<mu_tag<<".png'\n";
        ofs<<"set size ratio -1\n";
        ofs<<"set xrange [-pi:pi]\n";
        ofs<<"set yrange [-pi:pi]\n";
        ofs<<"set xlabel 'k_x'\n";
        ofs<<"set ylabel 'k_y'\n";
        ofs<<"set grid\n";
        ofs<<"set title 'FS (OpenMP): "<<tag<<", "<<mu_tag<<"'\n";
        ofs<<"pi=3.141592653589793\n";
        ofs<<"set xtics (-pi, -pi/2, 0, pi/2, pi)\n";
        ofs<<"set ytics (-pi, -pi/2, 0, pi/2, pi)\n";
        ofs<<"plot '"<<fs_dat<<"' using 1:2 with points pt 7 ps 0.25 title 'FS points'\n";
    }

    // 4) DOS
    compute_dos_gaussian(t, tp, Nk_mu, g, nE, eta, dos_dat);

    // gnuplot for DOS
    {
        std::ofstream ofs(dos_plt);
        ofs<<"set terminal pngcairo size 1000,700\n";
        ofs<<"set output 'dos_"<<tag<<"_eta"<<tag_num(eta)<<".png'\n";
        ofs<<"set xlabel 'E'\n";
        ofs<<"set ylabel 'DOS(E)  (per site, spin included)'\n";
        ofs<<"set grid\n";
        ofs<<"set title 'DOS (OpenMP): "<<tag<<", eta="<<eta<<"'\n";
        ofs<<"plot '"<<dos_dat<<"' using 1:2 with lines title 'DOS'\n";
    }

    std::cout << "\n[done] wrote:\n"
              << "  mu = " << mu << "\n"
              << "  " << band_dat << "\n"
              << "  " << fs_dat   << "\n"
              << "  " << dos_dat  << "\n"
              << "  " << band_plt << "\n"
              << "  " << fs_plt   << "\n"
              << "  " << dos_plt  << "\n\n";

    std::cout << "Run gnuplot:\n"
              << "  gnuplot " << band_plt << "\n"
              << "  gnuplot " << fs_plt   << "\n"
              << "  gnuplot " << dos_plt  << "\n";

    return 0;
}
