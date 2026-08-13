#include "native_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
inline bool is_finite(double x) { return std::isfinite(x); }
inline double qnan() { return std::numeric_limits<double>::quiet_NaN(); }

inline double stable_var(double sum, double sum2, long long n) {
    if (n <= 0) return qnan();
    double v = (sum2 - (sum * sum) / static_cast<double>(n)) / static_cast<double>(n);
    if (v < 0.0 && v > -1e-14 * (std::abs(sum2) + 1.0)) v = 0.0;
    return v;
}
}

void feature_bank_cpu(
    const double* z, std::size_t B, std::size_t T,
    const int32_t* spans, std::size_t S,
    int spectral_window, int min_periods,
    const int32_t* horizons, std::size_t H,
    double* psi, double* past, double* future) {

    const double NaN = qnan();
    const std::size_t BS = B * S;

    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t task = 0; task < static_cast<std::ptrdiff_t>(BS); ++task) {
        const std::size_t b = static_cast<std::size_t>(task) / S;
        const std::size_t s = static_cast<std::size_t>(task) % S;
        const int span = spans[s];
        const double nu = (static_cast<double>(span) - 1.0) / (static_cast<double>(span) + 1.0);
        const double alpha = 1.0 - nu;
        const double scale = nu / (1.0 - nu);
        const double* zr = z + b * T;
        double* out = psi + (b * S + s) * T;

        std::vector<double> lam_prev(T, NaN);
        double ewm = NaN;
        long long ewm_count = 0;
        for (std::size_t t = 0; t < T; ++t) {
            if (ewm_count >= span && is_finite(ewm)) lam_prev[t] = ewm;
            const double x = zr[t];
            if (is_finite(x)) {
                if (!is_finite(ewm)) ewm = x;
                else ewm = alpha * x + nu * ewm;
                ++ewm_count;
            }
        }

        double sumz = 0.0, sumz2 = 0.0;
        long long nz = 0;
        double sx = 0.0, sy = 0.0, sxy = 0.0;
        long long np = 0;

        for (std::size_t t = 0; t < T; ++t) {
            const double x = zr[t];
            const double y = lam_prev[t];
            if (is_finite(x)) { sumz += x; sumz2 += x*x; ++nz; }
            if (is_finite(x) && is_finite(y)) { sx += x; sy += y; sxy += x*y; ++np; }

            if (t >= static_cast<std::size_t>(spectral_window)) {
                const std::size_t old = t - static_cast<std::size_t>(spectral_window);
                const double xo = zr[old];
                const double yo = lam_prev[old];
                if (is_finite(xo)) { sumz -= xo; sumz2 -= xo*xo; --nz; }
                if (is_finite(xo) && is_finite(yo)) { sx -= xo; sy -= yo; sxy -= xo*yo; --np; }
            }

            if (nz >= min_periods && np >= min_periods && np > 1) {
                const double var = stable_var(sumz, sumz2, nz);
                const double cov_num = sxy - sx*sy/static_cast<double>(np);
                const double cov = cov_num / static_cast<double>(np - 1);  // pandas rolling cov default ddof=1
                out[t] = (is_finite(var) && var > 0.0) ? scale * cov / var : NaN;
            } else {
                out[t] = NaN;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t bi = 0; bi < static_cast<std::ptrdiff_t>(B); ++bi) {
        const std::size_t b = static_cast<std::size_t>(bi);
        const double* zr = z + b * T;
        std::vector<double> ps(T + 1, 0.0);
        std::vector<int32_t> pc(T + 1, 0);
        for (std::size_t t = 0; t < T; ++t) {
            const double x = zr[t];
            ps[t+1] = ps[t] + (is_finite(x) ? x : 0.0);
            pc[t+1] = pc[t] + (is_finite(x) ? 1 : 0);
        }
        for (std::size_t hi = 0; hi < H; ++hi) {
            const int h = horizons[hi];
            double* po = past + (b * H + hi) * T;
            double* fo = future + (b * H + hi) * T;
            for (std::size_t t = 0; t < T; ++t) {
                if (t + 1 >= static_cast<std::size_t>(h)) {
                    const std::size_t a = t + 1 - static_cast<std::size_t>(h), c = t + 1;
                    po[t] = (pc[c] - pc[a] == h) ? (ps[c] - ps[a]) : NaN;
                } else po[t] = NaN;

                const std::size_t a = t + 1;
                const std::size_t c = t + 1 + static_cast<std::size_t>(h);
                if (c <= T) fo[t] = (pc[c] - pc[a] == h) ? (ps[c] - ps[a]) : NaN;
                else fo[t] = NaN;
            }
        }
    }
}
