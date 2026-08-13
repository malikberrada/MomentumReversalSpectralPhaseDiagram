#pragma once
#include <cstddef>
#include <cstdint>

void feature_bank_cpu(
    const double* z, std::size_t B, std::size_t T,
    const int32_t* spans, std::size_t S,
    int spectral_window, int min_periods,
    const int32_t* horizons, std::size_t H,
    double* psi, double* past, double* future);

#ifdef MRSPD_WITH_CUDA
bool mrspd_cuda_available();
void feature_bank_cuda(
    const double* z, std::size_t B, std::size_t T,
    const int32_t* spans, std::size_t S,
    int spectral_window, int min_periods,
    const int32_t* horizons, std::size_t H,
    double* psi, double* past, double* future);
#endif
