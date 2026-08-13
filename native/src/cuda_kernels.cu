#include "native_api.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
inline void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
__device__ __forceinline__ bool fin(double x) { return isfinite(x); }
__device__ __forceinline__ double nanv() { return nan(""); }

__global__ void spectral_kernel(const double* z, std::size_t B, std::size_t T,
                                const int32_t* spans, std::size_t S,
                                int window, int minp, double* psi) {
    const std::size_t task = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t tasks = B * S;
    if (task >= tasks) return;
    const std::size_t b = task / S;
    const std::size_t s = task % S;
    const int span = spans[s];
    const double nu = (double(span) - 1.0) / (double(span) + 1.0);
    const double alpha = 1.0 - nu;
    const double scale = nu / (1.0 - nu);
    const double* zr = z + b*T;
    double* out = psi + task*T;

    // First pass stores EWM predecessor values directly in output as scratch.
    double ewm = nanv();
    int ecount = 0;
    for (std::size_t t=0; t<T; ++t) {
        out[t] = (ecount >= span && fin(ewm)) ? ewm : nanv();
        const double x = zr[t];
        if (fin(x)) {
            ewm = fin(ewm) ? (alpha*x + nu*ewm) : x;
            ++ecount;
        }
    }

    double sumz=0.0, sumz2=0.0, sx=0.0, sy=0.0, sxy=0.0;
    int nz=0, np=0;
    for (std::size_t t=0; t<T; ++t) {
        const double x=zr[t], y=out[t];
        if (fin(x)) { sumz+=x; sumz2+=x*x; ++nz; }
        if (fin(x) && fin(y)) { sx+=x; sy+=y; sxy+=x*y; ++np; }
        if (t >= (std::size_t)window) {
            const std::size_t old=t-(std::size_t)window;
            const double xo=zr[old], yo=out[old];
            if (fin(xo)) { sumz-=xo; sumz2-=xo*xo; --nz; }
            if (fin(xo) && fin(yo)) { sx-=xo; sy-=yo; sxy-=xo*yo; --np; }
        }
        double value=nanv();
        if (nz>=minp && np>=minp && np>1) {
            double var=(sumz2-sumz*sumz/double(nz))/double(nz);
            if (var < 0.0 && var > -1e-14*(fabs(sumz2)+1.0)) var=0.0;
            const double cov=(sxy-sx*sy/double(np))/double(np-1);
            if (isfinite(var) && var>0.0) value=scale*cov/var;
        }
        // Safe because old y is only read before this write for current t; future removals need historical y,
        // so we cannot overwrite scratch yet. This kernel therefore uses a second scratch array in host wrapper.
        // (This branch is unused; host wrapper launches scratch version below.)
        out[t]=value;
    }
}

__global__ void ewm_kernel(const double* z, std::size_t B, std::size_t T,
                           const int32_t* spans, std::size_t S, double* lam) {
    const std::size_t task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= B*S) return;
    const std::size_t b=task/S, s=task%S;
    const int span=spans[s];
    const double nu=(double(span)-1.0)/(double(span)+1.0), alpha=1.0-nu;
    const double* zr=z+b*T;
    double* lr=lam+task*T;
    double ewm=nanv(); int count=0;
    for (std::size_t t=0;t<T;++t) {
        lr[t]=(count>=span && fin(ewm)) ? ewm : nanv();
        const double x=zr[t];
        if (fin(x)) { ewm=fin(ewm) ? alpha*x+nu*ewm : x; ++count; }
    }
}

__global__ void psi_kernel(const double* z, const double* lam,
                           std::size_t B, std::size_t T,
                           const int32_t* spans, std::size_t S,
                           int window, int minp, double* psi) {
    const std::size_t task=blockIdx.x*blockDim.x+threadIdx.x;
    if (task>=B*S) return;
    const std::size_t b=task/S, s=task%S;
    const int span=spans[s];
    const double nu=(double(span)-1.0)/(double(span)+1.0), scale=nu/(1.0-nu);
    const double* zr=z+b*T;
    const double* lr=lam+task*T;
    double* out=psi+task*T;
    double sumz=0.0,sumz2=0.0,sx=0.0,sy=0.0,sxy=0.0; int nz=0,np=0;
    for (std::size_t t=0;t<T;++t) {
        const double x=zr[t],y=lr[t];
        if(fin(x)){sumz+=x;sumz2+=x*x;++nz;}
        if(fin(x)&&fin(y)){sx+=x;sy+=y;sxy+=x*y;++np;}
        if(t>=(std::size_t)window){
            const std::size_t old=t-(std::size_t)window;
            const double xo=zr[old],yo=lr[old];
            if(fin(xo)){sumz-=xo;sumz2-=xo*xo;--nz;}
            if(fin(xo)&&fin(yo)){sx-=xo;sy-=yo;sxy-=xo*yo;--np;}
        }
        double v=nanv();
        if(nz>=minp&&np>=minp&&np>1){
            double var=(sumz2-sumz*sumz/double(nz))/double(nz);
            if(var<0.0&&var>-1e-14*(fabs(sumz2)+1.0))var=0.0;
            const double cov=(sxy-sx*sy/double(np))/double(np-1);
            if(isfinite(var)&&var>0.0)v=scale*cov/var;
        }
        out[t]=v;
    }
}

__global__ void prefix_kernel(const double* z, std::size_t B, std::size_t T,
                              double* ps, int32_t* pc) {
    const std::size_t b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=B)return;
    const double* zr=z+b*T;
    double* sr=ps+b*(T+1); int32_t* cr=pc+b*(T+1);
    sr[0]=0.0; cr[0]=0;
    for(std::size_t t=0;t<T;++t){
        const double x=zr[t];
        sr[t+1]=sr[t]+(fin(x)?x:0.0);
        cr[t+1]=cr[t]+(fin(x)?1:0);
    }
}

__global__ void sum_kernel(const double* ps,const int32_t* pc,
                           std::size_t B,std::size_t T,
                           const int32_t* horizons,std::size_t H,
                           double* past,double* future){
    const std::size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t total=B*H*T;
    if(idx>=total)return;
    const std::size_t t=idx%T;
    const std::size_t q=idx/T;
    const std::size_t hidx=q%H,b=q/H;
    const int h=horizons[hidx];
    const double* sr=ps+b*(T+1); const int32_t* cr=pc+b*(T+1);
    double pv=nanv(),fv=nanv();
    if(t+1>=(std::size_t)h){
        const std::size_t a=t+1-(std::size_t)h,c=t+1;
        if(cr[c]-cr[a]==h)pv=sr[c]-sr[a];
    }
    const std::size_t a=t+1,c=t+1+(std::size_t)h;
    if(c<=T && cr[c]-cr[a]==h)fv=sr[c]-sr[a];
    past[idx]=pv; future[idx]=fv;
}
}

bool mrspd_cuda_available(){
    int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0;
}

void feature_bank_cuda(
    const double* z, std::size_t B, std::size_t T,
    const int32_t* spans, std::size_t S,
    int spectral_window, int min_periods,
    const int32_t* horizons, std::size_t H,
    double* psi, double* past, double* future) {
    if(!mrspd_cuda_available()) throw std::runtime_error("No CUDA device available");
    double *dz=nullptr,*dpsi=nullptr,*dlam=nullptr,*dpast=nullptr,*dfuture=nullptr,*dps=nullptr;
    int32_t *dspans=nullptr,*dh=nullptr,*dpc=nullptr;
    const std::size_t nZ=B*T,nPsi=B*S*T,nSum=B*H*T,nPref=B*(T+1);
    check(cudaMalloc((void**)&dz,nZ*sizeof(double)),"cudaMalloc z");
    check(cudaMalloc((void**)&dpsi,nPsi*sizeof(double)),"cudaMalloc psi");
    check(cudaMalloc((void**)&dlam,nPsi*sizeof(double)),"cudaMalloc lam");
    check(cudaMalloc((void**)&dpast,nSum*sizeof(double)),"cudaMalloc past");
    check(cudaMalloc((void**)&dfuture,nSum*sizeof(double)),"cudaMalloc future");
    check(cudaMalloc((void**)&dspans,S*sizeof(int32_t)),"cudaMalloc spans");
    check(cudaMalloc((void**)&dh,H*sizeof(int32_t)),"cudaMalloc horizons");
    check(cudaMalloc((void**)&dps,nPref*sizeof(double)),"cudaMalloc prefix sum");
    check(cudaMalloc((void**)&dpc,nPref*sizeof(int32_t)),"cudaMalloc prefix count");
    try {
        check(cudaMemcpy(dz,z,nZ*sizeof(double),cudaMemcpyHostToDevice),"copy z");
        check(cudaMemcpy(dspans,spans,S*sizeof(int32_t),cudaMemcpyHostToDevice),"copy spans");
        check(cudaMemcpy(dh,horizons,H*sizeof(int32_t),cudaMemcpyHostToDevice),"copy horizons");
        const int th=128;
        ewm_kernel<<<(int)((B*S+th-1)/th),th>>>(dz,B,T,dspans,S,dlam);
        psi_kernel<<<(int)((B*S+th-1)/th),th>>>(dz,dlam,B,T,dspans,S,spectral_window,min_periods,dpsi);
        prefix_kernel<<<(int)((B+th-1)/th),th>>>(dz,B,T,dps,dpc);
        const std::size_t total=B*H*T;
        sum_kernel<<<(int)((total+255)/256),256>>>(dps,dpc,B,T,dh,H,dpast,dfuture);
        check(cudaGetLastError(),"kernel launch");
        check(cudaDeviceSynchronize(),"kernel sync");
        check(cudaMemcpy(psi,dpsi,nPsi*sizeof(double),cudaMemcpyDeviceToHost),"copy psi");
        check(cudaMemcpy(past,dpast,nSum*sizeof(double),cudaMemcpyDeviceToHost),"copy past");
        check(cudaMemcpy(future,dfuture,nSum*sizeof(double),cudaMemcpyDeviceToHost),"copy future");
    } catch(...) {
        cudaFree(dz);cudaFree(dpsi);cudaFree(dlam);cudaFree(dpast);cudaFree(dfuture);cudaFree(dspans);cudaFree(dh);cudaFree(dps);cudaFree(dpc);
        throw;
    }
    cudaFree(dz);cudaFree(dpsi);cudaFree(dlam);cudaFree(dpast);cudaFree(dfuture);cudaFree(dspans);cudaFree(dh);cudaFree(dps);cudaFree(dpc);
}
