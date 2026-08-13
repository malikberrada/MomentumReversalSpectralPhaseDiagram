# `mrspd-native` (optional accelerator)

Optional C++/OpenMP and CUDA accelerator for MRSPD. It is built with **setuptools + the CPython/NumPy C APIs + nvcc**; no CMake is required.

The pure-Python implementation is the reproducibility baseline. Installing this extension is optional and should not change the scientific estimand.

## CPU/OpenMP build

```bash
MRSPD_NATIVE_CUDA=0 python -m pip install ./native --no-build-isolation
```

PowerShell:

```powershell
$env:MRSPD_NATIVE_CUDA="0"
python -m pip install .\native --no-build-isolation
```

## CUDA build

Requires the NVIDIA CUDA Toolkit (`nvcc`) and a host compiler. Auto mode falls back to CPU/OpenMP if CUDA compilation fails.

```bash
python -m pip install ./native --no-build-isolation
```

Force CUDA (fail closed if unavailable):

```bash
MRSPD_NATIVE_CUDA=1 python -m pip install ./native --no-build-isolation
```

Check the installed backend:

```bash
python -c "import mrspd_native as n; print(n.backend_info())"
```
