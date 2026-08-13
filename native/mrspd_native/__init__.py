from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

__all__ = ["feature_bank_batch", "cuda_available", "cuda_compiled", "backend_info"]
__version__ = "0.1.1"

# Python 3.8+ on Windows uses a restricted DLL search policy for extension
# module dependencies.  A CUDA-enabled _core.pyd linked against cudart therefore
# needs the CUDA bin directory registered explicitly before _core is imported.
_DLL_DIR_HANDLES = []
_DLL_DIRS = []


def _version_key(path: Path) -> tuple[int, ...]:
    m = re.search(r"v(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(path), re.I)
    if not m:
        return (0, 0, 0)
    return tuple(int(x or 0) for x in m.groups())


def _windows_cuda_bin_candidates() -> list[Path]:
    if os.name != "nt":
        return []

    candidates: list[Path] = []
    seen: set[str] = set()

    def add(p: str | Path | None) -> None:
        if not p:
            return
        path = Path(p)
        # Accept either a CUDA root or its bin directory.
        if path.name.lower() != "bin":
            path = path / "bin"
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = os.path.normcase(str(resolved))
        if resolved.is_dir() and key not in seen:
            seen.add(key)
            candidates.append(resolved)

    # Exact toolkit selected by the user's environment first.
    for name in ("CUDA_PATH", "CUDA_HOME"):
        add(os.environ.get(name))

    # NVIDIA installers also expose versioned CUDA_PATH_Vx_y variables.
    for name, value in os.environ.items():
        if name.upper().startswith("CUDA_PATH_V"):
            add(value)

    # Existing PATH entries can point directly at CUDA/bin.
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and "cuda" in entry.lower():
            p = Path(entry)
            if p.name.lower() == "bin":
                add(p)

    # Finally scan all locally installed CUDA Toolkits. This is important when
    # CUDA_PATH is stale but another toolkit was used by the build helper.
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    base = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if base.is_dir():
        roots = sorted((p for p in base.glob("v*") if p.is_dir()), key=_version_key, reverse=True)
        for root in roots:
            add(root)

    return candidates


def _configure_windows_cuda_dll_search() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    for bin_dir in _windows_cuda_bin_candidates():
        try:
            # Keep the handle alive: closing it removes the directory again.
            handle = os.add_dll_directory(str(bin_dir))
        except (FileNotFoundError, OSError):
            continue
        _DLL_DIR_HANDLES.append(handle)
        _DLL_DIRS.append(str(bin_dir))


_configure_windows_cuda_dll_search()

try:
    from ._core import cuda_available, cuda_compiled, feature_bank_batch as _feature_bank_batch
except ImportError as exc:
    if os.name == "nt" and "DLL load failed" in str(exc):
        searched = "; ".join(_DLL_DIRS) if _DLL_DIRS else "<no CUDA bin directory found>"
        raise ImportError(
            f"{exc}\n"
            "mrspd_native found its Python extension but Windows could not load one of its native DLL dependencies. "
            f"CUDA DLL directories registered before importing _core: {searched}. "
            "If this persists, run `where.exe cudart64_*.dll` and `where.exe nvcuda.dll` to identify the missing runtime dependency."
        ) from exc
    raise


def feature_bank_batch(z, spans, spectral_window, min_periods, horizons, backend="auto"):
    z = np.ascontiguousarray(z, dtype=np.float64)
    spans = np.ascontiguousarray(spans, dtype=np.int32)
    horizons = np.ascontiguousarray(horizons, dtype=np.int32)
    return _feature_bank_batch(z, spans, int(spectral_window), int(min_periods), horizons, str(backend))


def backend_info():
    info = {
        "version": __version__,
        "cuda_compiled": bool(cuda_compiled()),
        "cuda_available": bool(cuda_available()),
    }
    if os.name == "nt":
        info["cuda_dll_dirs"] = list(_DLL_DIRS)
    return info
