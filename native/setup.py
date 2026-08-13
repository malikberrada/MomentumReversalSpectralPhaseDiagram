from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def _nvcc_version(nvcc: str) -> tuple[int, int, int]:
    try:
        out = subprocess.check_output(
            [nvcc, "--version"], stderr=subprocess.STDOUT, text=True, errors="replace"
        )
    except Exception:
        return (0, 0, 0)
    m = re.search(r"release\s+(\d+)\.(\d+)(?:[^\n]*?V(\d+)\.(\d+)\.(\d+))?", out, re.I)
    if not m:
        m = re.search(r"V(\d+)\.(\d+)\.(\d+)", out)
        if not m:
            return (0, 0, 0)
        return tuple(int(x) for x in m.groups())
    major, minor = int(m.group(1)), int(m.group(2))
    patch = int(m.group(5) or 0)
    return (major, minor, patch)


def _nvcc_candidates() -> list[str]:
    # Explicit user override always wins.
    override = os.environ.get("CUDACXX") or os.environ.get("NVCC")
    if override:
        p = Path(override)
        if p.exists():
            return [str(p)]
        raise RuntimeError(f"CUDACXX/NVCC points to a missing file: {override}")

    candidates: list[str] = []

    def add(p: str | Path | None) -> None:
        if not p:
            return
        s = str(Path(p))
        if Path(s).exists() and s not in candidates:
            candidates.append(s)

    # CUDA_PATH/CUDA_HOME may be stale, so do not let it mask newer installed toolkits.
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env_name)
        if root:
            add(Path(root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))

    add(shutil.which("nvcc"))
    add(shutil.which("nvcc.exe"))

    if os.name == "nt":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if base.exists():
            for v in base.glob("v*"):
                add(v / "bin" / "nvcc.exe")
    else:
        for root in (Path("/usr/local"), Path("/opt")):
            if root.exists():
                for v in root.glob("cuda*"):
                    add(v / "bin" / "nvcc")

    return candidates


def find_nvcc() -> str | None:
    candidates = _nvcc_candidates()
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda p: (_nvcc_version(p), p), reverse=True)
    best = ranked[0]
    versions = ", ".join(f"{Path(p).parent.parent.name}:{'.'.join(map(str, _nvcc_version(p)))}" for p in ranked)
    print(f"[mrspd-native] nvcc candidates: {versions}")
    print(f"[mrspd-native] selected nvcc: {best}")
    return best


def cuda_root_from_nvcc(nvcc: str) -> Path:
    return Path(nvcc).resolve().parent.parent


def _run_cuda_compile(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    out = proc.stdout or ""
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return proc.returncode == 0, out


def _unsupported_compiler_error(output: str) -> bool:
    s = output.lower()
    return (
        "unsupported microsoft visual studio version" in s
        or "unsupported host compiler" in s
        or "unsupported compiler" in s
    )


class NativeBuildExt(build_ext):
    def build_extension(self, ext):
        mode = os.environ.get("MRSPD_NATIVE_CUDA", "auto").strip().lower()
        force_cuda = mode in {"1", "on", "force", "cuda"}
        cuda_requested = mode not in {"0", "off", "cpu", "false"}
        nvcc = find_nvcc() if cuda_requested else None

        if force_cuda and not nvcc:
            raise RuntimeError(
                "MRSPD_NATIVE_CUDA requests CUDA, but nvcc was not found. "
                "Set CUDACXX/NVCC to nvcc.exe or install the CUDA Toolkit."
            )

        cuda_enabled = False
        if nvcc:
            cuda_root = cuda_root_from_nvcc(nvcc)
            temp_dir = Path(self.build_temp)
            temp_dir.mkdir(parents=True, exist_ok=True)
            obj = temp_dir / ("cuda_kernels.obj" if os.name == "nt" else "cuda_kernels.o")

            cmd = [
                nvcc,
                "-c",
                str(SRC / "cuda_kernels.cu"),
                "-o",
                str(obj),
                "-O3",
                "--std=c++17",
                "--prec-div=true",
                "--prec-sqrt=true",
                "--ftz=false",
            ]

            # Local pip builds should optimize for the GPU that is actually present.
            # Override with MRSPD_CUDA_ARCH=sm_86 / 86 / all-major / none if desired.
            arch = os.environ.get("MRSPD_CUDA_ARCH", "native").strip().lower()
            if arch not in {"", "none", "off", "0"}:
                if arch.isdigit():
                    arch = f"sm_{arch}"
                cmd.append(f"-arch={arch}")

            if os.name == "nt":
                cmd += ["-Xcompiler", "/MD"]
            else:
                cmd += ["-Xcompiler", "-fPIC"]

            print("[mrspd-native] compiling CUDA kernels:", " ".join(cmd))
            ok, out = _run_cuda_compile(cmd)

            allow_mode = os.environ.get(
                "MRSPD_NVCC_ALLOW_UNSUPPORTED_COMPILER", "auto"
            ).strip().lower()
            may_retry = allow_mode in {"1", "on", "true", "yes", "auto"}
            should_retry = (
                os.name == "nt"
                and not ok
                and may_retry
                and (allow_mode != "auto" or _unsupported_compiler_error(out))
            )

            if should_retry:
                retry_cmd = cmd.copy()
                retry_cmd.insert(1, "--allow-unsupported-compiler")
                print(
                    "[mrspd-native] nvcc rejected the installed MSVC version; "
                    "retrying with NVIDIA's --allow-unsupported-compiler option."
                )
                print("[mrspd-native] CUDA retry:", " ".join(retry_cmd))
                ok, out = _run_cuda_compile(retry_cmd)

            if ok:
                cuda_enabled = True
                ext.extra_objects = list(ext.extra_objects or []) + [str(obj)]
                ext.define_macros = list(ext.define_macros or []) + [("MRSPD_WITH_CUDA", "1")]
                if os.name == "nt":
                    ext.library_dirs = list(ext.library_dirs or []) + [str(cuda_root / "lib" / "x64")]
                    ext.libraries = list(ext.libraries or []) + ["cudart"]
                else:
                    lib64 = cuda_root / "lib64"
                    if lib64.exists():
                        ext.library_dirs = list(ext.library_dirs or []) + [str(lib64)]
                    ext.libraries = list(ext.libraries or []) + ["cudart"]
            elif force_cuda:
                raise RuntimeError(
                    "CUDA compilation failed even after the compatibility retry. "
                    "Prefer a newer CUDA Toolkit compatible with your installed MSVC, "
                    "or set MRSPD_NATIVE_CUDA=0 for the C++/OpenMP backend."
                )
            else:
                print("[mrspd-native] CUDA build failed in auto mode; falling back to C++/OpenMP.")
        else:
            print("[mrspd-native] nvcc not found: building optimized C++/OpenMP backend only")

        if self.compiler.compiler_type == "msvc":
            ext.extra_compile_args = ["/O2", "/std:c++17", "/openmp", "/EHsc"]
        else:
            ext.extra_compile_args = ["-O3", "-std=c++17", "-fopenmp"]
            ext.extra_link_args = list(ext.extra_link_args or []) + ["-fopenmp"]

        super().build_extension(ext)


ext = Extension(
    "mrspd_native._core",
    sources=["src/bindings.cpp", "src/cpu_kernels.cpp"],
    include_dirs=[np.get_include(), str(SRC)],
    language="c++",
)

setup(ext_modules=[ext], cmdclass={"build_ext": NativeBuildExt})
