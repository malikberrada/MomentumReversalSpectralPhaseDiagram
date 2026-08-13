from __future__ import annotations

"""Non-monotone spectral phase-surface analysis for MRSPD.

This module deliberately does not assume a single monotone boundary.  It models

    g_h(psi) = E[M_{t,h} F_{t,h} | psi]

with cubic spline bases and searches *all* zero crossings.  A tensor-product
spline in (psi, log(h)) tests whether the boundary is horizon-dependent.

All test metrics are strict time-OOS.  Train/test purging uses the maximum
future horizon.  Descriptive quantile bins are also defined from the training
sample and then applied to the OOS block.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


@dataclass(frozen=True)
class PhaseSurfaceConfig:
    quantile_bins: int = 30
    spline_knots: int = 7
    spline_degree: int = 3
    ridge_alpha: float = 4.0
    logistic_c: float = 0.5
    root_grid_size: int = 2048
    root_domain_qlo: float = 0.005
    root_domain_qhi: float = 0.995
    bootstrap_reps: int = 500
    max_fit_rows: int = 250_000
    predict_chunk_size: int = 100_000
    random_seed: int = 20260812

    # Phase-transition certification.  Candidate roots are proposed on train
    # and certified only on the future OOS block.
    cert_alpha: float = 0.05
    cert_side_width_q: float = 0.08
    cert_side_gap_q: float = 0.01
    cert_min_effect_sigma: float = 0.02
    cert_abs_effect: float = 0.0
    cert_min_obs_per_side: int = 250
    cert_min_dates_per_side: int = 40
    cert_min_folds: int = 3
    cert_min_fold_fraction: float = 0.75
    cert_root_cluster_q: float = 0.06
    cert_max_root_iqr_q: float = 0.05
    cert_single_phase_bins: int = 5


def _date_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("date")["date"].transform("size").to_numpy(dtype=float)
    w = 1.0 / counts
    return w / np.mean(w)


def _safe_auc(y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score, sample_weight=w))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(x[ok], weights=w[ok]))


def _walkforward_date_blocks(df: pd.DataFrame, cfg) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = np.array(sorted(pd.to_datetime(df["date"].unique())))
    n = len(dates)
    start = max(1, int(cfg.min_train_frac * n))
    if n - start < cfg.n_walkforward_splits * 5:
        raise RuntimeError("Not enough dates for requested walk-forward splits")
    edges = np.linspace(start, n, cfg.n_walkforward_splits + 1, dtype=int)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(cfg.n_walkforward_splits):
        a, b = edges[i], edges[i + 1]
        if b > a:
            out.append((pd.Timestamp(dates[a]), pd.Timestamp(dates[b - 1])))
    return out


class SplineSurface:
    """Penalized cubic-spline surface with optional tensor interaction/context.

    The model is linear in spline basis functions, which gives a GAM-like
    additive model when ``interaction=False`` and a tensor-product spline
    surface when ``interaction=True``.  ``task`` is either ``continuous`` for
    E[phase_product|X] or ``logistic`` for P(phase_product>0|X).
    """

    def __init__(
        self,
        *,
        task: str,
        interaction: bool,
        context: bool,
        n_knots: int,
        degree: int,
        ridge_alpha: float,
        logistic_c: float,
    ) -> None:
        if task not in {"continuous", "logistic"}:
            raise ValueError(task)
        self.task = task
        self.interaction = interaction
        self.context = context
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_c = float(logistic_c)

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray) -> "SplineSurface":
        xpsi = df[["psi_primary"]].to_numpy(dtype=float)
        xh = np.log(df[["horizon"]].to_numpy(dtype=float))
        self.psi_lo_, self.psi_hi_ = np.nanquantile(xpsi[:, 0], [0.005, 0.995])
        xpsi = np.clip(xpsi, self.psi_lo_, self.psi_hi_)

        self.psi_spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
            knots="quantile",
            extrapolation="constant",
        )
        self.h_spline_ = SplineTransformer(
            n_knots=min(self.n_knots, max(2, len(np.unique(xh)))),
            degree=self.degree,
            include_bias=False,
            knots="quantile",
            extrapolation="constant",
        )
        self.psi_spline_.fit(xpsi)
        self.h_spline_.fit(xh)

        self.control_cols_ = ["vol20", "range20", "log_dollar_volume20"]
        if self.context:
            controls = df[self.control_cols_].to_numpy(dtype=float)
            self.control_medians_ = np.nanmedian(controls, axis=0)
            controls = np.where(np.isfinite(controls), controls, self.control_medians_)
            self.control_means_ = np.mean(controls, axis=0)
            self.control_stds_ = np.std(controls, axis=0)
            self.control_stds_[self.control_stds_ == 0.0] = 1.0
            self.asset_classes_ = sorted(df["asset_class"].astype(str).dropna().unique().tolist())
            self.asset_to_idx_ = {c: i for i, c in enumerate(self.asset_classes_)}
        else:
            self.control_medians_ = None
            self.control_means_ = None
            self.control_stds_ = None
            self.asset_classes_ = []
            self.asset_to_idx_ = {}

        X = self._design(df)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        if self.task == "continuous":
            self.estimator_ = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
            self.estimator_.fit(Xs, df["phase_product"].to_numpy(dtype=float), sample_weight=sample_weight)
        else:
            self.estimator_ = LogisticRegression(C=self.logistic_c, max_iter=700, solver="lbfgs")
            self.estimator_.fit(Xs, df["phase_label"].to_numpy(dtype=int), sample_weight=sample_weight)
        return self

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        psi = np.clip(df[["psi_primary"]].to_numpy(dtype=float), self.psi_lo_, self.psi_hi_)
        logh = np.log(df[["horizon"]].to_numpy(dtype=float))
        bp = self.psi_spline_.transform(psi)
        bh = self.h_spline_.transform(logh)
        parts = [bp, bh]
        if self.interaction:
            tensor = np.einsum("ij,ik->ijk", bp, bh, optimize=True).reshape(len(df), -1)
            parts.append(tensor)

        if self.context:
            c = df[self.control_cols_].to_numpy(dtype=float)
            c = np.where(np.isfinite(c), c, self.control_medians_)
            cz = (c - self.control_means_) / self.control_stds_
            parts.append(cz)
            # Allow volatility to alter the spectral shape without turning the
            # model into a high-dimensional black box.
            parts.append(bp * cz[:, [0]])
            onehot = np.zeros((len(df), len(self.asset_classes_)), dtype=float)
            for i, cls in enumerate(df["asset_class"].astype(str).to_numpy()):
                j = self.asset_to_idx_.get(cls)
                if j is not None:
                    onehot[i, j] = 1.0
            parts.append(onehot)
        return np.concatenate(parts, axis=1)

    def _predict_one(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler_.transform(self._design(df))
        if self.task == "continuous":
            return self.estimator_.predict(X)
        return self.estimator_.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, chunk_size: int = 100_000) -> np.ndarray:
        # Avoid materializing a multi-gigabyte tensor-product design matrix on
        # the full S&P 500 panel.  Only the prediction vector is retained.
        if len(df) <= chunk_size:
            return self._predict_one(df)
        out = np.empty(len(df), dtype=float)
        for start in range(0, len(df), chunk_size):
            stop = min(start + chunk_size, len(df))
            out[start:stop] = self._predict_one(df.iloc[start:stop])
        return out


def _roots_from_curve(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Return every finite zero crossing using linear interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out: list[dict] = []
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        return out
    eps_x = max(1e-12, float(np.ptp(x)) / max(2, len(x) - 1))
    last_root: float | None = None
    for i in range(len(x) - 1):
        x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
        root: float | None = None
        if y0 == 0.0:
            root = float(x0)
        elif y0 * y1 < 0.0:
            root = float(x0 - y0 * (x1 - x0) / (y1 - y0))
        if root is None:
            continue
        if last_root is not None and abs(root - last_root) <= 2.0 * eps_x:
            continue
        slope = float((y1 - y0) / (x1 - x0)) if x1 != x0 else float("nan")
        direction = "R_to_M" if y0 < 0.0 <= y1 else "M_to_R" if y0 > 0.0 >= y1 else "touch"
        out.append({"psi_star": root, "direction": direction, "local_slope": slope})
        last_root = root
    return out


def _phase_path(grid: np.ndarray, g: np.ndarray, roots: list[dict]) -> str:
    if len(grid) == 0:
        return ""
    cuts = [float(grid[0])] + [float(r["psi_star"]) for r in roots] + [float(grid[-1])]
    states: list[str] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (a + b)
        val = float(np.interp(mid, grid, g))
        states.append("M" if val > 0.0 else "R" if val < 0.0 else "0")
    return "→".join(states)


def _quantile_oos_rows(train: pd.DataFrame, test: pd.DataFrame, n_bins: int, meta: dict) -> list[dict]:
    rows: list[dict] = []
    for h, trh in train.groupby("horizon", sort=True):
        teh = test[test["horizon"] == h]
        if len(trh) < 200 or len(teh) < 50:
            continue
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.nanquantile(trh["psi_primary"].to_numpy(float), q))
        if len(edges) < 4:
            continue
        # Outer bins are open-ended so OOS extremes are not silently dropped.
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_id = np.searchsorted(edges, teh["psi_primary"].to_numpy(float), side="right") - 1
        bin_id = np.clip(bin_id, 0, len(edges) - 2)
        z = teh.copy()
        z["psi_bin"] = bin_id
        w = _date_weights(z)
        z["_w"] = w
        for b, gb in z.groupby("psi_bin", sort=True):
            ww = gb["_w"].to_numpy(float)
            rows.append({
                **meta,
                "horizon": int(h),
                "psi_bin": int(b),
                "train_edge_lo": float(edges[b]) if np.isfinite(edges[b]) else None,
                "train_edge_hi": float(edges[b + 1]) if np.isfinite(edges[b + 1]) else None,
                "mean_psi": _weighted_mean(gb["psi_primary"].to_numpy(float), ww),
                "mean_phase_product": _weighted_mean(gb["phase_product"].to_numpy(float), ww),
                "p_momentum": _weighted_mean(gb["phase_label"].to_numpy(float), ww),
                "n": int(len(gb)),
                "n_dates": int(gb["date"].nunique()),
            })
    return rows




def _balanced_fit_sample(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    """Deterministic date-balanced fit sample for very large panels.

    The scientific estimand weights dates equally.  Sampling with probability
    proportional to the inverse number of rows on each date preserves that
    objective while bounding the spline design matrix in RAM.
    """
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    # Pandas 2/3 with Copy-on-Write may expose transform().to_numpy() as a
    # read-only view.  Never normalize that buffer in-place: materialize our
    # own writable float64 array and normalize out-of-place.
    counts = df.groupby("date", sort=False)["date"].transform("size").to_numpy(
        dtype=np.float64, copy=True
    )
    if counts.size != len(df) or np.any(~np.isfinite(counts)) or np.any(counts <= 0):
        raise RuntimeError("Invalid date counts while constructing balanced fit sample")
    prob = np.reciprocal(counts)
    total = float(prob.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Invalid sampling probabilities in balanced fit sample")
    prob = prob / total
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df), size=max_rows, replace=False, p=prob)
    return df.iloc[np.sort(idx)].copy()


def _continuous_metrics(test: pd.DataFrame, pred: np.ndarray, train: pd.DataFrame, model_name: str, meta: dict) -> dict:
    y = test["phase_product"].to_numpy(float)
    w = _date_weights(test)
    base = _weighted_mean(train["phase_product"].to_numpy(float), _date_weights(train))
    bp = np.full_like(y, base)
    mse = mean_squared_error(y, pred, sample_weight=w)
    bmse = mean_squared_error(y, bp, sample_weight=w)
    return {
        **meta,
        "model": model_name,
        "n": int(len(test)),
        "mse": float(mse),
        "baseline_mse": float(bmse),
        "mse_skill": float(1.0 - mse / bmse) if bmse > 0 else float("nan"),
        "mae": float(mean_absolute_error(y, pred, sample_weight=w)),
        "spearman_pred_product": float(spearmanr(pred, y, nan_policy="omit").statistic),
        "phase_auc_from_g": _safe_auc(test["phase_label"].to_numpy(int), pred, w),
    }


def _logistic_metrics(test: pd.DataFrame, p: np.ndarray, train: pd.DataFrame, model_name: str, meta: dict) -> dict:
    y = test["phase_label"].to_numpy(int)
    w = _date_weights(test)
    base = _weighted_mean(train["phase_label"].to_numpy(float), _date_weights(train))
    bp = np.full_like(p, base)
    brier = brier_score_loss(y, p, sample_weight=w)
    bb = brier_score_loss(y, bp, sample_weight=w)
    return {
        **meta,
        "model": model_name,
        "n": int(len(test)),
        "auc": _safe_auc(y, p, w),
        "brier": float(brier),
        "baseline_brier": float(bb),
        "brier_skill": float(1.0 - brier / bb) if bb > 0 else float("nan"),
        "logloss": float(log_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6), sample_weight=w)),
    }


def _make_linear_comparator(num_cols: Sequence[str]) -> Pipeline:
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            list(num_cols),
        ))
    transformers.append(("h", OneHotEncoder(handle_unknown="ignore"), ["horizon"]))
    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LogisticRegression(C=1.0, max_iter=700, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def _datewise_pair_metrics(test: pd.DataFrame, p0: np.ndarray, p1: np.ndarray, meta: dict, names: tuple[str, str]) -> list[dict]:
    z = test[["date", "phase_label"]].copy()
    z["p0"] = p0
    z["p1"] = p1
    rows: list[dict] = []
    for date, g in z.groupby("date", sort=True):
        y = g["phase_label"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        a0 = float(roc_auc_score(y, g["p0"].to_numpy(float)))
        a1 = float(roc_auc_score(y, g["p1"].to_numpy(float)))
        b0 = float(np.mean((y - g["p0"].to_numpy(float)) ** 2))
        b1 = float(np.mean((y - g["p1"].to_numpy(float)) ** 2))
        rows.append({
            **meta,
            "date": pd.Timestamp(date),
            "model_0": names[0],
            "model_1": names[1],
            "auc_0": a0,
            "auc_1": a1,
            "delta_auc": a1 - a0,
            # Positive means model_1 has lower (better) Brier loss.
            "delta_brier_improvement": b0 - b1,
            "n": int(len(g)),
        })
    return rows


def _moving_block_bootstrap(values: np.ndarray, block_len: int, reps: int, rng: np.random.Generator) -> dict:
    """Moving-block bootstrap of a time-series mean.

    The implementation is vectorized over bootstrap replications and reports
    finite-sample-corrected one-sided p-values.  ``p_gt0`` tests H1: mean > 0;
    ``p_lt0`` tests H1: mean < 0.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {
            "estimate": None, "ci_lo": None, "ci_hi": None,
            "p_two_sided": None, "p_gt0": None, "p_lt0": None,
            "n_dates": 0,
        }
    if n < max(10, 2 * max(1, block_len)):
        est = float(np.mean(x))
        return {
            "estimate": est, "ci_lo": None, "ci_hi": None,
            "p_two_sided": None, "p_gt0": None, "p_lt0": None,
            "n_dates": n,
        }

    block_len = max(1, min(int(block_len), n))
    n_starts = n - block_len + 1
    n_blocks = int(math.ceil(n / block_len))
    tail_len = n - (n_blocks - 1) * block_len

    prefix = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    starts = rng.integers(0, n_starts, size=(reps, n_blocks))
    full = prefix[starts + block_len] - prefix[starts]

    if n_blocks == 1:
        sums = prefix[starts[:, 0] + tail_len] - prefix[starts[:, 0]]
    else:
        sums = np.sum(full[:, :-1], axis=1)
        last = starts[:, -1]
        sums += prefix[last + tail_len] - prefix[last]
    boot = sums / float(n)

    est = float(np.mean(x))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    # +1 correction prevents impossible p=0 with finite bootstrap reps.
    p_gt0 = float((1.0 + np.sum(boot <= 0.0)) / (reps + 1.0))
    p_lt0 = float((1.0 + np.sum(boot >= 0.0)) / (reps + 1.0))
    p_two = float(min(1.0, 2.0 * min(p_gt0, p_lt0)))
    return {
        "estimate": est,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_two_sided": p_two,
        "p_gt0": p_gt0,
        "p_lt0": p_lt0,
        "n_dates": n,
    }


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, preserving NaNs."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    order_local = np.argsort(p[idx])
    ordered_idx = idx[order_local]
    m = len(ordered_idx)
    running = 0.0
    for rank, original_idx in enumerate(ordered_idx):
        adj = min(1.0, (m - rank) * float(p[original_idx]))
        running = max(running, adj)
        out[original_idx] = running
    return out


def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    xx, ww = x[ok], w[ok]
    mu = np.average(xx, weights=ww)
    return float(np.sqrt(np.average((xx - mu) ** 2, weights=ww)))


def _datewise_phase_means(df: pd.DataFrame) -> np.ndarray:
    """Equal-date estimand: first average cross-sectionally within each date."""
    if df.empty:
        return np.empty(0, dtype=float)
    z = (
        df.groupby("date", sort=True)["phase_product"]
        .mean()
        .to_numpy(dtype=float)
    )
    return z[np.isfinite(z)]


def _bootstrap_phase_mean(
    df: pd.DataFrame,
    *,
    block_len: int,
    reps: int,
    seed: int,
) -> dict:
    vals = _datewise_phase_means(df)
    return _moving_block_bootstrap(
        vals,
        block_len,
        reps,
        np.random.default_rng(seed),
    )


def _candidate_root_oos_certification(
    train_h: pd.DataFrame,
    test_h: pd.DataFrame,
    root: dict,
    *,
    block_len: int,
    scfg: PhaseSurfaceConfig,
    seed: int,
) -> dict:
    """Certify a train-proposed root on future OOS observations only."""
    psi_train = train_h["psi_primary"].to_numpy(dtype=float)
    psi_train = psi_train[np.isfinite(psi_train)]
    if len(psi_train) < 200:
        return {"eligible": False, "reason": "insufficient_train"}

    root_psi = float(root["psi_star"])
    root_q = float(np.mean(psi_train <= root_psi))
    width = float(scfg.cert_side_width_q)
    gap = float(scfg.cert_side_gap_q)
    if not (0.0 < gap < width < 0.5):
        raise ValueError("Require 0 < cert_side_gap_q < cert_side_width_q < 0.5")
    if root_q - width <= 0.0 or root_q + width >= 1.0:
        return {
            "eligible": False,
            "reason": "root_too_close_to_domain_edge",
            "root_quantile": root_q,
        }

    qs = np.array([
        root_q - width, root_q - gap,
        root_q + gap, root_q + width,
    ])
    left_lo, left_hi, right_lo, right_hi = np.quantile(psi_train, qs)
    left = test_h[
        (test_h["psi_primary"] >= left_lo) &
        (test_h["psi_primary"] <= left_hi)
    ]
    right = test_h[
        (test_h["psi_primary"] >= right_lo) &
        (test_h["psi_primary"] <= right_hi)
    ]

    n_left_dates = int(left["date"].nunique())
    n_right_dates = int(right["date"].nunique())
    enough = (
        len(left) >= scfg.cert_min_obs_per_side
        and len(right) >= scfg.cert_min_obs_per_side
        and n_left_dates >= scfg.cert_min_dates_per_side
        and n_right_dates >= scfg.cert_min_dates_per_side
    )
    if not enough:
        return {
            "eligible": False,
            "reason": "insufficient_oos_support",
            "root_quantile": root_q,
            "left_n": int(len(left)),
            "right_n": int(len(right)),
            "left_n_dates": n_left_dates,
            "right_n_dates": n_right_dates,
        }

    train_scale = _weighted_std(
        train_h["phase_product"].to_numpy(float),
        _date_weights(train_h),
    )
    epsilon = max(
        float(scfg.cert_abs_effect),
        float(scfg.cert_min_effect_sigma) * train_scale if np.isfinite(train_scale) else 0.0,
    )

    lst = _bootstrap_phase_mean(
        left, block_len=block_len, reps=scfg.bootstrap_reps, seed=seed + 11
    )
    rst = _bootstrap_phase_mean(
        right, block_len=block_len, reps=scfg.bootstrap_reps, seed=seed + 29
    )
    direction = str(root.get("direction", ""))
    if direction == "R_to_M":
        left_expected, right_expected = -1, +1
        left_p, right_p = lst["p_lt0"], rst["p_gt0"]
    elif direction == "M_to_R":
        left_expected, right_expected = +1, -1
        left_p, right_p = lst["p_gt0"], rst["p_lt0"]
    else:
        return {
            "eligible": False,
            "reason": "touch_not_transition",
            "root_quantile": root_q,
        }

    def side_pass(st: dict, expected: int) -> bool:
        if st["estimate"] is None or st["ci_lo"] is None or st["ci_hi"] is None:
            return False
        est = float(st["estimate"])
        effect_ok = abs(est) >= epsilon
        if expected > 0:
            return effect_ok and est > 0.0 and float(st["ci_lo"]) > 0.0
        return effect_ok and est < 0.0 and float(st["ci_hi"]) < 0.0

    left_pass = side_pass(lst, left_expected)
    right_pass = side_pass(rst, right_expected)
    # Intersection-union test: both directional side hypotheses must pass.
    candidate_p = (
        max(float(left_p), float(right_p))
        if left_p is not None and right_p is not None
        else float("nan")
    )
    return {
        "eligible": True,
        "reason": "",
        "root_quantile": root_q,
        "epsilon": float(epsilon),
        "train_phase_product_std": float(train_scale),
        "left_psi_lo": float(left_lo),
        "left_psi_hi": float(left_hi),
        "right_psi_lo": float(right_lo),
        "right_psi_hi": float(right_hi),
        "left_n": int(len(left)),
        "right_n": int(len(right)),
        "left_n_dates": n_left_dates,
        "right_n_dates": n_right_dates,
        "left_g": lst["estimate"],
        "left_ci_lo": lst["ci_lo"],
        "left_ci_hi": lst["ci_hi"],
        "left_p_directional": left_p,
        "right_g": rst["estimate"],
        "right_ci_lo": rst["ci_lo"],
        "right_ci_hi": rst["ci_hi"],
        "right_p_directional": right_p,
        "left_effect_ci_pass": bool(left_pass),
        "right_effect_ci_pass": bool(right_pass),
        "local_effect_ci_pass": bool(left_pass and right_pass),
        "candidate_p_iut": candidate_p,
    }


CERTIFIED_TRANSITION_COLUMNS = [
    "return_mode", "horizon", "direction", "cluster_id",
    "folds_total", "folds_support", "support_fraction",
    "min_folds_required", "median_root_psi", "median_root_quantile",
    "root_quantile_q25", "root_quantile_q75", "root_quantile_iqr",
    "max_root_iqr_allowed", "certified_transition", "folds_json",
]

CERTIFIED_TOPOLOGY_COLUMNS = [
    "return_mode", "horizon", "n_certified_transitions",
    "certified_phase_path", "directions_json", "root_quantiles_json",
    "topology_consistent",
]

def _frame_with_schema(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Return a DataFrame that preserves headers even when ``rows`` is empty."""
    return pd.DataFrame(rows, columns=columns)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV safely; an empty/headerless file is a valid empty result."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def repair_empty_phase_certification_csvs(out_dir: Path) -> list[str]:
    """Repair v5 headerless empty certification CSVs in-place.

    Zero certified transitions is a legitimate scientific result.  v5 wrote an
    empty DataFrame with no columns as a headerless CSV; pandas then raised
    EmptyDataError during plotting.  v5.1 upgrades those files to header-only
    schema-preserving CSVs without touching any non-empty result.
    """
    out_dir = Path(out_dir)
    repaired: list[str] = []
    targets = {
        "certified_phase_transitions.csv": CERTIFIED_TRANSITION_COLUMNS,
        "certified_phase_topology.csv": CERTIFIED_TOPOLOGY_COLUMNS,
    }
    for name, columns in targets.items():
        path = out_dir / name
        if not path.exists():
            continue
        frame = _read_csv_or_empty(path)
        if frame.empty and len(frame.columns) == 0:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            repaired.append(name)
    return repaired


def _apply_root_multiplicity(cert: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Holm-correct candidate roots within each fold/horizon curve."""
    if cert.empty:
        return cert.copy()
    z = cert.copy()
    z["candidate_p_holm"] = np.nan
    for _, idx in z.groupby(["return_mode", "fold", "horizon"], sort=False).groups.items():
        arr_idx = np.asarray(list(idx), dtype=int)
        p = z.loc[arr_idx, "candidate_p_iut"].to_numpy(float)
        z.loc[arr_idx, "candidate_p_holm"] = _holm_adjust(p)
    z["passes_multiplicity"] = (
        z["eligible"].eq(True)
        & z["local_effect_ci_pass"].eq(True)
        & (z["candidate_p_holm"] <= float(alpha))
    )
    return z


def _cluster_certified_roots(
    cert: pd.DataFrame,
    scfg: PhaseSurfaceConfig,
    topology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cluster replicated roots in quantile space and require cross-fold stability.

    ``topology`` supplies the denominator of attempted folds, including folds
    with zero candidate roots.  This prevents zero-root folds from
    accidentally disappearing from the replication requirement.
    """
    if cert.empty:
        return _frame_with_schema([], CERTIFIED_TRANSITION_COLUMNS)
    rows: list[dict] = []
    passed = cert[cert["passes_multiplicity"].eq(True)].copy()
    for (mode, h, direction), g in passed.groupby(
        ["return_mode", "horizon", "direction"], sort=True
    ):
        if topology is not None and not topology.empty:
            all_folds = topology[
                (topology["return_mode"] == mode) & (topology["horizon"] == h)
            ]["fold"].nunique()
        else:
            all_folds = cert[
                (cert["return_mode"] == mode) & (cert["horizon"] == h)
            ]["fold"].nunique()
        if all_folds == 0:
            continue
        g = g.sort_values("root_quantile")
        clusters: list[list[int]] = []
        centers: list[float] = []
        for idx, row in g.iterrows():
            q = float(row["root_quantile"])
            if not clusters:
                clusters.append([idx])
                centers.append(q)
                continue
            distances = np.abs(np.asarray(centers) - q)
            j = int(np.argmin(distances))
            if distances[j] <= scfg.cert_root_cluster_q:
                clusters[j].append(idx)
                centers[j] = float(np.median(g.loc[clusters[j], "root_quantile"]))
            else:
                clusters.append([idx])
                centers.append(q)

        for cluster_id, members in enumerate(clusters, start=1):
            cg = g.loc[members].copy()
            # At most one candidate per fold: keep the one closest to the
            # cluster median so a wiggly fold cannot create artificial support.
            med0 = float(np.median(cg["root_quantile"]))
            cg["_dist"] = np.abs(cg["root_quantile"] - med0)
            cg = (
                cg.sort_values(["fold", "_dist"])
                .groupby("fold", as_index=False)
                .first()
            )
            support = int(cg["fold"].nunique())
            support_fraction = support / float(all_folds)
            qmed = float(np.median(cg["root_quantile"]))
            q25, q75 = np.quantile(cg["root_quantile"], [0.25, 0.75]) if len(cg) > 1 else (qmed, qmed)
            iqr = float(q75 - q25)
            min_required = max(
                int(scfg.cert_min_folds),
                int(math.ceil(scfg.cert_min_fold_fraction * all_folds)),
            )
            stable = support >= min_required and iqr <= scfg.cert_max_root_iqr_q
            rows.append({
                "return_mode": mode,
                "horizon": int(h),
                "direction": direction,
                "cluster_id": cluster_id,
                "folds_total": int(all_folds),
                "folds_support": support,
                "support_fraction": float(support_fraction),
                "min_folds_required": int(min_required),
                "median_root_psi": float(np.median(cg["psi_star"])),
                "median_root_quantile": qmed,
                "root_quantile_q25": float(q25),
                "root_quantile_q75": float(q75),
                "root_quantile_iqr": iqr,
                "max_root_iqr_allowed": float(scfg.cert_max_root_iqr_q),
                "certified_transition": bool(stable),
                "folds_json": json.dumps(sorted(int(x) for x in cg["fold"].unique())),
            })
    return _frame_with_schema(rows, CERTIFIED_TRANSITION_COLUMNS)


def _certified_topology(clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return _frame_with_schema([], CERTIFIED_TOPOLOGY_COLUMNS)
    rows: list[dict] = []
    z = clusters[clusters["certified_transition"].eq(True)].copy()
    for (mode, h), g in z.groupby(["return_mode", "horizon"], sort=True):
        g = g.sort_values("median_root_quantile")
        dirs = g["direction"].tolist()
        if not dirs:
            continue
        states = ["R" if dirs[0] == "R_to_M" else "M"]
        consistent = True
        for d in dirs:
            expected = "R_to_M" if states[-1] == "R" else "M_to_R"
            if d != expected:
                consistent = False
                break
            states.append("M" if states[-1] == "R" else "R")
        rows.append({
            "return_mode": mode,
            "horizon": int(h),
            "n_certified_transitions": int(len(g)),
            "certified_phase_path": "→".join(states) if consistent else "INCONSISTENT",
            "directions_json": json.dumps(dirs),
            "root_quantiles_json": json.dumps(
                [float(x) for x in g["median_root_quantile"].tolist()]
            ),
            "topology_consistent": bool(consistent),
        })
    return _frame_with_schema(rows, CERTIFIED_TOPOLOGY_COLUMNS)


def _single_phase_fold_certification(
    train_h: pd.DataFrame,
    test_h: pd.DataFrame,
    *,
    train_n_roots: int,
    block_len: int,
    scfg: PhaseSurfaceConfig,
    seed: int,
) -> dict:
    """Strong no-transition test: every OOS Psi band must have one sign."""
    if train_n_roots != 0:
        return {
            "eligible": False,
            "reason": "train_surface_has_candidate_roots",
            "certified_single_phase": False,
        }
    psi_train = train_h["psi_primary"].to_numpy(float)
    q = np.linspace(0.0, 1.0, scfg.cert_single_phase_bins + 1)
    edges = np.unique(np.quantile(psi_train[np.isfinite(psi_train)], q))
    if len(edges) != scfg.cert_single_phase_bins + 1:
        return {
            "eligible": False,
            "reason": "degenerate_psi_bins",
            "certified_single_phase": False,
        }
    edges[0], edges[-1] = -np.inf, np.inf

    train_scale = _weighted_std(
        train_h["phase_product"].to_numpy(float), _date_weights(train_h)
    )
    epsilon = max(
        float(scfg.cert_abs_effect),
        float(scfg.cert_min_effect_sigma) * train_scale if np.isfinite(train_scale) else 0.0,
    )
    signs: list[int] = []
    band_details: list[dict] = []
    for b in range(scfg.cert_single_phase_bins):
        gb = test_h[
            (test_h["psi_primary"] >= edges[b])
            & (test_h["psi_primary"] < edges[b + 1])
        ]
        if len(gb) < scfg.cert_min_obs_per_side or gb["date"].nunique() < scfg.cert_min_dates_per_side:
            return {
                "eligible": False,
                "reason": "insufficient_oos_band_support",
                "certified_single_phase": False,
            }
        st = _bootstrap_phase_mean(
            gb,
            block_len=block_len,
            reps=scfg.bootstrap_reps,
            seed=seed + 101 * (b + 1),
        )
        if st["estimate"] is None or st["ci_lo"] is None or st["ci_hi"] is None:
            return {
                "eligible": False,
                "reason": "bootstrap_unavailable",
                "certified_single_phase": False,
            }
        est = float(st["estimate"])
        if est >= epsilon and float(st["ci_lo"]) > 0.0:
            sign = +1
            pdir = st["p_gt0"]
        elif est <= -epsilon and float(st["ci_hi"]) < 0.0:
            sign = -1
            pdir = st["p_lt0"]
        else:
            sign = 0
            pdir = None
        signs.append(sign)
        band_details.append({
            "bin": b,
            "estimate": est,
            "ci_lo": st["ci_lo"],
            "ci_hi": st["ci_hi"],
            "p_directional": pdir,
            "n": int(len(gb)),
            "n_dates": int(gb["date"].nunique()),
        })

    all_pos = all(s == +1 for s in signs)
    all_neg = all(s == -1 for s in signs)
    phase = "M" if all_pos else "R" if all_neg else ""
    p_band = [d["p_directional"] for d in band_details if d["p_directional"] is not None]
    # IUT across all bands: all bands need the same-sign alternative.
    p_iut = max(float(x) for x in p_band) if len(p_band) == len(band_details) else float("nan")
    return {
        "eligible": True,
        "reason": "",
        "epsilon": float(epsilon),
        "phase": phase,
        "candidate_p_iut": p_iut,
        "certified_single_phase": bool((all_pos or all_neg) and p_iut <= scfg.cert_alpha),
        "bands_json": json.dumps(band_details),
    }


def _aggregate_single_phase(folds: pd.DataFrame, scfg: PhaseSurfaceConfig) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (mode, h), g in folds.groupby(["return_mode", "horizon"], sort=True):
        all_folds = int(g["fold"].nunique())
        gg = g[g["certified_single_phase"].eq(True)].copy()
        if gg.empty:
            rows.append({
                "return_mode": mode,
                "horizon": int(h),
                "folds_total": all_folds,
                "folds_support": 0,
                "phase": "",
                "certified_single_phase": False,
            })
            continue
        counts = gg["phase"].value_counts()
        phase = str(counts.index[0])
        support = int(counts.iloc[0])
        min_required = max(
            int(scfg.cert_min_folds),
            int(math.ceil(scfg.cert_min_fold_fraction * all_folds)),
        )
        rows.append({
            "return_mode": mode,
            "horizon": int(h),
            "folds_total": all_folds,
            "folds_support": support,
            "support_fraction": support / float(all_folds),
            "min_folds_required": min_required,
            "phase": phase,
            "certified_single_phase": bool(support >= min_required),
        })
    return pd.DataFrame(rows)


def _exploratory_hypothesis_adjustment(
    boot_int: pd.DataFrame,
    boot_lin: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Report discovery-sample H1/H2 with Holm correction, explicitly exploratory."""
    specs = [
        ("H1", boot_int, "spline_interaction_minus_additive"),
        ("H2", boot_lin, "spectral_plus_controls_minus_controls"),
    ]
    rows: list[dict] = []
    for hid, table, comp in specs:
        if table.empty:
            continue
        q = table[
            (table["return_mode"] == "raw")
            & (table["metric"] == "mean_datewise_auc_delta")
        ]
        if q.empty:
            continue
        r = q.iloc[0]
        rows.append({
            "hypothesis": hid,
            "comparison": comp,
            "return_mode": "raw",
            "metric": "mean_datewise_auc_delta",
            "estimate": float(r["estimate"]),
            "p_one_sided_gt0": float(r["p_gt0"]) if pd.notna(r.get("p_gt0")) else np.nan,
            "status": "EXPLORATORY_ONLY_HYPOTHESES_WERE_FORMULATED_FROM_THIS_SAMPLE",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_h1_h2"] = _holm_adjust(out["p_one_sided_gt0"].to_numpy(float))
        out["passes_holm_0_05_exploratory_only"] = (
            (out["estimate"] > 0.0)
            & (out["p_holm_h1_h2"] <= float(alpha))
        )
    return out


def _prospective_confirmatory_protocol(panel: pd.DataFrame, scfg: PhaseSurfaceConfig) -> dict:
    """Freeze the two discovery-derived hypotheses for a genuinely new sample."""
    discovery_end = pd.Timestamp(panel["date"].max()).date().isoformat()
    protocol = {
        "status": "PROSPECTIVE_ONLY_DO_NOT_TREAT_CURRENT_SAMPLE_AS_CONFIRMATORY",
        "discovery_sample_end": discovery_end,
        "holdout_requirement": (
            "Test on observations strictly after discovery_sample_end or on a "
            "pre-specified external universe not used to formulate H1/H2."
        ),
        "alpha_familywise": 0.05,
        "multiplicity_correction": "Holm-Bonferroni across H1 and H2",
        "one_sided": True,
        "frozen_model_hyperparameters": {
            "spline_knots": scfg.spline_knots,
            "spline_degree": scfg.spline_degree,
            "ridge_alpha": scfg.ridge_alpha,
            "logistic_c": scfg.logistic_c,
            "quantile_bins": scfg.quantile_bins,
        },
        "hypotheses": [
            {
                "id": "H1",
                "return_mode": "raw",
                "metric": "mean_datewise_auc_delta",
                "alternative": "greater",
                "comparison": "spline_interaction_minus_additive",
                "claim": "AUC[s(Psi,log h)] > AUC[s(Psi)+s(log h)]",
            },
            {
                "id": "H2",
                "return_mode": "raw",
                "metric": "mean_datewise_auc_delta",
                "alternative": "greater",
                "comparison": "spectral_plus_controls_minus_controls",
                "claim": "AUC[spectral+controls] > AUC[controls]",
            },
        ],
    }
    import hashlib
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    protocol["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return protocol

def _bootstrap_datewise_table(datewise: pd.DataFrame, cfg, scfg: PhaseSurfaceConfig, comparison: str) -> pd.DataFrame:
    rows: list[dict] = []
    # Panel dates are sampled every anchor_stride rows.  This number of sampled
    # dates therefore spans at least the maximum forward horizon.
    block_len = max(1, int(math.ceil(max(cfg.horizons) / cfg.anchor_stride)))
    for j, (mode, g) in enumerate(datewise.groupby("return_mode", sort=False)):
        g = g.sort_values("date")
        rng = np.random.default_rng(scfg.random_seed + 1009 * (j + 1))
        auc = _moving_block_bootstrap(g["delta_auc"].to_numpy(float), block_len, scfg.bootstrap_reps, rng)
        bri = _moving_block_bootstrap(g["delta_brier_improvement"].to_numpy(float), block_len, scfg.bootstrap_reps, rng)
        rows.append({
            "comparison": comparison,
            "return_mode": mode,
            "metric": "mean_datewise_auc_delta",
            "estimate": auc["estimate"],
            "ci_lo": auc["ci_lo"],
            "ci_hi": auc["ci_hi"],
            "p_two_sided": auc["p_two_sided"],
            "p_gt0": auc.get("p_gt0"),
            "p_lt0": auc.get("p_lt0"),
            "n_dates": auc["n_dates"],
            "block_len_sampled_dates": block_len,
            "block_span_trading_days_approx": block_len * cfg.anchor_stride,
            "bootstrap_reps": scfg.bootstrap_reps,
        })
        rows.append({
            "comparison": comparison,
            "return_mode": mode,
            "metric": "mean_datewise_brier_improvement",
            "estimate": bri["estimate"],
            "ci_lo": bri["ci_lo"],
            "ci_hi": bri["ci_hi"],
            "p_two_sided": bri["p_two_sided"],
            "p_gt0": bri.get("p_gt0"),
            "p_lt0": bri.get("p_lt0"),
            "n_dates": bri["n_dates"],
            "block_len_sampled_dates": block_len,
            "block_span_trading_days_approx": block_len * cfg.anchor_stride,
            "bootstrap_reps": scfg.bootstrap_reps,
        })
    return pd.DataFrame(rows)


def run_phase_surface_analysis(
    panel: pd.DataFrame,
    cfg,
    out_dir: Path,
    *,
    quantile_bins: int = 30,
    spline_knots: int = 7,
    bootstrap_reps: int = 500,
    max_fit_rows: int = 250_000,
    cert_alpha: float = 0.05,
    cert_side_width_q: float = 0.08,
    cert_side_gap_q: float = 0.01,
    cert_min_effect_sigma: float = 0.02,
    cert_abs_effect: float = 0.0,
    cert_min_obs_per_side: int = 250,
    cert_min_dates_per_side: int = 40,
    cert_min_folds: int = 3,
    cert_min_fold_fraction: float = 0.75,
    cert_root_cluster_q: float = 0.06,
    cert_max_root_iqr_q: float = 0.05,
    cert_single_phase_bins: int = 5,
) -> dict:
    """Run the non-monotone phase-surface research extension and write outputs."""
    scfg = PhaseSurfaceConfig(
        quantile_bins=quantile_bins,
        spline_knots=spline_knots,
        bootstrap_reps=bootstrap_reps,
        max_fit_rows=max_fit_rows,
        random_seed=cfg.random_seed,
        cert_alpha=cert_alpha,
        cert_side_width_q=cert_side_width_q,
        cert_side_gap_q=cert_side_gap_q,
        cert_min_effect_sigma=cert_min_effect_sigma,
        cert_abs_effect=cert_abs_effect,
        cert_min_obs_per_side=cert_min_obs_per_side,
        cert_min_dates_per_side=cert_min_dates_per_side,
        cert_min_folds=cert_min_folds,
        cert_min_fold_fraction=cert_min_fold_fraction,
        cert_root_cluster_q=cert_root_cluster_q,
        cert_max_root_iqr_q=cert_max_root_iqr_q,
        cert_single_phase_bins=cert_single_phase_bins,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    quantile_rows: list[dict] = []
    continuous_rows: list[dict] = []
    logistic_rows: list[dict] = []
    root_rows: list[dict] = []
    topology_rows: list[dict] = []
    root_cert_rows: list[dict] = []
    single_phase_fold_rows: list[dict] = []
    grid_rows: list[pd.DataFrame] = []
    interaction_datewise: list[dict] = []
    linear_datewise: list[dict] = []

    spec_cols = [f"psi_{s}" for s in cfg.spectral_spans]
    control_cols = ["vol20", "range20", "log_dollar_volume20"]

    for mode_i, (mode, pm) in enumerate(panel.groupby("return_mode", sort=False)):
        blocks = _walkforward_date_blocks(pm, cfg)
        for fold, (test_start, test_end) in enumerate(blocks, start=1):
            purge = pd.offsets.BDay(max(cfg.horizons))
            train = pm[pm["date"] < (test_start - purge)].copy()
            test = pm[(pm["date"] >= test_start) & (pm["date"] <= test_end)].copy()
            if len(train) < 1000 or len(test) < 200:
                continue
            meta = {
                "return_mode": mode,
                "fold": fold,
                "test_start": test_start,
                "test_end": test_end,
            }
            fit_seed = scfg.random_seed + 100_000 * (mode_i + 1) + fold
            fit_train = _balanced_fit_sample(train, scfg.max_fit_rows, fit_seed)
            wtr = _date_weights(fit_train)

            # 1) OOS non-parametric quantile curves.  Bin edges come only from train.
            quantile_rows.extend(_quantile_oos_rows(train, test, scfg.quantile_bins, meta))

            models: dict[str, SplineSurface] = {}
            for interaction, context, name in [
                (False, False, "spline_additive"),
                (True, False, "spline_interaction"),
                (True, True, "spline_interaction_context"),
            ]:
                cm = SplineSurface(
                    task="continuous",
                    interaction=interaction,
                    context=context,
                    n_knots=scfg.spline_knots,
                    degree=scfg.spline_degree,
                    ridge_alpha=scfg.ridge_alpha,
                    logistic_c=scfg.logistic_c,
                ).fit(fit_train, wtr)
                gp = cm.predict(test)
                continuous_rows.append(_continuous_metrics(test, gp, train, name, meta))
                models[f"continuous::{name}"] = cm

                lm = SplineSurface(
                    task="logistic",
                    interaction=interaction,
                    context=context,
                    n_knots=scfg.spline_knots,
                    degree=scfg.spline_degree,
                    ridge_alpha=scfg.ridge_alpha,
                    logistic_c=scfg.logistic_c,
                ).fit(fit_train, wtr)
                pp = lm.predict(test)
                logistic_rows.append(_logistic_metrics(test, pp, train, name, meta))
                models[f"logistic::{name}"] = lm

            # 2/3/4) All zeros of the continuous tensor-product spline surface.
            surface = models["continuous::spline_interaction"]
            lo, hi = np.nanquantile(train["psi_primary"].to_numpy(float), [scfg.root_domain_qlo, scfg.root_domain_qhi])
            psi_grid = np.linspace(lo, hi, scfg.root_grid_size)
            for h in cfg.horizons:
                eval_df = pd.DataFrame({
                    "psi_primary": psi_grid,
                    "horizon": np.full_like(psi_grid, h, dtype=float),
                })
                g = surface.predict(eval_df)
                roots = _roots_from_curve(psi_grid, g)
                path = _phase_path(psi_grid, g, roots)
                topology_rows.append({
                    **meta,
                    "horizon": int(h),
                    "model": "spline_interaction",
                    "psi_domain_lo": float(lo),
                    "psi_domain_hi": float(hi),
                    "n_roots": int(len(roots)),
                    "phase_path": path,
                    "roots_json": json.dumps([r["psi_star"] for r in roots]),
                })
                train_h = train[train["horizon"] == h]
                test_h = test[test["horizon"] == h]
                cert_block_len = max(
                    1, int(math.ceil(max(cfg.horizons) / cfg.anchor_stride))
                )
                for j, root in enumerate(roots, start=1):
                    root_rows.append({
                        **meta,
                        "horizon": int(h),
                        "model": "spline_interaction",
                        "root_index": j,
                        **root,
                    })
                    cert = _candidate_root_oos_certification(
                        train_h,
                        test_h,
                        root,
                        block_len=cert_block_len,
                        scfg=scfg,
                        seed=fit_seed + 10_000 * int(h) + 97 * j,
                    )
                    root_cert_rows.append({
                        **meta,
                        "horizon": int(h),
                        "model": "spline_interaction",
                        "root_index": j,
                        **root,
                        **cert,
                    })

                single = _single_phase_fold_certification(
                    train_h,
                    test_h,
                    train_n_roots=len(roots),
                    block_len=cert_block_len,
                    scfg=scfg,
                    seed=fit_seed + 50_000 * int(h),
                )
                single_phase_fold_rows.append({
                    **meta,
                    "horizon": int(h),
                    "model": "spline_interaction",
                    "train_n_roots": int(len(roots)),
                    **single,
                })

                # Thin grid persisted for plotting/reproducibility.
                take = np.linspace(0, len(psi_grid) - 1, 256, dtype=int)
                grid_rows.append(pd.DataFrame({
                    **{k: [v] * len(take) for k, v in meta.items()},
                    "horizon": int(h),
                    "psi": psi_grid[take],
                    "g_hat": g[take],
                }))

            # Context-conditioned roots g(psi,h,sigma,C)=0 at class-specific
            # train medians for the three OHLCV controls.
            csurface = models["continuous::spline_interaction_context"]
            for cls, trc in train.groupby("asset_class", sort=True):
                med = trc[control_cols].median(numeric_only=True)
                for h in cfg.horizons:
                    eval_df = pd.DataFrame({
                        "psi_primary": psi_grid,
                        "horizon": np.full_like(psi_grid, h, dtype=float),
                        "asset_class": str(cls),
                        "vol20": float(med.get("vol20", np.nan)),
                        "range20": float(med.get("range20", np.nan)),
                        "log_dollar_volume20": float(med.get("log_dollar_volume20", np.nan)),
                    })
                    g = csurface.predict(eval_df)
                    roots = _roots_from_curve(psi_grid, g)
                    for j, root in enumerate(roots, start=1):
                        root_rows.append({
                            **meta,
                            "horizon": int(h),
                            "model": "spline_interaction_context",
                            "asset_class": str(cls),
                            "root_index": j,
                            **root,
                        })

            # Test whether the interaction itself improves OOS discrimination.
            p_add = models["logistic::spline_additive"].predict(test)
            p_int = models["logistic::spline_interaction"].predict(test)
            interaction_datewise.extend(_datewise_pair_metrics(
                test, p_add, p_int, meta,
                ("spline_additive", "spline_interaction"),
            ))

            # 5) Paired time-OOS controls-only vs spectral+controls comparator.
            pair_preds: list[np.ndarray] = []
            for cols in [control_cols, spec_cols + control_cols]:
                pipe = _make_linear_comparator(cols)
                pipe.fit(fit_train, fit_train["phase_label"].astype(int), clf__sample_weight=wtr)
                pair_preds.append(pipe.predict_proba(test)[:, 1])
            linear_datewise.extend(_datewise_pair_metrics(
                test, pair_preds[0], pair_preds[1], meta,
                ("controls_only", "spectral_plus_controls"),
            ))

            print(f"phase-surface {mode} fold {fold}/{len(blocks)} complete")

    quantiles = pd.DataFrame(quantile_rows)
    continuous = pd.DataFrame(continuous_rows)
    logistic = pd.DataFrame(logistic_rows)
    roots = pd.DataFrame(root_rows)
    topology = pd.DataFrame(topology_rows)
    root_cert = _apply_root_multiplicity(pd.DataFrame(root_cert_rows), scfg.cert_alpha)
    transition_clusters = _cluster_certified_roots(root_cert, scfg, topology)
    certified_topology = _certified_topology(transition_clusters)
    single_phase_folds = pd.DataFrame(single_phase_fold_rows)
    single_phase_regimes = _aggregate_single_phase(single_phase_folds, scfg)
    grids = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame()
    int_dw = pd.DataFrame(interaction_datewise)
    lin_dw = pd.DataFrame(linear_datewise)

    quantiles.to_csv(out_dir / "quantile_phase_curve_oos.csv", index=False)
    continuous.to_csv(out_dir / "metrics_phase_product_spline.csv", index=False)
    logistic.to_csv(out_dir / "metrics_logistic_spline.csv", index=False)
    roots.to_csv(out_dir / "phase_roots.csv", index=False)
    topology.to_csv(out_dir / "phase_topology.csv", index=False)
    root_cert.to_csv(out_dir / "phase_root_certification_oos.csv", index=False)
    transition_clusters.to_csv(out_dir / "certified_phase_transitions.csv", index=False)
    certified_topology.to_csv(out_dir / "certified_phase_topology.csv", index=False)
    single_phase_folds.to_csv(out_dir / "single_phase_fold_certification.csv", index=False)
    single_phase_regimes.to_csv(out_dir / "certified_single_phase_regimes.csv", index=False)
    grids.to_csv(out_dir / "phase_surface_grid.csv.gz", index=False, compression="gzip")
    int_dw.to_csv(out_dir / "datewise_interaction_delta.csv", index=False)
    lin_dw.to_csv(out_dir / "datewise_model_delta.csv", index=False)

    boot_int = _bootstrap_datewise_table(int_dw, cfg, scfg, "spline_interaction_minus_additive") if not int_dw.empty else pd.DataFrame()
    boot_lin = _bootstrap_datewise_table(lin_dw, cfg, scfg, "spectral_plus_controls_minus_controls") if not lin_dw.empty else pd.DataFrame()
    boot_int.to_csv(out_dir / "bootstrap_interaction_delta.csv", index=False)
    boot_lin.to_csv(out_dir / "bootstrap_model_delta.csv", index=False)

    exploratory_h = _exploratory_hypothesis_adjustment(
        boot_int, boot_lin, alpha=scfg.cert_alpha
    )
    exploratory_h.to_csv(
        out_dir / "exploratory_h1_h2_holm.csv", index=False
    )
    protocol = _prospective_confirmatory_protocol(panel, scfg)
    with (out_dir / "prospective_confirmatory_protocol.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(protocol, f, indent=2, default=str)

    summary: dict = {
        "definition": "g(psi,h,...) = E[phase_product | spectral state]; momentum iff g>0, reversal iff g<0",
        "quantile_bins": scfg.quantile_bins,
        "spline_knots": scfg.spline_knots,
        "bootstrap_reps": scfg.bootstrap_reps,
        "max_fit_rows_per_fold": scfg.max_fit_rows,
        "phase_certification": {
            "principle": "train proposes candidate roots; future OOS data certify both sides",
            "alpha": scfg.cert_alpha,
            "side_width_quantile": scfg.cert_side_width_q,
            "side_gap_quantile": scfg.cert_side_gap_q,
            "min_effect_sigma": scfg.cert_min_effect_sigma,
            "absolute_effect_floor": scfg.cert_abs_effect,
            "min_obs_per_side": scfg.cert_min_obs_per_side,
            "min_dates_per_side": scfg.cert_min_dates_per_side,
            "within_curve_multiplicity": "Holm-Bonferroni",
            "min_folds": scfg.cert_min_folds,
            "min_fold_fraction": scfg.cert_min_fold_fraction,
            "root_cluster_tolerance_quantile": scfg.cert_root_cluster_q,
            "max_root_iqr_quantile": scfg.cert_max_root_iqr_q,
        },
        "prospective_confirmatory_protocol_sha256": protocol["protocol_sha256"],
        "prospective_confirmatory_status": protocol["status"],
    }
    if not continuous.empty:
        summary["continuous_spline_mean"] = (
            continuous.groupby(["return_mode", "model"])[["mse_skill", "spearman_pred_product", "phase_auc_from_g"]]
            .mean().reset_index().to_dict("records")
        )
    if not logistic.empty:
        summary["logistic_spline_mean"] = (
            logistic.groupby(["return_mode", "model"])[["auc", "brier_skill"]]
            .mean().reset_index().to_dict("records")
        )
    if not topology.empty:
        topo = topology.groupby(["return_mode", "horizon"]).agg(
            mean_n_roots=("n_roots", "mean"),
            folds_with_roots=("n_roots", lambda s: int(np.sum(np.asarray(s) > 0))),
            folds=("n_roots", "size"),
        ).reset_index()
        summary["phase_topology_by_horizon"] = topo.to_dict("records")
        path_counts = (
            topology.groupby(["return_mode", "horizon", "phase_path"]).size()
            .reset_index(name="fold_count")
            .sort_values(["return_mode", "horizon", "fold_count"], ascending=[True, True, False])
        )
        summary["phase_path_counts"] = path_counts.to_dict("records")
    if not boot_int.empty:
        summary["interaction_block_bootstrap"] = boot_int.to_dict("records")
    if not boot_lin.empty:
        summary["spectral_increment_block_bootstrap"] = boot_lin.to_dict("records")
    if not root_cert.empty:
        summary["root_certification_counts"] = {
            "candidate_roots": int(len(root_cert)),
            "eligible_oos": int(root_cert["eligible"].eq(True).sum()),
            "local_effect_ci_pass": int(root_cert["local_effect_ci_pass"].eq(True).sum()),
            "holm_pass": int(root_cert["passes_multiplicity"].eq(True).sum()),
        }
    if not transition_clusters.empty:
        summary["certified_transition_clusters"] = (
            transition_clusters[
                transition_clusters["certified_transition"].eq(True)
            ].to_dict("records")
        )
    if not certified_topology.empty:
        summary["certified_phase_topology"] = certified_topology.to_dict("records")
    if not single_phase_regimes.empty:
        summary["certified_single_phase_regimes"] = (
            single_phase_regimes[
                single_phase_regimes["certified_single_phase"].eq(True)
            ].to_dict("records")
        )
    if not exploratory_h.empty:
        summary["exploratory_h1_h2_holm"] = exploratory_h.to_dict("records")

    with (out_dir / "phase_surface_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    make_phase_surface_plots(out_dir)
    return summary


def make_phase_surface_plots(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Upgrade any legacy v5 headerless empty certification artifacts before
    # reading them.  This is intentionally idempotent.
    repair_empty_phase_certification_csvs(out_dir)

    qpath = out_dir / "quantile_phase_curve_oos.csv"
    if qpath.exists():
        q = _read_csv_or_empty(qpath)
        if not q.empty:
            for mode, gm in q.groupby("return_mode", sort=False):
                g = gm.groupby(["horizon", "psi_bin"], as_index=False).agg(
                    mean_psi=("mean_psi", "mean"),
                    mean_phase_product=("mean_phase_product", "mean"),
                )
                fig, ax = plt.subplots(figsize=(11, 6))
                for h, gh in g.groupby("horizon", sort=True):
                    ax.plot(gh["mean_psi"], gh["mean_phase_product"], marker="o", markersize=2.5, linewidth=1.0, label=str(h))
                ax.axhline(0.0, linewidth=1.0)
                ax.set_xlabel("Psi (OOS observations in train-defined quantile bins)")
                ax.set_ylabel("E[phase_product | Psi bin]")
                ax.set_title(f"Non-parametric spectral phase curves — {mode}")
                ax.legend(title="h", ncol=3)
                fig.tight_layout()
                fig.savefig(out_dir / f"quantile_phase_curves_{mode}.png", dpi=180)
                plt.close(fig)

    gpath = out_dir / "phase_surface_grid.csv.gz"
    if gpath.exists():
        g = _read_csv_or_empty(gpath)
        if not g.empty:
            for mode, gm in g.groupby("return_mode", sort=False):
                # Average fold-specific fitted surfaces on a common Psi rank grid.
                horizons = sorted(gm["horizon"].unique())
                fig, ax = plt.subplots(figsize=(11, 6))
                for h in horizons:
                    gh = gm[gm["horizon"] == h].copy()
                    # Curves can have slightly different Psi supports; interpolate
                    # each fold to a common overlap grid before averaging.
                    folds = []
                    los, his = [], []
                    for _, gf in gh.groupby("fold"):
                        gf = gf.sort_values("psi")
                        folds.append(gf)
                        los.append(gf["psi"].min())
                        his.append(gf["psi"].max())
                    if not folds:
                        continue
                    lo, hi = max(los), min(his)
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                        continue
                    x = np.linspace(lo, hi, 300)
                    yy = [np.interp(x, f["psi"], f["g_hat"]) for f in folds]
                    y = np.mean(np.vstack(yy), axis=0)
                    ax.plot(x, y, label=str(int(h)))
                ax.axhline(0.0, linewidth=1.0)
                ax.set_xlabel("Psi")
                ax.set_ylabel("Spline estimate of g_h(Psi)")
                ax.set_title(f"Non-monotone fitted phase curves — {mode}")
                ax.legend(title="h", ncol=3)
                fig.tight_layout()
                fig.savefig(out_dir / f"spline_phase_curves_{mode}.png", dpi=180)
                plt.close(fig)

    rpath = out_dir / "phase_roots.csv"
    if rpath.exists():
        r = _read_csv_or_empty(rpath)
        r = r[r.get("model", "") == "spline_interaction"] if not r.empty else r
        if not r.empty:
            for mode, gm in r.groupby("return_mode", sort=False):
                fig, ax = plt.subplots(figsize=(10, 6))
                for (fold, root_index), gr in gm.groupby(["fold", "root_index"]):
                    gr = gr.sort_values("horizon")
                    ax.plot(gr["horizon"], gr["psi_star"], marker="o", linewidth=1.0, alpha=0.75, label=f"fold {fold}, root {root_index}")
                ax.axhline(0.0, linewidth=1.0)
                ax.set_xscale("log")
                ax.set_xlabel("Horizon h (trading days, log scale)")
                ax.set_ylabel("Zero crossing Psi*(h)")
                ax.set_title(f"Multiple spectral phase boundaries — {mode}")
                if gm.groupby(["fold", "root_index"]).ngroups <= 12:
                    ax.legend(fontsize=8, ncol=2)
                fig.tight_layout()
                fig.savefig(out_dir / f"phase_roots_{mode}.png", dpi=180)
                plt.close(fig)


    cpath = out_dir / "certified_phase_transitions.csv"
    if cpath.exists():
        c = _read_csv_or_empty(cpath)
        if not c.empty and "certified_transition" in c.columns:
            c = c[c["certified_transition"].astype(str).str.lower().isin(["true", "1"])]
            for mode, gm in c.groupby("return_mode", sort=False):
                if gm.empty:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                for direction, gd in gm.groupby("direction", sort=False):
                    gd = gd.sort_values("horizon")
                    ax.scatter(
                        gd["horizon"],
                        gd["median_root_psi"],
                        label=direction,
                        s=45,
                    )
                ax.axhline(0.0, linewidth=1.0)
                ax.set_xscale("log")
                ax.set_xlabel("Horizon h (trading days, log scale)")
                ax.set_ylabel("Certified median Psi*(h)")
                ax.set_title(f"Certified OOS spectral phase transitions — {mode}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / f"certified_phase_transitions_{mode}.png", dpi=180)
                plt.close(fig)
