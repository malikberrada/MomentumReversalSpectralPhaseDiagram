import numpy as np
import pandas as pd

from mrspd.phase_surface import (
    PhaseSurfaceConfig,
    SplineSurface,
    _apply_root_multiplicity,
    _balanced_fit_sample,
    _candidate_root_oos_certification,
    _cluster_certified_roots,
    _date_weights,
    _phase_path,
    _roots_from_curve,
    _single_phase_fold_certification,
)


def _synthetic_phase(seed=7):
    rng = np.random.default_rng(seed)
    horizons = np.array([5, 10, 21, 42, 63, 126])
    dates = pd.bdate_range("2020-01-01", periods=180)
    rows = []
    for date in dates:
        for h in horizons:
            psi = rng.uniform(-1.8, 1.8, 28)
            # R -> M -> R, with horizon-dependent center.
            center = 0.16 * np.log(h / 21.0)
            g = 0.62 - (psi - center) ** 2
            y = g + rng.normal(0.0, 0.22, size=len(psi))
            for p, yy in zip(psi, y):
                rows.append((date, h, p, yy, float(yy > 0), "us_stock", 0.20, 0.02, 20.0))
    return pd.DataFrame(rows, columns=[
        "date", "horizon", "psi_primary", "phase_product", "phase_label",
        "asset_class", "vol20", "range20", "log_dollar_volume20",
    ])


def test_multiple_roots_and_phase_path():
    df = _synthetic_phase()
    model = SplineSurface(
        task="continuous", interaction=True, context=False,
        n_knots=7, degree=3, ridge_alpha=1.0, logistic_c=0.5,
    ).fit(df, _date_weights(df))

    h = 21
    x = np.linspace(-1.5, 1.5, 2000)
    pred = model.predict(pd.DataFrame({"psi_primary": x, "horizon": h}))
    roots = _roots_from_curve(x, pred)
    assert len(roots) == 2
    vals = np.array([r["psi_star"] for r in roots])
    assert np.allclose(vals, [-np.sqrt(0.62), np.sqrt(0.62)], atol=0.22)
    assert _phase_path(x, pred, roots) == "R→M→R"


def test_horizon_interaction_moves_roots():
    df = _synthetic_phase(11)
    model = SplineSurface(
        task="continuous", interaction=True, context=False,
        n_knots=7, degree=3, ridge_alpha=1.0, logistic_c=0.5,
    ).fit(df, _date_weights(df))

    centers = []
    for h in (5, 126):
        x = np.linspace(-1.7, 1.7, 2000)
        pred = model.predict(pd.DataFrame({"psi_primary": x, "horizon": h}))
        roots = _roots_from_curve(x, pred)
        assert len(roots) == 2
        centers.append(np.mean([r["psi_star"] for r in roots]))
    assert centers[1] > centers[0] + 0.20


def test_balanced_fit_sample_handles_readonly_numpy_views():
    # Regression test for pandas Copy-on-Write / read-only NumPy buffers.
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=40), 50)
    df = pd.DataFrame({
        "date": dates,
        "x": np.arange(len(dates), dtype=float),
    })
    try:
        old_cow = pd.options.mode.copy_on_write
        pd.options.mode.copy_on_write = True
    except Exception:
        old_cow = None
    try:
        out = _balanced_fit_sample(df, max_rows=500, random_state=123)
    finally:
        if old_cow is not None:
            pd.options.mode.copy_on_write = old_cow
    assert len(out) == 500
    assert out.index.is_monotonic_increasing
    assert out["date"].nunique() >= 30


def test_balanced_fit_sample_is_deterministic():
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=25), 32)
    df = pd.DataFrame({"date": dates, "x": np.arange(len(dates))})
    a = _balanced_fit_sample(df, max_rows=200, random_state=77)
    b = _balanced_fit_sample(df, max_rows=200, random_state=77)
    assert np.array_equal(a.index.to_numpy(), b.index.to_numpy())


def test_oos_root_certification_accepts_real_r_to_m_transition():
    rng = np.random.default_rng(123)
    train = _synthetic_phase(31)
    train_h = train[train["horizon"] == 21].copy()

    dates = pd.bdate_range("2022-01-03", periods=120)
    rows = []
    for date in dates:
        psi = rng.uniform(-1.6, 1.6, 80)
        # Clean R -> M transition at psi=0 for this unit test.
        y = 0.55 * psi + rng.normal(0.0, 0.12, len(psi))
        for pp, yy in zip(psi, y):
            rows.append((date, 21, pp, yy, float(yy > 0), "us_stock", 0.2, 0.02, 20.0))
    test_h = pd.DataFrame(rows, columns=[
        "date", "horizon", "psi_primary", "phase_product", "phase_label",
        "asset_class", "vol20", "range20", "log_dollar_volume20",
    ])
    cfg = PhaseSurfaceConfig(
        bootstrap_reps=300,
        cert_min_effect_sigma=0.01,
        cert_min_obs_per_side=100,
        cert_min_dates_per_side=30,
    )
    root = {"psi_star": 0.0, "direction": "R_to_M", "local_slope": 0.55}
    out = _candidate_root_oos_certification(
        train_h, test_h, root, block_len=8, scfg=cfg, seed=77
    )
    assert out["eligible"]
    assert out["local_effect_ci_pass"]
    assert out["left_g"] < 0 < out["right_g"]
    assert out["candidate_p_iut"] <= 0.05


def test_holm_and_cross_fold_root_clustering_require_replication():
    rows = []
    for fold, q in [(1, 0.39), (2, 0.41), (3, 0.40), (4, 0.72)]:
        rows.append({
            "return_mode": "raw", "fold": fold, "horizon": 21,
            "direction": "R_to_M", "psi_star": q * 2 - 1,
            "root_quantile": q, "eligible": True,
            "local_effect_ci_pass": True,
            "candidate_p_iut": 0.001 if fold <= 3 else 0.20,
        })
    cert = _apply_root_multiplicity(pd.DataFrame(rows), 0.05)
    scfg = PhaseSurfaceConfig(
        cert_min_folds=3,
        cert_min_fold_fraction=0.75,
        cert_root_cluster_q=0.05,
        cert_max_root_iqr_q=0.03,
    )
    clusters = _cluster_certified_roots(cert, scfg)
    good = clusters[clusters["certified_transition"]]
    assert len(good) == 1
    assert int(good.iloc[0]["folds_support"]) == 3
    assert abs(float(good.iloc[0]["median_root_quantile"]) - 0.40) < 0.02


def test_single_phase_certification_can_certify_reversal_only():
    rng = np.random.default_rng(2026)
    dates_train = pd.bdate_range("2020-01-01", periods=100)
    dates_test = pd.bdate_range("2021-01-01", periods=100)
    def make(dates):
        rows = []
        for d in dates:
            psi = rng.uniform(-1.5, 1.5, 100)
            y = -0.22 - 0.02 * psi**2 + rng.normal(0.0, 0.05, len(psi))
            for pp, yy in zip(psi, y):
                rows.append((d, 126, pp, yy, 0, "us_stock", 0.2, 0.02, 20.0))
        return pd.DataFrame(rows, columns=[
            "date", "horizon", "psi_primary", "phase_product", "phase_label",
            "asset_class", "vol20", "range20", "log_dollar_volume20",
        ])
    train_h, test_h = make(dates_train), make(dates_test)
    cfg = PhaseSurfaceConfig(
        bootstrap_reps=250,
        cert_min_effect_sigma=0.01,
        cert_min_obs_per_side=100,
        cert_min_dates_per_side=25,
        cert_single_phase_bins=5,
    )
    out = _single_phase_fold_certification(
        train_h, test_h, train_n_roots=0, block_len=8, scfg=cfg, seed=9
    )
    assert out["certified_single_phase"]
    assert out["phase"] == "R"


def test_empty_certified_transition_csv_is_schema_safe(tmp_path):
    from mrspd.phase_surface import (
        CERTIFIED_TRANSITION_COLUMNS,
        _certified_topology,
        make_phase_surface_plots,
    )

    # Simulate the scientifically valid outcome: zero replicated/certified roots.
    empty_cert = pd.DataFrame(columns=[
        "return_mode", "fold", "horizon", "direction", "psi_star",
        "root_quantile", "passes_multiplicity",
    ])
    cfg = PhaseSurfaceConfig()
    clusters = _cluster_certified_roots(empty_cert, cfg)
    assert clusters.empty
    assert list(clusters.columns) == CERTIFIED_TRANSITION_COLUMNS

    topology = _certified_topology(clusters)
    assert topology.empty
    assert "certified_phase_path" in topology.columns

    # Header-only files must be readable and plotting must simply skip them.
    clusters.to_csv(tmp_path / "certified_phase_transitions.csv", index=False)
    topology.to_csv(tmp_path / "certified_phase_topology.csv", index=False)
    make_phase_surface_plots(tmp_path)


def test_plotter_tolerates_legacy_headerless_empty_certification_csv(tmp_path):
    from mrspd.phase_surface import make_phase_surface_plots

    # v5 could emit a one-byte/headerless CSV when no transition survived.
    (tmp_path / "certified_phase_transitions.csv").write_text("\n", encoding="utf-8")
    make_phase_surface_plots(tmp_path)


def test_repair_legacy_empty_certification_artifacts(tmp_path):
    from mrspd.phase_surface import repair_empty_phase_certification_csvs

    for name in ("certified_phase_transitions.csv", "certified_phase_topology.csv"):
        (tmp_path / name).write_text("\n", encoding="utf-8")
    repaired = repair_empty_phase_certification_csvs(tmp_path)
    assert set(repaired) == {
        "certified_phase_transitions.csv", "certified_phase_topology.csv"
    }
    assert "certified_transition" in pd.read_csv(
        tmp_path / "certified_phase_transitions.csv"
    ).columns
    assert "certified_phase_path" in pd.read_csv(
        tmp_path / "certified_phase_topology.csv"
    ).columns
