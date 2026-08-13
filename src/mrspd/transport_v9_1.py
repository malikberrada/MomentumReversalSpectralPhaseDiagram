from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import transport_v9 as v9
from .critical_horizon import _find_onset, _holm_adjust
from .phase_surface import _moving_block_bootstrap, _walkforward_date_blocks

PROTOCOL_VERSION = "MRSPD-PERCENTILE-TRANSPORT-v9.1"
DEFAULT_Q0_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_EXTENDED_HORIZONS = (63,72,84,96,105,112,119,126,133,140,147,168,189,210,231,252)


@dataclass(frozen=True)
class V91Config:
    bootstrap_reps: int = 1000
    alpha: float = 0.05
    min_effect_sigma: float = 0.02
    abs_effect: float = 0.0
    tail_bins: int = 2
    min_obs_per_tail_bin: int = 250
    min_dates_per_tail_bin: int = 40
    min_train_dates_per_tail_bin: int = 40
    min_folds: int = 3
    min_fold_fraction: float = 0.75
    hc_min_consecutive: int = 3
    hc_min_tail_fraction: float = 0.80
    random_seed: int = 20260812
    min_sector_names_per_date: int = 20


def canonical_json_sha256(doc: dict) -> str:
    payload = dict(doc)
    payload.pop("protocol_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()


def _q_col(normalization: str) -> str:
    return "q_psi_global" if normalization == "global" else "q_psi_sector"


def add_normalized_coordinates(panel: pd.DataFrame) -> pd.DataFrame:
    z = panel
    z["q_psi_global"] = z.groupby(["date","return_mode","horizon"], observed=True, sort=False)["psi_primary"].rank(method="average", pct=True).astype("float32")
    if "sector" in z.columns:
        keys=["date","return_mode","horizon","sector"]
        counts=z.groupby(keys, observed=True, sort=False)["psi_primary"].transform("count")
        q=z.groupby(keys, observed=True, sort=False)["psi_primary"].rank(method="average", pct=True)
        z["q_psi_sector"] = q.where(counts >= 20).astype("float32")
    return z


def _tail_edges(q0: float, bins: int) -> np.ndarray:
    if not (0.0 < q0 < 1.0):
        raise ValueError("q0 must lie in (0,1)")
    return np.linspace(float(q0), 1.0, int(bins)+1)


def _bin_mask(frame: pd.DataFrame, qcol: str, lo: float, hi: float, is_last: bool) -> pd.Series:
    q=frame[qcol].astype(float)
    return (q >= lo) & (q <= hi if is_last else q < hi)


def _date_means(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("date", sort=True)["phase_product"].mean().to_numpy(float)


def _aligned_tail_fold_certification(
    train_h: pd.DataFrame,
    test_h: pd.DataFrame,
    *,
    qcol: str,
    q0: float,
    cfg: V91Config,
    block_len: int,
    seed: int,
) -> dict:
    """Effect size is scaled at the SAME unit as the estimand: daily tail-bin means."""
    edges=_tail_edges(q0,cfg.tail_bins)
    details=[]; pvals=[]; all_negative=True
    for b,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        last=b == len(edges)-2
        tr=train_h.loc[_bin_mask(train_h,qcol,lo,hi,last)]
        te=test_h.loc[_bin_mask(test_h,qcol,lo,hi,last)]
        if len(te)<cfg.min_obs_per_tail_bin or te["date"].nunique()<cfg.min_dates_per_tail_bin:
            return {"eligible":False,"reason":"insufficient_test_tail_bin_support","tail_negative":False,"candidate_p_iut":np.nan,"tail_bins_json":json.dumps(details)}
        tr_dm=_date_means(tr)
        te_dm=_date_means(te)
        tr_dm=tr_dm[np.isfinite(tr_dm)]; te_dm=te_dm[np.isfinite(te_dm)]
        if len(tr_dm)<cfg.min_train_dates_per_tail_bin:
            return {"eligible":False,"reason":"insufficient_train_tail_bin_dates","tail_negative":False,"candidate_p_iut":np.nan,"tail_bins_json":json.dumps(details)}
        scale=float(np.std(tr_dm,ddof=1)) if len(tr_dm)>1 else 0.0
        eps=max(float(cfg.abs_effect), float(cfg.min_effect_sigma)*scale)
        st=_moving_block_bootstrap(te_dm, block_len, cfg.bootstrap_reps, np.random.default_rng(seed+101*(b+1)))
        if st["estimate"] is None or st["ci_hi"] is None or st["p_lt0"] is None:
            return {"eligible":False,"reason":"bootstrap_unavailable","tail_negative":False,"candidate_p_iut":np.nan,"tail_bins_json":json.dumps(details)}
        est=float(st["estimate"]); ci_hi=float(st["ci_hi"]); p=float(st["p_lt0"])
        effect_pass=bool(est <= -eps); ci_pass=bool(ci_hi < 0.0)
        passed=effect_pass and ci_pass
        all_negative &= passed
        pvals.append(p)
        details.append({
            "bin":b,"q_lo":float(lo),"q_hi":float(hi),"estimate":est,"ci_lo":float(st["ci_lo"]),"ci_hi":ci_hi,"p_lt0":p,
            "train_date_mean_sd":scale,"epsilon_bin":eps,"effect_floor_pass":effect_pass,"ci_negative_pass":ci_pass,
            "n":int(len(te)),"n_dates":int(te["date"].nunique()),"train_n_dates":int(len(tr_dm)),
        })
    p_iut=max(pvals) if len(pvals)==cfg.tail_bins else np.nan
    return {"eligible":True,"reason":"","tail_negative":bool(all_negative and np.isfinite(p_iut) and p_iut<=cfg.alpha),"candidate_p_iut":float(p_iut),"tail_bins_json":json.dumps(details)}


def _apply_holm(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    z=df.copy(); z["p_holm_horizon_family"]=np.nan; z["passes_horizon_holm"]=False; z["certified_tail_negative_fold"]=False
    for _,idx in z.groupby(["universe","normalization","q0","return_mode","fold"],sort=False).groups.items():
        loc=list(idx); adj=_holm_adjust(z.loc[loc,"candidate_p_iut"].to_numpy(float))
        z.loc[loc,"p_holm_horizon_family"]=adj
        z.loc[loc,"passes_horizon_holm"]=np.isfinite(adj)&(adj<=alpha)
    z["certified_tail_negative_fold"]=(z["eligible"].fillna(False).astype(bool)&z["tail_negative"].fillna(False).astype(bool)&z["passes_horizon_holm"].fillna(False).astype(bool))
    return z


def evaluate_candidate(panel: pd.DataFrame, *, universe: str, normalization: str, q0: float, horizons: Sequence[int], wf_cfg, cfg: V91Config):
    qcol=_q_col(normalization)
    if qcol not in panel.columns or panel[qcol].notna().mean()<0.50:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),{"identified":False,"reason":f"{normalization}_coordinate_unavailable"}
    hs=tuple(sorted(set(int(x) for x in horizons))); max_h=max(hs); block_len=max(1,int(math.ceil(max_h/int(wf_cfg.anchor_stride))))
    rows=[]
    for mi,(mode,pm0) in enumerate(panel.groupby("return_mode",observed=True,sort=False)):
        pm=pm0[pm0["horizon"].isin(hs)]
        blocks=_walkforward_date_blocks(pm,wf_cfg)
        for fold,(test_start,test_end) in enumerate(blocks,1):
            purge=pd.offsets.BDay(max_h); train=pm[pm["date"]<test_start-purge]; test=pm[(pm["date"]>=test_start)&(pm["date"]<=test_end)]
            for h in hs:
                tr=train[train["horizon"]==h]; te=test[test["horizon"]==h]
                if len(tr)<200 or len(te)<100:
                    cert={"eligible":False,"reason":"insufficient_horizon_rows","tail_negative":False,"candidate_p_iut":np.nan,"tail_bins_json":"[]"}
                else:
                    cert=_aligned_tail_fold_certification(tr,te,qcol=qcol,q0=q0,cfg=cfg,block_len=block_len,seed=cfg.random_seed+300000*(mi+1)+10000*fold+int(h)+int(round(q0*1000)))
                rows.append({"universe":universe,"normalization":normalization,"q0":float(q0),"return_mode":str(mode),"fold":fold,"test_start":test_start,"test_end":test_end,"horizon":int(h),**cert})
    folds=_apply_holm(pd.DataFrame(rows),cfg.alpha)
    regs=[]
    for (u,norm,q,mode,h),g in folds.groupby(["universe","normalization","q0","return_mode","horizon"],sort=True):
        total=int(g["fold"].nunique()); req=max(cfg.min_folds,int(math.ceil(cfg.min_fold_fraction*total))); sup=int(g["certified_tail_negative_fold"].sum())
        regs.append({"universe":u,"normalization":norm,"q0":float(q),"return_mode":mode,"horizon":int(h),"folds_total":total,"min_folds_required":req,"negative_folds_support":sup,"support_fraction":sup/total if total else np.nan,"certified_tail_negative":bool(sup>=req)})
    regimes=pd.DataFrame(regs)
    hc_rows=[]
    for (u,norm,q,mode),g in regimes.groupby(["universe","normalization","q0","return_mode"],sort=False):
        g=g.sort_values("horizon")
        est=_find_onset(g["horizon"].to_numpy(int),g["certified_tail_negative"].to_numpy(bool),np.zeros(len(g),dtype=bool),min_consecutive=cfg.hc_min_consecutive,min_tail_fraction=cfg.hc_min_tail_fraction)
        hc_rows.append({"universe":u,"normalization":norm,"q0":float(q),"return_mode":mode,**est})
    hc=pd.DataFrame(hc_rows); consensus={"identified":False}
    if not hc.empty and hc["identified"].astype(bool).all():
        upper=int(hc["hc_grid"].max()); lower_vals=hc["hc_lower_exclusive"].dropna(); lower=int(lower_vals.max()) if len(lower_vals) else None
        consensus={"identified":True,"hc_grid":upper,"hc_lower_exclusive":lower,"hc_upper_inclusive":upper,"hc_midpoint":0.5*(lower+upper) if lower is not None else float(upper),"modes":hc["return_mode"].tolist()}
    return folds,regimes,hc,consensus


def _candidate_quality(regs: pd.DataFrame, consensus: dict) -> float:
    if not consensus.get("identified") or regs.empty: return 0.0
    hc=int(consensus["hc_grid"]); g=regs[regs["horizon"]>=hc]
    return float(g["support_fraction"].min()) if not g.empty else 0.0


def select_hypothesis(records: list[dict]) -> dict:
    eligible=[]
    for r in records:
        dc=r["discovery_consensus"]; vc=r["second_consensus"]
        if dc.get("identified") and vc.get("identified"):
            quality=min(float(r["discovery_min_post_onset_support"]),float(r["second_min_post_onset_support"]))
            delta=abs(int(dc["hc_grid"])-int(vc["hc_grid"]))
            eligible.append((-quality,delta,float(r["q0"]),str(r["normalization"]),r))
    if not eligible:
        return {"selected":False,"reason":"no_scale_aligned_percentile_tail_candidate_identified_in_both_universes_and_both_modes"}
    eligible.sort(key=lambda x:(x[0],x[1],x[2],x[3])); _,delta,_,_,best=eligible[0]
    d=int(best["discovery_consensus"]["hc_grid"]); s=int(best["second_consensus"]["hc_grid"])
    return {"selected":True,"normalization":best["normalization"],"q0":float(best["q0"]),"tail_bins":2,"effect_scale":"train_date_mean_sd_per_tail_bin","min_effect_sigma":float(best["min_effect_sigma"]),"discovery_hc_grid":d,"second_universe_hc_grid":s,"transport_delta_days":abs(d-s),"expected_hc_envelope":[min(d,s),max(d,s)],"selection_rule":"require both universes x both modes persistent onset; maximize worst post-onset fold support, then minimize hc gap, then prefer widest upper tail (smallest q0), deterministic normalization tie-break"}


def outlier_attribution(panel: pd.DataFrame, *, horizons: Sequence[int], wf_cfg, q_threshold: float=0.875, top_rows: int=20, top_dates: int=10):
    hs=tuple(sorted(set(int(x) for x in horizons))); max_h=max(hs); rows=[]; dates=[]
    for mode,pm0 in panel.groupby("return_mode",observed=True,sort=False):
        pm=pm0[pm0["horizon"].isin(hs)]; blocks=_walkforward_date_blocks(pm,wf_cfg)
        for fold,(test_start,test_end) in enumerate(blocks,1):
            te=pm[(pm["date"]>=test_start)&(pm["date"]<=test_end)]
            for h in hs:
                if h<119: continue
                g=te[(te["horizon"]==h)&(te["q_psi_global"]>=q_threshold)]
                if g.empty: continue
                dm=g.groupby("date",sort=True)["phase_product"].mean().sort_values(ascending=False)
                for d,val in dm.head(top_dates).items(): dates.append({"return_mode":str(mode),"fold":fold,"horizon":int(h),"date":d,"date_mean_phase_product":float(val),"q_threshold":q_threshold})
                top=g.assign(abs_phase=lambda x:x["phase_product"].abs()).nlargest(top_rows,"abs_phase")
                for _,r in top.iterrows():
                    item={"return_mode":str(mode),"fold":fold,"horizon":int(h),"date":r["date"],"ticker":str(r["ticker"]),"q_psi":float(r["q_psi_global"]),"phase_product":float(r["phase_product"]),"abs_phase_product":float(abs(r["phase_product"]))}
                    for c in ("source_index","cap_bucket","sector","asset_class"):
                        if c in r.index: item[c]=None if pd.isna(r[c]) else str(r[c])
                    rows.append(item)
    return pd.DataFrame(rows),pd.DataFrame(dates)


def subgroup_diagnostics(panel: pd.DataFrame, *, q0: float, horizons: Sequence[int], wf_cfg, cfg: V91Config, min_tickers: int=25) -> pd.DataFrame:
    out=[]
    for col in ("source_index","cap_bucket","sector","asset_class"):
        if col not in panel.columns: continue
        for val,g0 in panel.groupby(col,observed=True,sort=True):
            if pd.isna(val): continue
            nt=int(g0["ticker"].nunique())
            if nt<min_tickers: continue
            g=g0.copy(); g["q_psi_global"]=g.groupby(["date","return_mode","horizon"],observed=True,sort=False)["psi_primary"].rank(method="average",pct=True).astype("float32")
            _,_,hc,cons=evaluate_candidate(g,universe=f"second:{col}={val}",normalization="global",q0=q0,horizons=horizons,wf_cfg=wf_cfg,cfg=cfg)
            out.append({"subgroup_type":col,"subgroup":str(val),"ticker_count":nt,"q0":q0,"consensus_identified":bool(cons.get("identified")),"consensus_hc_grid":cons.get("hc_grid"),"per_mode_json":hc.to_json(orient="records")})
    return pd.DataFrame(out)


def run_exploratory_v91(*, discovery_panel_path: Path, second_panel_path: Path, out_dir: Path, discovery_metadata_path: Path|None=None, second_metadata_path: Path|None=None, horizons: Sequence[int]=v9.DEFAULT_COARSE_HORIZONS, q0_grid: Sequence[float]=DEFAULT_Q0_GRID, bootstrap_reps:int=1000, alpha:float=.05, min_effect_sigma:float=.02, min_folds:int=3, min_fold_fraction:float=.75, splits:int=4, min_train_frac:float=.5, stride:int=5, seed:int=20260812) -> dict:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); hs=tuple(sorted(set(int(x) for x in horizons)))
    cfg=V91Config(bootstrap_reps=bootstrap_reps,alpha=alpha,min_effect_sigma=min_effect_sigma,min_folds=min_folds,min_fold_fraction=min_fold_fraction,random_seed=seed)
    wf=v9._cfg_namespace(hs,seed=seed,splits=splits,train_frac=min_train_frac,stride=stride)
    panels={}; records=[]; all_f=[]; all_r=[]; all_h=[]
    for label,path,mpath in (("discovery",discovery_panel_path,discovery_metadata_path),("second_universe",second_panel_path,second_metadata_path)):
        p=v9.read_panel_streaming(Path(path),label=f"v9.1-{label}"); p=p[p["horizon"].isin(hs)].reset_index(drop=True); p=v9.attach_metadata(p,Path(mpath) if mpath else None); add_normalized_coordinates(p); panels[label]=p
    norms=["global"]
    sector_ok=all("q_psi_sector" in panels[u].columns and panels[u]["q_psi_sector"].notna().mean()>=.50 for u in panels)
    if sector_ok: norms.append("sector")
    for norm in norms:
        for q0 in q0_grid:
            per={}
            for label in ("discovery","second_universe"):
                f,r,h,c=evaluate_candidate(panels[label],universe=label,normalization=norm,q0=float(q0),horizons=hs,wf_cfg=wf,cfg=cfg)
                if not f.empty: all_f.append(f)
                if not r.empty: all_r.append(r)
                if not h.empty: all_h.append(h)
                per[label]=(r,c)
            dr,dc=per["discovery"]; sr,sc=per["second_universe"]
            records.append({"normalization":norm,"q0":float(q0),"min_effect_sigma":min_effect_sigma,"discovery_consensus":dc,"second_consensus":sc,"discovery_min_post_onset_support":_candidate_quality(dr,dc),"second_min_post_onset_support":_candidate_quality(sr,sc)})
    folds=pd.concat(all_f,ignore_index=True) if all_f else pd.DataFrame(); regs=pd.concat(all_r,ignore_index=True) if all_r else pd.DataFrame(); hc=pd.concat(all_h,ignore_index=True) if all_h else pd.DataFrame()
    folds.to_csv(out/"scale_aligned_tail_fold_certification.csv",index=False); regs.to_csv(out/"scale_aligned_tail_regimes.csv",index=False); hc.to_csv(out/"scale_aligned_tail_hc.csv",index=False)
    selected=select_hypothesis(records); (out/"selected_transport_hypothesis_v9_1.json").write_text(json.dumps(selected,indent=2,default=str),encoding="utf-8")
    top_rows,top_dates=outlier_attribution(panels["second_universe"],horizons=hs,wf_cfg=wf); top_rows.to_csv(out/"second_universe_upper_extreme_top_rows.csv",index=False); top_dates.to_csv(out/"second_universe_upper_extreme_top_dates.csv",index=False)
    diag_q0=float(selected.get("q0",0.75)); subgroup=subgroup_diagnostics(panels["second_universe"],q0=diag_q0,horizons=hs,wf_cfg=wf,cfg=cfg); subgroup.to_csv(out/"second_universe_subgroup_tail_hc_v9_1.csv",index=False)
    last=max(hs); near=[]
    if not regs.empty:
        for (u,norm,q,mode),g in regs.groupby(["universe","normalization","q0","return_mode"],sort=False):
            z=g[g["horizon"]==last]
            if not z.empty and float(z.iloc[0]["support_fraction"])>=.75: near.append({"universe":u,"normalization":norm,"q0":float(q),"return_mode":mode,"last_horizon":last,"support_fraction":float(z.iloc[0]["support_fraction"])})
    summary={"status":"EXPLORATORY_TRANSPORT_V9_1_SCALE_ALIGNED","scientific_change":"effect floor coefficient remains 0.02, but sigma is estimated from TRAIN DAILY TAIL-BIN MEANS, matching the OOS estimand unit; no confirmatory v8/v9 verdict is altered","horizons":list(hs),"q0_grid":[float(x) for x in q0_grid],"normalizations_tested":norms,"sector_normalization_available":sector_ok,"candidate_summaries":records,"selected_transport_hypothesis":selected,"near_right_boundary_evidence":near,"rules":{**asdict(cfg),"walkforward_splits":splits,"min_train_frac":min_train_frac,"stride":stride,"effect_scale":"train_date_mean_sd_per_tail_bin"},"guardrail":"Discovery and second universe are development data. Freeze only after v9.1 selection; third universe must remain untouched."}
    (out/"transport_exploratory_summary_v9_1.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    return summary


def freeze_protocol_v91(*, exploratory_summary: Path, selected_hypothesis: Path, out: Path) -> dict:
    s=json.loads(Path(exploratory_summary).read_text()); sel=json.loads(Path(selected_hypothesis).read_text())
    if not sel.get("selected"): raise RuntimeError("Cannot freeze v9.1: no transportable candidate selected on development universes")
    doc={"protocol_version":PROTOCOL_VERSION,"status":"FROZEN_BEFORE_THIRD_UNIVERSE","hypothesis":{"coordinate":sel["normalization"],"upper_tail_q0":sel["q0"],"tail_bins":sel["tail_bins"],"effect_scale":"train_date_mean_sd_per_tail_bin","endpoint":"persistent OOS negative upper percentile-tail in raw and market_residual"},"horizons":s["horizons"],"expected_hc_envelope":sel["expected_hc_envelope"],"rules":s["rules"],"development_summary_sha256":hashlib.sha256(Path(exploratory_summary).read_bytes()).hexdigest(),"selected_hypothesis_sha256":hashlib.sha256(Path(selected_hypothesis).read_bytes()).hexdigest(),"guardrail":"No q0, normalization, effect-scale, alpha, folds, horizon grid, or persistence rule may change after third-universe data are accessed."}
    doc["protocol_sha256"]=canonical_json_sha256(doc); Path(out).write_text(json.dumps(doc,indent=2),encoding="utf-8"); return doc
