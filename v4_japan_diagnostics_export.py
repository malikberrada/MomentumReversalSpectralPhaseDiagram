#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

MAD_NORMALIZER = 1.4826
HUBER_C = 1.345


def holm_adjust(pvals):
    p=np.asarray(pvals,dtype=float)
    n=len(p)
    order=np.argsort(np.where(np.isfinite(p),p,1.0))
    adj=np.ones(n,dtype=float)
    running=0.0
    for rank, idx in enumerate(order):
        mult=n-rank
        val=min(1.0, mult*(p[idx] if np.isfinite(p[idx]) else 1.0))
        running=max(running,val)
        adj[idx]=running
    return adj


def moving_block_bootstrap_mean(values,reps,seed,block=None):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    n=len(x)
    if n==0: return np.nan,np.nan,np.nan,np.nan
    est=float(x.mean())
    if n==1: return est,est,est,1.0 if est>=0 else 0.5
    if block is None: block=max(5,int(round(n**(1/3))))
    block=min(block,n)
    rng=np.random.default_rng(seed)
    means=np.empty(reps)
    starts_max=max(1,n-block+1)
    need=int(math.ceil(n/block))
    for r in range(reps):
        starts=rng.integers(0,starts_max,size=need)
        samp=np.concatenate([x[s:s+block] for s in starts])[:n]
        means[r]=samp.mean()
    lo,hi=np.quantile(means,[0.025,0.975])
    p=float((1+np.sum(means>=0))/(reps+1))
    return est,float(lo),float(hi),p


def preprocess_panel(path,horizons):
    use=['date','ticker','return_mode','horizon','psi_primary','phase_product']
    parts=[]
    for c in pd.read_csv(path,usecols=use,chunksize=250_000):
        c=c[c.horizon.isin(horizons)].copy()
        if len(c): parts.append(c)
    d=pd.concat(parts,ignore_index=True)
    d['date']=pd.to_datetime(d['date'])
    d['horizon']=d['horizon'].astype(int)
    g=['date','return_mode','horizon']
    d['q_psi']=d.groupby(g,observed=True)['psi_primary'].rank(method='average',pct=True)
    med=d.groupby(g,observed=True)['phase_product'].transform('median')
    mad=(d.phase_product-med).abs().groupby([d[x] for x in g],observed=True).transform('median')
    scale=MAD_NORMALIZER*mad
    std=d.groupby(g,observed=True)['phase_product'].transform('std').fillna(0.0)
    scale=scale.where(scale>0,std).where(lambda s:s>0,1.0)
    delta=d.phase_product-med
    cap=HUBER_C*scale
    d['phase_product_robust']=med+delta.clip(lower=-cap,upper=cap)
    return d


def expand_fold_csv(folds):
    rows=[]
    for _,r in folds.iterrows():
        bins=json.loads(r.tail_bins_json)
        for b in bins:
            rows.append({
                'return_mode':r.return_mode,'fold':int(r.fold),'horizon':int(r.horizon),
                'eligible':bool(r.eligible),'passes_holm':bool(r.passes_holm),
                'certified_fold':bool(r.certified_tail_negative_fold),
                'candidate_p_iut':float(r.candidate_p_iut),
                'test_start':r.test_start,'test_end':r.test_end,
                **{f'bin_{k}':v for k,v in b.items()}
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',required=True)
    ap.add_argument('--folds',required=True)
    ap.add_argument('--regimes',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--bootstrap-reps',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20260812)
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(a.folds)
    regs=pd.read_csv(a.regimes)
    horizons=sorted(map(int,folds.horizon.unique()))

    # Exact Holm-adjusted p-values within mode x fold horizon families.
    folds=folds.copy()
    folds['p_holm_adjusted']=np.nan
    for (mode,fold),g in folds.groupby(['return_mode','fold'],sort=False):
        idx=g.index.to_numpy()
        folds.loc[idx,'p_holm_adjusted']=holm_adjust(g.candidate_p_iut.to_numpy(float))
    folds.to_csv(out/'v4_fold_certification_with_holm_p.csv',index=False)

    expanded=expand_fold_csv(folds)
    # carry adjusted p into expanded rows
    pmap=folds.set_index(['return_mode','fold','horizon'])['p_holm_adjusted']
    expanded['p_holm_adjusted']=[pmap.loc[(r.return_mode,r.fold,r.horizon)] for r in expanded.itertuples()]
    expanded.to_csv(out/'v4_subband_diagnostics_exact.csv',index=False)

    d=preprocess_panel(Path(a.panel),horizons)
    summaries=[]
    for mode in sorted(d.return_mode.astype(str).unique()):
        mf=folds[folds.return_mode.astype(str)==mode]
        # test windows are fold-specific; union is the OOS evaluation region
        wins=[(pd.Timestamp(r.test_start),pd.Timestamp(r.test_end)) for r in mf[['fold','test_start','test_end']].drop_duplicates().itertuples()]
        for h in horizons:
            z=d[(d.return_mode.astype(str)==mode)&(d.horizon.eq(h))&(d.q_psi.ge(.85))].copy()
            mask=np.zeros(len(z),dtype=bool)
            for lo,hi in wins: mask |= z.date.between(lo,hi).to_numpy()
            z=z[mask]
            daily=z.groupby('date',observed=True).phase_product_robust.mean().sort_index()
            est,lo,hi,p= moving_block_bootstrap_mean(daily.to_numpy(),a.bootstrap_reps,a.seed+1000*h+(0 if mode=='raw' else 500000))
            row_sd=float(z.phase_product_robust.std(ddof=1)) if len(z)>1 else np.nan
            daily_sd=float(daily.std(ddof=1)) if len(daily)>1 else np.nan
            fr=mf[mf.horizon.eq(h)]
            cert=int(fr.certified_tail_negative_fold.sum())
            p_iut_vals=fr.candidate_p_iut.astype(float).to_numpy()
            p_holm_vals=fr.p_holm_adjusted.astype(float).to_numpy()
            reg=regs[(regs.return_mode.astype(str)==mode)&(regs.horizon.eq(h))]
            summaries.append({
                'return_mode':mode,'horizon':h,'tail_mean_oos':est,'ci95_lo':lo,'ci95_hi':hi,
                'p_tail_mean_one_sided':p,'effect_D_row_sd':est/row_sd if row_sd>0 else np.nan,
                'effect_D_daily_sd':est/daily_sd if daily_sd>0 else np.nan,
                'n_dates':int(daily.size),'n_obs':int(len(z)),'certified_folds':cert,'eligible_folds':int(fr.eligible.sum()),
                'p_iut_min':float(np.nanmin(p_iut_vals)),'p_iut_median':float(np.nanmedian(p_iut_vals)),'p_iut_max':float(np.nanmax(p_iut_vals)),
                'p_holm_min':float(np.nanmin(p_holm_vals)),'p_holm_median':float(np.nanmedian(p_holm_vals)),'p_holm_max':float(np.nanmax(p_holm_vals)),
                'horizon_certified':bool(reg.certified_tail_negative.iloc[0]) if len(reg) else False,
            })
    s=pd.DataFrame(summaries)
    s.to_csv(out/'v4_horizon_tail_summary_exact.csv',index=False)

    # Article-focused table: around the two onsets.
    keep={('market_residual',84),('market_residual',96),('market_residual',105),('market_residual',112),
          ('raw',140),('raw',147),('raw',168),('raw',189)}
    article=s[[ (r.return_mode,r.horizon) in keep for r in s.itertuples() ]].copy()
    article.to_csv(out/'v4_article_table_exact.csv',index=False)

    # Figure: empirical OOS tail mean with 95% CI; star = horizon-level Holm/certification pass.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8.4,7.0),sharex=True)
    for ax,mode in zip(axes,['market_residual','raw']):
        x=s[s.return_mode.eq(mode)].sort_values('horizon')
        ax.errorbar(x.horizon,x.tail_mean_oos,yerr=[x.tail_mean_oos-x.ci95_lo,x.ci95_hi-x.tail_mean_oos],fmt='o-',capsize=3,label='OOS robust-tail mean ± 95% CI')
        xp=x[x.horizon_certified]
        ax.scatter(xp.horizon,xp.tail_mean_oos,marker='*',s=90,label='Horizon certified (Holm + fold support)')
        ax.axhline(0,linewidth=1)
        hc=96 if mode=='market_residual' else 147
        ax.axvline(hc,linestyle='--',linewidth=1,label=f'hc={hc}')
        ax.set_ylabel('E[Yrob | qPsi >= 0.85]')
        ax.set_title('Market residual' if mode=='market_residual' else 'Raw returns')
        ax.legend(fontsize=8,frameon=False)
    axes[-1].set_xlabel('Horizon (trading sessions)')
    fig.suptitle('Japanese V4: prospective OOS robust-tail reversal by horizon')
    fig.tight_layout(rect=[0,0,1,.97])
    fig.savefig(out/'v4_japan_tail_mean_ci.png',dpi=220,bbox_inches='tight')
    print('PASS')
    print(out/'v4_article_table_exact.csv')
    print(out/'v4_japan_tail_mean_ci.png')

if __name__=='__main__': main()
