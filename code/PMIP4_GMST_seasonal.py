"""
Seasonal (DJF/MAM/JJA/SON) GMST anomalies for every registered PMIP4 model.

Seasonal counterpart of results/tables/PMIP4_GMST_anomalies.csv (annual,
written by PMIP4_GMST_sanity_check.py), built for comparison with LMR
Seasonal (warmth_and_trends/notebooks/LMR_volcanic_GMST.ipynb), whose
seasons follow the same convention:

* DJF of year Y = Dec(Y-1) + Jan(Y) + Feb(Y); seasons are labeled by the
  calendar year and month of their center (DJF -> Jan, MAM -> Apr,
  JJA -> Jul, SON -> Oct).
* Anomalies are taken relative to each season's own 1951-1980 mean, which
  removes the seasonal cycle.

The global mean is taken on each month's native grid *before* any time
averaging or stitching, so the paleo/historical grid mismatch that
stitch_annual_tas() has to regrid (CESM2) is irrelevant here. Otherwise the
paleo-run handling (corrected files, YEAR_OFFSETS, TRIM_END, incomplete-year
drop, STITCH_THRESHOLD, 1-year-overlap rule) mirrors
ce_fields.products.load_stitched_tas().

Monthly time stamps: CMORized output stamps months mid-month, but native CAM
output (CESM*) stamps each month at its *end* (January -> Feb 1), which would
shift every season by one month. The month of each record is therefore taken
from the midpoint of the time bounds when they exist; otherwise, all-day-1
stamps whose first record falls in February are treated as end-of-month
stamps and shifted back one day. The rule applied to each model is logged,
and the calendar-year means are checked against the annual CSV as a sanity
check (a one-month shift shows up there as a mismatch).

Also writes the global attributes of every source file to
results/tables/PMIP4_global_attrs.json, to help document each simulation's
forcing.

Outputs (in results/tables/):
    PMIP4_GMST_seasonal_anomalies.csv  -- columns: year, season, time, <model>...
    PMIP4_global_attrs.json
Log: results/logs/PMIP4_GMST_seasonal.log

Run on the HPC (data paths are HPC-only): sbatch code/PMIP4_GMST_seasonal.sh
"""

import sys
import json
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ce_fields.data_config import DataConfig
from ce_fields.io import open_nc_or_zarr
from ce_fields.pmip4_config import (get_pmip4_models, get_cmip_hist_models,
                                    MODEL_VAR, YEAR_OFFSETS, TRIM_END, STITCH_THRESHOLD)

CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
GMST_REF     = (1951, 1980)
SEASONS      = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
CENTER_MONTH = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}

TABLES_DIR = REPO_ROOT / 'results' / 'tables'
LOG_DIR    = REPO_ROOT / 'results' / 'logs'
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

class _Tee:
    """Write to multiple streams simultaneously (terminal + log file)."""
    def __init__(self, *streams): self._streams = streams
    def write(self, s): [f.write(s) for f in self._streams]
    def flush(self): [f.flush() for f in self._streams]



def _json_safe(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v)


def monthly_gmst(path, var, label):
    """Area-weighted monthly global mean of `var`, as a DataFrame (year, month, gmst).

    Also returns the file's global attributes and the time-stamp rule applied.
    """
    ds = open_nc_or_zarr(path, decode_times=CFTIME_CODER)
    attrs = {k: _json_safe(v) for k, v in ds.attrs.items()}

    bnds_name = ds['time'].attrs.get('bounds')
    if bnds_name is None:
        bnds_name = next((b for b in ('time_bnds', 'time_bounds', 'time_bnd') if b in ds), None)

    if bnds_name is not None and bnds_name in ds:
        b = ds[bnds_name].load()
        t = [lo + (hi - lo) / 2 for lo, hi in b.values]
        rule = f'midpoint of {bnds_name}'
    else:
        t = list(ds['time'].values)
        if all(d.day == 1 for d in t) and t[0].month == 2:
            t = [d - datetime.timedelta(days=1) for d in t]
            rule = 'end-of-month stamps (first record in Feb) -> shifted back 1 day'
        else:
            rule = 'stamps used as-is'

    da = ds[var]
    weights = np.cos(np.deg2rad(da['lat']))
    g = da.weighted(weights).mean(dim=['lat', 'lon']).load().values
    print(f"[{_ts()}]   {label}: {len(g)} months, first stamps "
          f"{[str(x) for x in ds['time'].values[:2]]} -> rule: {rule}")
    ds.close()

    years  = np.array([d.year for d in t])
    months = np.array([d.month for d in t])
    return pd.DataFrame({'year': years, 'month': months, 'gmst': g}), attrs, rule


def drop_incomplete_years(df, label):
    n = df.groupby('year')['month'].nunique()
    bad = n[n != 12]
    if len(bad):
        print(f"[{_ts()}]   {label}: dropping incomplete year(s) "
              + ", ".join(f"{y} ({k} months)" for y, k in bad.items()))
    return df[~df['year'].isin(bad.index)]


def stitched_monthly_gmst(name, pmip4_models, cmip_hist_models):
    """Monthly GMST for one model, with the same paleo handling + stitch rules as
    ce_fields.products.load_stitched_tas()."""
    var = MODEL_VAR.get(name, 'tas')
    df, attrs, rule = monthly_gmst(pmip4_models[name], var, f'{name} (paleo)')
    all_attrs = {'paleo': attrs}
    df = drop_incomplete_years(df, f'{name} (paleo)')
    df['year'] -= YEAR_OFFSETS.get(name, 0)
    n_trim = TRIM_END.get(name, 0)
    if n_trim:
        df = df[df['year'] <= df['year'].max() - n_trim]

    paleo_end = int(df['year'].max())
    hist_path = cmip_hist_models.get(name)
    stitched = hist_path is not None and paleo_end < STITCH_THRESHOLD
    if stitched:
        dh, h_attrs, _ = monthly_gmst(hist_path, 'tas', f'{name} (hist)')
        all_attrs['historical'] = h_attrs
        dh = drop_incomplete_years(dh, f'{name} (hist)')
        hist_start = int(dh['year'].min())
        if hist_start == paleo_end:
            print(f"[{_ts()}]   {name}: 1-year overlap at {hist_start}; dropping it from hist")
            dh = dh[dh['year'] > hist_start]
        elif hist_start != paleo_end + 1:
            raise ValueError(f'{name}: cannot stitch paleo (ends {paleo_end}) '
                             f'with hist (starts {hist_start})')
        df = pd.concat([df, dh], ignore_index=True)
    print(f"[{_ts()}]   {name}: CE {int(df['year'].min())}-{int(df['year'].max())}, stitched={stitched}")
    return df.sort_values(['year', 'month']).reset_index(drop=True), all_attrs, rule, stitched


def seasonal_anomalies(df):
    """Monthly (year, month, gmst) -> seasonal anomalies indexed by (year, season)."""
    df = df.copy()
    df['season'] = df['month'].map(SEASONS)
    df['season_year'] = df['year'] + (df['month'] == 12)
    grp = df.groupby(['season_year', 'season'])['gmst']
    seas = grp.mean()[grp.count() == 3]           # require all 3 months (drops first/last DJF)
    seas.index = seas.index.set_names(['year', 'season'])
    ref = seas[seas.index.get_level_values('year').to_series().between(*GMST_REF).values]
    if ref.empty:
        raise ValueError(f'No data in reference period {GMST_REF}')
    clim = ref.groupby(level='season').mean()
    return seas - clim.reindex(seas.index.get_level_values('season')).values


if __name__ == '__main__':
    _log_path = LOG_DIR / 'PMIP4_GMST_seasonal.log'
    _logfile  = open(_log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, _logfile)

    config = DataConfig()
    print(f"[{_ts()}] Log: {_log_path}")
    print(f"[{_ts()}] Environment: {config.environment}")
    pmip4_models = get_pmip4_models(config)
    cmip_hist_models = get_cmip_hist_models(config)
    print(f"[{_ts()}] Models: {list(pmip4_models)}\n")

    annual_csv = TABLES_DIR / 'PMIP4_GMST_anomalies.csv'
    annual = pd.read_csv(annual_csv, index_col=0)
    annual = annual[annual.index != 'has_stitch']
    annual.index = annual.index.astype(int)
    annual = annual.apply(pd.to_numeric, errors='coerce')

    series, global_attrs, summary = {}, {}, []
    for name in pmip4_models:
        t0 = time.time()
        try:
            df, attrs, rule, stitched = stitched_monthly_gmst(name, pmip4_models, cmip_hist_models)
        except Exception as e:
            print(f"[{_ts()}]   ERROR {name}: {e}\n")
            continue
        global_attrs[name] = attrs
        series[name] = seasonal_anomalies(df)

        # Sanity check against the annual product: calendar-year means of the monthly GMST,
        # as anomalies wrt 1951-1980, should match PMIP4_GMST_anomalies.csv.
        ann = df.groupby('year')['gmst'].mean()
        ann = ann - ann.loc[GMST_REF[0]:GMST_REF[1]].mean()
        common = ann.index.intersection(annual[name].dropna().index) if name in annual else []
        max_diff = float(np.abs(ann.loc[common] - annual.loc[common, name]).max()) if len(common) else np.nan
        print(f"[{_ts()}]   {name}: max |annual(from monthly) - annual CSV| = {max_diff:.4f} °C "
              f"over {len(common)} years  ({time.time() - t0:.0f}s)\n")
        summary.append({'model': name, 'time_rule': rule, 'stitched': stitched,
                        'n_seasons': len(series[name]), 'max_diff_vs_annual_csv': max_diff})

    out = pd.DataFrame(series)
    out = out.reindex(sorted(out.index, key=lambda ys: (ys[0], CENTER_MONTH[ys[1]])))
    out.insert(0, 'time', [y + (CENTER_MONTH[s] - 0.5) / 12 for y, s in out.index])
    csv_path = TABLES_DIR / 'PMIP4_GMST_seasonal_anomalies.csv'
    out.to_csv(csv_path, float_format='%.4f')

    attrs_path = TABLES_DIR / 'PMIP4_global_attrs.json'
    with open(attrs_path, 'w') as f:
        json.dump(global_attrs, f, indent=1)

    print(f"[{_ts()}] Summary:\n{pd.DataFrame(summary).to_string(index=False)}\n")
    print(f"[{_ts()}] Saved: {csv_path}  ({len(out)} seasons x {len(series)} models)")
    print(f"[{_ts()}] Saved: {attrs_path}")
