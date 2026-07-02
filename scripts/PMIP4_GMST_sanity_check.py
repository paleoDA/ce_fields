"""
CHECKS THAT past1000 --> hist stitching is sensible
"""

import sys
import time
import subprocess
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC
import matplotlib.pyplot as plt
import cftime as _cftime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '..')
from ce_simul.data_config import DataConfig
from ce_simul.io import open_nc_or_zarr
from ce_simul.pmip4_config import (get_pmip4_models, get_cmip_hist_models,
                                    MODEL_VAR, YEAR_OFFSETS, STITCH_THRESHOLD)
from ce_simul.products import load_stitched_tas, compute_gmst

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts():
    """Return a compact timestamp string for log lines."""
    return datetime.datetime.now().strftime('%H:%M:%S')

def _elapsed(t0):
    return f"{time.time() - t0:.1f}s"

class _Tee:
    """Write to multiple streams simultaneously (terminal + log file)."""
    def __init__(self, *streams): self._streams = streams
    def write(self, s): [f.write(s) for f in self._streams]
    def flush(self): [f.flush() for f in self._streams]

# =============================================================================
# CONFIGURATION
# =============================================================================

config = DataConfig()

_log_dir = Path('../results/logs')
_log_dir.mkdir(parents=True, exist_ok=True)
_log_path = _log_dir / 'PMIP4_GMST_sanity_check.log'
_logfile  = open(_log_path, 'w')
sys.stdout = _Tee(sys.__stdout__, _logfile)

print(f"[{_ts()}] Log: {_log_path.resolve()}")
print(f"[{_ts()}] Environment: {config.environment}")

PMIP4_MODELS     = get_pmip4_models(config)
CMIP_HIST_MODELS = get_cmip_hist_models(config)


CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

FIGURE_FOLDER = Path('../results/figures/')
DATA_FOLDER   = Path('../results/data/')
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"[{_ts()}] Models registered: {list(PMIP4_MODELS.keys())}")



# =============================================================================
# TIME-AXIS DIAGNOSTICS
# =============================================================================
print(f"\n[{_ts()}] ── Time-axis diagnostics ──────────────────────────────────")

def _model_year_range(path, name_tag, offset=0, var=None):
    try:
        ds = open_nc_or_zarr(path, decode_times=CFTIME_CODER)
        t0_val = ds.time.values[0]
        decoded = isinstance(t0_val, _cftime.datetime)
        cal = ds.time.encoding.get('calendar', ds.time.attrs.get('calendar', 'unknown'))
        if var is None:
            var = MODEL_VAR.get(name_tag, 'tas')
        ann = ds[var].isel(lat=0, lon=0).groupby('time.year').mean('time')
        y0 = int(ann.year.values[0])  - offset
        y1 = int(ann.year.values[-1]) - offset
        ds.close()
        return decoded, cal, y0, y1
    except Exception as e:
        return False, 'error', None, None, str(e)

_time_ok         = {}   # paleo run decoded OK
_hist_ok         = {}   # historical run decoded OK
_pmip4_end_year  = {}   # last CE year of PMIP4 run (drives stitch decision)
_do_stitch       = {}   # True when run ends before STITCH_THRESHOLD and hist is available

print("─" * 85)
print(f"{'Model':<22} {'run':<10} {'decoded?':<10} {'calendar':<22} {'year range (CE)'}")
print("─" * 85)

for name in PMIP4_MODELS:
    offset = YEAR_OFFSETS.get(name, 0)

    result = _model_year_range(PMIP4_MODELS[name], name, offset=offset)
    if len(result) == 5:
        decoded, cal, y0, y1, err = False, 'error', None, None, result[4]
        print(f"{name:<22} {'paleo':<10} ERROR: {err}")
    else:
        decoded, cal, y0, y1 = result
        print(f"{name:<22} {'paleo':<10} {'cftime' if decoded else 'OTHER':<10} {cal:<22} {y0}–{y1}")
    _time_ok[name]        = decoded and y0 is not None
    _pmip4_end_year[name] = y1

    hist_path = CMIP_HIST_MODELS.get(name)
    if hist_path is not None:
        result_h = _model_year_range(hist_path, name, var='tas')
        if len(result_h) == 5:
            decoded_h, cal_h, yh0, yh1 = False, 'error', None, None
            print(f"{name:<22} {'hist':<10} ERROR: {result_h[4]}")
        else:
            decoded_h, cal_h, yh0, yh1 = result_h
            print(f"{name:<22} {'hist':<10} {'cftime' if decoded_h else 'OTHER':<10} {cal_h:<22} {yh0}–{yh1}")
        _hist_ok[name] = yh0 is not None

        if _time_ok[name] and _hist_ok[name]:
            gap = yh0 - y1 - 1
            if gap == 0:
                status = "✓ clean stitch"
            elif gap == -1:
                status = f"⚠ 1-yr overlap at {y1} — will drop from hist"
            elif gap > 0:
                status = f"✗ GAP of {gap} years ({y1}→{yh0})"
            else:
                status = f"✗ OVERLAP of {-gap} years — cannot stitch"
            print(f"  → stitch check: {status}")
    else:
        _hist_ok[name] = False

    # Data-driven stitch decision: only stitch if the run ends before STITCH_THRESHOLD
    _do_stitch[name] = _hist_ok[name] and (y1 is not None) and (y1 < STITCH_THRESHOLD)
    if not _do_stitch[name]:
        label = 'self-contained' if (y1 is not None and y1 >= STITCH_THRESHOLD) else 'no hist available'
        print(f"  → no stitch ({label}, last model year: {y1} CE)")
    print()

# =============================================================================
# GLOBAL MEAN SURFACE TEMPERATURE
# =============================================================================
print(f"\n[{_ts()}] ── Computing GMST for all models ────────────────────────────")

GMST_REF    = (1951, 1980)
gmst_series = {}

fig, ax = plt.subplots(figsize=(12, 5))

for name in PMIP4_MODELS:
    if not _time_ok.get(name, False):
        print(f"[{_ts()}]   Skipping {name}: paleo time axis did not decode cleanly")
        continue

    t0 = time.time()
    print(f"[{_ts()}]   {name}: loading + stitching …", end='', flush=True)

    try:
        # load_stitched_tas() re-derives the same stitch decision from the
        # data itself, so this matches _do_stitch[name] above by construction.
        tas_ann = load_stitched_tas(name, config=config)
        print(f" done ({_elapsed(t0)})", flush=True)

        print(f"[{_ts()}]   {name}: computing lat-weighted GMST …", end='', flush=True)
        t1 = time.time()
        gmst_anom = compute_gmst(tas_ann, ref_period=GMST_REF)
        print(f" done ({_elapsed(t1)})", flush=True)

        gmst_series[name] = pd.Series(gmst_anom.values, index=gmst_anom.year.values, name=name)
        ax.plot(gmst_anom.year, gmst_anom.values, label=name, linewidth=1, alpha = 0.6)
        print(f"[{_ts()}]   {name}: GMST range "
              f"{float(gmst_anom.min()):.2f} – {float(gmst_anom.max()):.2f} °C  "
              f"(total {_elapsed(t0)})")

    except Exception as e:
        print(f"\n[{_ts()}]   Error processing {name}: {e}")

if gmst_series:
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.axvspan(GMST_REF[0], GMST_REF[1], color='gray', alpha=0.15,
               label=f'Reference ({GMST_REF[0]}–{GMST_REF[1]})')
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel(f'GMST anomaly (°C, ref. {GMST_REF[0]}–{GMST_REF[1]})')
    ax.set_title('PMIP4 + CMIP6-hist Annual Global Mean Surface Temperature', fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pmip_fig_dir = FIGURE_FOLDER / 'PMIP'
    pmip_fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pmip_fig_dir / 'PMIP4_GMST_anomalies.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    gmst_df = pd.DataFrame(gmst_series)
    gmst_df.index.name = 'year'
    tables_dir = Path('../results/tables')
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / 'PMIP4_GMST_anomalies.csv'
    gmst_df.to_csv(csv_path, float_format='%.4f')
    # Append stitch indicator: 1 if a separate CMIP6 historical run was stitched at 1850, 0 otherwise
    with open(csv_path, 'a') as _f:
        _vals = ','.join(str(int(_do_stitch.get(n, False))) for n in gmst_df.columns)
        _f.write(f'has_stitch,{_vals}\n')
    print(f"\n[{_ts()}] Saved: {pmip_fig_dir / 'PMIP4_GMST_anomalies.png'}")
    print(f"[{_ts()}] Saved: {csv_path}  ({len(gmst_df)} years × {len(gmst_df.columns)} models)")
    subprocess.run(
        ['git', 'add',
         str((pmip_fig_dir / 'PMIP4_GMST_anomalies.png').resolve()),
         str(csv_path.resolve())],
        cwd=Path(__file__).resolve().parent.parent, check=False,
    )
else:
    print(f"[{_ts()}] No models produced valid GMST — check diagnostics above.")

print(f"[{_ts()}] Log saved: {_log_path.resolve()}")
sys.stdout = sys.__stdout__
_logfile.close()


# =============================================================================
print(f"\n[{_ts()}] ── Pipeline complete ─────────────────────────────────────────")
