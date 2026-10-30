#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate PMIP4 zarr stores: open each file, load tas, compute lat-weighted
global mean surface temperature (GMST) anomalies, and plot all models together.

Seasonal cycle is removed by subtracting the per-month climatology.
Annual GMST anomalies are then referenced to REF_PERIOD = (1300, 1700).

Usage
-----
    python scripts/check_pmip4_zarr.py

On the HPC, submit via the companion SLURM script:
    sbatch scripts/check_pmip4_zarr.sh
"""

# Suppress OpenBLAS / MKL multi-threading warnings before numpy is imported
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import traceback
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

import dask
dask.config.set(scheduler='synchronous')

# ── locate package regardless of working directory ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ce_fields.data_config import DataConfig
from ce_fields.pmip4_config import get_pmip4_models

# ── configuration ────────────────────────────────────────────────────────────
REF_PERIOD = (1961, )

config = DataConfig()
print(f"Environment : {config.environment}")

PMIP4_MODELS = get_pmip4_models(config)

FIG_DIR = Path(__file__).resolve().parent.parent / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers (same logic as PaleoClimaX/PMIP_checks/gmst_utils.py) ───────────

def remove_seasonal_cycle(tas: xr.DataArray) -> xr.DataArray:
    """Subtract the per-month climatology computed over the full record."""
    climatology = tas.groupby('time.month').mean('time')
    return tas.groupby('time.month') - climatology


def lat_weighted_global_mean(da: xr.DataArray) -> xr.DataArray:
    """Latitude-weighted spatial mean over (lat, lon)."""
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = 'lat_weights'
    return da.weighted(weights).mean(('lat', 'lon'))


def process_model(name: str, zarr_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Open zarr, remove seasonal cycle, compute annual lat-weighted GMST anomaly
    relative to REF_PERIOD.

    Returns
    -------
    years : np.ndarray (int)
    gmst  : np.ndarray (K, anomaly)
    or None on failure.
    """
    print(f"\n  Loading  {name}  ({zarr_path})")
    ds  = xr.open_zarr(zarr_path)
    tas = ds['tas']

    print(f"  → time: {int(tas.time.dt.year[0])}–{int(tas.time.dt.year[-1])}"
          f"  shape: {tas.shape}")

    print(f"  Removing seasonal cycle ...")
    tas_anom = remove_seasonal_cycle(tas)

    print(f"  Computing lat-weighted global mean ...")
    gmst_monthly = lat_weighted_global_mean(tas_anom)

    print(f"  Annualising ...")
    gmst_annual = gmst_monthly.groupby('time.year').mean('time')

    years = gmst_annual.year.values.astype(int)
    gmst  = gmst_annual.values

    # Reference to REF_PERIOD
    ref_mask = (years >= REF_PERIOD[0]) & (years <= REF_PERIOD[1])
    if ref_mask.sum() == 0:
        print(f"  WARNING: REF_PERIOD {REF_PERIOD} not covered — skipping reference subtraction.")
    else:
        gmst = gmst - gmst[ref_mask].mean()

    ds.close()
    return years, gmst


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PMIP4 zarr validation — GMST anomalies")
    print("=" * 65)

    results = {}
    for name, path in PMIP4_MODELS.items():
        try:
            years, gmst = process_model(name, path)
            results[name] = (years, gmst)
            print(f"  ✓  {name}  ({len(years)} years, "
                  f"GMST range {gmst.min():.2f}–{gmst.max():.2f} K)")
        except Exception:
            print(f"  ✗  {name}  FAILED:")
            traceback.print_exc()

    if not results:
        print("\nNo data to plot — check that zarr paths are accessible.")
        return

    # ── plot ──────────────────────────────────────────────────────────────
    colors = cm.tab10(np.linspace(0, 0.9, len(results)))

    fig, ax = plt.subplots(figsize=(13, 5))

    for (name, (years, gmst)), color in zip(results.items(), colors):
        ax.plot(years, gmst, color=color, linewidth=1.2, alpha=0.85, label=name)

    ax.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.axvspan(*REF_PERIOD, color='grey', alpha=0.12,
               label=f'Reference {REF_PERIOD[0]}–{REF_PERIOD[1]}')

    ax.set_xlabel('Year (CE)', fontsize=12)
    ax.set_ylabel('GMST anomaly (K)', fontsize=12)
    ax.set_title('PMIP4 annual GMST anomalies\n'
                 f'(seasonal cycle removed; referenced to {REF_PERIOD[0]}–{REF_PERIOD[1]})',
                 fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_png = FIG_DIR / 'pmip4_gmst_check.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
