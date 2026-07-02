"""
tas_1850_adjustments.py
========================
Apply the paleo/historical (1850) stitch corrections derived in
notebooks/PMIP4_1850_stitch.ipynb to full 3-D (year, lat, lon) PMIP4 tas
fields — not just the GMST scalar time series used to derive them.

Two kinds of correction, both derived from GMST and then applied uniformly
across the spatial field (per the notebook's approach: the field-level
correction is a function of year only, identical at every grid point):

* Flat offset (ACCESS-ESM1-5, EC-Earth3-Veg-LR, MRI-ESM2-0): a constant
  shift added to every pre-1850 year, matching the past1000 1825-1849 mean
  to the historical 1850-1874 mean.
* CESM2 linear detrend: past1000.002 carries a spurious warming drift (see
  the notebook for the physical explanation — background volcanic forcing
  removed at run start). We fit a linear trend to the CESM2 GMST over
  850-1849 and subtract it from the pre-1850 segment. Algebraically this
  reduces to simply removing the fitted trend from the raw values (no
  separate re-centering step is needed — see notebook cell for CESM2).

Both corrections are derived from results/tables/PMIP4_GMST_anomalies.csv,
the same file the notebook reads, so this module and the notebook stay in
sync automatically when that file is regenerated.

Importable as a module:
    from ce_simul.tas_1850_adjustments import apply_1850_adjustment
    tas_annual = apply_1850_adjustment(tas_annual, model_name)

Run standalone to print the derived corrections (sanity check against the
notebook, no HPC data needed):
    python -m ce_simul.tas_1850_adjustments
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GMST_ANOMALIES_CSV = (
    Path(__file__).resolve().parent.parent / 'results' / 'tables' / 'PMIP4_GMST_anomalies.csv'
)

FLAT_OFFSET_MODELS = ['ACCESS-ESM1-5', 'EC-Earth3-Veg-LR', 'MRI-ESM2-0']
CESM2_MODEL = 'CESM2'
STITCH_YEAR = 1850


@lru_cache(maxsize=None)
def _load_gmst(csv_path=GMST_ANOMALIES_CSV):
    """Load PMIP4_GMST_anomalies.csv into a {model: pd.Series} DataFrame."""
    raw = pd.read_csv(csv_path, index_col='year', dtype=object)
    gmst = raw.drop(index='has_stitch').astype(float)
    gmst.index = gmst.index.astype(int)
    return gmst


def compute_flat_offsets(gmst, models=FLAT_OFFSET_MODELS):
    """Offset = mean(hist 1850-1874) - mean(paleo 1825-1849), per model."""
    offsets = {}
    for name in models:
        s = gmst[name].dropna()
        pre_mean  = s[(s.index >= 1825) & (s.index < STITCH_YEAR)].mean()
        post_mean = s[(s.index >= STITCH_YEAR) & (s.index < STITCH_YEAR + 25)].mean()
        offsets[name] = post_mean - pre_mean
    return offsets


def compute_cesm2_trend(gmst):
    """Fit a linear trend to CESM2 past1000 GMST (850-1849).

    Returns (slope, intercept) of trend(year) = slope*year + intercept, in
    °C/year and °C. Subtracting this trend from the raw pre-1850 field is
    equivalent to the notebook's detrend-and-recenter procedure (the
    recentering step is a no-op once the linear-fit residual mean is zero).
    """
    s = gmst[CESM2_MODEL].dropna()
    paleo = s[s.index < STITCH_YEAR]
    yrs = paleo.index.values.astype(float)
    slope, intercept = np.polyfit(yrs, paleo.values, 1)
    return slope, intercept


def apply_1850_adjustment(tas_annual, model_name, gmst=None):
    """Apply the pre-1850 stitch correction for `model_name` to a tas field.

    Parameters
    ----------
    tas_annual : xr.DataArray
        Annual-mean tas with a ``year`` coordinate (any other dims, e.g.
        lat/lon, are broadcast over unchanged). Must already be on the CE
        year axis (any model-year offset applied beforehand, as in
        ce_simul.pmip4_config.YEAR_OFFSETS).
    model_name : str
        PMIP4 model name, as registered in ce_simul.pmip4_config.
    gmst : pd.DataFrame, optional
        Pre-loaded PMIP4_GMST_anomalies.csv contents; loaded from disk (and
        cached) if not given.

    Returns
    -------
    xr.DataArray
        `tas_annual` with the pre-1850 segment corrected. Years >= 1850
        (historical) are returned unchanged. Models with no registered
        correction are returned unchanged.
    """
    if gmst is None:
        gmst = _load_gmst()

    if model_name == CESM2_MODEL:
        slope, intercept = compute_cesm2_trend(gmst)
        trend = slope * tas_annual.year + intercept
        return xr.where(tas_annual.year < STITCH_YEAR, tas_annual - trend, tas_annual, keep_attrs=True)

    if model_name in FLAT_OFFSET_MODELS:
        offset = compute_flat_offsets(gmst, [model_name])[model_name]
        return xr.where(tas_annual.year < STITCH_YEAR, tas_annual + offset, tas_annual, keep_attrs=True)

    return tas_annual


if __name__ == '__main__':
    gmst = _load_gmst()

    print(f"{'Model':<22} {'offset (°C)'}")
    print('-' * 40)
    for name, offset in compute_flat_offsets(gmst).items():
        print(f"{name:<22} {offset:+.4f}")

    slope, intercept = compute_cesm2_trend(gmst)
    print(f"\n{CESM2_MODEL} past1000 linear trend: {slope * 100:+.4f} °C/century")
    print(f"  trend(year) = {slope:+.6f}*year + {intercept:+.4f}")
