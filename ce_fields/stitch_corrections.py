"""
Pre-1850 stitch corrections for PMIP4 past1000 runs, derived from annual GMST.

Of the 9 PMIP4/CMIP6-hist models we carry, only ACCESS-ESM1-5 (flat offset)
and CESM2 (linear detrend of an unrelated warming-drift artifact) show a
1849->1850 jump larger than internal-variability noise (>0.1 degC); see
code/PMIP4_1850_stitch.ipynb for the derivation. The corrections are derived
from the annual GMST anomalies shipped with this package
(ce_fields/data/PMIP4_GMST_anomalies.csv, written by
code/PMIP4_GMST_sanity_check.py), so that downstream repositories can
reproduce them without the raw PMIP4 output. code/apply_1850_adjustments.py
uses the same functions to write the corrected monthly fields.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GMST_ANOMALIES_CSV = Path(__file__).resolve().parent / 'data' / 'PMIP4_GMST_anomalies.csv'
STITCH_YEAR = 1850

ACCESS_MODEL = 'ACCESS-ESM1-5'
CESM2_MODEL = 'CESM2'


def load_gmst_anomalies(csv_path: Path = GMST_ANOMALIES_CSV) -> pd.DataFrame:
    """Load PMIP4_GMST_anomalies.csv into a DataFrame (index: year CE, one column per model, °C).

    The trailing `has_stitch` row (1 if a CMIP6 historical run was stitched
    at 1850) is dropped.
    """
    raw = pd.read_csv(csv_path, index_col='year', dtype=object)
    gmst = raw.drop(index='has_stitch').astype(float)
    gmst.index = gmst.index.astype(int)
    return gmst


def compute_flat_offset(gmst: pd.DataFrame, model_name: str = ACCESS_MODEL) -> float:
    """Offset = v(1850) - v(1849) — the single-year restart discontinuity,
    not a difference of pre/post-1850 decadal means (see the notebook's
    "Choosing the right diagnostic" section for why)."""
    s = gmst[model_name].dropna()
    return s.loc[STITCH_YEAR] - s.loc[STITCH_YEAR - 1]


def compute_cesm2_trend(gmst: pd.DataFrame) -> tuple:
    """Fit a linear trend to CESM2 past1000 GMST (850-1849).

    Returns (slope, intercept) of trend(year) = slope*year + intercept, in
    °C/year and °C.
    """
    s = gmst[CESM2_MODEL].dropna()
    paleo = s[s.index < STITCH_YEAR]
    yrs = paleo.index.values.astype(float)
    slope, intercept = np.polyfit(yrs, paleo.values, 1)
    return slope, intercept


def apply_1850_adjustment(tas_monthly: xr.DataArray, model_name: str, gmst: pd.DataFrame) -> xr.DataArray:
    """Apply the pre-1850 correction to a monthly-resolution paleo field.

    The annual-derived correction is broadcast across every month of every
    year in `tas_monthly` — safe because the raw paleo files for both
    ACCESS-ESM1-5 and CESM2 only ever contain pre-1850 years.
    """
    if model_name == CESM2_MODEL:
        slope, intercept = compute_cesm2_trend(gmst)
        trend = slope * tas_monthly['time'].dt.year + intercept
        return tas_monthly - trend

    if model_name == ACCESS_MODEL:
        offset = compute_flat_offset(gmst, model_name)
        return tas_monthly + offset

    raise ValueError(f"No registered correction for '{model_name}'")
