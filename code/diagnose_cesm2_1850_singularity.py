"""
diagnose_cesm2_1850_singularity.py
====================================
Antarctic_BH_surface_temp_PMIP4.nc now stitches CESM2c cleanly (850-2014,
no gaps -- see diagnose_cesm2_gap2.py / 75a9985 / 212a5c4), but every one of
the 4 borehole sites shows a huge (+2.6 to +17.6 degC) one-year warm spike
at 1850, reverting the following year. Year 1850 comes entirely from the
PALEO segment (stitch_annual_tas's 1-year-overlap rule drops hist's 1850,
keeps paleo's -- see the "Dropping year 1850 from historical run" log line),
so this can't be caused by the regrid fix, which only touches hist years
(>=1851).

Two suspects:
  1. apply_1850_adjustments.py's docstring assumes "every year in [the raw
     paleo] file is pre-1850 by construction" -- but Stage 1 of
     diagnose_cesm2_gap2.py showed the corrected file spans 850-1850 (1001
     year labels), i.e. year 1850 IS present, contradicting that assumption.
     compute_cesm2_trend() fits the linear detrend on years <1850 only, so
     1850's correction is a 1-year extrapolation -- worth checking it isn't
     wildly different from 1849's.
  2. The SerializationWarning seen during diagnose_cesm2_gap2.py ("variable
     'tas'/'TREFHT' has multiple fill values... decoding all values to NaN")
     could be silently NaN-ing some months. xarray's .mean('time',
     skipna=True) default would then average only the SURVIVING months --
     if winter (JJA) months got wiped for exactly year 1850, the remaining
     (Antarctic summer) months would produce exactly this kind of large,
     single-year warm bias.

This checks, for both the CORRECTED and RAW archives:
  - How many monthly time steps exist in year 1850 vs neighboring years
    (partial-year check).
  - How many of those months are all-NaN in space (fill-value check).
  - The actual per-month site values at year 1850 vs 1849/1851, at WAIS
    (steepest anomaly: +14.75 degC).

Needs the real PMIP4 data mounted (HPC only). Run from the code/ directory:
    python diagnose_cesm2_1850_singularity.py
"""

import sys

import numpy as np
import xarray as xr

sys.path.insert(0, '..')
from ce_fields.data_config import DataConfig
from ce_fields.io import open_nc_or_zarr
from ce_fields.pmip4_config import get_pmip4_models, MODEL_VAR
from ce_fields.products import CFTIME_CODER

MODEL = 'CESM2'
WAIS_LAT, WAIS_LON = -79 - 28/60, 360 - (112 + 5/60)

config = DataConfig()
pmip4_models = get_pmip4_models(config)
var = MODEL_VAR.get(MODEL, 'tas')


def inspect(label, path):
    print(f"\n{'='*70}\n{label}: {path}\n{'='*70}")
    ds = open_nc_or_zarr(path, decode_times=CFTIME_CODER)
    da = ds[var]

    for year in (1848, 1849, 1850, 1851):
        yr_slice = da.sel(time=da['time'].dt.year == year)
        n_months = yr_slice.sizes.get('time', 0)
        if n_months == 0:
            print(f"  {year}: NOT PRESENT in this file")
            continue
        all_nan_months = int(
            yr_slice.isnull().all(dim=[d for d in yr_slice.dims if d != 'time']).sum()
        )
        print(f"  {year}: {n_months} months present, {all_nan_months} entirely NaN")

    # Per-month WAIS-nearest value across the transition, raw units (no
    # correction applied here even for the corrected file -- just reading
    # what's archived).
    print(f"\n  Monthly {var} nearest WAIS ({WAIS_LAT:.2f}, {WAIS_LON:.2f}), 1849-1851:")
    window = da.sel(time=(da['time'].dt.year >= 1849) & (da['time'].dt.year <= 1851))
    site = window.sel(lat=WAIS_LAT, lon=WAIS_LON, method='nearest').load()
    for t, v in zip(site['time'].values, site.values):
        flag = " <- NaN" if np.isnan(v) else ""
        print(f"    {t}: {v:.2f}{flag}")

    ds.close()


inspect('CORRECTED (data_config pmip4_CESM2)', pmip4_models[MODEL])

try:
    from ce_fields.data_config import DataConfig as _DC
    raw_path = config.get_file_path('raw_pmip4_CESM2')
    inspect('RAW (pre-correction, data_config raw_pmip4_CESM2)', raw_path)
except KeyError as e:
    print(f"\nNo raw_pmip4_CESM2 entry available: {e}")
