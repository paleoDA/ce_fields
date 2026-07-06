"""
diagnose_cesm2_gap2.py
========================
Follow-up to diagnose_cesm2_gap.py: that script showed the corrected
CESM2 file itself is fine (continuous 850-1850, 12000 monthly steps), so
the ~1000-year paleo gap seen in Antarctic_BH_surface_temp_PMIP4.nc must
appear somewhere in ce_simul.products.load_stitched_tas's groupby/stitch
logic, or in Antarctic_BH_surface_temp.py's own per-model combine step.

This calls the actual loading pipeline directly and inspects the annual
series at each stage, to localize exactly where the gap first appears:
  1. Right after ds_paleo[var].groupby('time.year').mean('time') --
     before any stitching.
  2. After load_stitched_tas() has run the full pipeline (stitched with
     CMIP6 historical).

Needs the real PMIP4 data mounted (HPC only). Run from the code/ directory:
    python diagnose_cesm2_gap2.py
"""

import sys

sys.path.insert(0, '..')
from ce_simul.data_config import DataConfig
from ce_simul.io import open_nc_or_zarr
from ce_simul.pmip4_config import get_pmip4_models, get_cmip_hist_models, MODEL_VAR
from ce_simul.products import load_stitched_tas, CFTIME_CODER

MODEL = 'CESM2'
config = DataConfig()

# ── Stage 1: raw groupby, no stitching ───────────────────────────────────
pmip4_models = get_pmip4_models(config)
ds_paleo = open_nc_or_zarr(pmip4_models[MODEL], decode_times=CFTIME_CODER)
var = MODEL_VAR.get(MODEL, 'tas')
tas_annual = ds_paleo[var].groupby('time.year').mean('time')

years = tas_annual.year.values
print(f"Stage 1 (raw groupby, pre-stitch): {years[0]}-{years[-1]}  ({len(years)} year labels)")
gaps = years[1:][years[1:] - years[:-1] > 1]
print(f"  Gaps in year labels: {gaps.tolist() if len(gaps) else 'none'}")
n_nan = int(tas_annual.isnull().all(dim=[d for d in tas_annual.dims if d != 'year']).sum())
print(f"  Year labels that are all-NaN over space: {n_nan}")
ds_paleo.close()

# ── Stage 2: full load_stitched_tas() pipeline ───────────────────────────
tas_stitched = load_stitched_tas(MODEL, config=config)
years2 = tas_stitched.year.values
print(f"\nStage 2 (after load_stitched_tas, post-stitch): {years2[0]}-{years2[-1]}  ({len(years2)} year labels)")
gaps2 = years2[1:][years2[1:] - years2[:-1] > 1]
print(f"  Gaps in year labels: {gaps2.tolist() if len(gaps2) else 'none'}")
n_nan2 = int(tas_stitched.isnull().all(dim=[d for d in tas_stitched.dims if d != 'year']).sum())
print(f"  Year labels that are all-NaN over space: {n_nan2}")
if n_nan2:
    nan_years = years2[tas_stitched.isnull().all(dim=[d for d in tas_stitched.dims if d != 'year']).values]
    print(f"  First/last all-NaN years: {nan_years[0]}-{nan_years[-1]}")
