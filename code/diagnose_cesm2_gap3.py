"""
diagnose_cesm2_gap3.py
========================
Follow-up to diagnose_cesm2_gap2.py, which showed load_stitched_tas('CESM2')
now fails with:
    AlignmentError: cannot align objects with join='override' with matching
    indexes along dimension 'lat' that don't have the same size

i.e. CESM2's paleo (native CAM run002) grid and its CMIP6 historical
(r11i1p1f1) grid have a different number of lat points -- a real spatial
grid mismatch, not a floating-point coordinate quirk. This prints both
grids' shapes and coordinate values so we can decide how to reconcile them
(regrid one to the other vs. some other resolution).

Needs the real PMIP4 data mounted (HPC only). Run from the code/ directory:
    python diagnose_cesm2_gap3.py
"""

import sys

sys.path.insert(0, '..')
from ce_fields.data_config import DataConfig
from ce_fields.io import open_nc_or_zarr
from ce_fields.pmip4_config import get_pmip4_models, get_cmip_hist_models
from ce_fields.products import CFTIME_CODER

MODEL = 'CESM2'
config = DataConfig()

pmip4_models = get_pmip4_models(config)
cmip_hist_models = get_cmip_hist_models(config)

ds_paleo = open_nc_or_zarr(pmip4_models[MODEL], decode_times=CFTIME_CODER)
ds_hist = open_nc_or_zarr(cmip_hist_models[MODEL], decode_times=CFTIME_CODER)

for label, ds in [('PALEO (native CAM run002)', ds_paleo), ('HISTORICAL (CMIP6 r11i1p1f1)', ds_hist)]:
    print(f"{label}:")
    print(f"  lat: size={ds.lat.size}  range=[{float(ds.lat.min()):.4f}, {float(ds.lat.max()):.4f}]")
    print(f"  lon: size={ds.lon.size}  range=[{float(ds.lon.min()):.4f}, {float(ds.lon.max()):.4f}]")
    print(f"  first 5 lat values: {ds.lat.values[:5]}")
    print(f"  last 5 lat values:  {ds.lat.values[-5:]}")
    print()

ds_paleo.close()
ds_hist.close()
