"""
diagnose_cesm2_gap.py
======================
One-off diagnostic: the borehole product (Antarctic_BH_surface_temp_PMIP4.nc)
shows CESM2c covering only 1851-2014, missing its ~1000-year past1000 paleo
segment (850-1850) that should be there per apply_1850_adjustments.py.

This checks two candidate points of failure:
  1. The corrected file itself (data_config.yml's pmip4_CESM2 entry) --
     if its time axis is already short, the bug is in
     apply_1850_adjustments.py's correction/write step.
  2. The raw source directory apply_1850_adjustments.py read from --
     if that's already short, the bug is upstream, in file discovery
     (open_mfdataset's glob over pmip4_root/netCDF/CESM2).

Needs the real PMIP4 data mounted (HPC only). Run from the code/ directory:
    python diagnose_cesm2_gap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, '..')
from ce_fields.data_config import DataConfig
from ce_fields.io import open_nc_or_zarr

config = DataConfig()
pmip4_root = Path(config.config['environments'][config.environment]['paths']['pmip4_data'])

# 1. The corrected file actually used by load_stitched_tas
corrected_path = config.get_file_path('pmip4_CESM2')
ds_c = open_nc_or_zarr(corrected_path, decode_times=True, use_cftime=True)
print('CORRECTED file:', corrected_path)
print(f'  time range: {ds_c.time.values[0]} -> {ds_c.time.values[-1]} ({ds_c.time.size} steps)')

# 2. The raw source directory apply_1850_adjustments.py globbed
raw_dir = pmip4_root / 'netCDF' / 'CESM2'
files = sorted(raw_dir.glob('*.nc'))
print()
print('RAW source dir:', raw_dir)
print(f'  {len(files)} .nc files:')
for f in files:
    print('   ', f.name)

ds_r = open_nc_or_zarr(raw_dir, decode_times=True, use_cftime=True)
print(f'  combined time range: {ds_r.time.values[0]} -> {ds_r.time.values[-1]} ({ds_r.time.size} steps)')
