"""
apply_1850_adjustments.py
==========================
One-off migration script: apply the pre-1850 stitch corrections derived in
PMIP4_1850_stitch.ipynb to the raw monthly paleo data for the two models
that need one, and write a corrected version of those netCDF files back
into the PMIP4 archive — so downstream code (ce_fields.products) never has
to correct these again; it just reads the archive as-is.

Background
----------
Of the 9 PMIP4/CMIP6-hist models we carry, only ACCESS-ESM1-5 (flat offset)
and CESM2 (linear detrend of an unrelated warming-drift artifact) show a
1849->1850 jump larger than internal-variability noise (>0.1 degC) — see
the notebook for the full derivation and why a decadal-mean-matching
diagnostic was the wrong tool for this. Both corrections are re-derived here
directly from ce_fields/data/PMIP4_GMST_anomalies.csv by the helpers in
ce_fields.stitch_corrections, so this script can't drift out of sync with
the notebook.

The correction (a constant offset for ACCESS-ESM1-5, a linear trend for
CESM2) is derived from annual GMST but applied uniformly to every month in
the raw monthly paleo file — every year in that file is pre-1850 by
construction, so there's no need to reload or restitch the historical run
here at all.

This script reads the ORIGINAL raw paleo data (hardcoded below, independent
of data_config.yml's pmip4_ACCESS-ESM1-5/pmip4_CESM2 entries, which have
been repointed to the corrected outputs written here) — do not rerun this
after that repointing without pointing RAW_PALEO_SOURCES back at the
original data, or the correction will be applied twice.

Needs the real PMIP4 data mounted (HPC only). Run from the code/ directory:
    python apply_1850_adjustments.py
"""

import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, '..')
from ce_fields.data_config import DataConfig
from ce_fields.pmip4_config import MODEL_VAR
from ce_fields.io import open_nc_or_zarr
from ce_fields.stitch_corrections import (
    ACCESS_MODEL, CESM2_MODEL, apply_1850_adjustment, load_gmst_anomalies,
)

CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


if __name__ == '__main__':
    config = DataConfig()
    env_paths = config.config['environments'][config.environment]['paths']
    pmip4_root = Path(env_paths['pmip4_data'])

    # Original, pre-correction source paths -- NOT the data_config.yml
    # pmip4_<model> entries, which now point at the corrected outputs below.
    RAW_PALEO_SOURCES = {
        ACCESS_MODEL: pmip4_root / 'zarr' / 'ACCESS-ESM1-5' / 'past1000' / 'r1i1p1f1' / 'Amon' / 'tas' / 'tas_raw.zarr',
        CESM2_MODEL: pmip4_root / 'netCDF' / 'CESM2',
    }

    CORRECTED_TARGETS = {
        ACCESS_MODEL: (pmip4_root / 'netCDF' / 'CSIRO' / 'ACCESS-ESM1-5' / 'past1000' / 'r1i1p1f1'
                       / 'Amon' / 'tas' / 'gn' / 'v20230508'
                       / 'tas_Amon_ACCESS-ESM1-5_past1000_r1i1p1f1_gn_1850corrected.nc'),
        CESM2_MODEL: (pmip4_root / 'netCDF' / 'CESM2' / 'corrected'
                      / 'TREFHT_Amon_CESM2_past1000_run002_1850corrected.nc'),
    }

    gmst = load_gmst_anomalies()

    for model_name in (ACCESS_MODEL, CESM2_MODEL):
        print(f"Applying 1850 adjustment to {model_name}...")
        var = MODEL_VAR.get(model_name, 'tas')

        ds = open_nc_or_zarr(RAW_PALEO_SOURCES[model_name], decode_times=CFTIME_CODER)
        corrected = apply_1850_adjustment(ds[var], model_name, gmst)
        corrected.attrs = ds[var].attrs
        corrected_ds = corrected.to_dataset(name=var)
        corrected_ds.attrs.update(ds.attrs)
        corrected_ds.attrs['pre_1850_correction'] = (
            'linear_detrend' if model_name == CESM2_MODEL else 'flat_offset'
        )
        corrected_ds.attrs['correction_source'] = 'code/apply_1850_adjustments.py'

        out_path = CORRECTED_TARGETS[model_name]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_ds.to_netcdf(out_path)
        ds.close()
        print(f"  saved: {out_path}")
