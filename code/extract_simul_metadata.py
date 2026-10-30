#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pull provenance metadata (institution, contact, resolution, variant label,
etc.) out of every dataset registered in data_config.yml and write it to
results/tables/PMIP4_simul_metadata.csv.

Only reads global/variable attributes and coordinate shapes/ranges -- never
loads the actual data arrays. For directory-of-netCDF entries (e.g.
cmip_hist_CESM2, pmip4_CESM1.1, raw_pmip4_CESM2), metadata is read from the
first file in the directory (sorted) and the file count is recorded; CMIP6
convention is that global attrs are identical across a run's files.

Usage
-----
    python code/extract_simul_metadata.py

On the HPC, submit via the companion SLURM script:
    sbatch code/extract_simul_metadata.sh
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ce_fields.data_config import DataConfig

# Global attributes worth surfacing if present (CMIP6/CMOR convention; raw
# native-model output typically has few or none of these).
GLOBAL_ATTRS = [
    'source_id', 'source', 'institution_id', 'institution', 'contact',
    'variant_label', 'realization_index', 'initialization_index',
    'physics_index', 'forcing_index', 'experiment_id', 'grid_label',
    'nominal_resolution', 'further_info_url', 'tracking_id', 'Conventions',
    'parent_source_id', 'parent_experiment_id', 'parent_variant_label',
]

OUT_CSV = Path(__file__).resolve().parent.parent / 'results' / 'tables' / 'PMIP4_simul_metadata.csv'


def first_nc_in(directory: Path) -> Path:
    files = sorted(directory.glob('*.nc'))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {directory}")
    return files[0]


def describe_dataset(key: str, path, description: str) -> dict:
    """Open just enough of `path` to harvest metadata; never loads data arrays."""
    p = Path(path)
    n_files = None

    if p.suffix == '.nc':
        fmt = 'netcdf-single'
        ds = xr.open_dataset(p, decode_times=False)
    elif p.is_dir() and any(p.glob('*.nc')):
        fmt = 'netcdf-multi'
        nc_files = sorted(p.glob('*.nc'))
        n_files = len(nc_files)
        ds = xr.open_dataset(nc_files[0], decode_times=False)
    else:
        fmt = 'zarr'
        ds = xr.open_zarr(p, decode_times=False)

    row = {
        'file_key': key,
        'model': key.split('_', 1)[1] if '_' in key else key,
        'path': str(p),
        'format': fmt,
        'n_files': n_files,
        'description': description,
    }

    # variable of interest: tas (CMIP6-standard) or TREFHT (native CAM)
    var_name = 'tas' if 'tas' in ds.data_vars else (
        'TREFHT' if 'TREFHT' in ds.data_vars else None)
    row['variable'] = var_name
    if var_name is not None:
        da = ds[var_name]
        row['variable_units'] = da.attrs.get('units', '')
        row['variable_long_name'] = da.attrs.get('long_name', '')
    else:
        row['variable_units'] = ''
        row['variable_long_name'] = ''

    # grid shape / resolution
    lat_name = 'lat' if 'lat' in ds.coords else ('latitude' if 'latitude' in ds.coords else None)
    lon_name = 'lon' if 'lon' in ds.coords else ('longitude' if 'longitude' in ds.coords else None)
    if lat_name is not None and lon_name is not None:
        row['n_lat'] = ds.sizes[lat_name]
        row['n_lon'] = ds.sizes[lon_name]
        lat_vals = np.asarray(ds[lat_name].values, dtype=float)
        lon_vals = np.asarray(ds[lon_name].values, dtype=float)
        row['lat_spacing_deg'] = round(float(np.abs(np.diff(lat_vals)).mean()), 3) if lat_vals.size > 1 else None
        row['lon_spacing_deg'] = round(float(np.abs(np.diff(lon_vals)).mean()), 3) if lon_vals.size > 1 else None
    else:
        row['n_lat'] = row['n_lon'] = row['lat_spacing_deg'] = row['lon_spacing_deg'] = None

    # time coverage (raw, undecoded units -- avoids cftime/calendar surprises)
    if 'time' in ds.coords:
        row['n_time_steps'] = ds.sizes['time']
        row['time_units'] = ds['time'].attrs.get('units', '')
        row['time_calendar'] = ds['time'].attrs.get('calendar', '')
    else:
        row['n_time_steps'] = row['time_units'] = row['time_calendar'] = None

    for attr in GLOBAL_ATTRS:
        row[attr] = ds.attrs.get(attr, '')

    ds.close()
    return row


def main():
    config = DataConfig()
    print(f"Environment : {config.environment}")

    columns = (
        ['file_key', 'model', 'variable', 'variable_units', 'variable_long_name',
         'n_lat', 'n_lon', 'lat_spacing_deg', 'lon_spacing_deg',
         'n_time_steps', 'time_units', 'time_calendar',
         'format', 'n_files', 'path']
        + GLOBAL_ATTRS
        + ['description']
    )

    rows = []
    for key, file_info in config.config['files'].items():
        path = config.get_file_path(key)
        print(f"\n  {key}  ({path})")
        try:
            row = describe_dataset(key, path, file_info.get('description', ''))
            rows.append(row)
            print(f"    ✓  {row.get('format')}"
                  + (f", {row['n_files']} files" if row.get('n_files') else "")
                  + (f", {row.get('n_lat')}x{row.get('n_lon')}" if row.get('n_lat') else ""))
        except Exception:
            print(f"    ✗  FAILED:")
            traceback.print_exc()
            rows.append({'file_key': key, 'path': str(path),
                         'description': file_info.get('description', ''),
                         'model': key.split('_', 1)[1] if '_' in key else key})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval='')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote metadata for {len(rows)} dataset(s) to {OUT_CSV}")


if __name__ == '__main__':
    main()
