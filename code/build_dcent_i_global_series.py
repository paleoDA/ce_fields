#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a DCENT-I global-mean ensemble series in the layout of HadCRUT5's
`analysis.ensemble_series.global.monthly.nc`, so that
`Instrumental.series('dcent_i', '1.1.0.0', region='global')` loads it exactly
as it loads HadCRUT5's producer series.

DCENT-I ships no global series, so this is derived: for each of the 200
member files, the cos(lat)-weighted mean over cells with data of the infilled
blended field `ts` (as `Instrumental.gmst()`), monthly, on the native
1982-2014 baseline. Written next to the member files in `{dcent_data}/v1.1.0/`.

Usage:
    python code/build_dcent_i_global_series.py           # refuses to overwrite
    python code/build_dcent_i_global_series.py --force

Created on Thu Nov 26 2026
@author: Julien Emile-Geay with Claude AI assistance
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from ce_fields.data_config import DataConfig

VERSION = '1.1.0.0'
OUT_NAME = f'DCENT_I_{VERSION}_ensemble_series.global.monthly.nc'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--force', action='store_true', help='overwrite an existing output file')
    args = parser.parse_args()

    member_glob = Path(DataConfig().get_file_path(f'instr_dcent_i_{VERSION}_tas_members'))
    files = sorted(member_glob.parent.glob(member_glob.name))
    assert len(files) == 200, f'expected 200 member files, found {len(files)}'
    out = member_glob.parent / OUT_NAME
    if out.exists() and not args.force:
        raise SystemExit(f'{out} exists; pass --force to rebuild')

    series, time, time_bnds = [], None, None
    for f in files:
        with xr.open_dataset(f, decode_times=False) as ds:
            assert VERSION in ds.attrs['version'], (f, ds.attrs['version'])
            ts = ds['ts'].squeeze(drop=True)
            gm = ts.weighted(np.cos(np.deg2rad(ts['lat']))).mean(['lat', 'lon']).values
            if time is None:
                time, time_bnds = ds['time'].copy(), ds['time_bnds'].values
            else:
                assert np.array_equal(ds['time'].values, time.values), f
            series.append(gm)
        print(f.name, end='\r')

    da = xr.DataArray(
        np.array(series, dtype='float64'), dims=('realization', 'time'),
        coords={'realization': np.arange(1, len(files) + 1), 'time': time.values},
        name='tas',
        attrs={'long_name': 'global mean blended surface temperature anomaly (cos-lat mean of ts)',
               'units': 'K'})
    da['time'].attrs = {k: v for k, v in time.attrs.items() if k != '_FillValue'}
    ds_out = da.to_dataset()
    ds_out['time_bnds'] = (('time', 'bnds'), time_bnds)
    ds_out.attrs = {
        'name': 'DCENT_I', 'version': VERSION,
        'title': f'DCENT-I v{VERSION} global-mean ensemble series (derived)',
        'comment': ('Derived by code/build_dcent_i_global_series.py:cos(lat)-weighted '
                    'mean over cells with data of each member\'s infilled blended field ts. Monthly '
                    'anomalies relative to the native 1982-2014 climatology. Not a producer series.'),
        'source': ', '.join(f.name for f in files[:1]) + f' ... ({len(files)} member files)',
        'history': f'Built on {date.today().isoformat()}',
    }
    ds_out.to_netcdf(out, encoding={'time': {'_FillValue': None}})
    print(f'\nwrote {out}  ({da.sizes["realization"]} members, {da.sizes["time"]} months)')


if __name__ == '__main__':
    main()
