#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instrumental gridded temperature products (HadCRUT5, CRUTEM5, Berkeley Earth,
DCENT-I, ...), loaded by explicit version.

Every load names its version: `Instrumental.from_version('hadcrut5', '5.1.0.0')`.
There is deliberately no default and no "latest", because these products
revise past values between releases, so a number computed from one release
is not reproducible from another. The catalog lives in data_config.yml (keys
`instr_<product>_<version>_<variable>_<kind>`, one per file), so a new release
is registered by adding an entry there, not by editing this module.

Each entry also carries a `version_check` (a global attribute of the file and
a string it must contain). The loader refuses a file that fails it, so a newer
download saved under the old file name cannot silently stand in for the
version that was asked for.

All products are returned on the same conventions as ce_fields' LMR loader:
dims (`year` or `time`, `lat`, `lon`[, `member`]), latitude ascending,
longitude 0-360 by default, and numeric years (integer `year` once annualized,
decimal mid-month `time` otherwise).

Created on Tue Nov 24 2026
@author: Julien Emile-Geay with Claude AI assistance
"""

import glob
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ce_fields.data_config import DataConfig

CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
CATALOG_PREFIX = 'instr_'

# Version-independent facts about each product. Version-specific facts
# (file, variable name, version check) live in data_config.yml.
PRODUCTS = {
    'hadcrut5': dict(
        name='HadCRUT5', native_baseline=(1961, 1990),
        reference='Morice et al. (2021), doi:10.1029/2019JD032361'),
    'crutem5': dict(
        name='CRUTEM5', native_baseline=(1961, 1990),
        reference='Osborn et al. (2021), doi:10.1029/2019JD032352'),
    'berkeley_earth': dict(
        name='Berkeley Earth', native_baseline=(1951, 1980),
        reference='Rohde & Hausfather (2020), doi:10.5194/essd-12-3469-2020'),
    'dcent_i': dict(
        name='DCENT-I', native_baseline=(1982, 2014),
        reference='Chan et al. (2026), Geosci. Data J. 13, e70054, doi:10.1002/gdj3.70054'),
}


def catalog(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """Every registered instrumental file, one row per (product, version,
    variable, kind), with whether it is found on this machine."""
    config = config or DataConfig()
    rows = []
    for key, info in config.config['files'].items():
        if not key.startswith(CATALOG_PREFIX):
            continue
        try:
            path = config.get_file_path(key)
            found = bool(glob.glob(str(path)))
        except FileNotFoundError:
            path, found = None, False
        rows.append(dict(product=info['product'], version=str(info['version']),
                         variable=info['variable'], kind=info['kind'],
                         found=found, key=key, path=path))
    return pd.DataFrame(rows)


def _find_entry(config, product, version, variable, kind):
    """Resolve (product, version, variable, kind) to one catalog key, with an
    error that lists what *is* registered when nothing matches."""
    if not isinstance(version, str):
        raise TypeError(f"version must be a string such as '5.1.0.0', got {version!r} "
                        f"({type(version).__name__}): a float would lose trailing zeros.")
    cat = catalog(config)
    if product not in set(cat['product']):
        raise ValueError(f"Unknown product '{product}'. Registered: {sorted(set(cat['product']))}")
    of_product = cat[cat['product'] == product]
    if version not in set(of_product['version']):
        raise ValueError(f"{product} version '{version}' is not registered. "
                         f"Registered versions: {sorted(set(of_product['version']))}. "
                         "To add one, copy an entry in ce_fields/data_config.yml.")
    match = of_product[(of_product['version'] == version)
                       & (of_product['variable'] == variable)
                       & (of_product['kind'] == kind)]
    if match.empty:
        have = of_product[of_product['version'] == version][['variable', 'kind']]
        raise ValueError(f"{product} {version} has no variable='{variable}', kind='{kind}'. "
                         f"Available: {list(map(tuple, have.values))}")
    return match['key'].iloc[0]


def _check_version(path, check, key):
    """Raise unless the file's global attribute `check['attr']` contains
    `check['contains']`."""
    with xr.open_dataset(path, decode_times=False) as ds:
        found = ds.attrs.get(check['attr'])
    if found is None or check['contains'] not in str(found):
        raise ValueError(
            f"Version check failed for '{key}': {path} has "
            f"{check['attr']}={found!r}, expected it to contain {check['contains']!r}. "
            "The file on disk is not the registered version (was it replaced by a "
            "newer download?). Register the new file under its own version instead.")
    return str(found)


def _year_month(time: xr.DataArray):
    """Integer year and month arrays for a monthly time axis, whether it is
    decoded (cftime) or decimal years (e.g. Berkeley Earth's 'year A.D.')."""
    values = time.values
    if np.issubdtype(values.dtype, np.number):
        years = np.floor(values).astype(int)
        months = np.floor((values - years) * 12).astype(int) + 1
    else:
        years = np.array([t.year for t in values])
        months = np.array([t.month for t in values])
    return years, months


def _harmonize(da: xr.DataArray, lon_convention: str) -> xr.DataArray:
    """Rename to lat/lon, sort latitude ascending and wrap longitude."""
    da = da.rename({k: v for k, v in {'latitude': 'lat', 'longitude': 'lon'}.items()
                    if k in da.dims})
    if lon_convention == '0_360':
        lon = da['lon'] % 360
    elif lon_convention == '-180_180':
        lon = ((da['lon'] + 180) % 360) - 180
    else:
        raise ValueError(f"lon_convention must be '0_360' or '-180_180', got '{lon_convention}'")
    return da.assign_coords(lon=lon).sortby('lat').sortby('lon')


def _recenter(da: xr.DataArray, ref_period: tuple) -> xr.DataArray:
    """Express `da` as anomalies relative to `ref_period` (inclusive years):
    the period mean for annual data, the period's monthly climatology for
    monthly data. Grid cells with no data in the period become NaN."""
    if 'year' in da.dims:
        ref = da.sel(year=slice(*ref_period))
        if ref.sizes['year'] == 0:
            raise ValueError(f"No data in reference period {ref_period}")
        return da - ref.mean('year')
    in_ref = (da['year_of'] >= ref_period[0]) & (da['year_of'] <= ref_period[1])
    if not bool(in_ref.any()):
        raise ValueError(f"No data in reference period {ref_period}")
    clim = da.isel(time=in_ref.values).groupby('month_of').mean('time')
    return da.groupby('month_of') - clim


class Instrumental:
    """A versioned instrumental temperature field.

    Attributes
    ----------
    data : xr.DataArray
        dims (`year` | `time`, `lat`, `lon`[, `member`]); provenance in `data.attrs`.
    product, version, variable, kind : str
        What was loaded, exactly as requested.

    Examples
    --------
    >>> hc = Instrumental.from_version('hadcrut5', '5.1.0.0')
    >>> gmst = hc.gmst(ref_period=(1951, 1980))
    >>> hc.label
    'HadCRUT5 v5.1.0.0'
    >>> Instrumental.catalog()          # what is registered, and found here
    """

    catalog = staticmethod(catalog)

    def __init__(self, data: xr.DataArray, product: str, version: str,
                 variable: str, kind: str):
        self.data = data
        self.product = product
        self.version = version
        self.variable = variable
        self.kind = kind

    @classmethod
    def from_version(cls, product: str, version: str,
                     variable: str = 'tas', kind: str = 'mean',
                     annualize: bool = True, min_months: int = 12,
                     time_slice: Optional[tuple] = None,
                     ref_period: Optional[tuple] = None,
                     lon_convention: str = '0_360',
                     config: Optional[DataConfig] = None) -> 'Instrumental':
        """Load one registered version of an instrumental product.

        Parameters
        ----------
        product : str
            e.g. 'hadcrut5', 'crutem5', 'berkeley_earth', 'dcent_i'.
        version : str
            Required, no default, e.g. '5.0.2.0'. See `Instrumental.catalog()`.
        variable : str, optional
            'tas' (headline blended land air + SST anomaly), 'lsat' (land air)
            or 'sst'. Default 'tas'.
        kind : str, optional
            'mean' (ensemble mean / best estimate) or 'members'. Default 'mean'.
        annualize : bool, optional
            Return calendar-year means on an integer `year` dim. Default True.
            Each cell's annual mean averages whichever of its months are present.
        min_months : int, optional
            When annualizing, drop years with fewer months than this on the
            time axis, i.e. an incomplete final year of an operational release.
            Default 12.
        time_slice : tuple, optional
            (start_year, end_year), inclusive; either may be None.
        ref_period : tuple, optional
            (start_year, end_year) to re-center to. Default None keeps the
            product's native baseline (`data.attrs['native_baseline']`),
            which differs between products.
        lon_convention : str, optional
            '0_360' (default, as LMR and PMIP4) or '-180_180' (as HadCRUT5's files).
        config : DataConfig, optional
            Defaults to ce_fields' own.
        """
        config = config or DataConfig()
        key = _find_entry(config, product, version, variable, kind)
        entry = config.config['files'][key]
        path = config.get_file_path(key)
        files = sorted(glob.glob(str(path)))
        if not files:
            raise FileNotFoundError(f"No file for '{key}' at {path}")
        file_version = _check_version(files[0], entry['version_check'], key)
        print(f"Loading {PRODUCTS.get(product, {}).get('name', product)} v{version} "
              f"({variable}, {kind}) from: {path}")

        if kind == 'members':
            def keep_var(ds):
                sub = ds[[entry['var']]]
                if 'realization' in sub.dims:
                    sub = sub.squeeze('realization', drop=True)
                return sub.drop_vars('realization', errors='ignore')
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='member',
                                   preprocess=keep_var, decode_times=False,
                                   parallel=False, combine_attrs='override')
            ds = ds.assign_coords(member=np.arange(1, len(files) + 1))
        else:
            ds = xr.open_dataset(files[0], decode_times=False)
        if 'since' in str(ds['time'].attrs.get('units', '')):
            ds = xr.decode_cf(ds, decode_times=CFTIME_CODER)
        da = ds[entry['var']]
        for coord in ('realization',):
            if coord in da.coords and coord not in da.dims:
                da = da.drop_vars(coord)

        years, months = _year_month(da['time'])
        da = da.assign_coords(time=(years + (months - 0.5) / 12).astype(float),
                              year_of=('time', years), month_of=('time', months))
        da = _harmonize(da, lon_convention)

        if annualize:
            counts = pd.Series(years).value_counts()
            complete = counts.index[counts >= min_months]
            dropped = sorted(set(counts.index) - set(complete))
            if dropped:
                print(f"Dropping years with < {min_months} months: {dropped}")
            da = da.isel(time=np.isin(years, complete))
            da = da.groupby('year_of').mean('time').rename(year_of='year')
            da['year'] = da['year'].astype(int)

        if time_slice is not None:
            dim, coord = ('year', 'year') if annualize else ('time', 'year_of')
            lo, hi = time_slice
            keep = xr.ones_like(da[coord], dtype=bool)
            if lo is not None:
                keep &= da[coord] >= lo
            if hi is not None:
                keep &= da[coord] <= hi
            da = da.isel({dim: keep.values})

        if ref_period is not None:
            da = _recenter(da, ref_period)

        meta = PRODUCTS.get(product, {})
        da.name = variable
        da.attrs.update(
            product=product, version=version, variable=variable, kind=kind,
            catalog_key=key, source_path=str(path), file_version=file_version,
            native_baseline=meta.get('native_baseline'),
            baseline=tuple(ref_period) if ref_period else meta.get('native_baseline'),
            reference=meta.get('reference', ''),
            units='K (anomaly)', label=f"{meta.get('name', product)} v{version}")
        return cls(da, product, version, variable, kind)

    @property
    def label(self) -> str:
        """Name and version, for plot legends and table rows."""
        return self.data.attrs['label']

    @property
    def years(self) -> np.ndarray:
        dim = 'year' if 'year' in self.data.dims else 'time'
        return self.data[dim].values

    def gmst(self, ref_period: Optional[tuple] = None) -> xr.DataArray:
        """Cos(lat)-weighted global mean over the cells with data, as in
        `LMR.compute_global_mean()`; re-centered to `ref_period` if given.
        For land- or ocean-only variables this is a land/ocean mean, not GMST."""
        weights = np.cos(np.deg2rad(self.data['lat']))
        gm = self.data.weighted(weights).mean(['lat', 'lon'])
        if ref_period is not None:
            gm = _recenter(gm, ref_period)
        gm.attrs.update(self.data.attrs)
        if ref_period is not None:
            gm.attrs['baseline'] = tuple(ref_period)
        gm.name = 'gmst' if self.variable == 'tas' else f'global_mean_{self.variable}'
        return gm

    def grid_bounds(self):
        """(lat_bnds, lon_bnds), each (n, 2), from cell-center midpoints, for
        conservative regridding (e.g. xESMF). Latitude edges are clipped to +-90."""
        def edges(c):
            mid = (c[1:] + c[:-1]) / 2
            return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
        lat_e = np.clip(edges(self.data['lat'].values), -90, 90)
        lon_e = edges(self.data['lon'].values)
        lat_b = xr.DataArray(np.stack([lat_e[:-1], lat_e[1:]], 1), dims=('lat', 'bnds'),
                             coords={'lat': self.data['lat']})
        lon_b = xr.DataArray(np.stack([lon_e[:-1], lon_e[1:]], 1), dims=('lon', 'bnds'),
                             coords={'lon': self.data['lon']})
        return lat_b, lon_b

    def __repr__(self):
        y = self.years
        return (f"Instrumental('{self.product}', '{self.version}', variable='{self.variable}', "
                f"kind='{self.kind}', shape={self.data.shape}, years={int(y.min())}-{int(y.max())})")
