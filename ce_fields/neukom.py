#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neukom et al. (2019) Common Era Multiproxy Temperature Field Reconstructions

Created on Mon Jul 20 2026
@author: Julien Emile-Geay with Claude AI assistance
"""

import numpy as np
import xarray as xr
from typing import Dict, Optional
from ce_fields.data_config import DataConfig

# Every archived file spans exactly this many consecutive annual steps
# (readme-neukom2019.txt: Earliest_Year: 1, Most_Recent_Year: 2000). Years
# are assigned by position rather than by decoding the raw time axis: the
# "days since" encoding in AM/CCA/CPS/PCR assumes a simplified leap-year
# rule that drifts by a day relative to the true Gregorian calendar at
# century years not divisible by 400 (e.g. year 300), which would otherwise
# produce duplicate/skipped calendar-year labels under strict CF decoding.
N_YEARS = 2000


class Neukom2019:
    """
    Neukom et al. (2019, Nature) global gridded Common Era temperature field
    reconstructions.

    Six independent reconstruction methods, each a 100-member ensemble at 5°
    resolution, annual (May-April aggregated) resolution, 1-2000 CE: AM
    (Analog Method), CCA (Canonical Correlation Analysis), CPS
    (Composite-Plus-Scale), DA (Data Assimilation), GraphEM (Graphical
    Expectation-Maximization), PCR (Principal Component Regression).

    Reference
    ---------
    Neukom, R., Steiger, N., Gómez-Navarro, J.J. et al. No evidence for
    globally coherent warm and cold periods over the preindustrial Common
    Era. Nature 571, 550-554 (2019). https://doi.org/10.1038/s41586-019-1401-2

    Attributes
    ----------
    data : xr.DataArray
        Gridded ensemble field with dims (year, member, lat, lon)
    method : str
        Reconstruction method name

    Examples
    --------
    Load one method and compute its GMST ensemble:

    >>> cps = Neukom2019.from_method('CPS')
    >>> gmst = cps.compute_gmst()

    Load all six methods:

    >>> methods = Neukom2019.load_all_methods()
    >>> for name, recon in methods.items():
    ...     print(name, recon.data.shape)
    """

    # config_key   : key in data_config.yml (files section)
    # var          : name of the temperature variable in the raw file
    # member_dim   : name of the ensemble-member dimension in the raw file
    METHOD_MAP = {
        'AM':      dict(config_key='neukom_am',     var='var0', member_dim='lev'),
        'CCA':     dict(config_key='neukom_cca',     var='tas',  member_dim='lev'),
        'CPS':     dict(config_key='neukom_cps',     var='tas',  member_dim='lev'),
        'DA':      dict(config_key='neukom_da',      var='temp', member_dim='ens'),
        'GraphEM': dict(config_key='neukom_graphem', var='temp', member_dim='ens'),
        'PCR':     dict(config_key='neukom_pcr',     var='tas',  member_dim='lev'),
    }

    def __init__(self, data: xr.DataArray, method: str):
        """
        Initialize Neukom2019 object.

        Parameters
        ----------
        data : xr.DataArray
            Gridded ensemble field with dims (year, member, lat, lon)
        method : str
            Reconstruction method name
        """
        self.data = data
        self.method = method

    @classmethod
    def from_method(cls, method: str, time_slice: Optional[tuple] = None,
                   chunks: Optional[dict] = None) -> 'Neukom2019':
        """
        Load one reconstruction method's gridded ensemble field.

        Parameters
        ----------
        method : str
            One of 'AM', 'CCA', 'CPS', 'DA', 'GraphEM', 'PCR'
        time_slice : tuple of (int, int), optional
            (start_year, end_year); either may be None for an open bound.
            Default None (no slicing).
        chunks : dict, optional
            Dask chunking passed to xr.open_dataset. Default {'time': 200},
            small enough to keep spatial reductions (e.g. compute_gmst)
            memory-light on these 1-3 GB files.

        Returns
        -------
        Neukom2019

        Raises
        ------
        ValueError
            If method is not recognised

        Examples
        --------
        >>> cps = Neukom2019.from_method('CPS', time_slice=(1000, 2000))
        """
        if method not in cls.METHOD_MAP:
            raise ValueError(f"Unknown method '{method}'. Must be one of {list(cls.METHOD_MAP)}")

        info = cls.METHOD_MAP[method]
        config = DataConfig()
        filepath = config.get_file_path(info['config_key'])
        print(f"Loading Neukom2019 {method} from: {filepath}")

        if chunks is None:
            chunks = {'time': 200}

        ds = xr.open_dataset(filepath, decode_times=False, chunks=chunks)
        data = ds[info['var']].rename({info['member_dim']: 'member'})
        data.name = 'tas'

        n_time = data.sizes['time']
        assert n_time == N_YEARS, (
            f"Expected {N_YEARS} annual time steps for Neukom2019 {method} "
            f"(readme-neukom2019.txt), got {n_time}")
        years = np.arange(1, N_YEARS + 1)
        data = data.assign_coords(time=years).rename(time='year')

        if time_slice is not None:
            start_year, end_year = time_slice
            data = data.sel(year=slice(start_year, end_year))

        return cls(data, method)

    @classmethod
    def load_all_methods(cls, time_slice: Optional[tuple] = None) -> Dict[str, 'Neukom2019']:
        """
        Load all six reconstruction methods.

        Parameters
        ----------
        time_slice : tuple of (int, int), optional
            (start_year, end_year) passed through to from_method

        Returns
        -------
        dict
            Mapping method name -> Neukom2019 object. Methods whose data
            file cannot be found or loaded are skipped with a warning,
            rather than aborting the whole load.

        Examples
        --------
        >>> methods = Neukom2019.load_all_methods(time_slice=(1000, 2000))
        """
        results = {}
        for method in cls.METHOD_MAP:
            try:
                results[method] = cls.from_method(method, time_slice=time_slice)
            except Exception as e:
                print(f"Warning: could not load Neukom2019 {method}: {e}")
        return results

    def compute_gmst(self, ref_period: tuple = (1961, 1990)) -> xr.DataArray:
        """
        Compute latitude-weighted global mean, as an anomaly relative to
        ref_period.

        Subtracting the reference-period mean also neutralizes the fact
        that the DA method's field is archived in absolute Kelvin while the
        other five methods are already anomalies in degrees Celsius: an
        additive offset (e.g. 273.15 K) cancels out of an anomaly regardless
        of the field's original units.

        Parameters
        ----------
        ref_period : tuple of (int, int), optional
            Reference period (inclusive) for the anomaly baseline.
            Default (1961, 1990).

        Returns
        -------
        xr.DataArray
            GMST anomaly ensemble, dims (year, member)

        Examples
        --------
        >>> cps = Neukom2019.from_method('CPS')
        >>> gmst = cps.compute_gmst()
        >>> gmst.mean('member').plot()
        """
        weights = np.cos(np.deg2rad(self.data.lat))
        gmst = self.data.weighted(weights).mean(dim=['lat', 'lon'])

        ref_slice = gmst.sel(year=slice(*ref_period))
        if len(ref_slice.year) == 0:
            raise ValueError(f"No data in reference period {ref_period}")
        ref_mean = ref_slice.mean('year')

        gmst = (gmst - ref_mean).load()
        gmst.attrs['long_name'] = f'Global Mean Surface Temperature anomaly ({self.method})'
        gmst.attrs['units'] = f'°C, ref. {ref_period[0]}-{ref_period[1]}'
        return gmst

    def to_grid_anomaly(self, ref_period: tuple = (1961, 1990)) -> xr.DataArray:
        """
        Re-express the gridded field as a per-gridpoint anomaly relative to
        ref_period, without any spatial reduction.

        Use this (rather than compute_gmst) when a spatial operation such as
        nearest-neighbour site extraction needs to happen before averaging.
        As in compute_gmst, subtracting the reference-period mean at each
        gridpoint also neutralizes the DA method's absolute-Kelvin units.

        The result is left dask-backed (not .load()ed): computing the
        reference mean only touches the chunks spanning ref_period, and the
        anomaly subtraction stays lazy, so a caller that subsets to a few
        sites (e.g. .sel(lat=..., lon=..., method='nearest')) before calling
        .load() only ever materializes those few grid columns rather than
        the full 1-3 GB field.

        Parameters
        ----------
        ref_period : tuple of (int, int), optional
            Reference period (inclusive) for the anomaly baseline.
            Default (1961, 1990).

        Returns
        -------
        xr.DataArray
            Anomaly field, dims (year, member, lat, lon), dask-backed

        Examples
        --------
        >>> da_recon = Neukom2019.from_method('DA')
        >>> anom = da_recon.to_grid_anomaly()
        >>> site = anom.sel(lat=-79, lon=248, method='nearest').load()
        """
        ref_slice = self.data.sel(year=slice(*ref_period))
        if len(ref_slice.year) == 0:
            raise ValueError(f"No data in reference period {ref_period}")
        ref_mean = ref_slice.mean('year')

        anom = self.data - ref_mean
        anom.attrs['long_name'] = f'Temperature anomaly ({self.method})'
        anom.attrs['units'] = f'°C, ref. {ref_period[0]}-{ref_period[1]}'
        return anom

    def __repr__(self):
        """String representation of Neukom2019 object."""
        y0, y1 = int(self.data.year.min()), int(self.data.year.max())
        return (f"Neukom2019(method='{self.method}', shape={self.data.shape}, "
                f"time_range={y0}-{y1})")
