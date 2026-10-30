#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Millennium Reanalysis (LMR) data loading and management

Created on Dec 4 2025
@author: Julien Emile-Geay with Claude AI assistance
"""

import xarray as xr
import numpy as np
import fsspec
from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Optional, Dict
from ce_fields.data_config import DataConfig


def load_available_vars(filepath: str,
                       target_vars: List[str],
                       return_first_match: bool = False,
                       show_available: bool = True,
                       decode_times = False,
                       **open_kwargs) -> Union[xr.Dataset, xr.DataArray, None]:
    """
    Load netCDF file and return only the variables that exist from target list.

    Parameters
    ----------
    filepath : str
        Path to netCDF file
    target_vars : List[str]
        List of variable names to search for
    return_first_match : bool, optional
        If True, return first matching variable as DataArray. Default False.
    show_available : bool, optional
        If True, print available variables when none match. Default True.
    decode_times : bool, optional
        Whether to decode time coordinates. Default False.
    **open_kwargs
        Additional keyword arguments passed to xr.open_dataset

    Returns
    -------
    Union[xr.Dataset, xr.DataArray, None]
        Dataset with matched variables, single DataArray if return_first_match=True,
        or None if no matches found

    Examples
    --------
    >>> target_vars = ['air', 'tas', 'sat']
    >>> data = load_available_vars('lmr_file.nc', target_vars, return_first_match=True)
    """
    try:
        # Try different engines if not specified
        engine = open_kwargs.pop('engine', None)
        if engine is None:
            # Try common engines in order of preference
            for eng in ['netcdf4', 'h5netcdf', 'scipy']:
                try:
                    with xr.open_dataset(filepath, engine=eng, decode_times=decode_times, **open_kwargs) as ds:
                        available_vars = list(ds.data_vars.keys())
                    engine = eng
                    break
                except (ImportError, ValueError):
                    continue
            if engine is None:
                raise ImportError("Could not find a suitable engine for reading netCDF files")
        else:
            # User specified engine
            with xr.open_dataset(filepath, engine=engine, decode_times=decode_times, **open_kwargs) as ds:
                available_vars = list(ds.data_vars.keys())

        # Find intersection of target and available variables
        found_vars = [var for var in target_vars if var in available_vars]

        if not found_vars:
            if show_available:
                print(f"None of the target variables {target_vars} found in file.")
                print(f"Available variables: {available_vars}")
            return None

        # Load only the found variables with decode_times parameter
        ds = xr.open_dataset(filepath, engine=engine, decode_times=decode_times, **open_kwargs)

        if return_first_match:
            first_var = found_vars[0]
            print(f"Loading variable: {first_var}")
            return ds[first_var]
        else:
            ds_subset = ds[found_vars]
            print(f"Loading variables: {found_vars}")
            return ds_subset

    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None


class LMR:
    """
    Last Millennium Reanalysis (LMR) gridded climate data handler.

    This class provides convenient loading and management of LMR datasets,
    returning xarray DataArrays that can be used with existing methods like
    DocuClimSeries.load_nearest_xarray() and HCLIMSeries.load_nearest_xarray().

    Attributes
    ----------
    version : str
        LMR version ('v2.1', 'seasonal', 'online')
    variable : str
        Climate variable loaded (e.g., 'tas', 'air')
    data : xr.DataArray
        The gridded climate data

    Examples
    --------
    Load LMR seasonal data and extract nearest gridpoint:

    >>> lmr = LMR.from_version(version='seasonal')
    >>> lmr_kyoto = sakura.load_nearest_xarray(lmr.data, annualized=True)

    Load all three LMR versions:

    >>> lmr_products = LMR.load_all_versions()
    >>> for version, lmr in lmr_products.items():
    ...     print(f"{version}: {lmr.data.shape}")

    See Also
    --------
    DocuClimSeries.load_nearest_xarray : Extract nearest gridpoint from LMR data
    HCLIMSeries.load_nearest_xarray : Extract nearest gridpoint from LMR data
    """

    # Standard variable names to search for in gridded files (in order of preference)
    TEMPERATURE_VARS = ['tas', 'air', 'air_mean', 'sat', 'tas_sfc']

    # Variable names in *_indices.nc files (LMR online ensemble members)
    INDICES_VARS = ['glob_mean', 'nh', 'sh', 'europe']

    # Config key prefix per version (append _tas_mean / _tas_ens / _gmst to get full key)
    # Ordered by publication date: v2.1 (2019), online (2021), seasonal (2025)
    VERSION_MAP = {
        'v2.1': 'lmr_v2_1',
        'online': 'lmr_online',
        'seasonal': 'lmr_seasonal',
    }

    def __init__(self, data: xr.DataArray, version: str, variable: str):
        """
        Initialize LMR object.

        Parameters
        ----------
        data : xr.DataArray
            The gridded climate data
        version : str
            LMR version ('v2.1', 'seasonal', 'online')
        variable : str
            Climate variable name
        """
        self.data = data
        self.version = version
        self.variable = variable

    @classmethod
    def from_version(cls, version: str = 'seasonal',
                    variable: str = 'tas',
                    annualize: bool = True,
                    time_slice: Optional[tuple] = None,
                    load_members: bool = False,
                    n_members: Optional[int] = None,
                    combine: str = 'nested') -> 'LMR':
        """
        Load LMR data from a specific version.

        Parameters
        ----------
        version : str, optional
            LMR version to load. Options: 'v2.1', 'seasonal', 'online'. Default 'seasonal'.
        variable : str, optional
            Variable to load:
            - 'tas'  : spatial temperature field (ensemble mean, or members when load_members=True)
            - 'gmst' : global mean surface temperature ensemble
            Default 'tas'.
        annualize : bool, optional
            If True, convert sub-annual data to annual means. Ignored for variable='gmst'
            (annualization is handled internally). Default True.
        time_slice : tuple, optional
            Tuple of (start_year, end_year). Either can be None (no bound). Default None.
        load_members : bool, optional
            For variable='tas', version='seasonal': stream all spatial members remotely.
            Ignored for variable='gmst' (gmst files always contain the full ensemble).
            Default False.
        n_members : int, optional
            Number of spatial members to load (seasonal only). Default None = all 800.
        combine : str, optional
            Unused; kept for backward compatibility.

        Returns
        -------
        LMR
            LMR object containing the data

        Raises
        ------
        ValueError
            If version or variable is not recognised, or file not found

        Examples
        --------
        >>> lmr = LMR.from_version('seasonal')                           # tas ensemble mean
        >>> lmr = LMR.from_version('seasonal', variable='gmst')          # GMST ensemble
        >>> lmr = LMR.from_version('seasonal', variable='tas', load_members=True)
        >>> lmr = LMR.from_version('v2.1', variable='gmst', time_slice=(1000, 2000))
        """
        if version not in cls.VERSION_MAP:
            raise ValueError(f"Unknown LMR version: {version}. "
                           f"Must be one of {list(cls.VERSION_MAP.keys())}")
        if variable not in ('tas', 'gmst'):
            raise ValueError(f"variable must be 'tas' or 'gmst', got '{variable}'")

        config = DataConfig()
        prefix = cls.VERSION_MAP[version]   # e.g. 'lmr_v2_1'

        if variable == 'gmst':
            filepath = config.get_file_path(f'{prefix}_gmst')
            print(f"Loading LMR {version} GMST from: {filepath}")
            data = cls._load_gmst(version, filepath)
            actual_var = 'gmst'

        else:  # variable == 'tas'
            if load_members and version == 'seasonal':
                config_key = f'{prefix}_tas_ens'
            elif load_members:
                raise NotImplementedError(
                    f"Spatial ensemble members are not available for version='{version}'. "
                    "Use variable='gmst' to access the full ensemble for this version.")
            else:
                config_key = f'{prefix}_tas_mean'

            filepath = config.get_file_path(config_key)
            print(f"Loading LMR {version} from: {filepath}")

            if load_members and version == 'seasonal':
                data = cls._load_ensemble_members(filepath, combine, annualize, n_members)
            else:
                data = load_available_vars(
                    filepath, cls.TEMPERATURE_VARS,
                    return_first_match=True,
                    decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
                if data is None:
                    raise ValueError(f"Could not load tas data from {filepath}")
                data = cls._process_time_dimension(data, version, annualize)

            actual_var = data.name

        if time_slice is not None:
            data = cls._slice_time(data, time_slice, version)

        return cls(data, version, actual_var)

    @classmethod
    def _load_gmst(cls, version: str, filepath: str) -> xr.DataArray:
        """
        Load pre-computed GMST ensemble for any LMR version.

        Returns DataArray with version-specific dimensions:
        - v2.1    : (time=2001, MCrun=20, members=100), time in integer years CE
        - online  : (mc_iteration=50, ensemble_member=100, year=1001)
        - seasonal: (year, ens_num=800), annualised from quarterly data
        """
        if version == 'v2.1':
            ds = xr.open_dataset(filepath, decode_times=False)
            years = (ds['time'].values / 365).astype(int)
            return ds['gmt'].assign_coords(time=years)
        elif version == 'online':
            return cls._load_online_members(filepath)
        elif version == 'seasonal':
            cftime_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            fp = str(filepath)
            _REMOTE_GMST = 'https://atmos.uw.edu/~zilumeng/LMR_Seasonal/data/index/GMT_ens.nc'
            if not fp.startswith(('http://', 'https://')) and not Path(fp).exists():
                print(f'Local GMT_ens.nc not found at {fp}, falling back to remote.')
                fp = _REMOTE_GMST
            if fp.startswith(('http://', 'https://')):
                print('Streaming LMR Seasonal GMST ensemble from remote server…')
                ds = xr.open_dataset(fsspec.filesystem('http').open(fp),
                                     engine='h5netcdf', decode_times=cftime_coder)
            else:
                ds = xr.open_dataset(fp, decode_times=cftime_coder)
            return ds['GMT'].groupby('time.year').mean('time')

    @classmethod
    def _load_ensemble_members(cls, filepath: str, combine: str, annualize: bool,
                               n_members: Optional[int] = None) -> xr.DataArray:
        """
        Load ensemble members using lazy loading with xr.open_mfdataset().

        For remote HTTPS URLs (not OPeNDAP), uses fsspec + h5netcdf to stream
        files without downloading them. Requires: fsspec, aiohttp, h5netcdf.

        Parameters
        ----------
        filepath : str
            Path pattern to ensemble member files (wildcard or URL template with *)
        combine : str
            Method for combining datasets when loading local files ('by_coords' or 'nested')
        annualize : bool
            Whether to annualize the data
        n_members : int, optional
            Number of members to load. Default None loads all available (100).

        Returns
        -------
        xr.DataArray
            Lazily loaded ensemble data
        """
        try:
            is_url = isinstance(filepath, str) and filepath.startswith(('http://', 'https://'))
            has_wildcard = isinstance(filepath, str) and '*' in filepath

            if is_url and has_wildcard:
                # Expand wildcard URL into individual member URLs using fsspec
                # URL pattern: https://.../tas_ens*.nc  → tas_ens0.nc, tas_ens1.nc, ...
                n = n_members if n_members is not None else 100
                base_url = filepath[:filepath.index('*')]
                suffix = filepath[filepath.index('*') + 1:]
                file_urls = [f"{base_url}{i}{suffix}" for i in range(n)]

                print(f"Streaming {n} members from remote server via fsspec...")
                fs = fsspec.filesystem("http")
                cftime_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                datasets = []
                for url in tqdm(file_urls, desc='Loading remote members'):
                    datasets.append(xr.open_dataset(
                        fs.open(url), engine='h5netcdf',
                        decode_times=cftime_coder, chunks={'time': 100}
                    ))
                ds = xr.concat(datasets, dim='ens_num')
            else:
                # Local wildcard path: let xarray expand it
                cftime_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                ds = xr.open_mfdataset(
                    filepath,
                    combine='nested',
                    decode_times=cftime_coder,
                    parallel=False,
                    concat_dim='member',
                    combine_attrs='override'
                )

            # Extract temperature variable
            for var in cls.TEMPERATURE_VARS:
                if var in ds.data_vars:
                    print(f"Loading variable: {var}")
                    data = ds[var]
                    break
            else:
                raise ValueError(f"None of {cls.TEMPERATURE_VARS} found in ensemble files")

            # Process time dimension
            data = cls._process_time_dimension(data, 'seasonal', annualize)

            n_loaded = data.sizes.get('ens_num', data.sizes.get('member', 'unknown'))
            print(f"Loaded {n_loaded} ensemble members, shape: {data.shape}")
            print(f"Lazily loaded: {data.chunks is not None}")

            return data

        except Exception as e:
            raise ValueError(f"Error loading ensemble members: {e}")

    @classmethod
    def _load_online_members(cls, filepath: str) -> xr.DataArray:
        """
        Load LMR online ensemble members from a *_indices.nc file.

        These files contain regional temperature indices with full MC/ensemble
        dimensions: (mc_iteration: 50, ensemble_member: 100, year: 1001).
        The 'year' coordinate is already numeric (integer years CE).

        Parameters
        ----------
        filepath : str
            Path to the *_indices.nc file (e.g., tas_sfc_indices.nc)

        Returns
        -------
        xr.DataArray
            Dataset variables as DataArray; 'glob_mean' is returned if present,
            otherwise the first available index variable.
        """
        ds = xr.open_dataset(filepath, decode_times=False)

        for var in cls.INDICES_VARS:
            if var in ds.data_vars:
                print(f"Loading variable: {var}")
                data = ds[var]
                n_mc = data.sizes.get('mc_iteration', '?')
                n_ens = data.sizes.get('ensemble_member', '?')
                print(f"Ensemble shape: mc_iteration={n_mc}, ensemble_member={n_ens}")
                return data

        available = list(ds.data_vars)
        raise ValueError(f"None of {cls.INDICES_VARS} found in {filepath}. "
                         f"Available: {available}")

    @classmethod
    def _cftime_to_decimal_year(cls, cftime_obj) -> float:
        """
        Convert cftime object to decimal year.

        Parameters
        ----------
        cftime_obj : cftime datetime object
            The cftime object to convert

        Returns
        -------
        float
            Decimal year (e.g., 2000.25 for April 1, 2000)
        """
        if not hasattr(cftime_obj, 'year'):
            # Already numeric
            return float(cftime_obj)

        year = cftime_obj.year

        # Get day of year (1-365/366)
        dayofyr = cftime_obj.dayofyr

        # Determine if leap year by checking if Dec 31 exists
        # For cftime, we can check the calendar type
        calendar = getattr(cftime_obj, 'calendar', 'standard')

        # Calculate days in year based on calendar
        if calendar in ['365_day', 'noleap']:
            days_in_year = 365
        elif calendar == '360_day':
            days_in_year = 360
        else:  # standard, gregorian, proleptic_gregorian
            # Check if leap year
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            days_in_year = 366 if is_leap else 365

        # Convert to decimal year
        decimal_year = year + (dayofyr - 1) / days_in_year

        return decimal_year

    @classmethod
    def _process_time_dimension(cls, data: xr.DataArray, version: str,
                                annualize: bool) -> xr.DataArray:
        """
        Process time dimension: handle cftime dates and annual averaging.

        Following the approach in notebooks/BTP_LMR_stacks.ipynb:
        - LMR_online: time values are already numeric years
        - Other versions: time values are cftime objects, convert to decimal years

        IMPORTANT: Always returns time coordinate as numeric decimal years (float)
        for compatibility with pyleoclim GeoSeries. Sub-annual data will have
        fractional years (e.g., 2000.25, 2000.50, etc.).

        Parameters
        ----------
        data : xr.DataArray
            Raw LMR data
        version : str
            LMR version
        annualize : bool
            Whether to convert to annual means

        Returns
        -------
        xr.DataArray
            Data with processed time dimension (always numeric decimal years)
        """
        # Get time coordinate name
        time_dim = 'time' if 'time' in data.dims else 'year'

        # Extract date values
        date = data[time_dim].values

        # For online version, time is already in numeric years
        if version == 'online':
            decimal_years = date
        else:
            # For v2.1 and seasonal: convert cftime objects to decimal years
            # This preserves sub-annual information
            vectorized_converter = np.vectorize(cls._cftime_to_decimal_year)
            decimal_years = vectorized_converter(date)

        # Check if sub-annual and annualize if requested
        if annualize and len(np.unique(np.round(decimal_years))) != len(decimal_years):
            print("Converting to annual means...")
            data = data.groupby('time.year').mean()
            # After groupby, time dimension is 'year' with numeric values
            return data

        # Replace time coordinate with decimal years
        # This ensures compatibility with pyleoclim GeoSeries
        if time_dim == 'time':
            data = data.assign_coords({time_dim: decimal_years.astype(float)})

        return data

    @classmethod
    def _slice_time(cls, data: xr.DataArray, time_slice: tuple,
                   version: str) -> xr.DataArray:
        """
        Slice data to specific time range.

        Note: After _process_time_dimension, all time coordinates are numeric years,
        so we use simple numeric comparisons for all versions.

        Parameters
        ----------
        data : xr.DataArray
            LMR data
        time_slice : tuple
            (start_year, end_year) - either can be None to indicate no bound
        version : str
            LMR version

        Returns
        -------
        xr.DataArray
            Time-sliced data
        """
        start_year, end_year = time_slice

        # After _process_time_dimension, all time coordinates are numeric years
        # So we can use simple numeric comparisons for all versions
        if 'year' in data.dims:
            # Seasonal after annualization has 'year' dimension
            if start_year is not None and end_year is not None:
                data = data.sel(year=(data.year >= start_year) & (data.year <= end_year))
            elif start_year is not None:
                data = data.sel(year=(data.year >= start_year))
            elif end_year is not None:
                data = data.sel(year=(data.year <= end_year))
        else:
            # All versions now have numeric 'time' coordinate
            if start_year is not None and end_year is not None:
                data = data.sel(time=(data.time >= start_year) & (data.time <= end_year))
            elif start_year is not None:
                data = data.sel(time=(data.time >= start_year))
            elif end_year is not None:
                data = data.sel(time=(data.time <= end_year))

        # Format output message
        if start_year is not None and end_year is not None:
            print(f"Time range: {start_year} - {end_year} CE")
        elif start_year is not None:
            print(f"Time range: {start_year} - end CE")
        elif end_year is not None:
            print(f"Time range: start - {end_year} CE")

        return data

    @classmethod
    def load_all_versions(cls, annualize: bool = True,
                         time_slice: Optional[tuple] = None) -> Dict[str, 'LMR']:
        """
        Load all available LMR versions.

        Parameters
        ----------
        annualize : bool, optional
            If True, convert sub-annual data to annual means. Default True.
        time_slice : tuple, optional
            Tuple of (start_year, end_year) to slice time dimension. Default None.

        Returns
        -------
        dict
            Dictionary mapping version names to LMR objects

        Examples
        --------
        >>> lmr_products = LMR.load_all_versions(time_slice=(850, 2000))
        >>> for version, lmr in lmr_products.items():
        ...     print(f"{version}: {lmr.data.shape}")
        """
        results = {}
        for version in cls.VERSION_MAP.keys():
            try:
                lmr = cls.from_version(version, annualize=annualize,
                                      time_slice=time_slice)
                results[version] = lmr
                print(f"Successfully loaded LMR {version}: {lmr.data.shape}\n")
            except Exception as e:
                print(f"Warning: Could not load LMR {version}: {e}\n")
                continue

        return results

    @property
    def years(self) -> np.ndarray:
        """
        Get years array from the time dimension.

        Returns
        -------
        np.ndarray
            Array of years (numeric)
        """
        time_dim = 'year' if 'year' in self.data.dims else 'time'
        return self.data[time_dim].values

    @property
    def time_range(self) -> tuple:
        """
        Get the time range (min, max) of the data.

        Returns
        -------
        tuple
            (min_year, max_year)
        """
        years = self.years
        return (int(years.min()), int(years.max()))

    def compute_global_mean(self) -> xr.DataArray:
        """
        Compute area-weighted global mean surface temperature.

        Returns
        -------
        xr.DataArray
            Global mean time series

        Examples
        --------
        >>> lmr = LMR.from_version('seasonal')
        >>> gmst = lmr.compute_global_mean()
        >>> gmst.plot()
        """
        # Create latitude weights (cosine of latitude)
        weights = np.cos(np.deg2rad(self.data.lat))
        weights.name = "weights"

        # Compute weighted mean
        gmst = self.data.weighted(weights.fillna(0)).mean(["lat", "lon"])
        gmst.attrs['long_name'] = f'Global Mean Surface Temperature ({self.version})'
        gmst.attrs['units'] = '°C'

        return gmst

    def ens_mean(self) -> xr.DataArray:
        """
        Compute ensemble mean across all members.

        Only applicable for seasonal_members version with member dimension.

        Returns
        -------
        xr.DataArray
            Ensemble mean field

        Raises
        ------
        ValueError
            If data does not have a 'member' dimension

        Examples
        --------
        >>> lmr = LMR.from_version('seasonal_members', load_members=True)
        >>> ensemble_mean = lmr.ens_mean()
        """
        if 'member' not in self.data.dims:
            raise ValueError("Data does not have a 'member' dimension. "
                           "Use from_version('seasonal', variable='tas', load_members=True)")

        return self.data.mean(dim='member')

    def ens_spread(self) -> xr.DataArray:
        """
        Compute ensemble spread (standard deviation) across all members.

        Only applicable when loaded with variable='tas', load_members=True.

        Returns
        -------
        xr.DataArray
            Ensemble spread field

        Raises
        ------
        ValueError
            If data does not have a 'member' dimension

        Examples
        --------
        >>> lmr = LMR.from_version('seasonal', variable='tas', load_members=True)
        >>> ensemble_spread = lmr.ens_spread()
        """
        if 'member' not in self.data.dims:
            raise ValueError("Data does not have a 'member' dimension. "
                           "Use from_version('seasonal', variable='tas', load_members=True)")

        return self.data.std(dim='member')

    def __repr__(self):
        """String representation of LMR object."""
        time_dim = 'year' if 'year' in self.data.dims else 'time'
        time_coord = self.data[time_dim]

        # After _process_time_dimension, all time coordinates are numeric
        min_year = int(time_coord.min().values)
        max_year = int(time_coord.max().values)

        time_range = f"{min_year} - {max_year}"
        return (f"LMR(version='{self.version}', variable='{self.variable}', "
                f"shape={self.data.shape}, time_range={time_range})")
