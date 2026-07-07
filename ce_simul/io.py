#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities for loading and stitching PMIP4/CMIP6-historical simulation output.

Created on Mon Jul 21 13:04:14 2025
@author: Julien Emile-Geay, Albus Shang with some help from Claude AI
"""

from pathlib import Path

import numpy as np
import xarray as xr


def _cell_bounds(coord: xr.DataArray) -> np.ndarray:
    """Infer 1D cell-edge bounds for a monotonic, roughly-regular coordinate.

    Midpoints between consecutive centers, with the first/last edges
    extrapolated symmetrically. Used to build the lat_b/lon_b arrays xESMF
    needs for conservative regridding, since raw PMIP4 paleo output
    typically carries no bounds variables of its own.
    """
    c = coord.values
    mid = (c[:-1] + c[1:]) / 2
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def regrid_conservative(
    source_da: xr.DataArray,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.DataArray:
    """Conservatively regrid ``source_da`` onto a (target_lat, target_lon) grid.

    Uses xESMF (install via the ``regrid`` extra: ``pip install
    ce-simul[regrid]``, or ``conda install -c conda-forge xesmf esmpy``);
    cell bounds are inferred on the fly via ``_cell_bounds`` since these are
    simple rectilinear atmosphere grids without their own bounds variables.
    Assumes a periodic (global, wrap-around) longitude axis, true of every
    PMIP4/CMIP6 model grid handled here.
    """
    import xesmf as xe

    # Clip to [-90, 90]: these are pole-inclusive FV grids (a grid point sits
    # at exactly +/-90 deg), so the unclipped midpoint-extrapolated polar
    # edge would otherwise land just outside the physical latitude domain.
    source_grid = xr.Dataset({
        'lat': source_da.lat, 'lon': source_da.lon,
        'lat_b': ('lat_b', np.clip(_cell_bounds(source_da.lat), -90, 90)),
        'lon_b': ('lon_b', _cell_bounds(source_da.lon)),
    })
    target_grid = xr.Dataset({
        'lat': target_lat, 'lon': target_lon,
        'lat_b': ('lat_b', np.clip(_cell_bounds(target_lat), -90, 90)),
        'lon_b': ('lon_b', _cell_bounds(target_lon)),
    })
    regridder = xe.Regridder(source_grid, target_grid, method='conservative', periodic=True)
    return regridder(source_da, keep_attrs=True)


def stitch_annual_tas(
    paleo_da: xr.DataArray,
    hist_da: xr.DataArray,
    model_name: str = '',
) -> xr.DataArray:
    """
    Verify temporal continuity and concatenate annual tas DataArrays.

    Both inputs must have an integer ``year`` coordinate in CE years (i.e.
    any model-year offset must already have been applied to ``paleo_da``
    before calling this function).

    Rules
    -----
    * Perfect stitch (paleo ends Y, hist starts Y+1): concatenate directly.
    * One-year overlap (paleo ends Y, hist starts Y): drop year Y from hist
      and concatenate. This is the known edge-case for models whose past1000
      simulation ends on 1850 instead of 1849.
    * Gap or multi-year overlap: raise ``ValueError``.
    * Grid mismatch (e.g. CESM2's past1000 was never CMORized and stayed on
      its coarser native f19_g17 grid, while its CMIP6 historical run is on
      the finer 0.9°x1.25° 'gn' grid): conservatively regrid the finer of
      the two onto the coarser one before concatenating.

    Parameters
    ----------
    paleo_da : xr.DataArray
        Annual-mean tas from the paleo run (past1000 / past2k) with ``year``
        as the sole dimension. Values are in CE years.
    hist_da : xr.DataArray
        Annual-mean tas from the historical run (1850–2014) with ``year``
        as the sole dimension.
    model_name : str, optional
        Used only in diagnostic messages.

    Returns
    -------
    xr.DataArray
        Concatenated annual tas spanning the full stitched period.

    Raises
    ------
    ValueError
        If there is a temporal gap between the two simulations, or an overlap
        of more than one year.
    """
    paleo_end  = int(paleo_da.year.values[-1])
    hist_start = int(hist_da.year.values[0])
    hist_end   = int(hist_da.year.values[-1])
    label = f"[{model_name}] " if model_name else ""

    if hist_start == paleo_end + 1:
        # Perfect stitch: no action needed
        pass
    elif hist_start == paleo_end:
        # One-year overlap: drop that year from hist to avoid duplication
        print(
            f"{label}1-year overlap at {hist_start} "
            f"(paleo ends {paleo_end}, hist starts {hist_start}). "
            f"Dropping year {hist_start} from historical run."
        )
        hist_da = hist_da.sel(year=hist_da.year > hist_start)
    elif hist_start > paleo_end + 1:
        gap = hist_start - paleo_end - 1
        raise ValueError(
            f"{label}Temporal gap of {gap} year(s) between paleo end "
            f"({paleo_end}) and historical start ({hist_start}). Cannot stitch."
        )
    else:
        overlap = paleo_end - hist_start + 1
        raise ValueError(
            f"{label}Overlap of {overlap} years between paleo (ends {paleo_end}) "
            f"and historical (starts {hist_start}). Cannot stitch automatically."
        )

    if paleo_da.lat.size != hist_da.lat.size or paleo_da.lon.size != hist_da.lon.size:
        # Different native grids (not just floating-point coordinate noise) --
        # regrid the finer one down onto the coarser one so they can be
        # concatenated.
        paleo_cells = paleo_da.lat.size * paleo_da.lon.size
        hist_cells = hist_da.lat.size * hist_da.lon.size
        if hist_cells > paleo_cells:
            print(f"{label}Grid mismatch: hist "
                  f"({hist_da.lat.size}x{hist_da.lon.size}) is finer than paleo "
                  f"({paleo_da.lat.size}x{paleo_da.lon.size}). Regridding hist onto paleo's grid.")
            hist_da = regrid_conservative(hist_da, paleo_da.lat, paleo_da.lon)
        else:
            print(f"{label}Grid mismatch: paleo "
                  f"({paleo_da.lat.size}x{paleo_da.lon.size}) is finer than hist "
                  f"({hist_da.lat.size}x{hist_da.lon.size}). Regridding paleo onto hist's grid.")
            paleo_da = regrid_conservative(paleo_da, hist_da.lat, hist_da.lon)

    # join='override': paleo and historical are the same model's grid, but
    # each file's lat/lon labels can carry negligible floating-point
    # differences depending on how that run was archived. 'override' keeps
    # paleo_da's coordinate labels as-is instead of unioning near-duplicate
    # values (join='outer') or hard-failing on them (the coming join='exact'
    # default).
    stitched = xr.concat([paleo_da, hist_da], dim='year', join='override')
    print(
        f"{label}Stitched: CE {int(stitched.year.values[0])}–"
        f"{int(stitched.year.values[-1])} ({len(stitched.year)} years)"
    )
    return stitched


def open_nc_or_zarr(path, **kwargs):
    """Open a climate dataset from a zarr store, single NC file, or multi-file NC directory.

    Parameters
    ----------
    path : str or Path
        Path to a zarr store (directory without .nc files), a single .nc file,
        or a directory containing multiple .nc files (opened with open_mfdataset).
    **kwargs
        Passed through to the underlying xarray open call (e.g. decode_times).

    Returns
    -------
    xr.Dataset
    """
    p = Path(path)
    if p.suffix == '.nc':
        return xr.open_dataset(p, **kwargs)
    elif p.is_dir() and any(p.glob('*.nc')):
        return xr.open_mfdataset(
            sorted(p.glob('*.nc')),
            combine='by_coords',
            data_vars='minimal',
            coords='minimal',
            compat='override',
            **kwargs,
        )
    else:
        return xr.open_zarr(p, **kwargs)
