#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities for loading and stitching PMIP4/CMIP6-historical simulation output.

Created on Mon Jul 21 13:04:14 2025
@author: Julien Emile-Geay, Albus Shang with some help from Claude AI
"""

from pathlib import Path

import xarray as xr


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

    stitched = xr.concat([paleo_da, hist_da], dim='year')
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
