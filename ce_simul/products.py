#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level API for producing corrected PMIP4 annual tas fields.

This module is the single source of truth for "load past1000/past2k,
stitch with CMIP6 historical where the run doesn't reach the present, and
apply the pre-1850 stitch-bias correction" — the pipeline previously
duplicated across BTP_PMIP4_stacks.py, Antarctic_BH_surface_temp.py, and
PMIP4_GMST_sanity_check.py in the ce_val repository. Other projects should
call load_corrected_tas() rather than re-deriving this logic.

Created on Wed Jul 1 2026
@author: Julien Emile-Geay
"""

import numpy as np
import xarray as xr

from .data_config import DataConfig
from .pmip4_config import (
    get_pmip4_models, get_cmip_hist_models,
    MODEL_VAR, YEAR_OFFSETS, TRIM_END, STITCH_THRESHOLD,
)
from .io import open_nc_or_zarr, stitch_annual_tas
from .tas_1850_adjustments import apply_1850_adjustment

CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def load_stitched_tas(model_name: str, config: DataConfig = None) -> xr.DataArray:
    """Load annual-mean tas for one PMIP4 model, stitched with CMIP6 historical.

    Applies the model's year-axis offset and end-of-run trim (see
    pmip4_config.YEAR_OFFSETS / TRIM_END), then stitches in the CMIP6
    historical run when the paleo run ends before pmip4_config.STITCH_THRESHOLD
    and a historical run is registered for that model.

    No pre-1850 bias correction is applied — see load_corrected_tas() for
    the full pipeline including that step.

    Parameters
    ----------
    model_name : str
        PMIP4 model name, as registered under the ``pmip4_<model_name>`` key
        in data_config.yml.
    config : DataConfig, optional
        Defaults to a fresh ``DataConfig()`` if not given.

    Returns
    -------
    xr.DataArray
        Annual-mean tas with a ``year`` coordinate in CE years, plus the
        original spatial dims (lat, lon). ``attrs['stitched']`` records
        whether a CMIP6 historical run was actually stitched in.
    """
    if config is None:
        config = DataConfig()

    pmip4_models = get_pmip4_models(config)
    cmip_hist_models = get_cmip_hist_models(config)

    if model_name not in pmip4_models:
        raise KeyError(f"Unknown PMIP4 model '{model_name}'; registered: {list(pmip4_models)}")

    ds_paleo = open_nc_or_zarr(pmip4_models[model_name], decode_times=CFTIME_CODER)
    var = MODEL_VAR.get(model_name, 'tas')
    tas_annual = ds_paleo[var].groupby('time.year').mean('time')

    offset = YEAR_OFFSETS.get(model_name, 0)
    if offset:
        tas_annual = tas_annual.assign_coords(year=tas_annual.year - offset)
    n_trim = TRIM_END.get(model_name, 0)
    if n_trim:
        tas_annual = tas_annual.isel(year=slice(None, -n_trim))
    ds_paleo.close()

    pmip4_end_year = int(tas_annual.year.values[-1])
    hist_path = cmip_hist_models.get(model_name)
    stitched = hist_path is not None and pmip4_end_year < STITCH_THRESHOLD
    if stitched:
        ds_hist = open_nc_or_zarr(hist_path, decode_times=CFTIME_CODER)
        tas_hist = ds_hist['tas'].groupby('time.year').mean('time')
        ds_hist.close()
        tas_annual = stitch_annual_tas(tas_annual, tas_hist, model_name=model_name)

    tas_annual.attrs['stitched'] = stitched
    return tas_annual


def load_corrected_tas(model_name: str, config: DataConfig = None) -> xr.DataArray:
    """Load, stitch, and pre-1850 bias-correct annual tas for one PMIP4 model.

    This is the recommended entry point for downstream consumers (e.g.
    borehole forward modelling, warmth-and-trends analysis): it returns the
    final, corrected annual tas field ready for use.

    Parameters
    ----------
    model_name : str
        PMIP4 model name, as registered under the ``pmip4_<model_name>`` key
        in data_config.yml.
    config : DataConfig, optional
        Defaults to a fresh ``DataConfig()`` if not given.

    Returns
    -------
    xr.DataArray
        Annual-mean tas, stitched and bias-corrected, with a ``year``
        coordinate in CE years.
    """
    tas_annual = load_stitched_tas(model_name, config=config)
    return apply_1850_adjustment(tas_annual, model_name)


def compute_gmst(tas_annual: xr.DataArray, ref_period: tuple = (1951, 1980)) -> xr.DataArray:
    """Latitude-weighted global mean, expressed as an anomaly relative to ref_period.

    Parameters
    ----------
    tas_annual : xr.DataArray
        Annual-mean tas with ``year``, ``lat``, ``lon`` dims.
    ref_period : tuple of (int, int), optional
        Reference period (inclusive) for the anomaly baseline. Default (1951, 1980).

    Returns
    -------
    xr.DataArray
        1-D GMST anomaly series indexed by ``year``.
    """
    weights = np.cos(np.deg2rad(tas_annual.lat))
    gmst = tas_annual.weighted(weights).mean(dim=['lat', 'lon'])

    ref_slice = gmst.sel(year=slice(*ref_period))
    if len(ref_slice.year) == 0:
        raise ValueError(f"No data in reference period {ref_period}")
    ref_mean = ref_slice.mean('year')

    return (gmst - ref_mean).load()
