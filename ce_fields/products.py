#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level API for producing PMIP4 annual tas fields, corrected or raw.

This module is the single source of truth for "load past1000/past2k,
stitch with CMIP6 historical where the run doesn't reach the present" — the
pipeline previously duplicated across BTP_PMIP4_stacks.py,
Antarctic_BH_surface_temp.py, and PMIP4_GMST_sanity_check.py in the former ce_val
repository. Other projects should call load_corrected_tas()/load_stitched_tas()
rather than re-deriving this logic.

The pre-1850 stitch-bias correction (ACCESS-ESM1-5, CESM2 — see
code/PMIP4_1850_stitch.ipynb) is baked into the archived data by
code/apply_1850_adjustments.py, not applied at runtime here. Both loaders
accept `use_corrected` to pick the corrected (default) or raw variant for
those two models, and tag the returned DataArray's attrs with
`corrected`/`display_name` so callers can label plots and tables correctly
without needing to know which models have a registered correction — see
ce_fields.pmip4_config.CORRECTED_MODELS / display_name().

Created on Wed Jul 1 2026
@author: Julien Emile-Geay
"""

import numpy as np
import xarray as xr

from .data_config import DataConfig
from .pmip4_config import (
    get_pmip4_models, get_cmip_hist_models,
    MODEL_VAR, YEAR_OFFSETS, TRIM_END, STITCH_THRESHOLD,
    CORRECTED_MODELS, display_name,
)
from .io import open_nc_or_zarr, stitch_annual_tas

CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def load_stitched_tas(model_name: str, config: DataConfig = None, use_corrected: bool = True) -> xr.DataArray:
    """Load annual-mean tas for one PMIP4 model, stitched with CMIP6 historical.

    Applies the model's year-axis offset and end-of-run trim (see
    pmip4_config.YEAR_OFFSETS / TRIM_END), then stitches in the CMIP6
    historical run when the paleo run ends before pmip4_config.STITCH_THRESHOLD
    and a historical run is registered for that model.

    Parameters
    ----------
    model_name : str
        PMIP4 model name, as registered under the ``pmip4_<model_name>`` key
        in data_config.yml.
    config : DataConfig, optional
        Defaults to a fresh ``DataConfig()`` if not given.
    use_corrected : bool, optional
        For the models in ``pmip4_config.CORRECTED_MODELS`` (currently
        ACCESS-ESM1-5, CESM2), whether to load the bias-corrected past1000
        data (default, ``True``) or the original raw data, registered under
        ``raw_pmip4_<model_name>`` in data_config.yml (``False``). No effect
        for other models, which have no correction registered either way.

    Returns
    -------
    xr.DataArray
        Annual-mean tas with a ``year`` coordinate in CE years, plus the
        original spatial dims (lat, lon).
        ``attrs['stitched']`` records whether a CMIP6 historical run was
        actually stitched in.
        ``attrs['corrected']`` records whether the bias-corrected data was
        used for this model.
        ``attrs['display_name']`` is the label to use in plots/tables (see
        ``pmip4_config.display_name``) — e.g. ``'CESM2c'`` when
        ``model_name='CESM2'`` and ``use_corrected=True``.
    """
    if config is None:
        config = DataConfig()

    pmip4_models = get_pmip4_models(config)
    cmip_hist_models = get_cmip_hist_models(config)

    if model_name not in pmip4_models:
        raise KeyError(f"Unknown PMIP4 model '{model_name}'; registered: {list(pmip4_models)}")

    corrected = use_corrected and model_name in CORRECTED_MODELS
    paleo_path = pmip4_models[model_name] if (use_corrected or model_name not in CORRECTED_MODELS) \
        else config.get_file_path(f'raw_pmip4_{model_name}')

    ds_paleo = open_nc_or_zarr(paleo_path, decode_times=CFTIME_CODER)
    var = MODEL_VAR.get(model_name, 'tas')
    tas_annual = ds_paleo[var].groupby('time.year').mean('time')

    # Some archived past1000 runs (e.g. CESM2 run002) truncate mid-year --
    # groupby('time.year').mean('time') would silently average whatever
    # month(s) happen to be present as if they were a full annual mean (a
    # single Antarctic-summer month standing in for the year average causes
    # a spurious multi-degree warm spike). Drop any year not backed by all
    # 12 months before it reaches offset/trim/stitch.
    n_months = ds_paleo['time'].groupby('time.year').count()
    incomplete = n_months.where(n_months != 12, drop=True)
    if incomplete.size:
        label = f"[{model_name}] " if model_name else ""
        print(
            f"{label}Dropping partial year(s) with <12 months present: "
            + ", ".join(f"{int(y)} ({int(n)} months)"
                        for y, n in zip(incomplete.year.values, incomplete.values))
        )
        tas_annual = tas_annual.sel(year=n_months.where(n_months == 12, drop=True).year)

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
    tas_annual.attrs['corrected'] = corrected
    tas_annual.attrs['display_name'] = display_name(model_name, corrected=corrected)
    # Force eager read: some sources (e.g. CESM1.1's native multi-file CAM
    # output) come back dask-backed with one chunk per year, so a lazy
    # array's .sel() cost scales with n_chunks on every call -- catastrophic
    # for the per-station loop callers run downstream.
    return tas_annual.load()


def load_corrected_tas(model_name: str, config: DataConfig = None, use_corrected: bool = True) -> xr.DataArray:
    """Load, stitch annual tas for one PMIP4 model.

    This is the recommended entry point for downstream consumers (e.g.
    borehole forward modelling, warmth-and-trends analysis): it returns the
    final tas field ready for use.

    The two models that needed a pre-1850 stitch-bias correction
    (ACCESS-ESM1-5, CESM2 — see code/PMIP4_1850_stitch.ipynb) have already
    had that correction applied and saved back to their archived data by
    code/apply_1850_adjustments.py, so no runtime correction step happens
    here: this is currently just an alias for load_stitched_tas(), kept as
    the stable name for callers that don't need to know which models did or
    didn't need a correction. Use ``tas_annual.attrs['display_name']`` when
    labeling models in plots/tables, so corrected series are always shown
    distinctly (e.g. 'CESM2c').

    Parameters
    ----------
    model_name : str
        PMIP4 model name, as registered under the ``pmip4_<model_name>`` key
        in data_config.yml.
    config : DataConfig, optional
        Defaults to a fresh ``DataConfig()`` if not given.
    use_corrected : bool, optional
        See load_stitched_tas — whether to load the bias-corrected data
        (default) or the original raw data for models in
        ``pmip4_config.CORRECTED_MODELS``.

    Returns
    -------
    xr.DataArray
        Annual-mean tas, stitched, with a ``year`` coordinate in CE years.
    """
    return load_stitched_tas(model_name, config=config, use_corrected=use_corrected)


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
