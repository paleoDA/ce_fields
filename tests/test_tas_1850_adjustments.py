"""
Tests for ce_simul.tas_1850_adjustments.apply_1850_adjustment.

Uses the real results/tables/PMIP4_GMST_anomalies.csv (checked into the
repo) so the derived corrections can be checked against the values in
notebooks/PMIP4_1850_stitch.ipynb — no HPC data needed.

Run from the repository root:
    pytest tests/test_tas_1850_adjustments.py
"""
import numpy as np
import pytest
import xarray as xr

from ce_simul.tas_1850_adjustments import (
    apply_1850_adjustment,
    compute_cesm2_trend,
    compute_flat_offsets,
    CESM2_MODEL,
    FLAT_OFFSET_MODELS,
    _load_gmst,
)


def _make_field(years, val=0.0):
    """Minimal annual tas DataArray with lat/lon dims, constant in space."""
    data = np.full((len(years), 2, 2), val, dtype=float)
    return xr.DataArray(
        data,
        dims=['year', 'lat', 'lon'],
        coords={'year': years, 'lat': [0.0, 45.0], 'lon': [0.0, 90.0]},
    )


@pytest.fixture(scope='module')
def gmst():
    return _load_gmst()


def test_flat_offset_models_have_offsets(gmst):
    offsets = compute_flat_offsets(gmst)
    assert set(offsets) == set(FLAT_OFFSET_MODELS)
    assert all(np.isfinite(v) for v in offsets.values())


def test_cesm2_trend_is_warming(gmst):
    slope, intercept = compute_cesm2_trend(gmst)
    # CESM2's past1000.002 drift is a warming artifact (see notebook).
    assert slope > 0


def test_flat_offset_applied_only_pre_1850(gmst):
    years = np.arange(1800, 1901)
    field = _make_field(years, val=0.0)
    name = FLAT_OFFSET_MODELS[0]
    offset = compute_flat_offsets(gmst, [name])[name]

    corrected = apply_1850_adjustment(field, name, gmst=gmst)

    pre  = corrected.sel(year=slice(1800, 1849)).values
    post = corrected.sel(year=slice(1850, 1900)).values
    assert np.allclose(pre, offset)
    assert np.allclose(post, 0.0)


def test_cesm2_detrend_applied_only_pre_1850(gmst):
    years = np.arange(1800, 1901)
    field = _make_field(years, val=0.0)
    slope, intercept = compute_cesm2_trend(gmst)

    corrected = apply_1850_adjustment(field, CESM2_MODEL, gmst=gmst)

    pre_years = np.arange(1800, 1850)
    expected_pre = -(slope * pre_years + intercept)
    pre  = corrected.sel(year=slice(1800, 1849)).values
    post = corrected.sel(year=slice(1850, 1900)).values

    assert np.allclose(pre[:, 0, 0], expected_pre)
    assert np.allclose(post, 0.0)


def test_unregistered_model_returned_unchanged(gmst):
    years = np.arange(1800, 1901)
    field = _make_field(years, val=1.23)
    corrected = apply_1850_adjustment(field, 'iCESM', gmst=gmst)
    assert corrected.identical(field)


def test_correction_uniform_across_grid(gmst):
    """The field-level correction must be spatially uniform (function of year only)."""
    years = np.arange(1800, 1860)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(years), 3, 4))
    field = xr.DataArray(
        data, dims=['year', 'lat', 'lon'],
        coords={'year': years, 'lat': np.arange(3.0), 'lon': np.arange(4.0)},
    )
    corrected = apply_1850_adjustment(field, CESM2_MODEL, gmst=gmst)
    delta = (corrected - field).sel(year=slice(1800, 1849))
    # Same additive shift at every grid point for a given year
    assert np.allclose(delta.std(dim=['lat', 'lon']).values, 0.0, atol=1e-10)
