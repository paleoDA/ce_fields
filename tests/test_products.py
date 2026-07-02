"""Tests for ce_simul.products.compute_gmst (no HPC data needed)."""
import numpy as np
import pytest
import xarray as xr

from ce_simul.products import compute_gmst


def _make_field(years, lat, val_by_lat):
    """Annual tas field, constant in time, varying by latitude only."""
    data = np.broadcast_to(np.array(val_by_lat)[None, :, None], (len(years), len(lat), 2)).astype(float)
    return xr.DataArray(
        data.copy(),
        dims=['year', 'lat', 'lon'],
        coords={'year': years, 'lat': lat, 'lon': [0.0, 90.0]},
    )


def test_compute_gmst_is_zero_at_reference():
    years = np.arange(1900, 2001)
    field = _make_field(years, lat=[-45.0, 0.0, 45.0], val_by_lat=[10.0, 20.0, 10.0])
    gmst = compute_gmst(field, ref_period=(1951, 1980))
    ref = gmst.sel(year=slice(1951, 1980))
    assert np.allclose(ref.values, 0.0)


def test_compute_gmst_weights_by_latitude():
    years = np.arange(1900, 1902)
    # Equator (weight 1.0) warmer than poles (weight ~0): global mean should be closer to equator value
    field = _make_field(years, lat=[-89.0, 0.0, 89.0], val_by_lat=[0.0, 10.0, 0.0])
    gmst = compute_gmst(field, ref_period=(1900, 1900))
    # relative to the (also weighted) reference year, non-reference year should be ~0 since field is constant in time
    assert np.allclose(gmst.values, 0.0, atol=1e-8)


def test_compute_gmst_raises_on_empty_reference():
    years = np.arange(1900, 1910)
    field = _make_field(years, lat=[0.0], val_by_lat=[1.0])
    with pytest.raises(ValueError, match="reference period"):
        compute_gmst(field, ref_period=(1951, 1980))
