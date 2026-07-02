"""Tests for ce_simul.io.stitch_annual_tas"""
import numpy as np
import pytest
import xarray as xr

from ce_simul.io import stitch_annual_tas


def _make_da(years, val=0.0):
    """Minimal annual tas DataArray with lat/lon dims for weighted-mean compat."""
    data = np.full((len(years), 2, 2), val, dtype=float)
    return xr.DataArray(
        data,
        dims=['year', 'lat', 'lon'],
        coords={'year': years, 'lat': [0.0, 45.0], 'lon': [0.0, 90.0]},
    )


def test_clean_stitch():
    """paleo ends 1849, hist starts 1850 → seamless concatenation."""
    paleo = _make_da(np.arange(850, 1850))
    hist  = _make_da(np.arange(1850, 2015))
    result = stitch_annual_tas(paleo, hist)
    assert int(result.year.values[0])  == 850
    assert int(result.year.values[-1]) == 2014
    assert len(result.year) == 1165   # 850–2014 inclusive


def test_one_year_overlap():
    """paleo ends 1850, hist starts 1850 → overlap year dropped from hist."""
    paleo = _make_da(np.arange(850, 1851))    # ends 1850
    hist  = _make_da(np.arange(1850, 2015))   # starts 1850
    result = stitch_annual_tas(paleo, hist, model_name='TestModel')
    assert int(result.year.values[0])  == 850
    assert int(result.year.values[-1]) == 2014
    # Year 1850 appears exactly once
    assert np.sum(result.year.values == 1850) == 1


def test_gap_raises():
    """Gap of 2 years between paleo and hist raises ValueError."""
    paleo = _make_da(np.arange(850, 1849))    # ends 1848
    hist  = _make_da(np.arange(1851, 2015))   # starts 1851
    with pytest.raises(ValueError, match="gap"):
        stitch_annual_tas(paleo, hist)


def test_multi_year_overlap_raises():
    """Multi-year overlap raises ValueError."""
    paleo = _make_da(np.arange(850, 1855))    # ends 1854
    hist  = _make_da(np.arange(1850, 2015))   # starts 1850
    with pytest.raises(ValueError, match="Overlap"):
        stitch_annual_tas(paleo, hist)


def test_values_preserved():
    """Values from each segment are preserved after stitching."""
    paleo = _make_da(np.arange(850, 1850), val=1.0)
    hist  = _make_da(np.arange(1850, 2015), val=2.0)
    result = stitch_annual_tas(paleo, hist)
    paleo_vals = result.sel(year=slice(850, 1849)).values
    hist_vals  = result.sel(year=slice(1850, 2014)).values
    assert np.allclose(paleo_vals, 1.0)
    assert np.allclose(hist_vals,  2.0)
