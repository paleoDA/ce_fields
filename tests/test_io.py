"""Tests for ce_fields.io.stitch_annual_tas"""
import sys
import types

import numpy as np
import pytest
import xarray as xr

from ce_fields.io import stitch_annual_tas, regrid_conservative, _cell_bounds


def _make_da(years, val=0.0):
    """Minimal annual tas DataArray with lat/lon dims for weighted-mean compat."""
    data = np.full((len(years), 2, 2), val, dtype=float)
    return xr.DataArray(
        data,
        dims=['year', 'lat', 'lon'],
        coords={'year': years, 'lat': [0.0, 45.0], 'lon': [0.0, 90.0]},
    )


def _make_grid_da(years, lat, lon, val=0.0):
    """Like _make_da but with a caller-chosen lat/lon grid, for regrid tests."""
    data = np.full((len(years), len(lat), len(lon)), val, dtype=float)
    return xr.DataArray(
        data,
        dims=['year', 'lat', 'lon'],
        coords={'year': years, 'lat': lat, 'lon': lon},
    )


class _FakeRegridder:
    """Stands in for xe.Regridder: records the grids it was built with and
    returns data reshaped onto the target grid (values don't need to be
    scientifically correct, just present, since these tests exercise branch
    selection and bounds-building, not regridding accuracy)."""

    def __init__(self, source_grid, target_grid, method, periodic):
        self.source_grid = source_grid
        self.target_grid = target_grid

    def __call__(self, da, keep_attrs=True):
        target_lat, target_lon = self.target_grid.lat, self.target_grid.lon
        data = np.full((len(da.year), target_lat.size, target_lon.size), float(da.mean()))
        return xr.DataArray(
            data, dims=['year', 'lat', 'lon'],
            coords={'year': da.year, 'lat': target_lat, 'lon': target_lon},
        )


@pytest.fixture
def fake_xesmf(monkeypatch):
    """Installs a fake xesmf module in sys.modules capturing the grids each
    Regridder call was built with, and how many times it was called --
    avoids needing the real (heavy, optional) xesmf/esmpy dependency."""
    captured = {'calls': []}

    def _regridder(source_grid, target_grid, method='conservative', periodic=True):
        rg = _FakeRegridder(source_grid, target_grid, method, periodic)
        captured['calls'].append(rg)
        return rg

    fake_module = types.SimpleNamespace(Regridder=_regridder)
    monkeypatch.setitem(sys.modules, 'xesmf', fake_module)
    return captured


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


# --- Grid-mismatch / regrid branch (CESM2c: native f19_g17 paleo vs. CMIP6
# 'gn' historical) --------------------------------------------------------

def test_matching_grids_skip_regrid(fake_xesmf):
    """Same lat/lon shape on both sides: no Regridder should be built at all."""
    lat, lon = [0.0, 45.0], [0.0, 90.0]
    paleo = _make_grid_da(np.arange(850, 1850), lat, lon, val=1.0)
    hist  = _make_grid_da(np.arange(1850, 2015), lat, lon, val=2.0)
    result = stitch_annual_tas(paleo, hist)
    assert fake_xesmf['calls'] == []
    assert int(result.year.values[-1]) == 2014


def test_finer_hist_regridded_onto_coarser_paleo(fake_xesmf):
    """CESM2c-shaped mismatch: hist grid finer than paleo -> hist gets
    regridded down onto paleo's (coarser) grid before concatenating."""
    paleo_lat, paleo_lon = np.linspace(-90, 90, 4), np.linspace(0, 270, 4)
    hist_lat,  hist_lon  = np.linspace(-90, 90, 8), np.linspace(0, 315, 8)
    paleo = _make_grid_da(np.arange(850, 1850), paleo_lat, paleo_lon, val=1.0)
    hist  = _make_grid_da(np.arange(1850, 2015), hist_lat, hist_lon, val=2.0)

    result = stitch_annual_tas(paleo, hist, model_name='CESM2')

    assert len(fake_xesmf['calls']) == 1
    call = fake_xesmf['calls'][0]
    assert call.source_grid.lat.size == 8   # regridding hist (finer)...
    assert call.target_grid.lat.size == 4   # ...onto paleo's grid
    assert result.lat.size == 4 and result.lon.size == 4
    assert int(result.year.values[-1]) == 2014


def test_finer_paleo_regridded_onto_coarser_hist(fake_xesmf):
    """Mirror case: paleo grid finer than hist -> paleo gets regridded onto
    hist's (coarser) grid."""
    paleo_lat, paleo_lon = np.linspace(-90, 90, 8), np.linspace(0, 315, 8)
    hist_lat,  hist_lon  = np.linspace(-90, 90, 4), np.linspace(0, 270, 4)
    paleo = _make_grid_da(np.arange(850, 1850), paleo_lat, paleo_lon, val=1.0)
    hist  = _make_grid_da(np.arange(1850, 2015), hist_lat, hist_lon, val=2.0)

    result = stitch_annual_tas(paleo, hist)

    assert len(fake_xesmf['calls']) == 1
    call = fake_xesmf['calls'][0]
    assert call.source_grid.lat.size == 8   # regridding paleo (finer)...
    assert call.target_grid.lat.size == 4   # ...onto hist's grid
    assert result.lat.size == 4 and result.lon.size == 4


def test_pole_inclusive_bounds_are_clipped(fake_xesmf):
    """CESM2's native f19_g17 grid has a lat point sitting exactly at +/-90.
    _cell_bounds' symmetric-extrapolation would otherwise place the polar
    edge just outside [-90, 90], which xESMF rejects -- this is the fix
    made after the AlignmentError seen against the real HPC files."""
    pole_lat = np.linspace(-90, 90, 6)   # pole-inclusive, like f19_g17
    lon = np.linspace(0, 300, 6)

    # Sanity check: the raw (unclipped) bounds really would breach the domain,
    # otherwise this test isn't exercising the fix.
    raw_bounds = _cell_bounds(xr.DataArray(pole_lat, dims=['lat'], coords={'lat': pole_lat}))
    assert raw_bounds.min() < -90 or raw_bounds.max() > 90

    da = _make_grid_da(np.arange(850, 1850), pole_lat, lon, val=1.0)
    regrid_conservative(da, xr.DataArray(pole_lat, dims=['lat']), xr.DataArray(lon, dims=['lon']))

    call = fake_xesmf['calls'][0]
    assert call.source_grid.lat_b.values.min() >= -90
    assert call.source_grid.lat_b.values.max() <= 90
    assert call.target_grid.lat_b.values.min() >= -90
    assert call.target_grid.lat_b.values.max() <= 90


def test_pole_inclusive_regrid_real_xesmf():
    """End-to-end (no mock): a real xESMF conservative regrid between two
    pole-inclusive grids of different resolution, mirroring CESM2c's native
    f19_g17 paleo grid vs. its finer CMIP6 'gn' historical grid. Without the
    -90/90 clip on the inferred bounds, xe.Regridder raises on this input
    (the AlignmentError precursor actually hit on HPC). Requires the
    optional 'regrid' extra (xesmf/esmpy); skipped if not installed."""
    pytest.importorskip('xesmf')

    src_lat, src_lon = np.linspace(-90, 90, 10), np.linspace(0, 320, 9)
    tgt_lat, tgt_lon = np.linspace(-90, 90, 20), np.linspace(0, 340.9, 18)
    da = _make_grid_da(np.arange(850, 853), src_lat, src_lon, val=1.0)

    out = regrid_conservative(da, xr.DataArray(tgt_lat, dims=['lat']), xr.DataArray(tgt_lon, dims=['lon']))

    assert out.shape == (3, len(tgt_lat), len(tgt_lon))
    assert np.allclose(out.values, 1.0)  # conservative regrid of a constant field preserves the constant
