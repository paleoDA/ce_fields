"""
Integration test for ce_fields.products against synthetic on-disk zarr stores.

Unlike test_products.py (pure in-memory arrays) and test_io.py (stitching
logic in isolation), this exercises the full load_stitched_tas /
load_corrected_tas pipeline end-to-end: open_nc_or_zarr -> groupby annual
mean -> stitch -> bias-correct, the exact path BTP_PMIP4_stacks.py (in
borehole_val) now calls into instead of re-deriving it locally.

No HPC data needed -- everything is synthetic and local to tmp_path.
"""
import numpy as np
import pytest
import xarray as xr
import yaml

from ce_fields.data_config import DataConfig
from ce_fields.products import load_stitched_tas, load_corrected_tas


def _make_monthly_ds(year_start, year_end, val=280.0):
    times = xr.date_range(start=f'{year_start}-01-01', end=f'{year_end}-12-31',
                           freq='MS', calendar='noleap', use_cftime=True)
    nlat, nlon = 3, 4
    data = np.full((len(times), nlat, nlon), val, dtype=float)
    return xr.Dataset(
        {'tas': (('time', 'lat', 'lon'), data)},
        coords={'time': times, 'lat': np.linspace(-60, 60, nlat), 'lon': np.linspace(0, 270, nlon)},
    )


@pytest.fixture
def synthetic_config(tmp_path):
    """A DataConfig pointing 'pmip4_TestModel' / 'cmip_hist_TestModel' at synthetic zarr stores."""
    paleo_path = tmp_path / 'paleo.zarr'
    hist_path  = tmp_path / 'hist.zarr'

    _make_monthly_ds(1800, 1849, val=280.0).to_zarr(paleo_path, mode='w')
    _make_monthly_ds(1850, 1900, val=282.0).to_zarr(hist_path, mode='w')

    yml = {
        'environments': {
            'laptop': {'root': str(tmp_path), 'paths': {}},
        },
        'files': {
            'pmip4_TestModel': {'path': str(paleo_path)},
            'cmip_hist_TestModel': {'path': str(hist_path)},
        },
        'current_environment': 'laptop',
    }
    yml_path = tmp_path / 'data_config.yml'
    yml_path.write_text(yaml.dump(yml))
    return DataConfig(config_path=yml_path)


def test_load_stitched_tas_clean_stitch(synthetic_config):
    tas = load_stitched_tas('TestModel', config=synthetic_config)
    assert int(tas.year.values[0])  == 1800
    assert int(tas.year.values[-1]) == 1900
    assert tas.attrs['stitched'] is True
    # values preserved from each segment
    assert np.allclose(tas.sel(year=slice(1800, 1849)).values, 280.0)
    assert np.allclose(tas.sel(year=slice(1850, 1900)).values, 282.0)


def test_load_corrected_tas_noop_for_unregistered_model(synthetic_config):
    """TestModel has no registered 1850-bias correction, so load_corrected_tas
    should match load_stitched_tas exactly."""
    stitched = load_stitched_tas('TestModel', config=synthetic_config)
    corrected = load_corrected_tas('TestModel', config=synthetic_config)
    assert corrected.identical(stitched)


def test_unknown_model_raises(synthetic_config):
    with pytest.raises(KeyError):
        load_stitched_tas('NotAModel', config=synthetic_config)


def _make_monthly_ds_var(year_start, year_end, var, val=280.0):
    """Like _make_monthly_ds but with a caller-chosen variable name (CESM2
    native output uses 'TREFHT', not 'tas' -- see pmip4_config.MODEL_VAR)."""
    ds = _make_monthly_ds(year_start, year_end, val=val)
    return ds.rename({'tas': var})


@pytest.fixture
def corrected_model_config(tmp_path):
    """A DataConfig for 'CESM2' (a real entry in pmip4_config.CORRECTED_MODELS)
    with distinct corrected/raw paleo paths, so use_corrected can be tested."""
    corrected_path = tmp_path / 'corrected.zarr'
    raw_path       = tmp_path / 'raw.zarr'
    hist_path      = tmp_path / 'hist.zarr'

    _make_monthly_ds_var(1800, 1849, 'TREFHT', val=280.0).to_zarr(corrected_path, mode='w')
    _make_monthly_ds_var(1800, 1849, 'TREFHT', val=270.0).to_zarr(raw_path, mode='w')
    _make_monthly_ds(1850, 1900, val=282.0).to_zarr(hist_path, mode='w')

    yml = {
        'environments': {
            'laptop': {'root': str(tmp_path), 'paths': {}},
        },
        'files': {
            'pmip4_CESM2': {'path': str(corrected_path)},
            'raw_pmip4_CESM2': {'path': str(raw_path)},
            'cmip_hist_CESM2': {'path': str(hist_path)},
        },
        'current_environment': 'laptop',
    }
    yml_path = tmp_path / 'data_config.yml'
    yml_path.write_text(yaml.dump(yml))
    return DataConfig(config_path=yml_path)


def test_use_corrected_true_reads_corrected_path(corrected_model_config):
    tas = load_stitched_tas('CESM2', config=corrected_model_config, use_corrected=True)
    assert np.allclose(tas.sel(year=slice(1800, 1849)).values, 280.0)
    assert tas.attrs['corrected'] is True
    assert tas.attrs['display_name'] == 'CESM2c'


def test_use_corrected_false_reads_raw_path(corrected_model_config):
    tas = load_stitched_tas('CESM2', config=corrected_model_config, use_corrected=False)
    assert np.allclose(tas.sel(year=slice(1800, 1849)).values, 270.0)
    assert tas.attrs['corrected'] is False
    assert tas.attrs['display_name'] == 'CESM2'


def test_load_corrected_tas_passes_through_use_corrected(corrected_model_config):
    raw = load_corrected_tas('CESM2', config=corrected_model_config, use_corrected=False)
    assert np.allclose(raw.sel(year=slice(1800, 1849)).values, 270.0)


@pytest.fixture
def partial_last_year_config(tmp_path):
    """Paleo run truncates mid-1850 (only January present, a warm outlier
    value), mirroring the real CESM2 run002 archive. hist starts at 1850
    with a full, unremarkable year."""
    paleo_path = tmp_path / 'paleo.zarr'
    hist_path  = tmp_path / 'hist.zarr'

    ds = _make_monthly_ds(1800, 1850, val=280.0)
    ds = ds.isel(time=slice(None, -11))  # keep all months through 1850-01 only
    ds['tas'][-1] = 400.0  # exaggerated outlier so a bug would be obvious
    ds.to_zarr(paleo_path, mode='w')

    _make_monthly_ds(1850, 1900, val=282.0).to_zarr(hist_path, mode='w')

    yml = {
        'environments': {
            'laptop': {'root': str(tmp_path), 'paths': {}},
        },
        'files': {
            'pmip4_TestModel': {'path': str(paleo_path)},
            'cmip_hist_TestModel': {'path': str(hist_path)},
        },
        'current_environment': 'laptop',
    }
    yml_path = tmp_path / 'data_config.yml'
    yml_path.write_text(yaml.dump(yml))
    return DataConfig(config_path=yml_path)


def test_load_stitched_tas_drops_partial_final_paleo_year(partial_last_year_config):
    tas = load_stitched_tas('TestModel', config=partial_last_year_config)
    # The partial (1-month, outlier) 1850 from paleo must not survive into
    # the stitched series -- 1850 should come from hist's full year instead.
    assert int(tas.year.values[0]) == 1800
    assert int(tas.year.values[-1]) == 1900
    assert np.allclose(tas.sel(year=1850).values, 282.0)
    assert np.allclose(tas.sel(year=slice(1800, 1849)).values, 280.0)
