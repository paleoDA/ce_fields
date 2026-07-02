"""
Integration test for ce_simul.products against synthetic on-disk zarr stores.

Unlike test_products.py (pure in-memory arrays) and test_io.py (stitching
logic in isolation), this exercises the full load_stitched_tas /
load_corrected_tas pipeline end-to-end: open_nc_or_zarr -> groupby annual
mean -> stitch -> bias-correct, the exact path BTP_PMIP4_stacks.py (in
ce_val) now calls into instead of re-deriving it locally.

No HPC data needed -- everything is synthetic and local to tmp_path.
"""
import numpy as np
import pytest
import xarray as xr
import yaml

from ce_simul.data_config import DataConfig
from ce_simul.products import load_stitched_tas, load_corrected_tas


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
