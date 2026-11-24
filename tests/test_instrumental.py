#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ce_fields.instrumental: version-explicit loading of instrumental
temperature products. Catalog/error-path tests run anywhere; loading tests
skip when the file is not on this machine.

Created on Tue Nov 24 2026
@author: Julien Emile-Geay with Claude AI assistance
"""

import numpy as np
import pytest
import xarray as xr

from ce_fields.data_config import DataConfig
from ce_fields.instrumental import Instrumental, catalog, PRODUCTS

CAT = catalog()
REQUIRED_FIELDS = ('product', 'version', 'variable', 'kind', 'var', 'version_check')


def _found(product, version, variable='tas', kind='mean'):
    row = CAT[(CAT['product'] == product) & (CAT['version'] == version)
              & (CAT['variable'] == variable) & (CAT['kind'] == kind)]
    return not row.empty and bool(row['found'].iloc[0])


class TestCatalog:

    def test_entries_are_complete_and_versioned(self):
        files = DataConfig().config['files']
        for key in CAT['key']:
            entry = files[key]
            for field in REQUIRED_FIELDS:
                assert field in entry, f"{key} lacks '{field}'"
            # the version must be spelled out in the key, never implied
            assert f"_{entry['version']}_" in key
            assert isinstance(entry['version'], str), f"{key}: quote the version in YAML"
            assert entry['product'] in PRODUCTS

    def test_keys_are_unique_per_version(self):
        assert not CAT.duplicated(['product', 'version', 'variable', 'kind']).any()

    def test_hadcrut5_versions_registered(self):
        versions = set(CAT.loc[CAT['product'] == 'hadcrut5', 'version'])
        assert {'5.0.2.0', '5.1.0.0'} <= versions


class TestErrors:

    def test_version_is_required(self):
        with pytest.raises(TypeError):
            Instrumental.from_version('hadcrut5')

    def test_numeric_version_rejected(self):
        with pytest.raises(TypeError, match='string'):
            Instrumental.from_version('hadcrut5', 5.1)

    def test_unknown_version_lists_registered(self):
        with pytest.raises(ValueError, match='Registered versions'):
            Instrumental.from_version('hadcrut5', '9.9.9.9')

    def test_unknown_product(self):
        with pytest.raises(ValueError, match='Unknown product'):
            Instrumental.from_version('bogus', '1.0')


@pytest.mark.skipif(not _found('hadcrut5', '5.1.0.0'), reason='HadCRUT5 5.1.0.0 not found')
class TestHadCRUT5:

    def test_harmonized_annual_field(self):
        hc = Instrumental.from_version('hadcrut5', '5.1.0.0')
        assert hc.data.dims == ('year', 'lat', 'lon')
        assert hc.data.attrs['version'] == '5.1.0.0'
        assert hc.label == 'HadCRUT5 v5.1.0.0'
        assert float(hc.data.lon.min()) >= 0 and float(hc.data.lon.max()) < 360
        assert np.all(np.diff(hc.data.lat) > 0)

    def test_incomplete_final_year_dropped(self):
        hc = Instrumental.from_version('hadcrut5', '5.1.0.0', annualize=False)
        last = int(hc.data['year_of'].max())
        n_last = int((hc.data['year_of'] == last).sum())
        annual = Instrumental.from_version('hadcrut5', '5.1.0.0')
        assert (int(annual.years.max()) == last) == (n_last == 12)

    def test_gmst_recentered(self):
        g = Instrumental.from_version('hadcrut5', '5.1.0.0').gmst(ref_period=(1951, 1980))
        assert abs(float(g.sel(year=slice(1951, 1980)).mean())) < 1e-10
        assert g.attrs['baseline'] == (1951, 1980)

    def test_lon_convention(self):
        hc = Instrumental.from_version('hadcrut5', '5.1.0.0', lon_convention='-180_180')
        assert float(hc.data.lon.min()) < 0

    def test_version_check_rejects_wrong_file(self):
        config = DataConfig()
        entry = config.config['files']['instr_hadcrut5_5.1.0.0_tas_mean']
        entry['version_check'] = {'attr': 'version', 'contains': 'HadCRUT.0.0.0.0'}
        with pytest.raises(ValueError, match='Version check failed'):
            Instrumental.from_version('hadcrut5', '5.1.0.0', config=config)


@pytest.mark.skipif(not _found('hadcrut5', '5.0.2.0'), reason='HadCRUT5 5.0.2.0 not found')
def test_versions_differ():
    """Releases revise the past: the reason versions must be explicit."""
    a = Instrumental.from_version('hadcrut5', '5.0.2.0').gmst()
    b = Instrumental.from_version('hadcrut5', '5.1.0.0').gmst()
    common = np.intersect1d(a.year, b.year)
    assert float(abs(a.sel(year=common) - b.sel(year=common)).max()) > 0


@pytest.mark.skipif(not _found('dcent_i', '1.1.0.0', kind='members'), reason='DCENT-I members not found')
def test_members_average_to_mean():
    mem = Instrumental.from_version('dcent_i', '1.1.0.0', kind='members', time_slice=(2000, 2002))
    mean = Instrumental.from_version('dcent_i', '1.1.0.0', time_slice=(2000, 2002))
    assert mem.data.sizes['member'] == 200
    xr.testing.assert_allclose(mem.gmst().mean('member'), mean.gmst(), atol=5e-3)
