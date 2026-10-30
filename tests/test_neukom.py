#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Neukom2019 class using pytest

Created on Mon Jul 20 2026
@author: Julien Emile-Geay with Claude AI assistance
"""

import os
import pytest
import numpy as np
import xarray as xr
from ce_fields.neukom import Neukom2019
from ce_fields.data_config import DataConfig


def _file_available(config_key):
    try:
        return os.path.exists(DataConfig().get_file_path(config_key))
    except Exception:
        return False


class TestNeukom2019Class:
    """Test suite for Neukom2019 class."""

    def test_method_map(self):
        """Test that all six methods are registered with the expected fields."""
        expected_methods = {'AM', 'CCA', 'CPS', 'DA', 'GraphEM', 'PCR'}
        assert set(Neukom2019.METHOD_MAP) == expected_methods
        for method, info in Neukom2019.METHOD_MAP.items():
            assert 'config_key' in info
            assert 'var' in info
            assert 'member_dim' in info

    def test_from_method_invalid(self):
        """Test error handling for an unrecognised method."""
        with pytest.raises(ValueError, match="Unknown method"):
            Neukom2019.from_method('bogus')

    @pytest.mark.parametrize('method', list(Neukom2019.METHOD_MAP))
    def test_from_method(self, method):
        """Test loading each of the six methods."""
        config_key = Neukom2019.METHOD_MAP[method]['config_key']
        if not _file_available(config_key):
            pytest.skip(f"Neukom2019 {method} data file not found")

        recon = Neukom2019.from_method(method, time_slice=(1900, 2000))

        assert isinstance(recon, Neukom2019)
        assert recon.method == method
        assert isinstance(recon.data, xr.DataArray)

        assert 'lat' in recon.data.dims
        assert 'lon' in recon.data.dims
        assert 'member' in recon.data.dims
        assert 'year' in recon.data.dims

        assert recon.data.lat.min() >= -90
        assert recon.data.lat.max() <= 90
        assert recon.data.sizes['member'] == 100

        years = recon.data.year.values
        assert years.min() >= 1900
        assert years.max() <= 2000

    @pytest.mark.skipif(not _file_available('neukom_cps'),
                        reason="Neukom2019 CPS data file not found")
    def test_compute_gmst(self):
        """Test GMST computation and its (year, member) shape."""
        cps = Neukom2019.from_method('CPS', time_slice=(1900, 2000))
        gmst = cps.compute_gmst(ref_period=(1961, 1990))

        assert isinstance(gmst, xr.DataArray)
        assert set(gmst.dims) == {'year', 'member'}
        assert gmst.sizes['member'] == 100
        # Reasonable temperature anomaly range
        assert -10 < float(gmst.mean()) < 10

    @pytest.mark.skipif(not _file_available('neukom_da'),
                        reason="Neukom2019 DA data file not found")
    def test_compute_gmst_da_kelvin(self):
        """DA is archived in absolute Kelvin; compute_gmst must still return
        a small anomaly once the reference period is subtracted."""
        da = Neukom2019.from_method('DA', time_slice=(1900, 2000))
        assert float(da.data.mean()) > 100  # sanity check: still absolute K

        gmst = da.compute_gmst(ref_period=(1961, 1990))
        assert -10 < float(gmst.mean()) < 10

    @pytest.mark.skipif(not _file_available('neukom_cps'),
                        reason="Neukom2019 CPS data file not found")
    def test_compute_gmst_no_data_in_ref_period(self):
        """Test error handling when the reference period is out of range."""
        cps = Neukom2019.from_method('CPS', time_slice=(1900, 2000))
        with pytest.raises(ValueError, match="No data in reference period"):
            cps.compute_gmst(ref_period=(2100, 2200))

    @pytest.mark.skipif(not _file_available('neukom_cps'),
                        reason="Neukom2019 CPS data file not found")
    def test_to_grid_anomaly(self):
        """Test that to_grid_anomaly preserves spatial dims and stays lazy."""
        cps = Neukom2019.from_method('CPS', time_slice=(1900, 2000))
        anom = cps.to_grid_anomaly(ref_period=(1961, 1990))

        assert isinstance(anom, xr.DataArray)
        assert set(anom.dims) == {'year', 'member', 'lat', 'lon'}
        assert anom.chunks is not None  # still dask-backed, not eagerly loaded

        # A single nearest-neighbour site extraction should be a small,
        # reasonable anomaly once loaded.
        site = anom.sel(lat=-79, lon=248, method='nearest').load()
        assert -10 < float(site.mean()) < 10

    @pytest.mark.skipif(not _file_available('neukom_da'),
                        reason="Neukom2019 DA data file not found")
    def test_to_grid_anomaly_da_kelvin(self):
        """DA is archived in absolute Kelvin; to_grid_anomaly must still
        return a small anomaly once loaded and the reference period
        subtracted."""
        da = Neukom2019.from_method('DA', time_slice=(1900, 2000))
        anom = da.to_grid_anomaly(ref_period=(1961, 1990))
        site = anom.sel(lat=-79, lon=248, method='nearest').load()
        assert -10 < float(site.mean()) < 10

    def test_load_all_methods(self):
        """Test loading all available methods."""
        available_count = sum(
            _file_available(info['config_key'])
            for info in Neukom2019.METHOD_MAP.values()
        )
        if available_count == 0:
            pytest.skip("No Neukom2019 data files found")

        methods = Neukom2019.load_all_methods(time_slice=(1900, 2000))

        assert isinstance(methods, dict)
        assert len(methods) == available_count
        for name, recon in methods.items():
            assert isinstance(recon, Neukom2019)
            assert name == recon.method

    @pytest.mark.skipif(not _file_available('neukom_cps'),
                        reason="Neukom2019 CPS data file not found")
    def test_repr(self):
        """Test string representation."""
        cps = Neukom2019.from_method('CPS', time_slice=(1900, 2000))
        repr_str = repr(cps)

        assert 'Neukom2019' in repr_str
        assert 'CPS' in repr_str
        assert 'shape' in repr_str
        assert 'time_range' in repr_str
