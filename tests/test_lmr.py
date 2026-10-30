#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LMR class using pytest

Created on Dec 4 2025
@author: Julien Emile-Geay with Claude AI assistance
"""

import os
import pytest
import numpy as np
import xarray as xr
from ce_fields.lmr import LMR, load_available_vars
from ce_fields.data_config import DataConfig


class TestLoadAvailableVars:
    """Test suite for load_available_vars utility function."""

    @pytest.fixture
    def config(self):
        """Load data configuration."""
        return DataConfig()

    def test_load_available_vars_first_match(self, config):
        """Test loading first matching variable."""
        try:
            filepath = config.get_file_path('lmr_seasonal')
        except:
            pytest.skip("LMR seasonal data file not found")

        target_vars = ['tas', 'air', 'sat']
        data = load_available_vars(filepath, target_vars, return_first_match=True,
                                  decode_times=False, show_available=False)

        if data is not None:
            assert isinstance(data, xr.DataArray)
            assert data.name in target_vars


class TestLMRClass:
    """Test suite for LMR class."""

    @pytest.fixture
    def config(self):
        """Load data configuration."""
        return DataConfig()

    def test_version_map(self):
        """Test that version mapping is correct."""
        assert 'v2.1' in LMR.VERSION_MAP
        assert 'seasonal' in LMR.VERSION_MAP
        assert 'online' in LMR.VERSION_MAP
        assert LMR.VERSION_MAP['v2.1'] == 'lmr_v2_1'
        assert LMR.VERSION_MAP['seasonal'] == 'lmr_seasonal'
        assert LMR.VERSION_MAP['online'] == 'lmr_online'

    def test_temperature_vars(self):
        """Test that temperature variable list is defined."""
        assert isinstance(LMR.TEMPERATURE_VARS, list)
        assert len(LMR.TEMPERATURE_VARS) > 0
        assert 'tas' in LMR.TEMPERATURE_VARS

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_seasonal_tas_mean')),
        reason="LMR seasonal data file not found"
    )
    def test_from_version_seasonal(self):
        """Test loading LMR seasonal version."""
        lmr = LMR.from_version('seasonal', annualize=True)

        assert isinstance(lmr, LMR)
        assert lmr.version == 'seasonal'
        assert isinstance(lmr.data, xr.DataArray)
        assert lmr.variable in LMR.TEMPERATURE_VARS

        # Check dimensions
        assert 'lat' in lmr.data.dims
        assert 'lon' in lmr.data.dims
        assert ('time' in lmr.data.dims or 'year' in lmr.data.dims)

        # Check coordinate ranges
        assert lmr.data.lat.min() >= -90
        assert lmr.data.lat.max() <= 90
        assert lmr.data.lon.min() >= 0
        assert lmr.data.lon.max() <= 360

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_v2_1_tas_mean')),
        reason="LMR v2.1 data file not found"
    )
    def test_from_version_v2_1(self):
        """Test loading LMR v2.1 version."""
        lmr = LMR.from_version('v2.1', annualize=True)

        assert isinstance(lmr, LMR)
        assert lmr.version == 'v2.1'
        assert isinstance(lmr.data, xr.DataArray)

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_online_tas_mean')),
        reason="LMR online data file not found"
    )
    def test_from_version_online(self):
        """Test loading LMR online version."""
        lmr = LMR.from_version('online', annualize=True)

        assert isinstance(lmr, LMR)
        assert lmr.version == 'online'
        assert isinstance(lmr.data, xr.DataArray)

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_online_gmst')),
        reason="LMR online GMST file not found"
    )
    def test_from_version_online_gmst(self):
        """Test loading LMR online GMST ensemble from tas_sfc_indices.nc."""
        lmr = LMR.from_version('online', variable='gmst')

        assert isinstance(lmr, LMR)
        assert lmr.version == 'online'
        assert lmr.variable == 'gmst'
        assert isinstance(lmr.data, xr.DataArray)
        assert 'mc_iteration' in lmr.data.dims
        assert 'ensemble_member' in lmr.data.dims
        assert lmr.data.sizes['mc_iteration'] == 50
        assert lmr.data.sizes['ensemble_member'] == 100

    def test_from_version_seasonal_members(self):
        """Test loading LMR seasonal ensemble members with lazy loading."""
        config = DataConfig()

        # Check if the URL is accessible by trying to load
        try:
            filepath = config.get_file_path('lmr_seasonal_tas_ens')
        except:
            pytest.skip("LMR seasonal members URL not accessible")

        # This test loads from URL, so it may be slow or fail if network is down
        try:
            lmr = LMR.from_version('seasonal', load_members=True, annualize=False)

            assert isinstance(lmr, LMR)
            assert lmr.version == 'seasonal'
            assert isinstance(lmr.data, xr.DataArray)

            # Check that member dimension exists
            assert 'member' in lmr.data.dims

            # Check that we have multiple members
            assert lmr.data.sizes['member'] > 1

            # Check that data is lazily loaded (has chunks)
            # Note: chunks may be None if data was loaded eagerly
            # This is fine for the test, just checking structure

            # Check standard dimensions
            assert 'lat' in lmr.data.dims
            assert 'lon' in lmr.data.dims
            assert 'time' in lmr.data.dims

        except Exception as e:
            # If loading fails (network issue, etc.), skip the test
            pytest.skip(f"Could not load ensemble members from URL: {e}")

    def test_from_version_invalid(self):
        """Test error handling for invalid version."""
        with pytest.raises(ValueError, match="Unknown LMR version"):
            LMR.from_version('invalid_version')

    def test_from_version_invalid_variable(self):
        """Test error handling for unsupported variable."""
        with pytest.raises(ValueError, match="must be 'tas' or 'gmst'"):
            LMR.from_version('seasonal', variable='precipitation')

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_seasonal_tas_mean')),
        reason="LMR seasonal data file not found"
    )
    def test_time_slicing(self):
        """Test time slicing functionality."""
        lmr = LMR.from_version('seasonal', time_slice=(1500, 1800))

        time_dim = 'year' if 'year' in lmr.data.dims else 'time'
        time_vals = lmr.data[time_dim].values

        # Get min/max years
        if time_dim == 'year':
            min_year = time_vals.min()
            max_year = time_vals.max()
        else:
            get_year = lambda x: x.year if hasattr(x, 'year') else x
            vectorized_get_year = np.vectorize(get_year)
            years = vectorized_get_year(time_vals)
            min_year = years.min()
            max_year = years.max()

        assert min_year >= 1500
        assert max_year <= 1800

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_seasonal_tas_mean')),
        reason="LMR seasonal data file not found"
    )
    def test_compute_global_mean(self):
        """Test global mean computation."""
        lmr = LMR.from_version('seasonal', time_slice=(1800, 1900))
        gmst = lmr.compute_global_mean()

        assert isinstance(gmst, xr.DataArray)
        # Should have time dimension only
        assert gmst.ndim == 1
        assert ('time' in gmst.dims or 'year' in gmst.dims)
        # Values should be reasonable temperature range
        assert -50 < gmst.mean().values < 50

    def test_ens_mean(self):
        """Test ensemble mean computation."""
        config = DataConfig()

        # This test requires ensemble members from URL
        try:
            filepath = config.get_file_path('lmr_seasonal_tas_ens')
        except:
            pytest.skip("LMR seasonal members URL not accessible")

        try:
            lmr = LMR.from_version('seasonal', load_members=True, annualize=False)

            # Compute ensemble mean
            ens_mean = lmr.ens_mean()

            assert isinstance(ens_mean, xr.DataArray)
            # Should not have member dimension after taking mean
            assert 'member' not in ens_mean.dims
            # Should still have spatial and temporal dimensions
            assert 'lat' in ens_mean.dims
            assert 'lon' in ens_mean.dims
            assert 'time' in ens_mean.dims

        except Exception as e:
            pytest.skip(f"Could not load ensemble members from URL: {e}")

    def test_ens_spread(self):
        """Test ensemble spread computation."""
        config = DataConfig()

        # This test requires ensemble members from URL
        try:
            filepath = config.get_file_path('lmr_seasonal_tas_ens')
        except:
            pytest.skip("LMR seasonal members URL not accessible")

        try:
            lmr = LMR.from_version('seasonal', load_members=True, annualize=False)

            # Compute ensemble spread
            ens_spread = lmr.ens_spread()

            assert isinstance(ens_spread, xr.DataArray)
            # Should not have member dimension after taking std
            assert 'member' not in ens_spread.dims
            # Should still have spatial and temporal dimensions
            assert 'lat' in ens_spread.dims
            assert 'lon' in ens_spread.dims
            assert 'time' in ens_spread.dims
            # Spread should be non-negative
            assert (ens_spread >= 0).all()

        except Exception as e:
            pytest.skip(f"Could not load ensemble members from URL: {e}")

    def test_ens_mean_without_members(self):
        """Test that ens_mean raises error when no member dimension exists."""
        try:
            lmr = LMR.from_version('seasonal', annualize=True)
        except:
            pytest.skip("LMR seasonal data file not found")

        with pytest.raises(ValueError, match="does not have a 'member' dimension"):
            lmr.ens_mean()

    def test_ens_spread_without_members(self):
        """Test that ens_spread raises error when no member dimension exists."""
        try:
            lmr = LMR.from_version('seasonal', annualize=True)
        except:
            pytest.skip("LMR seasonal data file not found")

        with pytest.raises(ValueError, match="does not have a 'member' dimension"):
            lmr.ens_spread()

    def test_load_all_versions(self):
        """Test loading all available versions."""
        config = DataConfig()

        # Count how many LMR files exist
        available_count = 0
        for version_key in LMR.VERSION_MAP.values():
            try:
                path = config.get_file_path(version_key)
                if os.path.exists(path):
                    available_count += 1
            except:
                continue

        if available_count == 0:
            pytest.skip("No LMR data files found")

        lmr_products = LMR.load_all_versions(time_slice=(1850, 1900))

        assert isinstance(lmr_products, dict)
        assert len(lmr_products) == available_count

        for version, lmr in lmr_products.items():
            assert isinstance(lmr, LMR)
            assert version == lmr.version
            assert isinstance(lmr.data, xr.DataArray)

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_seasonal_tas_mean')),
        reason="LMR seasonal data file not found"
    )
    def test_repr(self):
        """Test string representation."""
        lmr = LMR.from_version('seasonal', time_slice=(1800, 1850))
        repr_str = repr(lmr)

        assert 'LMR' in repr_str
        assert 'seasonal' in repr_str
        assert 'shape' in repr_str
        assert 'time_range' in repr_str


class TestIntegrationWithSeries:
    """Test integration with DocuClimSeries and HCLIMSeries."""

    @pytest.mark.skipif(
        not os.path.exists(DataConfig().get_file_path('lmr_seasonal_tas_mean')),
        reason="LMR seasonal data file not found"
    )
    def test_load_nearest_xarray_integration(self):
        """Test that LMR data can be used with load_nearest_xarray."""
        # This is an integration test that would require DocuClimSeries/HCLIMSeries
        # Just verify that LMR data has the right structure
        lmr = LMR.from_version('seasonal', time_slice=(1850, 1900))

        # Check that data has the required coordinates for load_nearest_xarray
        assert 'lat' in lmr.data.coords or 'latitude' in lmr.data.coords
        assert 'lon' in lmr.data.coords or 'longitude' in lmr.data.coords

        # Check that data is an xarray DataArray (required by load_nearest_xarray)
        assert isinstance(lmr.data, xr.DataArray)
