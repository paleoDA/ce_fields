"""
Tests for ce_fields.stitch_corrections: the packaged GMST table and the
pre-1850 correction helpers derived from it.
"""

import numpy as np
import pandas as pd
import pytest

from ce_fields.stitch_corrections import (
    ACCESS_MODEL, CESM2_MODEL, GMST_ANOMALIES_CSV, STITCH_YEAR,
    compute_cesm2_trend, compute_flat_offset, load_gmst_anomalies,
)


def test_packaged_table_is_installed_and_complete():
    assert GMST_ANOMALIES_CSV.exists()
    gmst = load_gmst_anomalies()
    assert {ACCESS_MODEL, CESM2_MODEL} <= set(gmst.columns)
    assert 'has_stitch' not in gmst.index.astype(str)
    assert gmst.index.dtype.kind == 'i'
    assert {STITCH_YEAR - 1, STITCH_YEAR} <= set(gmst[ACCESS_MODEL].dropna().index)


def test_flat_offset_is_single_year_jump():
    years = np.arange(1840, 1861)
    gmst = pd.DataFrame({ACCESS_MODEL: np.where(years < STITCH_YEAR, 0.0, 0.3)}, index=years)
    assert compute_flat_offset(gmst) == pytest.approx(0.3)


def test_cesm2_trend_fits_pre_1850_years_only():
    years = np.arange(850, 1900)
    # Linear drift before 1850, then an unrelated constant afterwards
    values = np.where(years < STITCH_YEAR, 0.001 * years - 1.0, 5.0)
    gmst = pd.DataFrame({CESM2_MODEL: values}, index=years)
    slope, intercept = compute_cesm2_trend(gmst)
    assert slope == pytest.approx(0.001)
    assert intercept == pytest.approx(-1.0)
