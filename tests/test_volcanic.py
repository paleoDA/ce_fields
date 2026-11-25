"""Tests for ce_fields.volcanic (eVolv2k loading and superposed epoch analysis)."""
import numpy as np
import pandas as pd
import pytest

from ce_fields.volcanic import load_evolv2k, largest_eruptions, sea


def test_load_evolv2k():
    ev = load_evolv2k()
    assert len(ev) == 256
    assert ev['yearCE'].is_monotonic_increasing
    # Samalas (1257) is the largest eruption in the record
    top = ev.loc[ev['vssi'].idxmax()]
    assert top['yearCE'] == 1257 and top['vssi'] == pytest.approx(59.42)


def test_largest_eruptions_respects_bounds_and_order():
    ev = largest_eruptions(n=5, start=1500, end=1900)
    assert len(ev) == 5
    assert ev['yearCE'].between(1500, 1900).all()
    assert ev['yearCE'].is_monotonic_increasing
    assert {1815, 1809}.issubset(set(ev['yearCE']))


def test_sea_recovers_imposed_response():
    years = np.arange(1000, 1100)
    s = pd.Series(1.0, index=years)          # constant offset: removed by the baseline
    response = {0: -0.2, 1: -0.5, 2: -0.3}
    events = [1020, 1050, 1080]
    for e in events:
        for lag, v in response.items():
            s[e + lag] += v
    out = sea(s, events, before=5, after=10)
    assert list(out.columns) == list(range(-5, 11))
    assert list(out.index) == events
    comp = out.mean(axis=0)
    assert np.allclose(comp.loc[-5:-1], 0.0)
    for lag, v in response.items():
        assert comp[lag] == pytest.approx(v)
    assert np.allclose(comp.loc[3:], 0.0)


def test_sea_missing_years_are_nan_not_shifted():
    s = pd.Series(np.arange(100, dtype=float), index=np.arange(1000, 1100))
    s = s.drop(1052)                          # absent year inside the window
    out = sea(s, [1050], before=5, after=10)
    assert np.isnan(out.loc[1050, 2])
    # a linear ramp: lag k sits k - (-3) = k + 3 above the baseline mean (lags -5..-1)
    assert out.loc[1050, 3] == pytest.approx(6.0)


def test_sea_requires_baseline():
    with pytest.raises(ValueError):
        sea(pd.Series([0.0], index=[1000]), [1000], before=0)
