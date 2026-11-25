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


GAO_SNIPPET = """Version 2 of IVI2 hemispheric and global volcanic stratospheric sulfate injection

  1       2        3       4       5
Year     NH       SH     Global  Revised?

1256    0.00     0.00     0.00
1257    0.00     0.00     0.00
1258  145.80   112.11   257.91   ***
1259    1.00     0.50     1.50
1260    0.00     0.00     0.00
1261    2.00     0.00     2.00
"""


def test_parse_gao2008_and_group_eruptions():
    from ce_fields.volcanic import parse_gao2008, gao2008_eruptions
    annual = parse_gao2008(GAO_SNIPPET)
    assert list(annual['year']) == list(range(1256, 1262))
    assert annual.set_index('year').loc[1258, 'sulfate'] == pytest.approx(257.91)
    ev = gao2008_eruptions(annual)
    # consecutive non-zero years (1258-1259) merge into one eruption dated 1258
    assert list(ev['yearCE']) == [1258, 1261]
    assert ev.loc[0, 'sulfate'] == pytest.approx(259.41)


def test_largest_eruptions_rejects_unknown_source():
    with pytest.raises(ValueError):
        largest_eruptions(source='nope')


def test_sea_ensemble_matches_sea_per_member():
    from ce_fields.volcanic import sea_ensemble
    rng = np.random.default_rng(0)
    years = np.arange(1000, 1200)
    vals = rng.normal(size=(len(years), 4))
    vals[110, 2] = np.nan                     # a gap in one member
    events = [1020, 1050, 1105, 1150]
    comp = sea_ensemble(vals, years, events, before=5, after=10)
    assert comp.shape == (4, 16)
    for m in range(4):
        ref = sea(pd.Series(vals[:, m], index=years), events, 5, 10).mean(axis=0)
        np.testing.assert_allclose(comp[m], ref.values)


def test_sea_baseline_shorter_than_pre_window():
    from ce_fields.volcanic import sea_ensemble
    years = np.arange(1000, 1100)
    s = pd.Series(np.arange(100, dtype=float), index=years)   # linear ramp
    out = sea(s, [1050], before=5, after=10, baseline=3)
    # baseline = mean of lags -3..-1 = lag -2, so lag k sits k + 2 above it
    assert out.loc[1050, 3] == pytest.approx(5.0)
    assert out.loc[1050].loc[-3:-1].mean() == pytest.approx(0.0)
    comp = sea_ensemble(s.to_numpy()[:, None], years, [1050], before=5, after=10, baseline=3)
    np.testing.assert_allclose(comp[0], out.loc[1050].to_numpy())
    with pytest.raises(ValueError):
        sea(s, [1050], before=5, after=10, baseline=6)
