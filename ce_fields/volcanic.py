"""
Volcanic forcing (eVolv2k v3) and superposed epoch analysis (SEA).

eVolv2k (Toohey and Sigl, 2017) is the volcanic forcing of the PMIP4
past1000/past2k experiments; it ships with the package (see
ce_fields/data/eVolv2k_v3/README.md for why it lives here).

The SEA follows the LMR volcanic analysis (LMR_VolcanicResponse_sea.py /
gmt_volcanicSEA2.py, Emile-Geay 2017): for each event, take the window
[-before, +after] years around the event year, subtract the mean of the
`before` years preceding it, and average the anomalies over events.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

EVOLV2K_FILE = Path(__file__).resolve().parent / 'data' / 'eVolv2k_v3' / 'eVolv2k_v3_ds_1.nc'


def load_evolv2k() -> pd.DataFrame:
    """Load the eVolv2k v3 eruption list.

    Returns
    -------
    pd.DataFrame
        One row per eruption, sorted by date, with columns ``yearCE`` (int),
        ``month``, ``day``, ``lat``, ``vssi`` (TgS), ``sigma_vssi`` (TgS)
        and ``hemi``. Unknown eruption dates are coded as 1 January in the
        dataset.
    """
    with xr.open_dataset(EVOLV2K_FILE) as ds:
        df = ds[['yearCE', 'month', 'day', 'lat', 'vssi', 'sigma_vssi', 'hemi']].to_dataframe()
    df['yearCE'] = df['yearCE'].astype(int)
    return df.sort_values(['yearCE', 'month', 'day']).reset_index(drop=True)


def largest_eruptions(n: int = 20, start: int = None, end: int = None) -> pd.DataFrame:
    """The `n` largest eVolv2k eruptions by VSSI with ``start <= yearCE <= end``.

    Returned in chronological order. Either bound may be None (no bound).
    """
    ev = load_evolv2k()
    if start is not None:
        ev = ev[ev['yearCE'] >= start]
    if end is not None:
        ev = ev[ev['yearCE'] <= end]
    return ev.nlargest(n, 'vssi').sort_values('yearCE').reset_index(drop=True)


def sea(series: pd.Series, event_years, before: int = 5, after: int = 10) -> pd.DataFrame:
    """Superposed epoch analysis of an annual series.

    Parameters
    ----------
    series : pd.Series
        Annual values indexed by integer year. Missing years (absent or NaN)
        are allowed and propagate as NaN.
    event_years : sequence of int
        Key (lag-0) years, e.g. eruption years.
    before, after : int
        Window extent: lags ``-before .. +after``. The mean over lags
        ``-before .. -1`` is the baseline subtracted from each event's window.

    Returns
    -------
    pd.DataFrame
        Anomalies relative to each event's pre-event baseline, indexed by
        event year, with one column per lag. The composite is
        ``.mean(axis=0)`` (pandas skips NaN).
    """
    if before < 1:
        raise ValueError("before must be >= 1: the pre-event years form the baseline")
    lags = np.arange(-before, after + 1)
    s = series.copy()
    s.index = s.index.astype(int)
    rows = {}
    for y in event_years:
        y = int(y)
        win = s.reindex(y + lags).to_numpy(dtype=float)
        rows[y] = win - np.nanmean(win[:before]) if np.isfinite(win[:before]).any() else np.full(len(lags), np.nan)
    out = pd.DataFrame.from_dict(rows, orient='index', columns=lags)
    out.index.name = 'event_year'
    out.columns.name = 'lag'
    return out
