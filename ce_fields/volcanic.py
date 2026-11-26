"""
Volcanic forcing (eVolv2k v3, Gao et al. 2008) and superposed epoch analysis (SEA).

eVolv2k (Toohey and Sigl, 2017) is the volcanic forcing of the PMIP4
past1000/past2k experiments; it ships with the package (see
ce_fields/data/eVolv2k_v3/README.md for why it lives here). iCESM (see
pmip4_config.VOLCANIC_FORCING) was instead forced with Gao et al. (2008,
IVI2), which is read remotely from Rutgers.

The SEA follows the LMR volcanic analysis (LMR_VolcanicResponse_sea.py /
gmt_volcanicSEA2.py, Emile-Geay 2017): for each event, take the window
[-before, +after] years around the event year, subtract the mean of the
`before` years preceding it, and average the anomalies over events.
"""
import warnings
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

EVOLV2K_FILE = Path(__file__).resolve().parent / 'data' / 'eVolv2k_v3' / 'eVolv2k_v3_ds_1.nc'

# Gao, Robock and Ammann (2008), doi:10.1029/2008JD010239: IVI2 annual
# stratospheric sulfate injection (Tg), 501-2000 CE, version 2 (2012).
GAO2008_URL = 'https://climate.envsci.rutgers.edu/IVI2/IVI2TotalInjection_501-2000Version2.txt'


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


def parse_gao2008(text: str) -> pd.DataFrame:
    """Parse the IVI2 text file: one row per year with columns ``year``,
    ``nh``, ``sh``, ``sulfate`` (global; all Tg). Header lines and the
    optional trailing "Revised?" flag are ignored."""
    rows = []
    for line in text.splitlines():
        tok = line.split()
        # data rows: integer year, then decimal values (excludes the "1 2 3 4 5" column-number line)
        if len(tok) >= 4 and tok[0].isdigit() and '.' in tok[1]:
            rows.append((int(tok[0]), float(tok[1]), float(tok[2]), float(tok[3])))
    return pd.DataFrame(rows, columns=['year', 'nh', 'sh', 'sulfate'])


def load_gao2008(url: str = GAO2008_URL) -> pd.DataFrame:
    """Gao et al. (2008) eruption list, read from `url` (remote by default).

    Consecutive years with non-zero injection are merged into one eruption,
    dated to its first year.

    Returns
    -------
    pd.DataFrame
        One row per eruption, chronological, with columns ``yearCE``, ``nh``,
        ``sh`` and ``sulfate`` (global stratospheric sulfate injection, Tg,
        summed over the eruption's years).
    """
    with urlopen(url, timeout=60) as f:
        annual = parse_gao2008(f.read().decode('utf-8', errors='replace'))
    return gao2008_eruptions(annual)


def gao2008_eruptions(annual: pd.DataFrame) -> pd.DataFrame:
    """Group an annual IVI2 table (see parse_gao2008) into eruptions."""
    nz = annual[annual['sulfate'] > 0].sort_values('year')
    run = (nz['year'].diff() != 1).cumsum()
    ev = nz.groupby(run).agg(yearCE=('year', 'min'), nh=('nh', 'sum'),
                             sh=('sh', 'sum'), sulfate=('sulfate', 'sum'))
    return ev.reset_index(drop=True)


def largest_eruptions(n: int = 20, start: int = None, end: int = None,
                      source: str = 'evolv2k') -> pd.DataFrame:
    """The `n` largest eruptions with ``start <= yearCE <= end``.

    Parameters
    ----------
    source : {'evolv2k', 'gao2008'}
        Eruption record: eVolv2k v3 ranked by VSSI (TgS), or Gao et al.
        (2008) ranked by global sulfate injection (Tg; read remotely).

    Returned in chronological order. Either bound may be None (no bound).
    """
    if source == 'evolv2k':
        ev, size = load_evolv2k(), 'vssi'
    elif source == 'gao2008':
        ev, size = load_gao2008(), 'sulfate'
    else:
        raise ValueError(f"source must be 'evolv2k' or 'gao2008', got {source!r}")
    if start is not None:
        ev = ev[ev['yearCE'] >= start]
    if end is not None:
        ev = ev[ev['yearCE'] <= end]
    return ev.nlargest(n, size).sort_values('yearCE').reset_index(drop=True)


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


def sea_ensemble(values, years, event_years, before: int = 5, after: int = 10) -> np.ndarray:
    """Composite SEA response of every member of an ensemble.

    Applies the same windowing and baseline as `sea` to each member
    independently, then averages over events.

    Parameters
    ----------
    values : array-like, shape (n_years, n_members)
        Annual values, one column per ensemble member.
    years : array-like of int, shape (n_years,)
        Year of each row. Need not be consecutive; missing years are NaN.
    event_years, before, after
        As in `sea`.

    Returns
    -------
    np.ndarray, shape (n_members, before + after + 1)
        Per-member composite (mean over events, NaN-skipping), lags
        ``-before .. +after`` along the second axis.
    """
    if before < 1:
        raise ValueError("before must be >= 1: the pre-event years form the baseline")
    values = np.asarray(values, dtype=float)
    years = np.asarray(years).astype(int)
    if values.ndim != 2 or values.shape[0] != len(years):
        raise ValueError("values must be 2-D with one row per year")
    lags = np.arange(-before, after + 1)
    row_of = {y: i for i, y in enumerate(years)}
    wins = np.full((len(event_years), len(lags), values.shape[1]), np.nan)
    for k, e in enumerate(event_years):
        for j, lag in enumerate(lags):
            i = row_of.get(int(e) + lag)
            if i is not None:
                wins[k, j] = values[i]
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN slices -> NaN
        wins -= np.nanmean(wins[:, :before], axis=1, keepdims=True)
        return np.nanmean(wins, axis=0).T
