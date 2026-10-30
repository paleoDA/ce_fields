"""
Extract annual near-surface air temperature from all PMIP4 model simulations
at the locations of 4 Antarctic ice-core boreholes (Lyu et al., 2020).

Sites (Table 1 of Lyu et al., 2020, Clim. Past 16, 1411–1428):
    WAIS         79°28′ S,   112°05′ W
    Larissa      66°02′ S,    64°04′ W
    Mill Island  65°33′26″ S, 100°47′11″ E
    Styx         73°51.10′ S, 163°41.22′ E

For each model, ce_fields.products.load_corrected_tas() loads the past1000/
past2k run and stitches it with the CMIP6 historical run (1850–2014/2005)
where needed — the same pipeline used throughout ce_fields. Only
ACCESS-ESM1-5 and CESM2 carry a pre-1850 stitch-bias correction, and it is
already baked into their archived data by code/apply_1850_adjustments.py;
no correction is applied here at runtime for any model.

Output
------
results/data/Antarctic_BH_surface_temp_PMIP4.nc
    xr.DataArray  dims (model, site, year)
    Units: K (native model output)

Run from the scripts/ directory:
    python Antarctic_BH_surface_temp.py
"""

import sys
import time
import datetime
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.insert(0, '..')
from ce_fields.data_config import DataConfig
from ce_fields.pmip4_config import get_pmip4_models
from ce_fields.products import load_corrected_tas


def _ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

def _elapsed(t0):
    return f"{time.time() - t0:.1f}s"


# =============================================================================
# SITE DEFINITIONS  (Lyu et al., 2020, Table 1)
# Longitudes stored in 0–360° convention; converted to model convention at
# query time if the model grid uses –180 to +180.
# =============================================================================

SITES = {
    #              lat (°N)    lon (0–360°E)
    'WAIS':        (-79 - 28/60,          360 - (112 + 5/60)),
    'Larissa':     (-66 - 2/60,           360 - (64  + 4/60)),
    'Mill Island': (-(65 + 33/60 + 25.84/3600),  100 + 47/60 + 11.44/3600),
    'Styx':        (-(73 + 51.10/60),             163 + 41.22/60),
}

SITE_NAMES = list(SITES.keys())
SITE_LATS  = np.array([SITES[s][0] for s in SITE_NAMES])
SITE_LONS  = np.array([SITES[s][1] for s in SITE_NAMES])   # 0–360°E

# =============================================================================
# CONFIGURATION
# =============================================================================

config = DataConfig()
print(f"[{_ts()}] Environment : {config.environment}")

PMIP4_MODELS = get_pmip4_models(config)

DATA_FOLDER = Path('../results/data')
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"[{_ts()}] Sites       : {SITE_NAMES}")
print(f"[{_ts()}] Models      : {list(PMIP4_MODELS.keys())}")

# =============================================================================
# EXTRACTION LOOP
# =============================================================================
print(f"\n[{_ts()}] ── Extracting site temperatures ─────────────────────────────")

per_model = []

for name in PMIP4_MODELS:
    t_model = time.time()
    print(f"\n[{_ts()}]   {name}")

    try:
        print(f"[{_ts()}]     loading + stitching …", end='', flush=True)
        t0 = time.time()
        tas_annual = load_corrected_tas(name, config=config)
        label = tas_annual.attrs['display_name']
        tag = f" (pre-corrected archive, labelled {label})" if tas_annual.attrs.get('corrected') else ""
        print(f" done{tag} ({_elapsed(t0)})", flush=True)

        year0, year1 = int(tas_annual.year.values[0]), int(tas_annual.year.values[-1])
        print(f"[{_ts()}]     time axis: {year0}–{year1} CE  ({len(tas_annual.year)} years)")

        # ── nearest-neighbour extraction ────────────────────────────────────
        print(f"[{_ts()}]     extracting {len(SITE_NAMES)} sites …", end='', flush=True)
        t0 = time.time()

        # Harmonise query longitudes to model grid convention
        lon_min = float(tas_annual.lon.min())
        if lon_min < -90:
            query_lons = np.where(SITE_LONS > 180, SITE_LONS - 360, SITE_LONS)
        else:
            query_lons = SITE_LONS

        site_lat_da = xr.DataArray(SITE_LATS,  dims='site', coords={'site': SITE_NAMES})
        site_lon_da = xr.DataArray(query_lons, dims='site', coords={'site': SITE_NAMES})

        tas_sites = tas_annual.sel(lat=site_lat_da, lon=site_lon_da, method='nearest')
        tas_sites = tas_sites.load()
        # Some models carry a scalar 'height' coord (e.g. 2m) for tas; others
        # don't. Irrelevant here, and inconsistent presence breaks xr.concat.
        tas_sites = tas_sites.drop_vars('height', errors='ignore')
        print(f" done ({_elapsed(t0)})", flush=True)

        # Wrap in a DataArray labelled by model (display_name distinguishes
        # corrected runs, e.g. 'CESM2' -> 'CESM2c')
        da = tas_sites.assign_coords(model=label).expand_dims('model')
        per_model.append(da)
        print(f"[{_ts()}]     {name} complete ({_elapsed(t_model)})")

    except Exception as e:
        print(f"\n[{_ts()}]   ERROR processing {name}: {e}")

# =============================================================================
# COMBINE AND SAVE
# =============================================================================
print(f"\n[{_ts()}] ── Combining {len(per_model)} models ──────────────────────────────────")

combined = xr.concat(per_model, dim='model', join='outer')

# Attach site lat/lon as non-dimension coordinates
combined = combined.assign_coords(
    site_lat=('site', SITE_LATS),
    site_lon=('site', SITE_LONS),
)
combined.attrs = {
    'title':       'Annual near-surface air temperature at 4 Antarctic borehole sites',
    'sites':       ', '.join(SITE_NAMES),
    'source':      'PMIP4 past1000 + CMIP6 historical; sites from Lyu et al. (2020) Table 1',
    'units':       'K',
    'lon_convention': '0–360°E',
}
combined.name = 'tas'

out_path = DATA_FOLDER / 'Antarctic_BH_surface_temp_PMIP4.nc'
out_path.unlink(missing_ok=True)
combined.to_netcdf(out_path)

models_done = combined.model.values.tolist()
year0 = int(combined.year.values[0])
year1 = int(combined.year.values[-1])
print(f"[{_ts()}] Saved: {out_path}")
print(f"[{_ts()}] Shape : {dict(zip(combined.dims, combined.shape))}")
print(f"[{_ts()}] Models: {models_done}")
print(f"[{_ts()}] Years : {year0}–{year1} CE")

print(f"\n[{_ts()}] ── Pipeline complete ─────────────────────────────────────────")
