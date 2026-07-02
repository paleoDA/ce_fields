"""Central registry of PMIP4 model metadata.

All scripts and notebooks import from here.  Adding a model, changing a
calibration constant, or registering a new path override requires editing
only this file.
"""

# ---------------------------------------------------------------------------
# Path-independent metadata
# ---------------------------------------------------------------------------

# Surface-temperature variable name (default 'tas'; native CAM output uses 'TREFHT')
MODEL_VAR = {
    'CESM2':   'TREFHT',
    'CESM1.1': 'TREFHT',
}

# Offset to convert model-internal year to CE year (default 0)
YEAR_OFFSETS = {
    'MPI-ESM1-2-LR': 7001,   # model years 7001–8850 → CE 0–1849
}

# PMIP4 paleo runs whose last CE year is below this threshold need a CMIP6
# historical stitch to reach the present.  1900 gives comfortable margin:
# some past1000 runs end at 1850 (not 1849), so 1851 would misclassify them;
# no self-contained past2k/combined run ends before ~2000.
STITCH_THRESHOLD = 1900

# Drop N years from the end of the paleo run before use (spurious tail values)
TRIM_END = {
    'CESM1.1': 1,
}

# Pre-1850 past1000 stitch corrections (flat offset for ACCESS-ESM1-5,
# EC-Earth3-Veg-LR, MRI-ESM2-0; linear detrend for CESM2) are computed
# directly from results/tables/PMIP4_GMST_anomalies.csv by
# scripts/PMIP4_tas_1850_adjustments.py, which stays in sync with
# notebooks/PMIP4_1850_stitch.ipynb. See apply_1850_adjustment() there.

# ---------------------------------------------------------------------------
# Path-dependent registries (require a DataConfig instance)
# ---------------------------------------------------------------------------

def get_pmip4_models(config):
    """Return alphabetically-sorted {model: path} dict, auto-discovered from data_config.yml.

    Any file key with prefix 'pmip4_' is included; the model name is the suffix.
    To register a new model, add a 'pmip4_<ModelName>' entry to data_config.yml — no
    changes to this file are needed.
    """
    prefix = 'pmip4_'
    models = {
        k[len(prefix):]: config.get_file_path(k)
        for k in config.config['files']
        if k.startswith(prefix)
    }
    return dict(sorted(models.items(), key=lambda x: x[0].lower()))


def get_cmip_hist_models(config):
    """Return {model: path} dict for available CMIP6 historical runs, auto-discovered from data_config.yml.

    Any file key with prefix 'cmip_hist_' is included; the model name is the suffix.
    Whether a given model is actually stitched is determined at runtime from
    _pmip4_end_year < STITCH_THRESHOLD — not from membership in this dict.
    """
    prefix = 'cmip_hist_'
    return {
        k[len(prefix):]: config.get_file_path(k)
        for k in config.config['files']
        if k.startswith(prefix)
    }
