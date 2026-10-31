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

# Pre-1850 past1000 stitch corrections (flat offset for ACCESS-ESM1-5;
# linear detrend for CESM2 — the only 2 models whose 1849->1850 jump exceeds
# noise, see code/PMIP4_1850_stitch.ipynb) were computed from
# results/tables/PMIP4_GMST_anomalies.csv and applied in-place to the
# archived data by code/apply_1850_adjustments.py. load_stitched_tas() reads
# the corrected files by default (use_corrected=True); pass
# use_corrected=False to read the original raw data instead (registered
# under raw_pmip4_<model> in data_config.yml).

# Raw model name -> display name for models with a registered correction, so
# downstream repos (borehole_val, docu_val, warmth_and_trends) can label corrected series
# distinctly in plots/tables instead of silently treating them the same as
# an uncorrected run. Use display_name() below rather than reading this
# dict directly.
CORRECTED_MODELS = {
    'ACCESS-ESM1-5': 'ACCESS-ESM1-5c',
    'CESM2': 'CESM2c',
}


def display_name(model_name: str, corrected: bool = True) -> str:
    """Return the label to use for `model_name` in plots/tables/legends.

    Appends the corrected-run suffix (see CORRECTED_MODELS, e.g.
    'CESM2' -> 'CESM2c') when `corrected` is True and the model actually has
    a registered correction; otherwise returns `model_name` unchanged. Pass
    the same `corrected`/`use_corrected` value used to load the data, so the
    label always reflects what was actually plotted.
    """
    if corrected and model_name in CORRECTED_MODELS:
        return CORRECTED_MODELS[model_name]
    return model_name

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


# Per-model color overrides, applied on top of the tab10 assignment in
# get_model_colors() below. MPI-ESM1-2-LR's tab10 slot (index 7) is gray,
# which reads poorly / collides with gray-toned "individual observation"
# elements in several figures (e.g. borehole_val's stack plots), so it is
# fixed to black everywhere a per-model color is assigned from this registry.
MODEL_COLOR_OVERRIDES = {
    'MPI-ESM1-2-LR': 'black',
}


def get_model_colors(config, corrected: bool = True):
    """Return {display_name: color} for every registered PMIP4 model.

    Colors are tab10, cycled over the raw model names in alphabetical
    (case-insensitive) order -- the same assignment used by
    code/PMIP4_1850_stitch.ipynb's "Full time series" figure and
    code/plot_Antarctic_BH_surface_temp.py -- so a given model reads as the
    same color in every figure across this project and borehole_val, except for
    the fixed overrides in MODEL_COLOR_OVERRIDES. Keyed by display_name (see
    `display_name()`) so corrected models (e.g. CESM2c) get their raw
    model's color under their display label.

    Parameters
    ----------
    config : DataConfig
        Used to auto-discover the registered models (see get_pmip4_models).
    corrected : bool, optional
        Passed through to `display_name()`. Default True.
    """
    import matplotlib.pyplot as plt

    raw_models = list(get_pmip4_models(config))
    tab10 = plt.get_cmap('tab10').colors
    colors = {
        display_name(raw, corrected=corrected): tab10[i % len(tab10)]
        for i, raw in enumerate(raw_models)
    }
    colors.update(MODEL_COLOR_OVERRIDES)
    return colors


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
