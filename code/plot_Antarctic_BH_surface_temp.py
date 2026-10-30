"""
plot_Antarctic_BH_surface_temp.py
==================================
Plot near-surface air temperature at the 4 Antarctic borehole sites
(Lyu et al., 2020) for every PMIP4 model.

Uses the same per-model color assignment as the "Full time series" figure
in code/PMIP4_1850_stitch.ipynb (PMIP4_GMST_full_timeseries.pdf) — tab10
colors cycled over the raw model names in alphabetical (case-insensitive)
order — so a given model reads as the same color in both figures.
Corrected models (ACCESS-ESM1-5, CESM2) are keyed by their display_name
(e.g. 'CESM2c') to match the model coordinate written by
Antarctic_BH_surface_temp.py.

Input
-----
results/data/Antarctic_BH_surface_temp_PMIP4.nc
    Produced by code/Antarctic_BH_surface_temp.py.

Output
------
results/figures/Antarctic_BH_surface_temp_PMIP4.pdf
    2x2 grid, one panel per site, one line per model.

Run from the code/ directory:
    python plot_Antarctic_BH_surface_temp.py
"""

import sys
from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '..')
from ce_simul.data_config import DataConfig
from ce_simul.pmip4_config import get_model_colors

DATA_FOLDER = Path('../results/data')
FIGURE_FOLDER = Path('../results/figures')
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COLORS — canonical per-model registry (ce_simul.pmip4_config.get_model_colors):
# tab10 cycled alphabetically over the raw model names, keyed by display_name
# so corrected models (e.g. CESM2c) get their raw model's color, with
# MPI-ESM1-2-LR fixed to black (MODEL_COLOR_OVERRIDES) so it reads the same
# way in every figure across this project and ce_val.
# =============================================================================
config = DataConfig()
MODEL_COLORS = get_model_colors(config)

# =============================================================================
# LOAD
# =============================================================================
da = xr.open_dataarray(DATA_FOLDER / 'Antarctic_BH_surface_temp_PMIP4.nc')
tas_c = da - 273.15  # native K -> degC

sites = list(da.site.values)
site_lat = dict(zip(sites, da.site_lat.values))
site_lon = dict(zip(sites, da.site_lon.values))  # 0-360°E, per file attrs

# =============================================================================
# PLOT — one panel per site, one line per model
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axs = axs.ravel()

for ax, site in zip(axs, sites):
    sub = tas_c.sel(site=site)
    for model in sub.model.values:
        s = sub.sel(model=model)
        ax.plot(s.year, s.values, linewidth=0.8, alpha=0.75, color=MODEL_COLORS.get(model, 'gray'))
    ax.set_title(f'{site}  ({abs(site_lat[site]):.2f}°S, {site_lon[site]:.2f}°E)',
                 fontweight='bold')
    ax.set_xlabel('Year (CE)')
    ax.set_ylabel('Near-surface air temperature (°C)')
    ax.grid(True, alpha=0.3)

legend_handles = [
    Line2D([0], [0], color=color, linewidth=2, label=label)
    for label, color in MODEL_COLORS.items()
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('PMIP4 near-surface air temperature at Antarctic borehole sites',
             fontweight='bold')
fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = FIGURE_FOLDER / 'Antarctic_BH_surface_temp_PMIP4.pdf'
fig.savefig(out_path, bbox_inches='tight')
plt.close(fig)

print(f"Saved: {out_path}")
