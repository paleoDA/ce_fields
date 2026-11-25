"""
PMIP4_LMR_volcanic_SEA.py
=========================
Superposed epoch analysis (SEA) of annual GMST around the largest eVolv2k
eruptions, for every PMIP4 model and the three LMR versions -- a test of
whether each product shows post-volcanic cooling (issues #4, #5: ACCESS-ESM1-5
and EC-Earth3-Veg-LR).

Method (as in the LMR volcanic analysis, cf. Fig. 4 of the LMR volcanic GRL
paper): for each eruption take the window of lags -5..+10 years around the
eruption year, subtract the mean of the 5 pre-eruption years, and average
over eruptions (ce_fields.volcanic.sea).

Eruptions: the N_EVENTS largest eVolv2k v3 eruptions by VSSI whose whole
window lies within both the PMIP4 paleo runs (to 1849; no stitch to CMIP6
historical inside a window) and LMR Online (from 1000 CE), i.e. lag-0 years
1005-1839. The same eruptions are used for every product. Lag 0 is the
eVolv2k eruption year.

Forcing caveat: all PMIP4 runs are treated as eVolv2k-forced. CESM1.1 (the
CESM Last Millennium Ensemble) and iCESM were forced with Gao et al. (2008)
instead, whose eruption years can differ by a year from eVolv2k's.

Input
-----
ce_fields/data/PMIP4_GMST_anomalies.csv  (packaged PMIP4 annual GMST)
ce_fields/data/eVolv2k_v3/               (packaged eruption list)
LMR GMST ensembles (lmr_*_gmst keys in data_config.yml; ensemble means used)

Output
------
results/figures/PMIP4_LMR_volcanic_SEA.pdf
    (a) PMIP4 models, (b) LMR v2.1 / Online / Seasonal; one composite line each.
results/tables/PMIP4_LMR_volcanic_SEA.csv
    The composites (rows: product, columns: lag), plus the eruption list in
    results/tables/PMIP4_LMR_volcanic_SEA_eruptions.csv.

Run from the code/ directory:
    python PMIP4_LMR_volcanic_SEA.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from ce_fields import DataConfig, LMR, display_name
from ce_fields.pmip4_config import get_model_colors
from ce_fields.stitch_corrections import load_gmst_anomalies
from ce_fields.volcanic import largest_eruptions, sea

FIGURE_FOLDER = Path('../results/figures')
TABLE_FOLDER = Path('../results/tables')
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
TABLE_FOLDER.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMETERS
# =============================================================================
BEFORE, AFTER = 5, 10            # SEA window: lags -BEFORE..+AFTER; baseline = lags -BEFORE..-1
N_EVENTS = 20                    # number of eruptions, as in the 2017 LMR analysis
EVENT_RANGE = (1000 + BEFORE,    # LMR Online starts in 1000 CE
               1849 - AFTER)     # PMIP4 paleo runs end in 1849/1850

LMR_VERSIONS = {                 # LMR version -> (label, color key in data_config.yml)
    'v2.1': ('LMR v2.1', 'lmr_v2_1'),
    'online': ('LMR Online', 'lmr_online'),
    'seasonal': ('LMR Seasonal', 'lmr_seasonal'),
}

config = DataConfig()
MODEL_COLORS = get_model_colors(config)

# =============================================================================
# ERUPTIONS
# =============================================================================
erup = largest_eruptions(N_EVENTS, *EVENT_RANGE)
event_years = erup['yearCE'].tolist()
print(f"{len(event_years)} largest eVolv2k eruptions in {EVENT_RANGE[0]}-{EVENT_RANGE[1]}:")
print(erup[['yearCE', 'month', 'lat', 'vssi']].to_string(index=False))
erup.to_csv(TABLE_FOLDER / 'PMIP4_LMR_volcanic_SEA_eruptions.csv', index=False)

# =============================================================================
# SEA: PMIP4 models
# =============================================================================
gmst = load_gmst_anomalies()
composites = {}
pmip_labels = []
for model in gmst.columns:
    label = display_name(model)          # the CSV holds the corrected ACCESS-ESM1-5/CESM2 data
    composites[label] = sea(gmst[model], event_years, BEFORE, AFTER).mean(axis=0)
    pmip_labels.append(label)

# =============================================================================
# SEA: LMR versions (ensemble-mean GMST)
# =============================================================================
lmr_labels = []
for version, (label, _) in LMR_VERSIONS.items():
    da = LMR.from_version(version, variable='gmst').data
    ydim = 'year' if 'year' in da.dims else 'time'
    gm = da.mean([d for d in da.dims if d != ydim])
    s = pd.Series(gm.values, index=gm[ydim].values.astype(int))
    composites[label] = sea(s, event_years, BEFORE, AFTER).mean(axis=0)
    lmr_labels.append(label)

comp = pd.DataFrame(composites).T
comp.index.name = 'product'
comp.to_csv(TABLE_FOLDER / 'PMIP4_LMR_volcanic_SEA.csv', float_format='%.4f')

print(f"\nComposite GMST response (°C, relative to lags -{BEFORE}..-1):")
summary = pd.DataFrame({
    'lag 0': comp[0],
    'lag 1': comp[1],
    'min': comp.loc[:, 0:AFTER].min(axis=1),
    'lag of min': comp.loc[:, 0:AFTER].idxmin(axis=1),
    'mean lags 0-2': comp.loc[:, 0:2].mean(axis=1),
})
print(summary.round(3).to_string())

# =============================================================================
# PLOT
# =============================================================================
lags = comp.columns.values
panels = [('(a)', 'PMIP4', pmip_labels, lambda l: MODEL_COLORS.get(l, 'gray')),
          ('(b)', 'LMR', lmr_labels,
           lambda l: config.get_plot_color(dict((v[0], v[1]) for v in LMR_VERSIONS.values())[l]))]

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (tag, title, labels, color_of) in zip(axs, panels):
    for label in labels:
        ax.plot(lags, comp.loc[label], color=color_of(label), linewidth=2, label=label)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_xlabel('Years relative to eruption year')
    ax.set_title(f'{tag} {title}', fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axs[0].set_ylabel(f'GMST anomaly (°C, rel. to {BEFORE} pre-eruption years)')
fig.suptitle(f'Superposed epoch analysis: {len(event_years)} largest eVolv2k eruptions, '
             f'{EVENT_RANGE[0]}–{EVENT_RANGE[1]} CE', fontsize=12)
fig.tight_layout()

out = FIGURE_FOLDER / 'PMIP4_LMR_volcanic_SEA.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out.resolve()}")
