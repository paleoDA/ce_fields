"""
PMIP4_LMR_volcanic_SEA.py
=========================
Superposed epoch analysis (SEA) of annual GMST around the largest eVolv2k
eruptions, for every PMIP4 model and the three LMR versions -- a test of
whether each product shows post-volcanic cooling (issues #4, #5: ACCESS-ESM1-5
and EC-Earth3-Veg-LR).

Method (as in the LMR volcanic analysis, cf. Fig. 4 of the LMR volcanic GRL
paper): for each eruption take the window of lags -3..+10 years around the
eruption year, subtract the mean of its 3 pre-eruption years, and average
over eruptions (ce_fields.volcanic.sea).

Eruptions: the N_EVENTS largest eruptions of each volcanic forcing record
whose whole window lies within both the PMIP4 paleo runs (to 1849; no stitch
to CMIP6 historical inside a window) and LMR Online (from 1000 CE), i.e.
lag-0 years 1005-1839. Lag 0 is the eruption year of that record.
  - eVolv2k v3 (ranked by VSSI): the PMIP4 protocol forcing, used for most
    PMIP4 runs and for all LMR versions.
  - Gao et al. (2008) IVI2 (ranked by global sulfate injection; read
    remotely): for the runs forced with it, CESM1.1 and iCESM
    (pmip4_config.VOLCANIC_FORCING).

LMR: the SEA is applied to every ensemble member. The line is the composite
of the ensemble-mean GMST; the envelope spans the 2.5-97.5% quantiles of the
member composites.

Input
-----
ce_fields/data/PMIP4_GMST_anomalies.csv  (packaged PMIP4 annual GMST)
ce_fields/data/eVolv2k_v3/               (packaged eruption list)
Gao et al. (2008) IVI2, ce_fields.volcanic.GAO2008_URL (network access needed)
LMR GMST ensembles (lmr_*_gmst keys in data_config.yml)

Output
------
results/figures/PMIP4_LMR_volcanic_SEA.pdf
    (a) PMIP4 models, (b) LMR v2.1 / Online / Seasonal with 95% envelopes.
results/tables/PMIP4_LMR_volcanic_SEA.csv
    The composites (rows: product, columns: lag), with the LMR quantiles as
    '<product> q2.5' / '<product> q97.5' rows, plus the eruption lists
    (column 'forcing') in results/tables/PMIP4_LMR_volcanic_SEA_eruptions.csv.

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
from ce_fields.pmip4_config import get_model_colors, VOLCANIC_FORCING
from ce_fields.stitch_corrections import load_gmst_anomalies
from ce_fields.volcanic import largest_eruptions, sea, sea_ensemble

FIGURE_FOLDER = Path('../results/figures')
TABLE_FOLDER = Path('../results/tables')
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
TABLE_FOLDER.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMETERS
# =============================================================================
BEFORE, AFTER = 3, 10            # SEA window: lags -BEFORE..+AFTER; baseline = lags -BEFORE..-1
N_EVENTS = 20                    # number of eruptions, as in the 2017 LMR analysis
EVENT_RANGE = (1000 + BEFORE,    # LMR Online starts in 1000 CE
               1849 - AFTER)     # PMIP4 paleo runs end in 1849/1850
QUANTILES = (2.5, 97.5)          # LMR ensemble envelope (central 95%)
ENVELOPE_ALPHA = 0.2

LMR_VERSIONS = {                 # LMR version -> (label, color key in data_config.yml)
    'v2.1': ('LMR v2.1', 'lmr_v2_1'),
    'online': ('LMR Online', 'lmr_online'),
    'seasonal': ('LMR Seasonal', 'lmr_seasonal'),
}
FORCING_NAMES = {'evolv2k': 'eVolv2k v3', 'gao2008': 'Gao et al. (2008)'}

config = DataConfig()
MODEL_COLORS = get_model_colors(config)

# =============================================================================
# ERUPTIONS (one list per forcing record)
# =============================================================================
eruptions = {f: largest_eruptions(N_EVENTS, *EVENT_RANGE, source=f) for f in FORCING_NAMES}
events = {f: ev['yearCE'].tolist() for f, ev in eruptions.items()}
for f, ev in eruptions.items():
    print(f"{len(ev)} largest {FORCING_NAMES[f]} eruptions in {EVENT_RANGE[0]}-{EVENT_RANGE[1]}: "
          f"{events[f]}")
pd.concat([ev.assign(forcing=f) for f, ev in eruptions.items()], ignore_index=True) \
  .to_csv(TABLE_FOLDER / 'PMIP4_LMR_volcanic_SEA_eruptions.csv', index=False)

# =============================================================================
# SEA: PMIP4 models, each composited on the eruptions of its own forcing
# =============================================================================
gmst = load_gmst_anomalies()
composites = {}
pmip_labels = []
for model in gmst.columns:
    label = display_name(model)          # the CSV holds the corrected ACCESS-ESM1-5/CESM2 data
    forcing = VOLCANIC_FORCING.get(model, 'evolv2k')
    composites[label] = sea(gmst[model], events[forcing], BEFORE, AFTER).mean(axis=0)
    pmip_labels.append(label)

# =============================================================================
# SEA: LMR versions -- ensemble-mean composite + member-composite quantiles
# =============================================================================
lmr_labels = []
envelopes = {}
for version, (label, _) in LMR_VERSIONS.items():
    da = LMR.from_version(version, variable='gmst').data
    ydim = 'year' if 'year' in da.dims else 'time'
    years = da[ydim].values.astype(int)
    members = da.transpose(ydim, ...).values.reshape(len(years), -1)   # (year, member)
    s = pd.Series(np.nanmean(members, axis=1), index=years)
    composites[label] = sea(s, events['evolv2k'], BEFORE, AFTER).mean(axis=0)
    member_comp = sea_ensemble(members, years, events['evolv2k'], BEFORE, AFTER)
    lo, hi = np.nanpercentile(member_comp, QUANTILES, axis=0)
    envelopes[label] = (lo, hi)
    composites[f'{label} q{QUANTILES[0]:g}'] = pd.Series(lo, index=composites[label].index)
    composites[f'{label} q{QUANTILES[1]:g}'] = pd.Series(hi, index=composites[label].index)
    print(f"{label}: {members.shape[1]} members")
    lmr_labels.append(label)

comp = pd.DataFrame(composites).T
comp.index.name = 'product'
comp.to_csv(TABLE_FOLDER / 'PMIP4_LMR_volcanic_SEA.csv', float_format='%.4f')

print(f"\nComposite GMST response (°C, relative to lags -{BEFORE}..-1):")
main = comp.loc[pmip_labels + lmr_labels]
summary = pd.DataFrame({
    'forcing': [VOLCANIC_FORCING.get(m, 'evolv2k') for m in gmst.columns] + ['evolv2k'] * len(lmr_labels),
    'lag 0': main[0],
    'lag 1': main[1],
    'min': main.loc[:, 0:AFTER].min(axis=1),
    'lag of min': main.loc[:, 0:AFTER].idxmin(axis=1),
    'mean lags 0-2': main.loc[:, 0:2].mean(axis=1),
}, index=main.index)
print(summary.round(3).to_string())

# =============================================================================
# PLOT
# =============================================================================
lags = comp.columns.values
lmr_color = {label: config.get_plot_color(key) for label, key in LMR_VERSIONS.values()}
gao_models = {display_name(m) for m, f in VOLCANIC_FORCING.items() if f == 'gao2008'}
panels = [('(a)', 'PMIP4', pmip_labels, lambda l: MODEL_COLORS.get(l, 'gray')),
          ('(b)', 'LMR', lmr_labels, lambda l: lmr_color[l])]

fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, (tag, title, labels, color_of) in zip(axs, panels):
    ax.grid(True, color='lightgray', linewidth=0.6)
    ax.set_axisbelow(True)
    for label in labels:
        if label in envelopes:
            ax.fill_between(lags, *envelopes[label], color=color_of(label),
                            alpha=ENVELOPE_ALPHA, linewidth=0)
        legend_label = f'{label} (Gao)' if label in gao_models else label
        ax.plot(lags, comp.loc[label], color=color_of(label), linewidth=2, label=legend_label)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_xlabel('Years relative to eruption year')
    ax.set_title(f'{tag} {title}', fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axs[0].set_ylabel(f'GMST anomaly (°C, rel. to {BEFORE} pre-eruption years)')
axs[1].text(0.02, 0.02, f'Envelopes: {QUANTILES[0]:g}–{QUANTILES[1]:g}% of ensemble-member composites',
            transform=axs[1].transAxes, fontsize=8, color='dimgray')
fig.suptitle(f'Superposed epoch analysis: {N_EVENTS} largest eruptions, '
             f'{EVENT_RANGE[0]}–{EVENT_RANGE[1]} CE', fontsize=16, fontweight='bold', y=0.995)
fig.text(0.5, 0.935, 'eVolv2k v3 eruptions; Gao et al. (2008) for the runs it forced (CESM1.1, iCESM)',
         ha='center', fontsize=10, color='dimgray')
fig.tight_layout(rect=(0, 0, 1, 0.92))

out = FIGURE_FOLDER / 'PMIP4_LMR_volcanic_SEA.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out.resolve()}")
