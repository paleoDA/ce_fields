# ce_fields

Common Era climate fields to be validated, in one place so that several
validation projects can share them without duplicating loading logic:

- **Simulations:** loading, stitching, and bias-correcting PMIP4 past1000/past2k
  and CMIP6-historical runs (`products`, `pmip4_config`, `io`).
- **Reconstructions:** the Last Millennium Reanalysis, v2.1, Online and Seasonal
  (`lmr`), and the six Neukom et al. (2019) field reconstructions (`neukom`).

Formerly `ce_simul`, which held the simulations only; renamed when the LMR and
Neukom2019 loaders moved here from `ce_val`.

## Install

```bash
pip install -e .
```

## Usage

```python
from ce_fields import DataConfig, load_corrected_tas

config = DataConfig()
tas = load_corrected_tas('ACCESS-ESM1-5', config=config)  # stitched annual tas, 1850-bias-corrected by default
print(tas.attrs['display_name'])  # 'ACCESS-ESM1-5c' -- use this label in plots/tables

# To audit against the original, uncorrected data instead:
raw = load_corrected_tas('ACCESS-ESM1-5', config=config, use_corrected=False)
print(raw.attrs['display_name'])  # 'ACCESS-ESM1-5'
```

Only ACCESS-ESM1-5 and CESM2 have a registered pre-1850 correction (see
`ce_fields.pmip4_config.CORRECTED_MODELS`); `use_corrected` has no effect for
any other model. Always read the label from `attrs['display_name']` rather
than hardcoding it, so plots/tables stay honest about which series were
corrected.

See `ce_fields/products.py` for the full API (`load_stitched_tas`, `load_corrected_tas`, `compute_gmst`).

Reconstructions:

```python
from ce_fields.lmr import LMR
from ce_fields.neukom import Neukom2019
```

## Data configuration

`data_config.yml` at the repo root defines PMIP4/CMIP6-historical, LMR and
Neukom2019 file locations per environment (laptop/HPC), auto-detected or
overridden via the `DATA_ENV` environment variable. Update the `pmip4_data` /
`cmip_hist_data` / `lmr_data` roots for your environment before running scripts
that touch real data.

## Repository layout

- `ce_fields/` — installable package (config, I/O, stitching, bias correction, high-level `products` API; LMR and Neukom2019 loaders)
- `scripts/` — diagnostic/pipeline scripts (`PMIP4_GMST_sanity_check.py`, `check_pmip4_zarr.py`, `Antarctic_BH_surface_temp.py`)
- `notebooks/` — `PMIP4_1850_stitch.ipynb`, the calibration notebook deriving the pre-1850 bias corrections
- `tests/` — pytest suite (no HPC data required)
- `results/` — checked-in data products (`PMIP4_GMST_anomalies.csv`, stitching diagnostic figures)
