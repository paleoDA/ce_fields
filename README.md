# ce_simul

Loading, stitching, and bias-correcting PMIP4 past1000/past2k and CMIP6-historical
simulations. Split out of [ce_val](https://github.com/paleoDA/ce_val) so its
outputs (corrected annual tas fields, GMST tables) can be reused across projects
without duplicating the loading/stitching logic.

## Install

```bash
pip install -e .
```

## Usage

```python
from ce_simul import DataConfig, load_corrected_tas

config = DataConfig()
tas = load_corrected_tas('ACCESS-ESM1-5', config=config)  # stitched + 1850-bias-corrected annual tas
```

See `ce_simul/products.py` for the full API (`load_stitched_tas`, `load_corrected_tas`, `compute_gmst`).

## Data configuration

`data_config.yml` at the repo root defines PMIP4/CMIP6-historical file locations
per environment (laptop/HPC), auto-detected or overridden via the `DATA_ENV`
environment variable. Update the `pmip4_data` / `cmip_hist_data` roots for your
environment before running scripts that touch real data.

## Repository layout

- `ce_simul/` — installable package (config, I/O, stitching, bias correction, high-level `products` API)
- `scripts/` — diagnostic/pipeline scripts (`PMIP4_GMST_sanity_check.py`, `check_pmip4_zarr.py`, `Antarctic_BH_surface_temp.py`)
- `notebooks/` — `PMIP4_1850_stitch.ipynb`, the calibration notebook deriving the pre-1850 bias corrections
- `tests/` — pytest suite (no HPC data required)
- `results/` — checked-in data products (`PMIP4_GMST_anomalies.csv`, stitching diagnostic figures)
