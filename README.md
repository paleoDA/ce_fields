# ce_fields

Utilities to access Common Era climate fields, to enable several
projects can share them without duplicating loading logic:

- **Simulations:** loading, stitching, and bias-correcting PMIP4 past1000/past2k
  and CMIP6-historical runs (`products`, `pmip4_config`, `io`).
- **Reconstructions:** the Last Millennium Reanalysis, v2.1, Online and Seasonal
  (`lmr`), and the six Neukom et al. (2019) field reconstructions (`neukom`).

## Install

With conda (pinned, tested versions; includes the regridding and script
dependencies):

```bash
conda env create -f environment.yml
conda activate ce_fields
```

or with pip only (minimum versions):

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

## Data

The fields themselves are not distributed with this package: they are large
and published elsewhere. To work with them, download local copies and tell
`ce_fields` where they are (see [Data configuration](#data-configuration)).
Repositories such as
[borehole_val](https://github.com/LinkedEarth/borehole_val) archive the
intermediate products derived from these fields, so their figures and tables
can be rebuilt without them.

| Dataset | Root key | Files used | Size | Source |
|---|---|---|---|---|
| PMIP4 `past1000`/`past2k` and CMIP6 `historical` monthly `tas` (9 models) | `pmip4_data`, `cmip_hist_data` | see `ce_fields/data_config.yml` | several GB per model and experiment | [ESGF](https://esgf.llnl.gov/) |
| LMR v2.1 ([Tardif et al., 2019](https://doi.org/10.5194/cp-15-1251-2019)) | `lmr_data` | `LMRv2.1/`: gridded ensemble mean and spread, GMST ensemble | 2.3 GB (GMST only: 12 MB) | NCEI Paleoclimatology, <https://www.atmos.uw.edu/~hakim/lmr/> |
| LMR Online ([Perkins and Hakim, 2021](https://doi.org/10.1029/2020PA003959)) | `lmr_data` | `LMRonline/…/`: gridded ensemble mean and variance, GMST indices | 1.4 GB (GMST only: 153 MB) | NCEI Paleoclimatology, <https://www.atmos.uw.edu/~hakim/lmr/> |
| LMR Seasonal ([Meng et al., 2025](https://doi.org/10.1175/JCLI-D-25-0048.1)) | `lmr_data` | `LMR_seasonal/`: gridded ensemble mean, GMST ensemble | 0.6 GB (GMST only: 29 MB) | <https://atmos.uw.edu/~zilumeng/LMR_Seasonal/data/> |
| Neukom et al. (2019) gridded reconstructions, 6 methods × 100 members ([doi](https://doi.org/10.1038/s41586-019-1401-2)) | `neukom_data` | `AM.nc`, `CCA.nc`, `CPS.nc`, `DA.nc`, `GraphEM.nc`, `PCR.nc` | 15 GB | [NCEI](https://www.ncei.noaa.gov/pub/data/paleo/pages2k/neukom2019temp/) |

The PMIP4/CMIP6 output used here (on USC's CARC cluster) amounts to several GB
per model and experiment, so regenerating products from it requires a serious
commitment of storage and compute, in practice on an HPC cluster. It is stored
as zarr or netCDF under a shared root; the exact path of each model's file for CommonClimate is in
`ce_fields/data_config.yml`. The annual GMST derived from it ships with the
package (`ce_fields/data/PMIP4_GMST_anomalies.csv`), together with the
pre-1850 stitch corrections derived from that table
(`ce_fields.stitch_corrections`), so GMST-based analyses need none of the raw
output.

## Data configuration

`ce_fields/data_config.yml` (shipped with the package) registers every file,
with `/path/to/...` placeholders for the external roots above. To make it work for you, set the roots
that exist on your machine in an override file, which is merged on top of it
key by key:

```yaml
# data_config.local.yml
environments:
  laptop:          # or hpc
    paths:
      lmr_data: "/my/data/LMR"
      neukom_data: "/my/data/Neukom_Nature2019"
```

Put it either next to `ce_fields/data_config.yml` in a clone (the name
`data_config.local.yml` is gitignored), or anywhere, and point the
`CE_FIELDS_LOCAL_CONFIG` environment variable at it. The latter also works
when ce_fields was installed with `pip`. Asking for a file whose root is still
a placeholder raises an error naming the dataset; `DataConfig().list_files()`
shows which files are found. The environment (`laptop` or `hpc`) is detected
from the hostname, or set with the `DATA_ENV` environment variable.

## Repository layout

- `ce_fields/` — installable package (config, I/O, stitching, bias correction, high-level `products` API; LMR and Neukom2019 loaders), with `data_config.yml` and the packaged GMST table in `data/`
- `code/` — pipeline and diagnostic scripts (`PMIP4_GMST_sanity_check.py`, `apply_1850_adjustments.py`, `check_pmip4_zarr.py`, `Antarctic_BH_surface_temp.py`), and `PMIP4_1850_stitch.ipynb`, the calibration notebook deriving the pre-1850 bias corrections
- `tests/` — pytest suite (no HPC data required)
- `results/` — checked-in figures and tables (stitching diagnostics, CESM2 GMST)

## Citation

Citation metadata are in `CITATION.cff`.

## License

MIT License (see LICENSE file).
