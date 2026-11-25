# eVolv2k v3 volcanic forcing

`eVolv2k_v3_ds_1.nc`: ice-core-derived volcanic stratospheric sulfur injection
(VSSI) for 256 eruptions, 500 BCE–1900 CE (year, month, latitude, VSSI and its
1σ uncertainty, hemispheric asymmetry). Version 3 of
Toohey, M. and Sigl, M. (2017), *Volcanic stratospheric sulfur injections and
aerosol optical depth from 500 BCE to 1900 CE*, Earth Syst. Sci. Data 9,
809–831, [doi:10.5194/essd-9-809-2017](https://doi.org/10.5194/essd-9-809-2017).
eVolv2k is the volcanic forcing prescribed for the PMIP4 `past1000`/`past2k`
experiments (Jungclaus et al., 2017).

Load it with `ce_fields.load_evolv2k()`; `ce_fields.EVOLV2K_FILE` is its path.

## Why it lives here

This file was moved here from `warmth_and_trends` (`data/eVolv2k_v3/`) in
November 2026. More than one project now needs it: warmth_and_trends uses it to
place eruptions on LMR GMST and to find post-volcanic warming trends, and
ce_fields uses it for the superposed epoch analysis of PMIP4 and LMR GMST
(`code/PMIP4_LMR_volcanic_SEA.py`, issues #4 and #5). `ce_fields` is the shared
package that the downstream repos already depend on for their Common Era data,
so the forcing ships with it, like `PMIP4_GMST_anomalies.csv`. That way there is
one copy and one loader, instead of a copy per repo that can drift.

The companion aerosol optical depth field
(`eVolv2k_v3_EVA_AOD_-500_1900_1.nc`, 21 MB) was never tracked in
warmth_and_trends and is not used here, so it was not moved.
