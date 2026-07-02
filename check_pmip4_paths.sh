#!/usr/bin/env bash
# check_pmip4_paths.sh
#
# Recursively scans PMIP4_data/netCDF/ to discover the actual directory structure,
# then cross-checks it against the pmip4_* entries in data_config.yml.
#
# Usage (run on HPC):
#   bash check_pmip4_paths.sh [NETCDF_ROOT] [YML_FILE]
#
# Defaults:
#   NETCDF_ROOT = /home1/julieneg/common_data/common/PMIP4_data/netCDF
#   YML_FILE    = data_config.yml  (relative to script location)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Inline Python diagnostics for a zarr store ───────────────────────────────
# Checks: openability, time range, grid shape, global mean tas, NaN fraction.
# Pass the zarr path as $1.
check_zarr_python() {
    local zarr_path="$1"
    python3 - "${zarr_path}" <<'PYEOF'
import sys
try:
    import xarray as xr, numpy as np
except ImportError as e:
    print(f"    [SKIP] Python import failed: {e}")
    sys.exit(0)

path = sys.argv[1]
try:
    ds = xr.open_zarr(path)
    if 'tas' not in ds:
        print(f"    [ERR]  'tas' variable not found. Variables present: {list(ds.data_vars)}")
        sys.exit(0)
    tas = ds['tas']
    # Time range
    times = ds['time'].values
    t0, t1 = str(times[0])[:10], str(times[-1])[:10]
    ntime = len(times)
    # Grid shape
    shape_str = " × ".join(f"{d}={tas.sizes[d]}" for d in tas.dims)
    # Lat-weighted global mean of first time step (K)
    lat_name = next((d for d in tas.dims if 'lat' in d.lower()), None)
    if lat_name:
        weights = np.cos(np.deg2rad(ds[lat_name].values))
        t0_slice = tas.isel(time=0).values
        valid = ~np.isnan(t0_slice)
        nan_pct = (~valid).mean() * 100
        # broadcast weights to 2-D
        w2d = np.broadcast_to(weights[:, None], t0_slice.shape) if t0_slice.ndim == 2 \
              else weights
        gmst = float(np.average(t0_slice[valid], weights=w2d.ravel()[valid.ravel()]))
        print(f"    time   : {t0} → {t1}  ({ntime} monthly steps)")
        print(f"    shape  : {shape_str}")
        print(f"    GMST[0]: {gmst:.2f} K  (expect ~270–295 K for past1000)")
        print(f"    NaN%%   : {nan_pct:.1f}%%")
    else:
        print(f"    time   : {t0} → {t1}  ({ntime} monthly steps)")
        print(f"    shape  : {shape_str}")
        print(f"    NaN%%   : (lat dim not identified — skipping weighted mean)")
except Exception as e:
    print(f"    [ERR]  Failed to open zarr: {e}")
PYEOF
}
NETCDF_ROOT="${1:-/home1/julieneg/common_data/common/PMIP4_data/netCDF}"
YML_FILE="${2:-${SCRIPT_DIR}/data_config.yml}"
LOG_FILE="${SCRIPT_DIR}/pmip4_path_check_$(date +%Y%m%d_%H%M%S).txt"

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[0;34m'
NC='\033[0m'

main() {

echo -e "${BLU}=== PMIP4 path checker ===${NC}"
echo "netCDF root : ${NETCDF_ROOT}"
echo "YAML file   : ${YML_FILE}"
echo

# ── 1. Sanity checks ──────────────────────────────────────────────────────────
if [[ ! -d "${NETCDF_ROOT}" ]]; then
    echo -e "${RED}ERROR: netCDF root not found: ${NETCDF_ROOT}${NC}"
    echo "       Run this script on the HPC, or adjust NETCDF_ROOT."
    exit 1
fi
if [[ ! -f "${YML_FILE}" ]]; then
    echo -e "${RED}ERROR: YAML file not found: ${YML_FILE}${NC}"
    exit 1
fi

# ── 2. Parse HPC pmip4_data base path from yml ────────────────────────────────
# Extract the pmip4_data value from the hpc: block
HPC_PMIP4_BASE=$(awk '
    /^  hpc:/ { in_hpc=1 }
    in_hpc && /^  [a-z]/ && !/^  hpc:/ { in_hpc=0 }
    in_hpc && /pmip4_data:/ {
        match($0, /"([^"]+)"/, arr)
        print arr[1]
        exit
    }
' "${YML_FILE}" || true)

if [[ -z "${HPC_PMIP4_BASE}" ]]; then
    echo -e "${YLW}WARNING: Could not parse hpc.pmip4_data from yml. Path substitution will be skipped.${NC}"
    HPC_PMIP4_BASE="{pmip4_data}"
fi
echo -e "${BLU}HPC pmip4_data base (yml):${NC} ${HPC_PMIP4_BASE}"
echo

# ── 3. Discover actual structure under netCDF root ───────────────────────────
echo -e "${BLU}--- Actual directory tree (model / experiment / variant / var) ---${NC}"
# Expected depth from NETCDF_ROOT:
#   institution(1) / model(2) / experiment(3) / variant(4) / freq(5) / var(6) / grid(7) / version(8) / file.nc

declare -A FOUND_MODELS   # "model|experiment|variant|var" -> "institution grid version"

while IFS= read -r ncfile; do
    rel="${ncfile#${NETCDF_ROOT}/}"
    IFS='/' read -ra parts <<< "${rel}"
    if [[ ${#parts[@]} -ge 8 ]]; then
        institution="${parts[0]}"
        model="${parts[1]}"
        experiment="${parts[2]}"
        variant="${parts[3]}"
        var="${parts[5]}"
        grid="${parts[6]}"
        version="${parts[7]}"
        key="${model}|${experiment}|${variant}|${var}"
        FOUND_MODELS["${key}"]="${institution} ${grid} ${version}"
    fi
done < <(find "${NETCDF_ROOT}" -name "*.nc" 2>/dev/null)

if [[ ${#FOUND_MODELS[@]} -eq 0 ]]; then
    echo -e "${YLW}  No .nc files found under ${NETCDF_ROOT}${NC}"
else
    for key in $(printf '%s\n' "${!FOUND_MODELS[@]}" | sort); do
        IFS='|' read -r model experiment variant var <<< "${key}"
        printf "  %-30s  exp=%-12s  variant=%-14s  var=%s\n" \
               "${model}" "${experiment}" "${variant}" "${var}"
    done
fi
echo

# ── 4. Parse pmip4_* entries from yml ────────────────────────────────────────
echo -e "${BLU}--- YAML pmip4_* entries (HPC paths) ---${NC}"

declare -A YML_MODELS   # yml_key -> resolved path

in_files=0
current_key=""
while IFS= read -r line; do
    [[ "${line}" =~ ^files: ]] && { in_files=1; continue; }
    [[ ${in_files} -eq 0 ]] && continue

    if [[ "${line}" =~ ^[[:space:]]{2}([a-zA-Z0-9_-]+): ]]; then
        current_key="${BASH_REMATCH[1]}"
    fi

    if [[ "${current_key}" == pmip4_* ]] && [[ "${line}" =~ path:.*\"(.*)\" ]]; then
        raw_path="${BASH_REMATCH[1]}"
        resolved="${raw_path/\{pmip4_data\}/${HPC_PMIP4_BASE}}"
        YML_MODELS["${current_key}"]="${resolved}"
        printf "  %-35s -> %s\n" "${current_key}" "${resolved}"
    fi
done < "${YML_FILE}"
echo

# ── 5. Cross-check yml entries against discovered netCDF structure ────────────
echo -e "${BLU}--- Cross-check results ---${NC}"

PASS=0; WARN=0; FAIL=0

for yml_key in $(printf '%s\n' "${!YML_MODELS[@]}" | sort); do
    yml_path="${YML_MODELS[$yml_key]}"
    model_name="${yml_key#pmip4_}"

    # Extract experiment and variant from the path
    experiment=$(echo "${yml_path}" | grep -oP "(?<=${model_name}/)[^/]+" | head -1 || true)
    variant=$(echo "${yml_path}" | grep -oP "(?<=${experiment}/)[^/]+" | head -1 || true)
    var_name=$(echo "${yml_path}" | grep -oP "(?<=Amon/)[^/]+" | head -1 || true)

    lookup_key="${model_name}|${experiment}|${variant}|${var_name}"

    if [[ -n "${FOUND_MODELS[$lookup_key]+x}" ]]; then
        info="${FOUND_MODELS[$lookup_key]}"
        institution=$(echo "${info}" | awk '{print $1}')
        grid=$(echo "${info}" | awk '{print $2}')
        version=$(echo "${info}" | awk '{print $3}')
        nc_dir="${NETCDF_ROOT}/${institution}/${model_name}/${experiment}/${variant}/Amon/${var_name}/${grid}/${version}"
        nc_size=$(du -sh "${nc_dir}" 2>/dev/null | awk '{print $1}' || echo "?")

        if [[ -d "${yml_path}" ]]; then
            zarr_size=$(du -sh "${yml_path}" 2>/dev/null | awk '{print $1}' || echo "?")
            echo -e "  ${GRN}[OK]${NC}   ${yml_key}"
            echo    "         zarr  : ${yml_path}  (${zarr_size})"
            echo    "         netCDF: ${nc_dir}  (${nc_size})"
            check_zarr_python "${yml_path}"
            ((PASS++)) || true
        else
            echo -e "  ${YLW}[WARN]${NC} ${yml_key}: netCDF found but zarr store missing"
            echo    "         netCDF: ${nc_dir}  (${nc_size})"
            echo    "         zarr  : ${yml_path}  (not yet created)"
            ((WARN++)) || true
        fi
    else
        # Check if the model appears under a different experiment/variant
        alternatives=$(printf '%s\n' "${!FOUND_MODELS[@]}" | grep "^${model_name}|" || true)
        if [[ -n "${alternatives}" ]]; then
            echo -e "  ${YLW}[WARN]${NC} ${yml_key}: model found but experiment/variant mismatch"
            echo    "         yml expects: experiment=${experiment}, variant=${variant}"
            echo    "         Found in netCDF tree:"
            while IFS= read -r alt; do
                IFS='|' read -r m e v vv <<< "${alt}"
                echo "           experiment=${e}, variant=${v}, var=${vv}"
            done <<< "${alternatives}"
            ((WARN++)) || true
        else
            echo -e "  ${RED}[FAIL]${NC} ${yml_key}: '${model_name}' not found anywhere under ${NETCDF_ROOT}"
            ((FAIL++)) || true
        fi
    fi
done

echo
echo -e "${BLU}--- Summary ---${NC}"
echo -e "  ${GRN}OK  :${NC} ${PASS}"
echo -e "  ${YLW}WARN:${NC} ${WARN}  (netCDF exists but zarr missing, or experiment/variant mismatch)"
echo -e "  ${RED}FAIL:${NC} ${FAIL}  (model not found in netCDF tree)"
echo

# ── 6. Models in netCDF tree not referenced in yml ───────────────────────────
echo -e "${BLU}--- Models in netCDF tree NOT referenced in yml ---${NC}"
unreferenced=0
for key in $(printf '%s\n' "${!FOUND_MODELS[@]}" | sort); do
    IFS='|' read -r model experiment variant var <<< "${key}"
    yml_key="pmip4_${model}"
    if [[ -z "${YML_MODELS[$yml_key]+x}" ]]; then
        echo -e "  ${YLW}[UNREF]${NC} model=${model}, experiment=${experiment}, variant=${variant}, var=${var}"
        ((unreferenced++)) || true
    fi
done
[[ ${unreferenced} -eq 0 ]] && echo "  (none)"
echo

# ── 7. Structural note ───────────────────────────────────────────────────────
echo -e "${BLU}--- Notes ---${NC}"
echo "  yml 'pmip4_data' (hpc) : ${HPC_PMIP4_BASE}"
echo "  actual netCDF root     : ${NETCDF_ROOT}"
echo "  The netCDF tree has an extra institution-level directory"
echo "  (e.g., EC-Earth-Consortium/) between the root and model name."
echo "  yml path templates do not include this level."

} # end main()

# Run main, tee to terminal and log file (strip colors in log)
main 2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' > "${LOG_FILE}")

echo
echo "Output saved to: ${LOG_FILE}"
