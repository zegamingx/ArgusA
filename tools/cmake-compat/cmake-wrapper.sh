#!/usr/bin/env bash
set -euo pipefail

wrapper_bin_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
wrapper_root="$(cd -- "${wrapper_bin_dir}/.." && pwd -P)"
real_cmake="${ARGUS_REAL_CMAKE:-}"
if [[ -z "${real_cmake}" && -f "${wrapper_root}/real-cmake.path" ]]; then
    IFS= read -r real_cmake < "${wrapper_root}/real-cmake.path" || true
fi

if [[ -z "${real_cmake}" || ! -x "${real_cmake}" ]]; then
    echo "Argus CMake compatibility launcher cannot find its real CMake 4.5.0 executable." >&2
    echo "Run tools/cmake-compat/install-wrapper.sh with the official cmake executable path." >&2
    exit 127
fi

# AGP currently generates -H<source-dir> internally. CMake 4.5 deprecated
# that command-line spelling in favor of -S. Rewrite only that launcher
# argument; all other arguments and the pinned Kitware binary pass through.
rewritten_arguments=()
while (($# > 0)); do
    case "$1" in
        -H)
            if (($# < 2)); then
                echo "CMake received -H without a source directory." >&2
                exit 2
            fi
            rewritten_arguments+=("-S" "$2")
            shift 2
            ;;
        -H?*)
            rewritten_arguments+=("-S${1#-H}")
            shift
            ;;
        *)
            rewritten_arguments+=("$1")
            shift
            ;;
    esac
done

exec "${real_cmake}" "${rewritten_arguments[@]}"
