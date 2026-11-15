#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "${script_directory}/../.." && pwd -P)"

if (($# < 1 || $# > 2)); then
    echo "Usage: $0 <real-cmake-4.5.0-executable> [wrapper-root]" >&2
    exit 2
fi

real_cmake_input="$1"
if [[ "${real_cmake_input}" != /* ]]; then
    real_cmake_input="$(command -v -- "${real_cmake_input}" || true)"
fi
if [[ -z "${real_cmake_input}" || ! -x "${real_cmake_input}" ]]; then
    echo "The supplied CMake executable does not exist or is not executable." >&2
    exit 1
fi

real_cmake_directory="$(cd -- "$(dirname -- "${real_cmake_input}")" && pwd -P)"
real_cmake="${real_cmake_directory}/$(basename -- "${real_cmake_input}")"
expected_version="${ARGUS_EXPECTED_CMAKE_VERSION:-4.5.0}"
reported_version="$("${real_cmake}" --version | sed -n '1p')"
if [[ "${reported_version}" != "cmake version ${expected_version}" ]]; then
    echo "Expected CMake ${expected_version}, received: ${reported_version}" >&2
    exit 1
fi

wrapper_root="${2:-${script_directory}/runtime}"
mkdir -p "${wrapper_root}/bin"
wrapper_root="$(cd -- "${wrapper_root}" && pwd -P)"
cp "${script_directory}/cmake-wrapper.sh" "${wrapper_root}/bin/cmake"
chmod 0755 "${wrapper_root}/bin/cmake"
printf '%s\n' "${real_cmake}" > "${wrapper_root}/real-cmake.path"

ninja_source="${real_cmake_directory}/ninja"
if [[ ! -x "${ninja_source}" ]]; then
    ninja_source="$(command -v ninja || true)"
fi
if [[ -z "${ninja_source}" || ! -x "${ninja_source}" ]]; then
    echo "Ninja was not found beside CMake or on PATH." >&2
    exit 1
fi
ln -sfn "${ninja_source}" "${wrapper_root}/bin/ninja"

# AGP resolves CMake before the CMake configure task runs. A machine-local
# cmake.dir override guarantees it selects this launcher instead of another
# CMake installation in the Android SDK.
local_properties="${repository_root}/local.properties"
temporary_properties="$(mktemp)"
if [[ -f "${local_properties}" ]]; then
    awk -F= '$1 !~ /^[[:space:]]*cmake[.]dir[[:space:]]*$/ { print }' \
        "${local_properties}" > "${temporary_properties}"
fi
escaped_wrapper_root="${wrapper_root//\\/\\\\}"
escaped_wrapper_root="${escaped_wrapper_root// /\\ }"
escaped_wrapper_root="${escaped_wrapper_root//:/\\:}"
printf 'cmake.dir=%s\n' "${escaped_wrapper_root}" >> "${temporary_properties}"
mv "${temporary_properties}" "${local_properties}"

if [[ -n "${GITHUB_PATH:-}" ]]; then
    printf '%s\n' "${wrapper_root}/bin" >> "${GITHUB_PATH}"
fi

"${script_directory}/test-wrapper.sh" "${wrapper_root}"
"${wrapper_root}/bin/cmake" --version | grep -Fx "cmake version ${expected_version}"
printf 'AGP CMake compatibility launcher installed at %s\n' "${wrapper_root}"
printf 'Real CMake remains %s\n' "${real_cmake}"
