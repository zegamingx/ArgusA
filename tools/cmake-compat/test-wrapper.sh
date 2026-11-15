#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
    echo "Usage: $0 <wrapper-root>" >&2
    exit 2
fi

wrapper_root="$1"
wrapper="${wrapper_root}/bin/cmake"
test -x "${wrapper}"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "${temporary_directory}"' EXIT
fake_cmake="${temporary_directory}/fake-cmake"
capture_file="${temporary_directory}/arguments.txt"

cat > "${fake_cmake}" <<'FAKE_CMAKE'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
    printf 'cmake version 4.5.0\n'
    exit 0
fi
printf '%s\n' "$@" > "${ARGUS_CMAKE_TEST_CAPTURE}"
if [[ "${1:-}" == "--fail" ]]; then
    exit 37
fi
FAKE_CMAKE
chmod +x "${fake_cmake}"

export ARGUS_REAL_CMAKE="${fake_cmake}"
export ARGUS_CMAKE_TEST_CAPTURE="${capture_file}"

"${wrapper}" "-H/opt/Argus source" "-B/tmp/Argus build" "-DARGUS_TEST=1"
mapfile -t arguments < "${capture_file}"
[[ "${#arguments[@]}" -eq 3 ]]
[[ "${arguments[0]}" == "-S/opt/Argus source" ]]
[[ "${arguments[1]}" == "-B/tmp/Argus build" ]]
[[ "${arguments[2]}" == "-DARGUS_TEST=1" ]]

"${wrapper}" -H "/opt/Argus separate source" --build "/tmp/Argus build"
mapfile -t arguments < "${capture_file}"
[[ "${#arguments[@]}" -eq 4 ]]
[[ "${arguments[0]}" == "-S" ]]
[[ "${arguments[1]}" == "/opt/Argus separate source" ]]
[[ "${arguments[2]}" == "--build" ]]
[[ "${arguments[3]}" == "/tmp/Argus build" ]]

"${wrapper}" "-S/explicit/source" "-B/explicit/build"
mapfile -t arguments < "${capture_file}"
[[ "${#arguments[@]}" -eq 2 ]]
[[ "${arguments[0]}" == "-S/explicit/source" ]]
[[ "${arguments[1]}" == "-B/explicit/build" ]]

set +e
"${wrapper}" --fail >/dev/null 2>&1
status=$?
set -e
[[ "${status}" -eq 37 ]]

printf 'CMake launcher compatibility tests passed.\n'
