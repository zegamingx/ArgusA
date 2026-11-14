#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY_ROOT="${GITHUB_WORKSPACE:-$PWD}"
readonly APK_RELATIVE_PATH="app/build/outputs/apk/release/app-release.apk"
readonly PACKAGE_NAME="com.argus.agi"
readonly ACTIVITY_COMPONENT="${PACKAGE_NAME}/.MainActivity"
readonly EXPECTED_API_LEVEL="35"
readonly CORE_INIT_TIMEOUT_SECONDS="${ARGUS_CORE_INIT_TIMEOUT_SECONDS:-180}"
readonlly SERVICE_START_TIMEOUT_SECONDS="${ARGUS_SERVICE_START_TIMEOUT_SECONDS:-30}"
readonly STABILITY_SECONDS="${ARGUS_STABILITY_SECONDS:-15}"

cd "${REPOSITORY_ROOT}"

log() {
    printf '[Argus smoke test] %s\n' "$*"
}

dump_diagnostics() {
    printf '\n===== Recent Android logcat =====\n' >&2
    adb_cmd logcat -d -v time -t 1000 2>/dev/null | tail -n 350 >&2 || true
    printf '===== End Android logcat =====\n' >&2
}

fail() {
    printf '::error::%s\n' "$*" >&2
    dump_diagnostics
    exit 1
}

on_error() {
    local exit_code=$?
    trap - ERR
    printf '::error::Smoke-test command failed at line %s (exit %s).\n' "${BASH_LINENO[0]:-unknown}" "${exit_code}" >&2
    dump_diagnostics
    exit "${exit_code}"
}
trap on_error ERR

for timeout_value in \
    "${CORE_INIT_TIMEOUT_SECONDS}" \
    "${SERVICE_START_TIMEOUT_SECONDS}" \
    "${STABILITY_SECONDS}"; do
    [[ "${timeout_value}" =~ ^[1-9][0-9]*$ ]] || fail "Smoke-test timeouts must be positive integers."
done

command -v adb >/dev/null 2>&1 || fail "Android Debug Bridge (adb) is not available on PATH."
readonly APK_PATH="${REPOSITORY_ROOT}/${APK_RELATIVE_PATH}"
[[ -s "${APK_PATH}" ]] || fail "Signed release APK is missing or empty: ${APK_PATH}"

ADB=(adb)
if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    ADB+=(-s "${ANDROID_SERIAL}")
fi

adb_cmd() {
    "${ADB[@]}" "$@"
}

log "Waiting for the Android emulator."
adb_cmd wait-for-device
DEVICE_STATE="$(adb_cmd get-state | tr -d '\r')"
[[ "${DEVICE_STATE}" == "device" ]] || fail "Emulator is not in the device state (state: ${DEVICE_STATE})."

API_LEVEL="$(adb_cmd shell getprop ro.build.version.sdk | tr -d '\r')"
[[ "${API_LEVEL}" == "${EXPECTED_API_LEVEL}" ]] || \
    fail "Expected Android API ${EXPECTED_API_LEVEL}, received API ${API_LEVEL:-unknown}."

log "Installing signed release APK (${APK_PATH})."
adb_cmd logcat -c
adb_cmd install -r "${APK_PATH}"
adb_cmd shell am force-stop "${PACKAGE_NAME}"

log "Launching ${ACTIVITY_COMPONENT}."
if ! START_OUTPUT="$(adb_cmd shell am start -W -n "${ACTIVITY_COMPONENT}" 2>&1)"; then
    printf '%s\n' "${START_OUTPUT}" >&2
    fail "Android rejected the Argus launcher activity."
fi
printf '%s\n' "${START_OUTPUT}"
grep -Fq 'Status: ok' <<< "${START_OUTPUUT}" || \
    fail "The Argus launcher activity did not report a successful start."

log "Waiting up to ${CORE_INIT_TIMEOUT_SECONDS}s for core initialization."
CORE_READY=false
for (( elapsed = 0; elapsed < CORE_INIT_TIMEOUT_SECONDS; elapsed += 1 )); do
    CRASH_LOG="$(adb_cmd logcat -d -b crash -v time 2>/dev/null || true)"
    if grep -Eiq 'FATAL EXCEPTION|Fatal signal|Process: com\.argus\.agi' <<< "${CRASH_LOG}"; then
        fail "Argus crashed while initializing its core runtime."
    fi

    APP_LOG="$(adb_cmd logcat -d -s ArgusApp:I 2>/dev/null || true)"
    if grep -Eiq 'Core initialization failed at|Insufficient memory while initializing Argus' <<< "${APP_LOG}"; then
        fail "Argus reported a core initialization failure."
    fi
    if grep -Fq 'Core runtime initialization completed.' <<< "${APP_LOG}"; then
        CORE_READY=true
        break
    fi
    sleep 1
done

[[ "${CORE_READY}" == "true" ]] || \
    fail "Argus did not complete core initialization within ${CORE_INIT_TIMEOUT_SECONDS}s."

PROCESS_ID="$(adb_cmd shell pidof "${PACKAGE_NAME}" 2>/dev/null | tr -d '\r' || true)"
[[ -n "${PROCESS_ID}" ]] || fail "Argus process exited immediately after core initialization."

log "Waiting up to ${SERVICE_START_TIMEOUT_SECONDS}s for ArgusForegroundService."
SERVICE_READY=false
for (( elapsed = 0; elapsed < SERVICE_START_TIMEOUT_SECONDS; elapsed += 1 )); do
    SERVICE_DUMP="$(adb_cmd shell dumpsys activity services "${PACKAGE_NAME}" 2>/dev/null || true)"
    if grep -Fq 'ArgusForegroundService' <<< "${SERVICE_DUMP}" && \
        grep -Fq 'isForeground=true' <<< "${SERVICE_DUMP}"; then
        SERVICE_READY=true
        break
    fi

    CRASH_LOG="$(adb_cmd logcat -d -b crash -v time 2>/dev/null || true)"
    if grep -Eiq 'FATAL EXCEPTION|Fatal signal|Process: com\.argus\.agi' <<< "${CRASH_LOG}"; then
        fail "Argus crashed while starting its foreground service."
    fi
    sleep 1
done

if [[ "${SERVICE_READY}" != "true" ]]; then
    printf '%s\n' "${SERVICE_DUMP:-Foreground-service dump unavailable.}" >&2
    fail "ArgusForegroundService did not reach the foreground state within ${SERVICE_START_TIMEOUT_SECONDS}s."
fi

log "Checking process and foreground-service stability for ${STABILITY_SECONDS}s."
sleep "${STABILITY_SECONDS}"
PROCESS_ID="$(adb_cmd shell pidof "${PACKAGE_NAME}" 2>/dev/null | tr -d '\r' || true)"
[[ -n "${PROCESS_ID}" ]] || fail "Argus process exited after core initialization."

SERVICE_DUMP="$(adb_cmd shell dumpsys activity services "${PACKAGE_NAME}" 2>/dev/null || true)"
grep -Fq 'ArgusForegroundService' <<< "${SERVICE_DUMP}" || \
    fail "ArgusForegroundService stopped during the stability window."
grep -Fq 'isForeground=true' <<< "${SERVICE_DUMP}" || \
    fail "ArgusForegroundService is no longer in the foreground state."

CRASH_LOG="$(adb_cmd logcat -d -b crash -v time 2>/dev/null || true)"
if grep -Eiq 'FATAL EXCEPTION|Fatal signal|Process: com\.argus\.agi' <<< "${CRASH_LOG}"; then
    fail "Argus crashed during the post-initialization stability window."
fi

log "PASS: Android ${EXPECTED_API_LEVEL}; release APK installed; core runtime initialized; foreground service remained active; no crash detected."
