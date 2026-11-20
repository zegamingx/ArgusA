#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Sign an Argus release APK with the offline Argus 2026 PKCS#12 identity.
# The private key is supplied by a protected GitHub Environment secret and is
# written only to a short-lived, permission-restricted runner temp directory.

readonly EXPECTED_CERT_SHA256="40969E172BC0F8CB18ED2DDF274CAF3D3DB1456EDAEAF03939C0C0511797E64E"
readonly MINIMUM_ANDROID_API="34"
readonly EXPECTED_APPLICATION_ID="com.argus.agi"
readonly DEFAULT_KEY_ALIAS="argus-release-2026"

fail() {
  printf '::error::%s\n' "$*" >&2
  exit 1
}

usage() {
  echo "Usage: $0 <unsigned-apk> <signed-output-apk>" >&2
  exit 2
}

[[ "$#" -eq 2 ]] || usage

input_apk="$(realpath "$1")"
output_apk="$(realpath -m "$2")"

[[ -s "$input_apk" ]] || fail "Unsigned APK does not exist or is empty: $input_apk"
[[ "$input_apk" != "$output_apk" ]] || fail "Input and output APK paths must be different."

: "${ARGUS_RELEASE_KEYSTORE_BASE64:?Missing ARGUS_RELEASE_KEYSTORE_BASE64 secret.}"
: "${ARGUS_RELEASE_STORE_PASSWORD:?Missing ARGUS_RELEASE_STORE_PASSWORD secret.}"
: "${ARGUS_RELEASE_KEY_PASSWORD:?Missing ARGUS_RELEASE_KEY_PASSWORD secret.}"

key_alias="${ARGUS_RELEASE_KEY_ALIAS:-$DEFAULT_KEY_ALIAS}"
[[ -n "$key_alias" ]] || fail "The release signing key alias is empty."

sdk_root="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
[[ -n "$sdk_root" ]] || fail "ANDROID_HOME or ANDROID_SDK_ROOT must identify the Android SDK."

build_tools="${sdk_root}/build-tools/37.0.0"
zipalign="${build_tools}/zipalign"
apksigner="${build_tools}/apksigner"
aapt="${build_tools}/aapt"

[[ -x "$zipalign" ]] || fail "Android zipalign was not found at $zipalign"
[[ -x "$apksigner" ]] || fail "Android apksigner was not found at $apksigner"
[[ -x "$aapt" ]] || fail "Android aapt was not found at $aapt"

output_dir="$(dirname "$output_apk")"
mkdir -p "$output_dir"
[[ ! -e "$output_apk" ]] || fail "Refusing to overwrite existing APK: $output_apk"

temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
mkdir -p "$temp_root"
work_dir="$(mktemp -d "${temp_root%/}/argus-signing.XXXXXXXX")"
cleanup() {
  rm -rf -- "$work_dir"
}
trap cleanup EXIT HUP INT TERM

keystore="${work_dir}/argus-release-2026.p12"
aligned_apk="${work_dir}/ARGUS-aligned.apk"
signed_apk="${work_dir}/ARGUS-signed.apk"
verification_log="${work_dir}/apksigner-verify.txt"

printf '%s' "$ARGUS_RELEASE_KEYSTORE_BASE64" | base64 --decode > "$keystore"
[[ -s "$keystore" ]] || fail "Decoded release keystore is empty."
chmod 600 "$keystore"

echo "Aligning native libraries for 16 KiB memory pages..."
"$zipalign" -f -P 16 4 "$input_apk" "$aligned_apk"
"$zipalign" -c -P 16 4 "$aligned_apk"

echo "Signing release APK with the protected Argus 2026 identity..."
"$apksigner" sign \
  --ks "$keystore" \
  --ks-type PKCS12 \
  --ks-key-alias "$key_alias" \
  --ks-pass env:ARGUS_RELEASE_STORE_PASSWORD \
  --key-pass env:ARGUS_RELEASE_KEY_PASSWORD \
  --min-sdk-version "$MINIMUM_ANDROID_API" \
  --v1-signing-enabled true \
  --v2-signing-enabled true \
  --v3-signing-enabled true \
  --v4-signing-enabled false \
  --out "$signed_apk" \
  "$aligned_apk"

echo "Verifying APK signature and pinned certificate fingerprint..."
"$apksigner" verify \
  --verbose \
  --print-certs \
  --min-sdk-version "$MINIMUM_ANDROID_API" \
  "$signed_apk" > "$verification_log" 2>&1 || {
    cat "$verification_log" >&2
    fail "apksigner rejected the produced APK."
  }

fingerprint_line="$(grep -m1 'Signer #1 certificate SHA-256 digest:' "$verification_log" || true)"
[[ -n "$fingerprint_line" ]] || {
  cat "$verification_log" >&2
  fail "No signer SHA-256 fingerprint was reported by apksigner."
}
actual_cert_sha256="$(
  printf '%s\n' "$fingerprint_line" \
    | sed -E 's/.*digest:[[:space:]]*//' \
    | tr -d '[:space:]:' \
    | tr '[:lower:]' '[:upper:]'
)"
[[ "$actual_cert_sha256" =~ ^[A-F0-9]{64}$ ]] \
  || fail "The reported signing certificate fingerprint is malformed."
[[ "$actual_cert_sha256" == "$EXPECTED_CERT_SHA256" ]] \
  || fail "Signing certificate mismatch. Expected $EXPECTED_CERT_SHA256, got $actual_cert_sha256."

"$zipalign" -c -P 16 4 "$signed_apk"
package_metadata="$("$aapt" dump badging "$signed_apk" | grep -m1 "^package:")"
[[ -n "$package_metadata" ]] || fail "Could not read Android package metadata."
application_id="$(printf '%s\n' "$package_metadata" | sed -nE "s/^package: name='([^']+)'.*/\1/p")"
[[ "$application_id" == "$EXPECTED_APPLICATION_ID" ]] \
  || fail "Unexpected package ID: expected $EXPECTED_APPLICATION_ID, got $application_id."

mv -- "$signed_apk" "$output_apk"
chmod 644 "$output_apk"

apk_sha256="$(sha256sum "$output_apk" | awk '{print $1}')"
printf '%s  %s\n' "$apk_sha256" "$(basename "$output_apk")" > "${output_apk}.sha256"
printf '%s\n' "$actual_cert_sha256" > "${output_apk}.certificate-sha256"

echo "Installable Argus APK created and verified."
echo "Package: $application_id"
echo "Certificate SHA-256: $actual_cert_sha256"
echo "APK SHA-256: $apk_sha256"
echo "Output: $output_apk"
