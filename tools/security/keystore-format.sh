#!/usr/bin/env bash
set -Eeuo pipefail

detect_argus_keystore_type() {
  local keystore_path="${1:?Keystore path is required.}"
  [[ -s "$keystore_path" ]] || return 1
  local magic
  magic="$(od -An -N4 -tx1 "$keystore_path" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  case "$magic" in
    feedfeed) printf 'JKS\n' ;;
    cececece) printf 'JCEKS\n' ;;
    30*) printf 'PKCS12\n' ;;
    *) return 2 ;;
  esac
}

decode_argus_keystore_base64() {
  local encoded_secret="${1:?Base64 keystore secret is required.}"
  local output_path="${2:?Output keystore path is required.}"
  local nested_path="${output_path}.nested"
  local detected_type=""

  if ! printf '%s' "$encoded_secret" | base64 --decode > "$output_path"; then
    rm -f -- "$output_path"
    return 1
  fi
  if detected_type="$(detect_argus_keystore_type "$output_path" 2>/dev/null)"; then
    printf '%s\n' "$detected_type"
    return 0
  fi

  # Some secret managers store Base64 text rather than the original keystore.
  # Accept one extra Base64 layer only when it yields a recognized keystore header.
  if base64 --decode "$output_path" > "$nested_path" 2>/dev/null; then
    if detected_type="$(detect_argus_keystore_type "$nested_path" 2>/dev/null)"; then
      mv -- "$nested_path" "$output_path"
      printf 'Detected and removed one extra Base64 encoding layer.\n' >&2
      printf '%s\n' "$detected_type"
      return 0
    fi
  fi

  rm -f -- "$nested_path"
  local magic
  magic="$(od -An -N4 -tx1 "$output_path" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  printf 'Unsupported decoded keystore header: %s (expected JKS, JCEKS or PKCS12).\n' "${magic:-unavailable}" >&2
  return 2
}
