#!/usr/bin/env bash
set -Eeuo pipefail

detect_argus_keystore_type() {
  local keystore_path="${1:?Keystore path is required.}"
  [[ -s "$keystore_path" ]] || return 1
  local magic
  magic="$(od -An -N4 -tx1 "$keystore_path" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  case "$magic" in
    feedfeed) printf 'JKS\n' ;;
    30*) printf 'PKCS12\n' ;;
    *) return 2 ;;
  esac
}
