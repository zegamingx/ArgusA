#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly EXPECTED_CERT_SHA256="40969E172BC0F8CB18ED2DDF274CAF3D3DB1456EDAEAF03939C0C0511797E64E"
readonly DEFAULT_KEY_ALIAS="argus-release-2026"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/keystore-format.sh"

fail() {
  printf '::error::%s\n' "$*" >&2
  exit 1
}

: "${ARGUS_RELEASE_KEYSTORE_BASE64:?Missing ARGUS_RELEASE_KEYSTORE_BASE64 secret.}"
: "${ARGUS_RELEASE_STORE_PASSWORD:?Missing ARGUS_RELEASE_STORE_PASSWORD secret.}"
: "${ARGUS_RELEASE_KEY_PASSWORD:?Missing ARGUS_RELEASE_KEY_PASSWORD secret.}"

key_alias="${ARGUS_RELEASE_KEY_ALIAS:-$DEFAULT_KEY_ALIAS}"
[[ -n "$key_alias" ]] || fail "The production signing key alias is empty."

temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
mkdir -p "$temp_root"
work_dir="$(mktemp -d "${temp_root%/}/argus-keystore-preflight.XXXXXXXX")"
cleanup() { rm -rf -- "$work_dir"; }
trap cleanup EXIT HUP INT TERM

keystore="${work_dir}/argus-release-keystore"
java_source="${work_dir}/ArgusKeystorePreflight.java"

printf '%s' "$ARGUS_RELEASE_KEYSTORE_BASE64" | base64 --decode > "$keystore"
[[ -s "$keystore" ]] || fail "Decoded production keystore is empty."
keystore_type="$(detect_argus_keystore_type "$keystore")" || {
  fail "The configured production keystore is not a supported JKS or PKCS12 file."
}

cat > "$java_source" <<'JAVA'
import java.io.FileInputStream;
import java.security.Key;
import java.security.KeyStore;
import java.security.MessageDigest;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.util.Arrays;
import java.util.HexFormat;

public final class ArgusKeystorePreflight {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) throw new IllegalArgumentException("Expected keystore path, type and alias.");
        char[] storePassword = System.getenv("ARGUS_RELEASE_STORE_PASSWORD").toCharArray();
        char[] keyPassword = System.getenv("ARGUS_RELEASE_KEY_PASSWORD").toCharArray();
        try {
            KeyStore keyStore = KeyStore.getInstance(args[1]);
            try (FileInputStream input = new FileInputStream(args[0])) {
                keyStore.load(input, storePassword);
            }
            if (!keyStore.containsAlias(args[2])) {
                throw new IllegalStateException("The configured keystore alias does not exist.");
            }
            Key key = keyStore.getKey(args[2], keyPassword);
            if (!(key instanceof PrivateKey)) {
                throw new IllegalStateException("The configured alias does not contain a private key.");
            }
            Certificate certificate = keyStore.getCertificate(args[2]);
            if (certificate == null) throw new IllegalStateException("The private key has no certificate.");
            String fingerprint = HexFormat.of().withUpperCase()
                .formatHex(MessageDigest.getInstance("SHA-256").digest(certificate.getEncoded()));
            String expected = "40969E172BC0F8CB18ED2DDF274CAF3D3DB1456EDAEAF03939C0C0511797E64E";
            if (!expected.equals(fingerprint)) {
                throw new IllegalStateException("The configured private key does not match the pinned Argus 2026 certificate.");
            }
            System.out.println("Production keystore preflight passed.");
            System.out.println("Keystore format: " + args[1]);
            System.out.println("Private-key alias: " + args[2]);
            System.out.println("Certificate SHA-256: " + fingerprint);
        } finally {
            Arrays.fill(storePassword, '\0');
            Arrays.fill(keyPassword, '\0');
        }
    }
}
JAVA

javac -d "$work_dir" "$java_source" || fail "Could not compile the release keystore preflight verifier."
java -cp "$work_dir" ArgusKeystorePreflight "$keystore" "$keystore_type" "$key_alias" ||
  fail "The active production keystore, password, alias or pinned certificate is invalid."
