#!/usr/bin/env bash
set -euo pipefail

# Get the repo root (one level up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CERT_DIR="${MIDNIGHT_CERT_DIR:-${ROOT_DIR}/infra/nginx/certs}"
DOMAIN="${1:-midnight.local}"
DAYS_VALID="${DAYS_VALID:-365}"

mkdir -p "${CERT_DIR}"

CERT_PATH="${CERT_DIR}/${DOMAIN}.crt"
KEY_PATH="${CERT_DIR}/${DOMAIN}.key"
OPENSSL_CONFIG="$(mktemp)"

cleanup() {
    rm -f "${OPENSSL_CONFIG}"
}
trap cleanup EXIT

cat <<EOF > "${OPENSSL_CONFIG}"
[ req ]
default_bits       = 2048
distinguished_name = req_distinguished_name
req_extensions     = req_ext
x509_extensions    = v3_ca
prompt             = no

[ req_distinguished_name ]
CN = ${DOMAIN}
O  = Midnight Renegade
OU = Dev

[ req_ext ]
subjectAltName = @alt_names

[ v3_ca ]
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = ${DOMAIN}
DNS.2 = localhost
IP.1  = 127.0.0.1
EOF

echo "Generating self-signed certificate for ${DOMAIN} (valid ${DAYS_VALID} days)"
openssl req \
    -x509 \
    -nodes \
    -days "${DAYS_VALID}" \
    -newkey rsa:2048 \
    -keyout "${KEY_PATH}" \
    -out "${CERT_PATH}" \
    -config "${OPENSSL_CONFIG}" >/dev/null 2>&1

echo "Certificate: ${CERT_PATH}"
echo "Private key: ${KEY_PATH}"
echo ""
echo "⚠️  Add '${DOMAIN}' to /etc/hosts pointing to this server's IP so browsers trust the hostname."
