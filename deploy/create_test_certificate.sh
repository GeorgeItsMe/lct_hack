#!/usr/bin/env bash
# Local QA only. Never replace a pre-existing certificate or change system trust.
set -euo pipefail
cd "$(dirname "$0")/.."
test_cert_dir=deploy/certs
if [[ -e "$test_cert_dir/privkey.pem" || -e "$test_cert_dir/fullchain.pem" ]]; then
  echo 'Certificate files already exist; refusing to overwrite them.' >&2
  exit 2
fi
umask 077
mkdir -p "$test_cert_dir"
cat > "$test_cert_dir/local-test.cnf" <<'CONFIG'
[req]
distinguished_name=dn
x509_extensions=extensions
prompt=no
[dn]
CN=localhost
[extensions]
subjectAltName=DNS:localhost,IP:127.0.0.1
basicConstraints=critical,CA:TRUE
keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign
extendedKeyUsage=serverAuth
CONFIG
openssl req -x509 -newkey rsa:2048 -nodes -days 2 \
  -config "$test_cert_dir/local-test.cnf" \
  -keyout "$test_cert_dir/privkey.pem" -out "$test_cert_dir/fullchain.pem"
echo 'Local test certificate created for two days; system trust was not changed.'
