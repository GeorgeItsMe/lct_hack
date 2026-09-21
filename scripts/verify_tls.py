"""Verify the local Caddy profile with a explicitly trusted test certificate.

No operating-system/browser trust stores are changed. An untrusted client must
reject the certificate, and legacy TLS must be rejected by the server.
"""

import http.client
import json
import socket
import ssl
import warnings
from pathlib import Path

certificate = Path("deploy/certs/fullchain.pem")
report = {"target": "localhost:8443", "certificate_scope": "local_test_only", "checks": {}}
for protocol in (ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3):
    context = ssl.create_default_context(cafile=str(certificate))
    context.minimum_version = context.maximum_version = protocol
    client = http.client.HTTPSConnection("localhost", 8443, context=context, timeout=10)
    client.connect()
    negotiated = client.sock.version()
    client.request("GET", "/api/health")
    response = client.getresponse()
    body = json.loads(response.read())
    assert response.status == 200 and body["database"] == "postgresql"
    assert response.getheader("Strict-Transport-Security") == "max-age=31536000"
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    report["checks"][negotiated] = "passed_hostname_certificate_and_api"
    client.close()

try:
    client = http.client.HTTPSConnection("localhost", 8443, context=ssl.create_default_context(), timeout=10)
    client.request("GET", "/api/health")
    raise AssertionError("Untrusted local certificate was accepted")
except ssl.SSLCertVerificationError:
    report["checks"]["untrusted_certificate"] = "rejected"
finally:
    client.close()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    context = ssl.create_default_context(cafile=str(certificate))
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_1
    # This client-only setting permits sending a legacy handshake for the
    # negative test. The server configuration is never weakened.
    context.set_ciphers("ALL:@SECLEVEL=0")
    try:
        with socket.create_connection(("localhost", 8443), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname="localhost"):
                raise AssertionError("Legacy TLS 1.1 was accepted")
    except ssl.SSLError as error:
        assert "PROTOCOL_VERSION" in str(error), f"No server rejection evidence: {error}"
        report["checks"]["TLSv1.1"] = "server_rejected_protocol_version"

report["customer_certificate_tested"] = False
Path("artifacts/tls_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
