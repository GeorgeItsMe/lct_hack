"""Local LDAPS integration check; uses only the disposable fixture directory."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from moscollector.authentication import authenticate
from moscollector.database import hash_password

os.environ["CONTOUR_LDAP_URL"] = "ldaps://localhost:1636"
os.environ["CONTOUR_LDAP_BIND_TEMPLATE"] = "uid={username},dc=contour,dc=test"
os.environ["CONTOUR_LDAP_CA_FILE"] = str(Path("deploy/certs/fullchain.pem").resolve())
user = SimpleNamespace(username="employee", password_hash=hash_password("different-local-password"))
assert authenticate(user, "local-ldap-fixture-password")
assert not authenticate(user, "wrong-directory-password")
assert not authenticate(user, "different-local-password")
unknown = SimpleNamespace(username="unknown", password_hash=hash_password("unknown"))
assert not authenticate(unknown, "local-ldap-fixture-password")
os.environ.pop("CONTOUR_LDAP_CA_FILE")
assert not authenticate(user, "local-ldap-fixture-password")
report = {
    "server": "local OpenLDAP fixture on loopback:1636",
    "valid_ldaps_bind": "passed",
    "wrong_password": "rejected",
    "unknown_user": "rejected",
    "local_password_fallback_for_ldap_user": "rejected",
    "untrusted_certificate": "rejected",
    "customer_directory_tested": False,
}
Path("artifacts/ldap_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
