"""Optional LDAPS bind for explicitly provisioned users; local admin is break-glass."""

import os
import re
import ssl
from urllib.parse import urlparse

from moscollector.database import verify_password


def authenticate(user, password):
    uri = os.getenv("CONTOUR_LDAP_URL", "")
    if not uri or user.username == "admin":
        return verify_password(password, user.password_hash)
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,80}", user.username) or not password:
        return False
    address = urlparse(uri)
    if address.scheme != "ldaps" or not address.hostname or address.username or address.password:
        raise ValueError("CONTOUR_LDAP_URL должен использовать ldaps:// без логина и пароля")
    template = os.getenv("CONTOUR_LDAP_BIND_TEMPLATE", "")
    if "{username}" not in template:
        raise ValueError("Настройте CONTOUR_LDAP_BIND_TEMPLATE")
    from ldap3 import Connection, Server, Tls
    from ldap3.core.exceptions import LDAPException

    tls = Tls(
        validate=ssl.CERT_REQUIRED,
        ca_certs_file=os.getenv("CONTOUR_LDAP_CA_FILE") or None,
        version=ssl.PROTOCOL_TLS_CLIENT,
    )
    server = Server(address.hostname, port=address.port or 636, use_ssl=True, tls=tls, connect_timeout=4)
    try:
        with Connection(
            server,
            user=template.format(username=user.username),
            password=password,
            auto_bind=True,
            auto_referrals=False,
            receive_timeout=4,
            read_only=True,
        ) as connection:
            return bool(connection.bound)
    except LDAPException:
        return False
