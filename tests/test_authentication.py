from types import SimpleNamespace

import pytest

from moscollector.authentication import authenticate
from moscollector.database import hash_password, make_database, seed_users


def test_production_rejects_existing_demo_accounts(tmp_path):
    engine, factory = make_database(f"sqlite:///{tmp_path / 'users.db'}")
    seed_users(factory, True)
    with pytest.raises(RuntimeError, match="Demo accounts"):
        seed_users(factory, False)
    engine.dispose()


def test_ldap_rejects_cleartext_transport_before_sending_password(monkeypatch):
    monkeypatch.setenv("CONTOUR_LDAP_URL", "ldap://directory.example")
    with pytest.raises(ValueError, match="ldaps"):
        authenticate(SimpleNamespace(username="employee"), "example-secret")


def test_local_admin_remains_available_when_directory_is_offline(monkeypatch):
    monkeypatch.setenv("CONTOUR_LDAP_URL", "ldaps://directory.example")
    user = SimpleNamespace(username="admin", password_hash=hash_password("local-admin-test-password"))
    assert authenticate(user, "local-admin-test-password")
    assert not authenticate(user, "wrong-password")


def test_ldap_rejects_unsafe_username_before_connection(monkeypatch):
    monkeypatch.setenv("CONTOUR_LDAP_URL", "ldaps://directory.example")
    assert not authenticate(SimpleNamespace(username="user,cn=admin"), "example-secret")
