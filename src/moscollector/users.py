"""Provision local roles; LDAP users receive an unusable random local password."""

import argparse
import getpass
import secrets

from sqlalchemy import select

from moscollector.database import AuditLog, User, hash_password, make_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", choices=["dispatcher", "analyst", "admin"], required=True)
    parser.add_argument("--ldap", action="store_true")
    args = parser.parse_args()
    password = (
        secrets.token_urlsafe(40) if args.ldap else getpass.getpass("Password (at least 12 characters): ")
    )
    if len(password) < 12:
        raise SystemExit("Password is too short")
    engine, factory = make_database()
    with factory() as db:
        if db.scalar(select(User).where(User.username == args.username)):
            raise SystemExit("User already exists; refusing to change permissions implicitly")
        db.add(
            User(
                username=args.username,
                display_name=args.name,
                role=args.role,
                password_hash=hash_password(password),
            )
        )
        db.add(
            AuditLog(
                action="user_provisioned_cli",
                detail=f"User {args.username}, role {args.role}, ldap={args.ldap}",
            )
        )
        db.commit()
    engine.dispose()
    print("User created")


if __name__ == "__main__":
    main()
