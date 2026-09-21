#!/usr/bin/env bash
set -euo pipefail
umask 077
mkdir -p /backups
while true; do
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  backup_file="/backups/contour-${stamp}.dump"
  if pg_dump --format=custom --file="${backup_file}.partial"; then
    mv "${backup_file}.partial" "$backup_file"
    (cd /backups && sha256sum "contour-${stamp}.dump") > "${backup_file}.sha256"
    echo "Backup completed: ${stamp}"
  else
    echo "Backup failed: ${stamp}" >&2
  fi
  sleep 21600
done
