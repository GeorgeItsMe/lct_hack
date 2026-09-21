#!/usr/bin/env bash
set -euo pipefail
umask 077
mkdir -p /backups
while true; do
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  backup_file="/backups/contour-${stamp}.dump"
  runtime_file="/backups/contour-${stamp}.runtime.tar.gz"
  if pg_dump --format=custom --file="${backup_file}.partial"; then
    # Completed snapshots are immutable and are never deleted by the service.
    # Dump first, then copy snapshots: all decisions in the DB dump reference
    # files that already existed when the runtime archive started.
    runtime_status=0
    tar --exclude='*/processed' --exclude='*/artifacts' --exclude='*/worker.log' \
      --exclude='*/status.tmp' -czf "${runtime_file}.partial" -C /runtime . || runtime_status=$?
    if [[ "$runtime_status" -le 1 ]]; then
      # Exit 1 means an in-flight job changed. On restore it is marked failed
      # and can be retried; completed snapshots remain consistent.
      mv "${backup_file}.partial" "$backup_file"
      mv "${runtime_file}.partial" "$runtime_file"
      (cd /backups && sha256sum "contour-${stamp}.dump" "contour-${stamp}.runtime.tar.gz") > "${backup_file}.sha256"
      echo "Database and runtime backup completed: ${stamp} (tar status ${runtime_status})"
    else
      echo "Runtime backup failed: ${stamp}; partial files preserved" >&2
    fi
  else
    echo "Backup failed: ${stamp}" >&2
  fi
  sleep 21600
done
