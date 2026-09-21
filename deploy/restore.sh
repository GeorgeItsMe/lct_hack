#!/usr/bin/env bash
# Restore into a NEW database to preserve the existing instance.
set -euo pipefail
backup_file=${1:?Usage: deploy/restore.sh backups/file.dump new_database_name}
target_db=${2:?Specify a new database name}
if [[ ! "$target_db" =~ ^contour_restore_[a-zA-Z0-9_]+$ ]]; then echo 'Use contour_restore_ followed by letters, digits, underscores.' >&2; exit 2; fi
docker compose exec -T db createdb -U contour "$target_db"
docker compose exec -T db pg_restore -U contour --exit-on-error --dbname="$target_db" < "$backup_file"
echo "Restored into $target_db. Verify before switching DATABASE_URL."
