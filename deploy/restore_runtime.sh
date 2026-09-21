#!/usr/bin/env bash
# Restore snapshots into a NEW Docker volume; never touch the active runtime.
set -euo pipefail
backup_file=${1:?Usage: deploy/restore_runtime.sh backups/file.runtime.tar.gz contour_restore_runtime_name}
target_volume=${2:?Specify a new Docker volume name}
if [[ ! "$target_volume" =~ ^contour_restore_runtime_[a-zA-Z0-9_]+$ ]]; then
  echo 'Use contour_restore_runtime_ followed by letters, digits, underscores.' >&2
  exit 2
fi
if docker volume inspect "$target_volume" >/dev/null 2>&1; then
  echo 'Target volume already exists; refusing to overwrite it.' >&2
  exit 2
fi
backup_parent=$(cd "$(dirname "$backup_file")" && pwd)
backup_name=$(basename "$backup_file")
docker volume create "$target_volume" >/dev/null
docker run --rm --network none \
  --mount "type=volume,source=$target_volume,target=/restore" \
  --mount "type=bind,source=$backup_parent,target=/backup,readonly" \
  postgres:16-bookworm tar -xzf "/backup/$backup_name" -C /restore
echo "Restored into $target_volume. Verify with the restored database before switching the API."
