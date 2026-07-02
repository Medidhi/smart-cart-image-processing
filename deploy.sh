#!/usr/bin/env bash
# Deploy the Pi side of grocery-detect to the Raspberry Pi 5.
#   ./deploy.sh            # sync files only
#   ./deploy.sh --run      # sync, then start the server on the Pi (with preview off)
set -euo pipefail

# Configure via env vars (do NOT hardcode secrets here):
#   PI_HOST=admin@<pi-ip>   PI_PASS=<ssh-password>   ./deploy.sh
# Better: set up SSH keys (ssh-copy-id) and leave PI_PASS unset.
PI_HOST="${PI_HOST:-admin@192.168.68.62}"
PI_PASS="${PI_PASS:-}"
DEST="~/grocery-detect"

# If PI_PASS is set, prefix ssh/rsync with sshpass; otherwise use plain ssh (keys).
if [[ -n "$PI_PASS" ]]; then
  SSH_PREFIX=(sshpass -p "$PI_PASS")
else
  SSH_PREFIX=()
fi

echo "[deploy] syncing pi/ -> $PI_HOST:$DEST"
"${SSH_PREFIX[@]}" rsync -az --delete \
  -e "ssh -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/pi/" "$PI_HOST:$DEST/"

# Link the pre-installed Hailo model zoo so models/ resolves on the Pi.
"${SSH_PREFIX[@]}" ssh -o StrictHostKeyChecking=no "$PI_HOST" '
  cd ~/grocery-detect &&
  mkdir -p models &&
  for m in yolov8s_h8 yolov8m_h10 yolov11m_h10; do
    [ -f /usr/share/hailo-models/$m.hef ] && ln -sf /usr/share/hailo-models/$m.hef models/$m.hef;
  done &&
  echo "[deploy] models: $(ls models)"
'

if [[ "${1:-}" == "--run" ]]; then
  echo "[deploy] starting server on Pi (Ctrl-C to stop) …"
  "${SSH_PREFIX[@]}" ssh -t -o StrictHostKeyChecking=no "$PI_HOST" \
    'cd ~/grocery-detect && python3 server.py'
fi
echo "[deploy] done"
