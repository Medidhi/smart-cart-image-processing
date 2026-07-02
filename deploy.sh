#!/usr/bin/env bash
# Deploy the Pi side of grocery-detect to the Raspberry Pi 5.
#   ./deploy.sh            # sync files only
#   ./deploy.sh --run      # sync, then start the server on the Pi (with preview off)
set -euo pipefail

PI_HOST="${PI_HOST:-admin@192.168.68.62}"
PI_PASS="${PI_PASS:-admin}"
DEST="~/grocery-detect"

echo "[deploy] syncing pi/ -> $PI_HOST:$DEST"
sshpass -p "$PI_PASS" rsync -az --delete \
  -e "ssh -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/pi/" "$PI_HOST:$DEST/"

# Link the pre-installed Hailo model zoo so models/ resolves on the Pi.
sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=no "$PI_HOST" '
  cd ~/grocery-detect &&
  mkdir -p models &&
  for m in yolov8s_h8 yolov8m_h10 yolov11m_h10; do
    [ -f /usr/share/hailo-models/$m.hef ] && ln -sf /usr/share/hailo-models/$m.hef models/$m.hef;
  done &&
  echo "[deploy] models: $(ls models)"
'

if [[ "${1:-}" == "--run" ]]; then
  echo "[deploy] starting server on Pi (Ctrl-C to stop) …"
  sshpass -p "$PI_PASS" ssh -t -o StrictHostKeyChecking=no "$PI_HOST" \
    'cd ~/grocery-detect && python3 server.py'
fi
echo "[deploy] done"
