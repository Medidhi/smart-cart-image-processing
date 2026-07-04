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

# If PI_PASS is set, prefix ssh/rsync with sshpass; otherwise use plain ssh
# (keys). NOTE: expanding an empty array under `set -u` errors on the bash 3.2
# that macOS ships, so wrap the prefix in a function instead.
run() {
  if [[ -n "$PI_PASS" ]]; then
    sshpass -p "$PI_PASS" "$@"
  else
    "$@"
  fi
}

echo "[deploy] syncing pi/ -> $PI_HOST:$DEST"
run rsync -az --delete \
  -e "ssh -o StrictHostKeyChecking=no" \
  "$(dirname "$0")/pi/" "$PI_HOST:$DEST/"

# Link the pre-installed Hailo model zoo so models/ resolves on the Pi.
# A custom-compiled .hef (e.g. grocery_yolov8n.hef, see training/HAILO.md)
# dropped into local pi/models/ is carried up by the rsync above; the guard
# below never overwrites a real file with a symlink.
run ssh -o StrictHostKeyChecking=no "$PI_HOST" '
  cd ~/grocery-detect &&
  mkdir -p models &&
  for m in yolov8s_h8 yolov8m_h10 yolov11m_h10; do
    if [ -f /usr/share/hailo-models/$m.hef ] && { [ ! -e models/$m.hef ] || [ -L models/$m.hef ]; }; then
      ln -sf /usr/share/hailo-models/$m.hef models/$m.hef;
    fi;
  done &&
  echo "[deploy] models: $(ls models)"
'

if [[ "${1:-}" == "--run" ]]; then
  echo "[deploy] starting server on Pi (Ctrl-C to stop) …"
  run ssh -t -o StrictHostKeyChecking=no "$PI_HOST" \
    'cd ~/grocery-detect && python3 server.py'
fi
echo "[deploy] done"
