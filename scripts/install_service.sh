#!/usr/bin/env bash
# Install and start the crypto-oracle systemd service.
# Run with: bash scripts/install_service.sh

set -euo pipefail

SERVICE_NAME="crypto-oracle"
SERVICE_FILE="deploy/crypto-oracle.service"
SYSTEMD_DIR="/etc/systemd/system"

cd "$(dirname "$0")/.."

echo "Installing ${SERVICE_NAME} paper trader service..."
sudo cp "deploy/crypto-oracle.service"       "${SYSTEMD_DIR}/crypto-oracle.service"

echo "Installing ${SERVICE_NAME} nightly learning timer..."
sudo cp "deploy/crypto-oracle-learn.service" "${SYSTEMD_DIR}/crypto-oracle-learn.service"
sudo cp "deploy/crypto-oracle-learn.timer"   "${SYSTEMD_DIR}/crypto-oracle-learn.timer"

sudo systemctl daemon-reload

sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

sudo systemctl enable crypto-oracle-learn.timer
sudo systemctl start  crypto-oracle-learn.timer

echo ""
echo "Services installed and started."
echo ""
echo "Paper trader:"
echo "  Status:   sudo systemctl status crypto-oracle"
echo "  Logs:     journalctl -u crypto-oracle -f"
echo "  Stop:     sudo systemctl stop crypto-oracle"
echo "  Restart:  sudo systemctl restart crypto-oracle"
echo ""
echo "Learning cycle (runs nightly at midnight UTC):"
echo "  Timer status:  sudo systemctl status crypto-oracle-learn.timer"
echo "  Run now:       sudo systemctl start crypto-oracle-learn.service"
echo "  Logs:          journalctl -u crypto-oracle-learn -f"
echo ""
echo "Health endpoints (once paper trader is running):"
echo "  curl http://localhost:8080/health"
echo "  curl http://localhost:8080/status"
echo "  curl http://localhost:8080/trades"
