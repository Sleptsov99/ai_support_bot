#!/usr/bin/env bash
# Upload the project to the server and (re)start the container. Run from the project folder on your laptop:
#   SSH_KEY=~/.ssh/your_key ./deploy/deploy.sh
set -euo pipefail
HOST=${HOST:-root@159.195.248.101}
SSH_KEY=${SSH_KEY:-~/.ssh/id_ed25519}
APP_DIR=/opt/ai_support_bot
SSH="ssh -i $SSH_KEY"

cd "$(dirname "$0")/.."
rsync -az --delete -e "$SSH" \
    --exclude .venv --exclude .git --exclude .env --exclude __pycache__ --exclude .pytest_cache \
    ./ "$HOST:$APP_DIR/"

$SSH "$HOST" "cd $APP_DIR && test -f .env || { echo 'No .env on the server: see DEPLOY.md step 4'; exit 1; }
              docker compose up -d --build && docker image prune -f >/dev/null && docker compose ps"
echo "Deployed: https://ai.avalink.co/demo"
