#!/usr/bin/env bash
# Start the dashboard on 127.0.0.1:${PORT:-3200} (reach it through an SSH tunnel).
#
#   dashboard/run.sh          # development: reloads on code changes
#   dashboard/run.sh prod     # production build, then serve it
set -euo pipefail
cd "$(dirname "$0")"
export NEXT_TELEMETRY_DISABLED=1

if [[ ! -f .env.local ]] || ! grep -q '^SESSION_SECRET=' .env.local; then
    echo "no login configured yet: (cd dashboard && npm run set-password)" >&2
    exit 1
fi
[[ -d node_modules ]] || npm ci

case "${1:-dev}" in
    dev)  npm run dev ;;
    prod) npm run build && npm run start ;;
    *)    echo "usage: $0 [dev|prod]" >&2; exit 2 ;;
esac
