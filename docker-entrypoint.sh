#!/bin/sh
# PolyBotKing entrypoint
# Auto-fixes permissions on bind-mounted volumes so the bot can write
# its SQLite database and log files regardless of host UID.

set -e

# Ensure runtime directories exist (idempotent).
mkdir -p /app/data /app/logs

# If we are running as root, fix ownership of mounted volumes and drop to botuser.
# If we are already non-root, just continue (permissions assumed correct).
if [ "$(id -u)" = "0" ]; then
    chown -R botuser:botuser /app/data /app/logs 2>/dev/null || true
    # Re-exec the command as botuser
    exec runuser -u botuser -- "$@"
fi

exec "$@"
