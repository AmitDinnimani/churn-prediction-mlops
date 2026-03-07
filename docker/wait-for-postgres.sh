#!/bin/sh
# =============================================================================
# wait-for-postgres.sh
# =============================================================================
# Blocks until Postgres accepts connections, then exec's the given command.
#
# Usage inside Dockerfile CMD:
#   /wait-for-postgres.sh <host> <cmd> [args...]
#
# Requires env vars: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
#
# FIX: redirect psql stderr to /dev/null so "could not connect" noise doesn't
#      spam the log on every retry — only our own message is shown.
# =============================================================================
set -e

host="$1"
shift
cmd="$@"

echo "Waiting for Postgres at $host..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' 2>/dev/null; do
  >&2 echo "  Postgres is unavailable — retrying in 2s"
  sleep 2
done

>&2 echo "Postgres is up — starting: $cmd"
exec $cmd