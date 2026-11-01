#!/usr/bin/env bash
# Generate observability/.env with random secrets for the self-hosted Langfuse stack.
# Refuses to overwrite an existing file, because changing secrets under existing data breaks it.
set -euo pipefail
cd "$(dirname "$0")"
[ -e .env ] && { echo ".env already exists; delete it (and the docker volumes) to start over" >&2; exit 1; }
rand() { openssl rand -hex "${1:-16}"; }
cat > .env <<EOF
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=$(rand 32)
SALT=$(rand 32)
ENCRYPTION_KEY=$(rand 32)
POSTGRES_PASSWORD=$(rand)
DATABASE_URL=postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/postgres
CLICKHOUSE_PASSWORD=$(rand)
REDIS_AUTH=$(rand)
MINIO_ROOT_PASSWORD=$(rand)
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=\${MINIO_ROOT_PASSWORD}
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=\${MINIO_ROOT_PASSWORD}
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=\${MINIO_ROOT_PASSWORD}
TELEMETRY_ENABLED=false
LANGFUSE_INIT_ORG_ID=opt-compass
LANGFUSE_INIT_ORG_NAME=opt-compass
LANGFUSE_INIT_PROJECT_ID=opt-compass
LANGFUSE_INIT_PROJECT_NAME=opt-compass
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-$(rand)
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-$(rand)
LANGFUSE_INIT_USER_EMAIL=admin@opt-compass.local
LANGFUSE_INIT_USER_NAME=admin
LANGFUSE_INIT_USER_PASSWORD=$(rand)
EOF
chmod 600 .env
echo "wrote observability/.env (gitignored)."
echo "Point the app at it:  export \$(grep -E '^LANGFUSE_INIT_PROJECT_(PUBLIC|SECRET)_KEY' observability/.env | sed 's/INIT_PROJECT_//' | xargs) LANGFUSE_HOST=http://localhost:3000"
