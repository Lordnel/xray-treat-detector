#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/xray-treat-detector"
SERVICE_NAME="xray-app"
PROJECT_NAME="xray-treat-detector"

echo "==> Deploy in: $APP_DIR"

echo "==> Stop old containers"
docker-compose down --remove-orphans || true

echo "==> Remove old container (if exists)"
docker rm -f "${PROJECT_NAME}_${SERVICE_NAME}_1" 2>/dev/null || true

echo "==> Build"
docker-compose build

echo "==> Up"
docker-compose up -d --force-recreate --remove-orphans

echo "==> Status"
docker-compose ps

echo "✅ Deploy done"
