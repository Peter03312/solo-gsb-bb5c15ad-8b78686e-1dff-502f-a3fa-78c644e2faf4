#!/bin/sh
# 容器启动链（依赖已在镜像构建阶段装齐，此处不再联网安装）：
#   1) 对持久卷上的 SQLite 数据库执行幂等初始化/轻量迁移（新卷建库、旧卷补齐）
#   2) 启动 API；FastAPI lifespan 启动时自动接管未决任务（pending 立即领取、
#      心跳陈旧的 running 抢占重跑），守护线程之后周期重扫
set -eu

echo "[entrypoint] using bundled dependencies (installed at image build time)"
echo "[entrypoint] initializing database at ${EXAM_DB_PATH:-/data/exam.db} ..."
EXAM_DB_PATH="${EXAM_DB_PATH:-/data/exam.db}" python3 -c "from app import database; database.init_db()"

echo "[entrypoint] starting API on 0.0.0.0:8000 (pending/stale tasks will be reclaimed on startup) ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
