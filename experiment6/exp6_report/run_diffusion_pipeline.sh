#!/usr/bin/env bash
# 训练结束后自动执行：评估 → 生成扩散模型插图 → 重新生成报告
set -u
cd "$(dirname "$0")"
LOG=diffusion_run/pipeline.log
mkdir -p diffusion_run
{
  echo "=== $(date '+%F %T') watcher started ==="
  while pgrep -f 'python3 diffusion_train.py' > /dev/null 2>&1; do sleep 30; done
  sleep 10
  echo "=== $(date '+%F %T') training process exited ==="
  echo "=== start diffusion_eval.py ==="
  python3 diffusion_eval.py
  echo "=== start diffusion_report.py ==="
  python3 diffusion_report.py
  echo "=== start gen_report.py ==="
  python3 gen_report.py
  echo "=== PIPELINE DONE $(date '+%F %T') ==="
} >> "$LOG" 2>&1
