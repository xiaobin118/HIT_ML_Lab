#!/usr/bin/env bash
# 目标：次日 08:00 前交付。20:50 开始训练，训练上限 9.4 小时（约 06:15 结束），
# 随后自动评估、出图、重新生成报告（预计 07:00 前完成），留出人工核查时间。
set -u
cd "$(dirname "$0")"
LOG=diffusion_run/overnight.log
mkdir -p diffusion_run
{
  echo "=== $(date '+%F %T') 立即启动训练（目标 08:00 前交付）==="
  python3 diffusion_train.py --steps 172000 --max-hours 9.4
  echo "=== $(date '+%F %T') 训练结束（exit=$?），开始评估 ==="
  python3 diffusion_eval.py
  echo "=== $(date '+%F %T') 评估结束（exit=$?），生成插图 ==="
  python3 diffusion_report.py
  echo "=== $(date '+%F %T') 插图结束（exit=$?），重新生成报告 ==="
  python3 gen_report.py
  echo "=== $(date '+%F %T') PIPELINE DONE（exit=$?）==="
} >> "$LOG" 2>&1
