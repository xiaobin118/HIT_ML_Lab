#!/usr/bin/env bash
# 按用户要求：22:00 启动扩散模型训练，次日 10:00 前完成评估与报告。
# 时间预算：22:00 → 08:00 训练（10 小时，约 18 万步）；08:00 → 09:00 评估、出图、生成报告。
set -u
cd "$(dirname "$0")"
LOG=diffusion_run/overnight.log
mkdir -p diffusion_run
{
  echo "=== $(date '+%F %T') overnight 调度启动 ==="
  target=$(date -d "$(date +%F) 22:00:00" +%s)
  now=$(date +%s)
  if [ "$now" -lt "$target" ]; then
    echo "等待至 22:00，需等待 $(( (target - now) / 60 )) 分钟 ..."
    sleep $((target - now))
  fi
  echo "=== $(date '+%F %T') 开始训练（180,000 步，上限 10 小时）==="
  python3 diffusion_train.py --steps 180000 --max-hours 10.0
  echo "=== $(date '+%F %T') 训练进程结束（exit=$?），开始评估 ==="
  python3 diffusion_eval.py
  echo "=== $(date '+%F %T') 评估结束（exit=$?），开始生成插图 ==="
  python3 diffusion_report.py
  echo "=== $(date '+%F %T') 插图结束（exit=$?），重新生成报告 ==="
  python3 gen_report.py
  echo "=== $(date '+%F %T') OVERNIGHT PIPELINE DONE（exit=$?）==="
} >> "$LOG" 2>&1
