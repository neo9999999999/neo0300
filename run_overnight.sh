#!/bin/bash
# 밤새 자동 백테스트 + 가중치 탐색 파이프라인
# 1) Walk-Forward OOS (9 프리셋, 30~60분)
# 2) Overnight Optimizer (수천 가중치 조합, 4~6시간)

set -e
cd "$(dirname "$0")"
source venv/bin/activate

LOG=/tmp/overnight_pipeline.log
echo "===== 파이프라인 시작: $(date) =====" | tee "$LOG"

# Step 1: Walk-Forward (이미 실행 중인 게 있으면 그게 끝날 때까지 대기)
echo "" | tee -a "$LOG"
echo "[Step 1] Walk-Forward 9 프리셋 OOS 검증" | tee -a "$LOG"

# 캐시 체크
if ls cache/wf_oos_*.json 1>/dev/null 2>&1; then
  echo "  ✓ Walk-Forward 캐시 존재. Skip." | tee -a "$LOG"
else
  echo "  → 실행 중 (또는 시작)..." | tee -a "$LOG"
  python3 -W ignore -u run_walk_forward.py 2>&1 | tee -a "$LOG"
fi

# Step 2: Overnight Optimizer
echo "" | tee -a "$LOG"
echo "[Step 2] Overnight Optimizer 시작 (수천 조합 탐색)" | tee -a "$LOG"
echo "  N_RANDOM=${N_RANDOM:-1500}  N_REFINE=${N_REFINE:-1000}" | tee -a "$LOG"

N_RANDOM="${N_RANDOM:-1500}" N_REFINE="${N_REFINE:-1000}" \
  python3 -W ignore -u overnight_optimizer.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== 파이프라인 완료: $(date) =====" | tee -a "$LOG"
echo "결과: cache/wf_oos_*.json (Walk-Forward)" | tee -a "$LOG"
echo "       cache/overnight_final.json (조합 탐색)" | tee -a "$LOG"
