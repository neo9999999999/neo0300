#!/bin/bash
# 기존 walk_forward 종료 대기 → overnight optimizer 자동 시작

cd "$(dirname "$0")"
source venv/bin/activate

CHAIN_LOG=/tmp/chain_overnight.log
echo "===== 체이닝 시작: $(date) =====" | tee "$CHAIN_LOG"

# 1) walk_forward 종료 대기
echo "[대기] run_walk_forward.py 종료 기다리는 중..." | tee -a "$CHAIN_LOG"
while pgrep -f "run_walk_forward.py" > /dev/null; do
  sleep 60
done
echo "[OK] walk_forward 종료 확인: $(date)" | tee -a "$CHAIN_LOG"

# 2) Overnight Optimizer 실행
echo "" | tee -a "$CHAIN_LOG"
echo "[시작] Overnight Optimizer: $(date)" | tee -a "$CHAIN_LOG"
N_RANDOM=1500 N_REFINE=1000 python3 -W ignore -u overnight_optimizer.py 2>&1 | tee -a "$CHAIN_LOG"

echo "" | tee -a "$CHAIN_LOG"
echo "===== 체이닝 완료: $(date) =====" | tee -a "$CHAIN_LOG"
