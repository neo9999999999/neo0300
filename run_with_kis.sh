#!/bin/bash
# KIS API 키 로드 + Streamlit + cloudflared 자동 실행
# 사용: ./run_with_kis.sh

cd "$(dirname "$0")"
source venv/bin/activate

# .env 로드
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo "✓ .env 로드: KIS_APP_KEY=${KIS_APP_KEY:0:10}..."
fi

# 기존 프로세스 정리
pkill -f "streamlit run app.py" 2>/dev/null
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 2

mkdir -p logs

# Streamlit 백그라운드
nohup streamlit run app.py --server.port 8501 --server.headless true \
  --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &
echo "Streamlit PID: $!"
sleep 5

# 헬스 체크
if curl -s http://localhost:8501/_stcore/health | grep -q ok; then
  echo "✓ Streamlit OK"
else
  echo "✗ Streamlit 실패"
  tail -5 logs/streamlit.log
  exit 1
fi

# cloudflared 백그라운드
nohup cloudflared tunnel --url http://localhost:8501 > logs/tunnel.log 2>&1 &
echo "Cloudflared PID: $!"
sleep 8

# URL 추출
URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/tunnel.log | tail -1)
if [ -n "$URL" ]; then
  echo "✓ URL: $URL"
else
  echo "✗ Cloudflared URL 추출 실패"
  tail -5 logs/tunnel.log
fi

echo ""
echo "비밀번호: 123456"
