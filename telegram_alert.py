"""
텔레그램 V/S/A/B 등급별 추천 알림.

[사전 준비]
1. @BotFather 로 새 봇 만들기 → TOKEN 받기
2. 본인 텔레그램에서 봇 검색 → /start
3. https://api.telegram.org/bot<TOKEN>/getUpdates 에서 chat_id 확인
4. 환경변수 설정:
   export TG_BOT_TOKEN='123:abc'
   export TG_CHAT_ID='123456789'

[실행]
python3 telegram_alert.py             # 캐시 데이터 기반 즉시 발송
python3 telegram_alert.py --live      # 실시간 스캔 후 발송 (5~10분)
python3 telegram_alert.py --dry-run   # 메시지만 출력 (발송 안 함)

[자동화]
cron 예시 (매주 평일 15:25 KST):
  25 15 * * 1-5 cd /Users/neo/Desktop/jongga_picker && \
      ./venv/bin/python3 telegram_alert.py --live >> logs/tg.log 2>&1
"""
import os
import sys
import json
import argparse
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

from grade import (
    GRADE_INFO, GRADE_WEIGHTS, PRESETS_4,
    classify_one, classify_candidates, build_grade_buckets,
    build_ensemble_all_enriched,
)


TG_API = "https://api.telegram.org/bot{token}/{method}"


def send_message(token: str, chat_id: str, text: str,
                   parse_mode: str = "HTML") -> dict:
    url = TG_API.format(token=token, method="sendMessage")
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Telegram error: {e}")
        return {"ok": False, "error": str(e)}


def format_grade_section(grade: str, sub: pd.DataFrame) -> str:
    info = GRADE_INFO[grade]
    header = f"{info['emoji']} <b>{info['name']}</b> · {info['weight_str']}"
    if sub.empty:
        return f"{header}\n   <i>오늘은 없음</i>"
    lines = [header]
    for _, r in sub.iterrows():
        name = r.get("Name", "")
        code = r.get("Code", "")
        close = int(r.get("Close", 0) or 0)
        cr = r.get("ChangeRatio", 0) or 0
        score = r.get("avg_score", r.get("Score", 0)) or 0
        n_p = int(r.get("n_presets", 0) or 0)
        n_shares = int(GRADE_WEIGHTS[grade] / close) if close > 0 else 0
        invest = n_shares * close
        lines.append(
            f"   <b>{name}</b> ({code})\n"
            f"      종가 {close:,}원 ({cr:+.2f}%) · 점수 {score:.1f} · {n_p}/4\n"
            f"      → {n_shares}주 매수 = {invest:,}원"
        )
    return "\n".join(lines)


def build_message(buckets, source: str = "cache", scan_time = None) -> str:
    now = scan_time or datetime.now()
    time_str = now.strftime("%Y-%m-%d") if hasattr(now, "strftime") else str(now)
    source_str = "🔴 실시간 스캔" if source == "live" else "📦 캐시 데이터"

    parts = []
    parts.append(f"📋 <b>오늘의 V/S/A/B 추천</b>")
    parts.append(f"<i>{source_str} · {time_str}</i>")
    parts.append("")

    total_invest = 0
    counts = {}
    for g in ["V", "S", "A", "B"]:
        sub = buckets[g]
        counts[g] = len(sub)
        parts.append(format_grade_section(g, sub))
        parts.append("")
        for _, r in sub.iterrows():
            close = r.get("Close", 0) or 0
            if close > 0:
                total_invest += int(GRADE_WEIGHTS[g] / close) * close

    parts.append("━━━━━━━━━━━━━━")
    parts.append(
        f"💰 <b>총 매수 필요액: {total_invest:,}원</b>\n"
        f"📊 V {counts['V']} · S {counts['S']} · A {counts['A']} · B {counts['B']}"
    )
    parts.append("")
    parts.append("📅 <b>매도: 매수 후 180일 자동 청산</b>")
    parts.append("⚠️ 손절/익절 없음")

    return "\n".join(parts)


def get_picks_from_cache():
    """캐시 enriched에서 가장 최근 거래일 후보 빌드."""
    df = build_ensemble_all_enriched()
    if df.empty:
        return None, None
    last_date = df["Date"].max()
    df_today = df[df["Date"] == last_date].copy()
    df_today = classify_candidates(df_today)
    buckets = build_grade_buckets(df_today, vs_max=3, ab_only_top1=True)
    return buckets, last_date


def get_picks_live():
    """실시간 스캔."""
    from config import FilterConfig, SignalParams
    from scanner import scan_ensemble

    fc = FilterConfig(
        min_amount=50 * 100_000_000,
        min_marcap=2000 * 100_000_000,
        change_min=7.0,
        change_max=25.0,
        include_kospi=False, include_kosdaq=True,
    )
    params = SignalParams()

    def cb(i, t):
        if i % 50 == 0:
            print(f"  진행: {i}/{t}")

    print("실시간 스캔 시작 (5~10분)...")
    df = scan_ensemble(fc, params, PRESETS_4, progress_callback=cb)
    if df.empty:
        return None, None
    df = df[df["Market"] == "KOSDAQ"]
    if "TradeType" in df.columns:
        df = df[df["TradeType"] == "돌파매매"]
    df = classify_candidates(df)
    buckets = build_grade_buckets(df, vs_max=3, ab_only_top1=True)
    return buckets, datetime.now()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                     help="실시간 스캔 (default: 캐시 데이터)")
    ap.add_argument("--dry-run", action="store_true",
                     help="발송하지 않고 메시지만 출력")
    args = ap.parse_args()

    token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")

    if not args.dry_run:
        if not token or not chat_id:
            print("ERROR: 환경변수 TG_BOT_TOKEN, TG_CHAT_ID 설정 필요.")
            print("예시:")
            print("  export TG_BOT_TOKEN='123:abc'")
            print("  export TG_CHAT_ID='123456789'")
            sys.exit(1)

    # 후보 가져오기
    if args.live:
        buckets, scan_time = get_picks_live()
    else:
        buckets, scan_time = get_picks_from_cache()

    if buckets is None:
        msg = "⚠️ 데이터 없음 — 추천 종목 X"
    else:
        source = "live" if args.live else "cache"
        msg = build_message(buckets, source=source, scan_time=scan_time)

    print("=" * 60)
    print(msg)
    print("=" * 60)

    if args.dry_run:
        print("\n(--dry-run 모드: 발송 안 함)")
        return

    print("\n텔레그램 발송 중...")
    result = send_message(token, chat_id, msg)
    if result.get("ok"):
        print("✅ 발송 완료")
    else:
        print(f"❌ 발송 실패: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
