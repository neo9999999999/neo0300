"""
Walk-Forward OOS 백그라운드 실행 스크립트.
진행 상황을 stdout으로 출력 + cache에 결과 저장.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
from datetime import datetime, timedelta
from walk_forward import walk_forward_validation


def cb(i, total, label="", phase=""):
    """진행률을 한 줄씩 stdout으로."""
    pct = i / max(total, 1) * 100
    msg = label or phase
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{i}/{total} {pct:.0f}%] {msg}",
          flush=True)


def main():
    print(f"=== Walk-Forward OOS 검증 시작 ===", flush=True)
    print(f"시작 시각: {datetime.now()}", flush=True)
    print(f"기간: 2020-01-01 ~ {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}", flush=True)
    print(f"유니버스: 시총 상위 1000", flush=True)
    print(f"학습기간: 252일 / OOS 청크: 63일", flush=True)
    print(f"예상 소요: 30~60분", flush=True)
    print(flush=True)

    t0 = time.time()
    try:
        result = walk_forward_validation(
            start_date="2020-01-01",
            end_date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            train_min_days=252,
            test_chunk_days=63,
            universe_limit=1000,
            sell_strategy="next_open",
            progress_callback=cb,
            force_refresh=False,
        )
    except KeyboardInterrupt:
        print("\n⚠️ 중단됨", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(flush=True)
    print(f"=== 완료 ({elapsed/60:.1f}분 소요) ===", flush=True)

    if "error" in result:
        print(f"❌ {result['error']}", flush=True)
        sys.exit(1)

    print(f"결정 시점 수: {result['n_decision_points']}회", flush=True)
    print(f"안정성: {result.get('stability_score', 0)*100:.0f}%", flush=True)
    print(flush=True)
    print("🏆 Consensus TOP 3:", flush=True)
    for i, item in enumerate(result.get("consensus_top3", [])[:3]):
        medal = ["🥇", "🥈", "🥉"][i]
        print(f"  {medal} {item['preset_name']}", flush=True)
        print(f"     TOP3 빈도: {item['votes_in_top3']}/{result['n_decision_points']}회", flush=True)
        print(f"     OOS 평균: {item.get('avg_return', 0):+.2f}%", flush=True)
        print(f"     OOS 승률: {item.get('win_rate', 0):.1f}%", flush=True)
        print(f"     OOS 샤프: {item.get('sharpe_annual', 0):.2f}", flush=True)
        print(f"     OOS 거래수: {item.get('n_trades', 0):,}", flush=True)
        print(flush=True)

    print("📊 9개 프리셋 OOS 성과:", flush=True)
    per_preset = list(result.get("per_preset_oos", {}).values())
    per_preset.sort(key=lambda p: p.get("votes_in_top3", 0), reverse=True)
    for p in per_preset:
        print(f"  {p['preset_name']:30s} | TOP3 {p['votes_in_top3']:2d}회 | "
              f"평균 {p.get('avg_return', 0):+.2f}% | "
              f"승률 {p.get('win_rate', 0):.1f}% | "
              f"샤프 {p.get('sharpe_annual', 0):.2f}", flush=True)

    print(flush=True)
    print(f"✅ 결과 캐시: cache/wf_oos_*.json", flush=True)


if __name__ == "__main__":
    main()
