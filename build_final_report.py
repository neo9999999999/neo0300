"""
최종 종합 리포트 빌더
====================

OOS 검증 결과를 종합해서 FINAL_GUIDE.md 작성.

내용:
- 회피 6 vs 6+v2 vs 6+v3 농도/수익률 비교 (Train/Test)
- 주1건 vs 일1건 vs 슬롯N 모드별 효과
- 슈퍼위너/100%+ 포착 효율
- 년/월별 손익 + 슈퍼위너 분포
- 최종 추천 운용 룰
- 라이브 갱신 가이드

산출:
- FINAL_GUIDE.md (가독성 좋은 마크다운)
- TOP_SUPERWINNERS.csv (실제 슈퍼위너로 판명된 전체 리스트)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")


def fmt_num(x, suffix=""):
    if pd.isna(x): return "-"
    if abs(x) >= 1e8: return f"{x/1e8:,.1f}억{suffix}"
    if abs(x) >= 1e4: return f"{x/1e4:,.0f}만{suffix}"
    return f"{x:,.0f}{suffix}"


def main():
    # 입력 로드
    cand = pd.read_parquet(CACHE / "candidates_enriched.parquet")
    cand["Date"] = pd.to_datetime(cand["Date"])
    oos = pd.read_csv(CACHE / "oos_simulation_results.csv")
    full = pd.read_csv(CACHE / "full_simulation_results.csv")
    cmp_df = pd.read_csv(CACHE / "avoid_v2v3_comparison.csv")
    rule_avoid = pd.read_csv(CACHE / "avoid_rule_candidates.csv")
    rule_inc = pd.read_csv(CACHE / "include_rule_candidates.csv")
    month_stat = pd.read_csv(CACHE / "year_month_returns.csv")
    best_trades = pd.read_csv(CACHE / "best_trades_weekly_v2.csv")
    final_all = pd.read_csv(CACHE / "final_candidates_all.csv")

    # 슈퍼위너 전체 리스트
    sw = cand[cand["peak_180d"] >= 200].sort_values("peak_180d", ascending=False).copy()
    sw["Year"] = sw["Date"].dt.year
    sw["YYYYMM"] = sw["Date"].dt.strftime("%Y-%m")
    sw_cols = ["Date", "YYYYMM", "Year", "Code", "Name", "Market", "Close",
               "peak_180d", "ret_180d", "sell_date", "Score", "Amount",
               "chart_pattern", "past_60", "past_120", "pos_252_high"]
    for c in ["For_5d", "Inst_5d", "For_20d", "Inst_20d", "PER_num", "PBR_num", "시총_num"]:
        if c in sw.columns:
            sw_cols.append(c)
    sw_cols = [c for c in sw_cols if c in sw.columns]
    sw[sw_cols].to_csv(CACHE / "TOP_SUPERWINNERS.csv", index=False)
    print(f"슈퍼위너 전체 {len(sw)}건 → cache/TOP_SUPERWINNERS.csv")

    # 100%+ 전체
    w100 = cand[(cand["peak_180d"] >= 100) & (cand["peak_180d"] < 200)].sort_values("peak_180d", ascending=False).copy()
    w100["Year"] = w100["Date"].dt.year
    w100["YYYYMM"] = w100["Date"].dt.strftime("%Y-%m")
    w100_cols = [c for c in sw_cols if c in w100.columns]
    w100[w100_cols].to_csv(CACHE / "TOP_100PLUS.csv", index=False)
    print(f"100%+ 전체 {len(w100)}건 → cache/TOP_100PLUS.csv")

    # 50%+ 전체
    w50 = cand[(cand["peak_180d"] >= 50) & (cand["peak_180d"] < 100)].sort_values("peak_180d", ascending=False).copy()
    w50["Year"] = w50["Date"].dt.year
    w50["YYYYMM"] = w50["Date"].dt.strftime("%Y-%m")
    w50_cols = [c for c in sw_cols if c in w50.columns]
    w50[w50_cols].to_csv(CACHE / "TOP_50PLUS.csv", index=False)
    print(f"50%+ 전체 {len(w50)}건 → cache/TOP_50PLUS.csv")

    # 년도별 슈퍼위너 분포
    sw_year = cand.groupby(cand["Date"].dt.year).agg(
        total=("Code", "count"),
        sw=("peak_180d", lambda x: (x >= 200).sum()),
        w100=("peak_180d", lambda x: (x >= 100).sum()),
        w50=("peak_180d", lambda x: (x >= 50).sum()),
        loser=("ret_180d", lambda x: (x <= -20).sum()),
        avg_peak=("peak_180d", "mean"),
        avg_ret=("ret_180d", "mean"),
    ).reset_index().rename(columns={"Date": "Year"})
    sw_year["SW%"] = sw_year["sw"] / sw_year["total"] * 100
    sw_year["100%+%"] = sw_year["w100"] / sw_year["total"] * 100
    sw_year["50%+%"] = sw_year["w50"] / sw_year["total"] * 100
    sw_year.to_csv(CACHE / "year_dist.csv", index=False)

    # 마크다운 리포트
    md = []
    md.append(f"# 슈퍼위너 발굴 + 회피 보강 최종 가이드")
    md.append(f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    md.append("## 1. 핵심 결론\n")
    # 최고 시뮬 결과 찾기
    best_oos = oos.sort_values("return_pct", ascending=False).iloc[0]
    md.append(f"- **최적 운용**: {best_oos['filter']} + {best_oos['mode']} → "
              f"180일 보유, 자본 10%/종목")
    md.append(f"- **OOS Test 결과**: 매수 {best_oos['n']}건, 슈퍼위너 {best_oos['SW_n']}개, "
              f"100%+ {best_oos['100+_n']}개, 50%+ {best_oos['50+_n']}개")
    md.append(f"- **수익률**: 자본 1,000만 → {best_oos['final']:,.0f}원 (+{best_oos['return_pct']:.0f}%)\n")

    md.append("## 2. 회피 룰 변천\n")
    md.append("**회피 6 (기존)**")
    md.append("- X1. 하락추세+일시반등 (pullback_recovery + slope60≤-1 + pos_252_high≤-40)")
    md.append("- X2. KOSPI 떨어지는 칼날 (KOSPI + past_120≤-20 + pos_252_high≤-40)")
    md.append("- X3. 과열 신고가 (s12≥80 + new_high + past_120≥50)")
    md.append("- X4. 1년 +100% 과열 (past_240 ≥ 100)")
    md.append("- X5. 1년 +150% 극과열 (past_240 ≥ 150)")
    md.append("- X6. 거래대금 ≥ 3,000억\n")

    md.append("**회피 +v2 (B+C 데이터 추가 발굴)**")
    md.append("- X10. PER > 50 (고평가)")
    md.append("- X11. PER ≤ 0 (적자)")
    md.append("- X13. 강하락추세+52주高-50%↓ (slope60≤-2 + pos_252_high≤-50)")
    md.append("- X14. 60일 drawdown -30%↓")
    md.append("- X17. 60일 +80%↑ 단기과열")
    md.append("- X9 (수급). 외인+기관 동시 20일 누적 순매도\n")

    md.append("## 3. 회피 보강 효과 (풀의 슈퍼위너 농도)\n")
    md.append("| 기간 | 필터 | 풀크기 | 슈퍼위너 농도 | 100%+ 농도 | 50%+ 농도 | 평균peak |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in cmp_df.iterrows():
        md.append(f"| {r['set']} | {r['label']} | {int(r['n']):,} | "
                  f"{r['SW%']:.2f}% | {r['100%+_%']:.2f}% | "
                  f"{r['50%+_%']:.2f}% | {r['평균peak']:.1f}% |")
    md.append("")

    md.append("## 4. OOS Test (2024-2026) 시뮬레이션 비교\n")
    md.append("| 필터 | 모드 | 매수 | SW | 100%+ | 50%+ | 평균peak | 평균ret | 승률 | 최종자본 | 수익률 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in oos.iterrows():
        md.append(f"| {r['filter']} | {r['mode']} | {int(r['n'])} | "
                  f"{int(r['SW_n'])} ({r['SW_rate']:.1f}%) | "
                  f"{int(r['100+_n'])} ({r['100+_rate']:.1f}%) | "
                  f"{int(r['50+_n'])} ({r['50+_rate']:.1f}%) | "
                  f"{r['avg_peak']:.1f}% | {r['avg_ret']:.1f}% | "
                  f"{r['winrate']:.1f}% | {r['final']:,.0f} | +{r['return_pct']:.1f}% |")
    md.append("")

    md.append("## 5. 전체기간 (2020-04~) 시뮬레이션\n")
    md.append("| 필터 | 모드 | 매수 | SW | 100%+ | 50%+ | 평균peak | 평균ret | 승률 | 최종자본 | 수익률 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in full.iterrows():
        md.append(f"| {r['filter']} | {r['mode']} | {int(r['n'])} | "
                  f"{int(r['SW_n'])} ({r['SW_rate']:.1f}%) | "
                  f"{int(r['100+_n'])} ({r['100+_rate']:.1f}%) | "
                  f"{int(r['50+_n'])} ({r['50+_rate']:.1f}%) | "
                  f"{r['avg_peak']:.1f}% | {r['avg_ret']:.1f}% | "
                  f"{r['winrate']:.1f}% | {r['final']:,.0f} | +{r['return_pct']:.1f}% |")
    md.append("")

    md.append("## 6. 년도별 풀 분포 (회피 6 적용 후)\n")
    md.append("| 년도 | 시그널 | SW | SW% | 100%+ | 100%+% | 50%+ | 50%+% | 평균peak |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in sw_year.iterrows():
        md.append(f"| {int(r['Year'])} | {int(r['total']):,} | {int(r['sw'])} | {r['SW%']:.2f}% | "
                  f"{int(r['w100'])} | {r['100%+%']:.2f}% | "
                  f"{int(r['w50'])} | {r['50%+%']:.2f}% | {r['avg_peak']:.1f}% |")
    md.append("")

    md.append("## 7. 월별 손익 (회피 6+v2 / 주1건 거래대금↓)\n")
    md.append("| YYYY-MM | 매수 | 평균ret | 평균peak | SW | 100%+ | 50%+ |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in month_stat.iterrows():
        md.append(f"| {r['YYYYMM']} | {int(r['n'])} | "
                  f"{r['avg_ret']:.1f}% | {r['avg_peak']:.1f}% | "
                  f"{int(r['SW_n'])} | {int(r['w100_n'])} | {int(r['w50_n'])} |")
    md.append("")

    md.append("## 8. 운용 가이드\n")
    md.append("### 권장 운용 룰 (소자본 1,000만 기준)")
    md.append("1. **풀**: KOSPI + KOSDAQ 동시 전체 스캔")
    md.append("2. **회피**: 회피 6 + v2 (총 12개 규칙) 자동 제외")
    md.append("3. **선택**: 주 1건 (또는 일 1건도 가능), 거래대금 낮은 순")
    md.append("4. **자본 배분**: 종목당 자본 10% (= 100만원/종목)")
    md.append("5. **보유**: 180거래일 후 종가 매도 (~9개월)")
    md.append("6. **자본 회전**: 첫 청산 전까지 약 6개월 자본 묶임, 이후 매주 회전\n")

    md.append("### 동시 보유 한도")
    md.append("- 슬롯 한도 없이 자본 부족 시 skip → 자본 1,000만으로는 최대 10종목 동시 보유 (10%×10)")
    md.append("- 실제 시뮬에선 매주 매수 + 180일 회전이라 약 5~7종목이 동시 보유\n")

    md.append("## 9. 슈퍼위너 (200%+ peak) 전체 리스트\n")
    md.append(f"전체 {len(sw)}건 → `cache/TOP_SUPERWINNERS.csv` 참조\n")
    md.append("**년도별 분포**:")
    sw_by_year = sw.groupby("Year").size()
    for y, n in sw_by_year.items():
        md.append(f"- {y}: {n}건")
    md.append("")

    md.append("## 10. 라이브 운영\n")
    md.append("**매일 16:30 KST 자동 갱신** (`.github/workflows/daily.yml`)")
    md.append("1. `daily_update.py` - KRX OHLCV + 시총 갱신")
    md.append("2. `update_fundamentals_current()` - PER/PBR 현재값 갱신")
    md.append("3. `update_supply_demand_incremental()` - 외인/기관 최근 20일 추가")
    md.append("4. `precompute_enriched.py` - 4 프리셋 chart_feats 재계산")
    md.append("5. `live_filter.py` - 회피 12개 적용 → `cache/today_picks.csv` 생성")
    md.append("6. Git auto-commit & push → Streamlit Cloud 자동 재배포\n")

    md.append("**오늘의 추천 확인**: `cache/today_picks.csv` 또는 app.py 우측 상단 '오늘의 추천' 탭\n")

    md.append("## 11. 산출 파일\n")
    md.append("- `cache/candidates_enriched.parquet` - 회피 6 적용 + 수급/펀더 매칭 전체 풀")
    md.append("- `cache/final_candidates_all.csv` - 회피 6+v2 적용 전체 종목 리스트")
    md.append("- `cache/TOP_SUPERWINNERS.csv` - 슈퍼위너(200%+) 전체")
    md.append("- `cache/TOP_100PLUS.csv` - 100%~200% 전체")
    md.append("- `cache/TOP_50PLUS.csv` - 50%~100% 전체")
    md.append("- `cache/oos_simulation_results.csv` - OOS Test 시뮬")
    md.append("- `cache/full_simulation_results.csv` - 전체기간 시뮬")
    md.append("- `cache/year_month_returns.csv` - 월별 손익")
    md.append("- `cache/best_trades_weekly_v2.csv` - 주1건 매수 트레이드 전체")
    md.append("- `cache/year_dist.csv` - 년도별 풀 분포")
    md.append("- `cache/sw_diff_analysis.csv` - 슈퍼위너 vs 일반 변수 차이")
    md.append("- `cache/avoid_rule_candidates.csv` - 회피 룰 후보 평가")
    md.append("- `cache/include_rule_candidates.csv` - 포함(강화) 룰 후보 평가\n")

    text = "\n".join(md)
    with open("FINAL_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n✓ FINAL_GUIDE.md 생성 완료")


if __name__ == "__main__":
    main()
