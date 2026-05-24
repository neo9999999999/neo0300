"""
100x 그리드 백테스트 결과를 GRID_100_GUIDE.md 로 정리
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")
df = pd.read_csv(CACHE / "grid_100_summary.csv")

md = []
md.append(f"# 100가지 백테스트 그리드 서치 결과")
md.append(f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
md.append("자본 1,000만 / 종목당 10% / 180일 보유\n")

md.append("## 필터 10가지")
md.append("- F01_회피6_기본 — 회피 6개만")
md.append("- F02_slope_05 — slope60 ≥ 0.5 (상승추세)")
md.append("- F03_slope_10 — slope60 ≥ 1.0 (강한 상승추세)")
md.append("- F04_외인상위25 — 외국인 20일 누적매수 상위 25%")
md.append("- F05_slope05_For양 — 상승추세 + 외인 누적매수 양수")
md.append("- F06_외인기관동시매수 — 외인+기관 동시 20일 순매수")
md.append("- F07_눌림목30_10 — 52주高 -30~-10% (눌림목)")
md.append("- F08_안정추세60 — 60일 -10~+30%")
md.append("- F09_slope05_과열X — 상승추세 + 1년 -20~+60% (과열 제외)")
md.append("- F10_KOSDAQ만 — KOSDAQ 종목만\n")

md.append("## 정렬 10가지")
md.append("- S01_거래대금낮 - 거래대금 ↑ → 소형주 가중")
md.append("- S02_거래대금높 - 거래대금 ↑ → 대형주 가중")
md.append("- S03_점수낮 / S04_점수높 - 점수 기준")
md.append("- S05_변동성낮 / S06_변동성높 - vol60 기준")
md.append("- S07_slope강한 - slope60 ↑")
md.append("- S08_외인매수강 - For_20d ↑")
md.append("- S09_52주고점근접 - pos_252_high ↑")
md.append("- S10_랜덤\n")

# 모드별 Top3 (수익률 기준)
def top3_section(df, set_label, title):
    sub = df[df["set"] == set_label].sort_values("ret_pct", ascending=False).head(3).copy()
    md.append(f"## {title}\n")
    md.append("| 순위 | 필터 | 정렬 | 매수 | 슈퍼위너 | 100%+ | 50%+ | 평균peak | 평균ret | 승률 | 최종자본 | **수익률** | 수익금 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        md.append(f"| {i} | {r['filter']} | {r['sort']} | {int(r['n'])} | "
                  f"{int(r['SW_n'])} ({r['SW_rate']:.1f}%) | "
                  f"{int(r['w100_n'])} ({r['w100_rate']:.1f}%) | "
                  f"{int(r['w50_n'])} ({r['w50_rate']:.1f}%) | "
                  f"{r['avg_peak']:.1f}% | {r['avg_ret']:.1f}% | "
                  f"{r['winrate']:.1f}% | {r['final']:,.0f} | "
                  f"**+{r['ret_pct']:.1f}%** | {r['profit']:,.0f} |")
    md.append("")
    return sub


md.append("# OOS Test (2024-2026)\n")
top_oos_d = top3_section(df, "OOS_daily_1", "OOS · 매일 1건 (TOP3)")
top_oos_w2 = top3_section(df, "OOS_weekly_2", "OOS · 주 2건 (TOP3)")
top_oos_w3 = top3_section(df, "OOS_weekly_3", "OOS · 주 3건 (TOP3)")

md.append("# 전체기간 (2020-04 ~ 2025-08)\n")
top_full_d = top3_section(df, "FULL_daily_1", "전체 · 매일 1건 (TOP3)")
top_full_w2 = top3_section(df, "FULL_weekly_2", "전체 · 주 2건 (TOP3)")
top_full_w3 = top3_section(df, "FULL_weekly_3", "전체 · 주 3건 (TOP3)")

# OOS Test 100%+/SW 비율 기준 best
md.append("# OOS 슈퍼위너 비율 기준 TOP3\n")
for set_label, title in [("OOS_daily_1", "OOS · 매일 1건 · SW 비율 기준"),
                          ("OOS_weekly_2", "OOS · 주 2건 · SW 비율 기준"),
                          ("OOS_weekly_3", "OOS · 주 3건 · SW 비율 기준")]:
    sub = df[df["set"] == set_label].sort_values("SW_rate", ascending=False).head(3)
    md.append(f"### {title}\n")
    md.append("| 순위 | 필터 | 정렬 | 매수 | SW% | 100%+% | 50%+% | 수익률 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        md.append(f"| {i} | {r['filter']} | {r['sort']} | {int(r['n'])} | "
                  f"{r['SW_rate']:.1f}% | {r['w100_rate']:.1f}% | "
                  f"{r['w50_rate']:.1f}% | +{r['ret_pct']:.1f}% |")
    md.append("")

# 최강 요약
md.append("# 최종 추천 (모드별 1등)\n")
md.append("## OOS Test (2024-2026 검증)")
best_d = top_oos_d.iloc[0]
best_w2 = top_oos_w2.iloc[0]
best_w3 = top_oos_w3.iloc[0]
md.append(f"\n**매일 1건 모드** — `{best_d['filter']}` + `{best_d['sort']}`")
md.append(f"- {int(best_d['n'])}건 매수, 슈퍼위너 {int(best_d['SW_n'])}개, 100%+ {int(best_d['w100_n'])}개, 50%+ {int(best_d['w50_n'])}개")
md.append(f"- 자본 1,000만 → **{best_d['final']:,.0f}원 (+{best_d['ret_pct']:.0f}%)**\n")

md.append(f"**주 2건 모드** — `{best_w2['filter']}` + `{best_w2['sort']}`")
md.append(f"- {int(best_w2['n'])}건 매수, 슈퍼위너 {int(best_w2['SW_n'])}개, 100%+ {int(best_w2['w100_n'])}개, 50%+ {int(best_w2['w50_n'])}개")
md.append(f"- 자본 1,000만 → **{best_w2['final']:,.0f}원 (+{best_w2['ret_pct']:.0f}%)**\n")

md.append(f"**주 3건 모드** — `{best_w3['filter']}` + `{best_w3['sort']}`")
md.append(f"- {int(best_w3['n'])}건 매수, 슈퍼위너 {int(best_w3['SW_n'])}개, 100%+ {int(best_w3['w100_n'])}개, 50%+ {int(best_w3['w50_n'])}개")
md.append(f"- 자본 1,000만 → **{best_w3['final']:,.0f}원 (+{best_w3['ret_pct']:.0f}%)**\n")

md.append("## 전체기간 (2020-04 ~ 2025-08)\n")
fd = top_full_d.iloc[0]; fw2 = top_full_w2.iloc[0]; fw3 = top_full_w3.iloc[0]
md.append(f"- 매일 1건: `{fd['filter']}` + `{fd['sort']}` → +{fd['ret_pct']:.0f}% ({fd['final']:,.0f}원)")
md.append(f"- 주 2건:   `{fw2['filter']}` + `{fw2['sort']}` → +{fw2['ret_pct']:.0f}% ({fw2['final']:,.0f}원)")
md.append(f"- 주 3건:   `{fw3['filter']}` + `{fw3['sort']}` → +{fw3['ret_pct']:.0f}% ({fw3['final']:,.0f}원)\n")

md.append("# 산출 파일\n")
md.append("- `cache/grid_100_summary.csv` — 600조합 전체 결과")
md.append("- `cache/grid_100_top_combos.json` — Top3 조합 목록")
md.append("- `cache/BEST_POOL_daily_1_OOS_TOP1.csv` — OOS 매일1건 최고 추천방식의 풀 전체")
md.append("- `cache/BEST_POOL_weekly_2_OOS_TOP1.csv` — OOS 주2건 최고 추천방식의 풀 전체")
md.append("- `cache/BEST_POOL_daily_1_FULL.csv` — 매일1건 풀 전체 (2020-2026)")
md.append("- `cache/BEST_TRADES_FULL_daily_1.csv` — 전체기간 매일1건 Top3 매수 종목")
md.append("- `cache/BEST_TRADES_FULL_weekly_2.csv` — 전체기간 주2건 Top3 매수 종목")
md.append("- `cache/BEST_TRADES_FULL_weekly_3.csv` — 전체기간 주3건 Top3 매수 종목")
md.append("- `cache/BEST_TRADES_OOS_daily_1.csv` — OOS 매일1건 Top3 매수 종목")
md.append("- `cache/BEST_TRADES_OOS_weekly_2.csv` — OOS 주2건 Top3 매수 종목")

text = "\n".join(md)
with open("GRID_100_GUIDE.md", "w", encoding="utf-8") as f:
    f.write(text)
print(f"GRID_100_GUIDE.md 생성 완료 ({len(text):,} chars)")
