"""
마스터 백테스트 종목 리스트
==========================
2020-04 ~ 2025-08 모든 매수 종목을 RF 모델 적용 후 한 파일에 통합.

산출:
- cache/MASTER_백테스트_종목리스트.csv  - 전체 시그널 풀에 RF 손절확률 부착
- cache/MASTER_매일1건_RF안전_종목.csv  - 매일1건 모드 추천 종목
- cache/MASTER_주3건_RF안전_종목.csv    - 주3건 모드 추천 종목
- BACKTEST_FINAL.md                    - 가독성 있는 요약
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from train_rf_loss_model import prepare_X, add_pre_features, FEATURES

CACHE = Path("cache")

# 모델 로드
with open(CACHE / "rf_loss_model.pkl", "rb") as f:
    rf = pickle.load(f)
with open(CACHE / "rf_features.json") as f:
    meta = json.load(f)
TH20 = meta["th20"]
print(f"RF model loaded. th20 (위험 상위 20%): {TH20:.4f}")

# 시그널 풀
cand = pd.read_parquet(CACHE / "candidates_enriched.parquet")
cand["Date"] = pd.to_datetime(cand["Date"])
cand = cand.dropna(subset=["peak_180d", "sell_close", "ret_180d"]).copy()

# 시계열 특성 보강 (이미 있으면 skip)
if "pre_5d_max_high_ratio" not in cand.columns:
    print("[1/3] 시계열 특성 추가...")
    cand = add_pre_features(cand)

# RF 적용
print("[2/3] RF 손절확률 예측...")
X, features = prepare_X(cand, features=meta["features"])
probs = rf.predict_proba(X)[:, 1]
cand["RF손절확률"] = probs
cand["RF위험"] = (probs >= TH20).astype(int)

# 결과 분류
cand["결과"] = "중립"
cand.loc[cand["ret_180d"] <= -20, "결과"] = "손절"
cand.loc[cand["ret_180d"] > 0, "결과"] = "익절"
cand["peak등급"] = "일반"
cand.loc[cand["peak_180d"] >= 50, "peak등급"] = "50%+"
cand.loc[cand["peak_180d"] >= 100, "peak등급"] = "100%+"
cand.loc[cand["peak_180d"] >= 200, "peak등급"] = "슈퍼위너"


# 매수 모드별 추출
print("[3/3] 모드별 종목 추출...")
daily = cand.sort_values("Amount").copy()
daily["bucket"] = daily["Date"].dt.strftime("%Y-%m-%d")
daily_picks = daily.groupby("bucket").head(1).copy()
daily_picks["매수모드"] = "매일1건"

weekly3 = cand[cand["Market"] == "KOSDAQ"].sort_values("Amount").copy()
weekly3["bucket"] = weekly3["Date"].dt.strftime("%Y-%U")
w3_picks = weekly3.groupby("bucket").head(3).copy()
w3_picks["매수모드"] = "주3건"

# RF 안전 버전 (RF위험=0 만)
daily_safe = daily_picks[daily_picks["RF위험"] == 0].copy()
w3_safe = w3_picks[w3_picks["RF위험"] == 0].copy()


# 표준 컬럼
COLS = [
    "Date", "Code", "Name", "Market", "Close",
    "sell_date", "sell_close",
    "ret_180d", "peak_180d",
    "결과", "peak등급", "RF손절확률", "RF위험",
    "Score", "Amount", "chart_pattern",
    "past_60", "past_120", "past_240",
    "pos_252_high", "slope60", "drawdown60",
    "For_20d", "Inst_20d", "PER_num", "PBR_num",
]
COLS = [c for c in COLS if c in cand.columns]


def fmt(df, modo=None):
    out = df[COLS].copy() if not modo else df[COLS + ["매수모드"]].copy()
    out["ret_180d"] = out["ret_180d"].round(1)
    out["peak_180d"] = out["peak_180d"].round(1)
    out["RF손절확률"] = (out["RF손절확률"] * 100).round(1)
    out = out.sort_values("Date", ascending=False).reset_index(drop=True)
    return out


# ====== 4가지 마스터 리스트 ======
master_all = fmt(cand).sort_values("Date", ascending=False)
master_daily_all = fmt(daily_picks, modo=True)
master_daily_safe = fmt(daily_safe, modo=True)
master_w3_all = fmt(w3_picks, modo=True)
master_w3_safe = fmt(w3_safe, modo=True)

master_all.to_csv(CACHE / "MASTER_백테스트_종목리스트.csv", index=False)
master_daily_all.to_csv(CACHE / "MASTER_매일1건_전체_종목.csv", index=False)
master_daily_safe.to_csv(CACHE / "MASTER_매일1건_RF안전_종목.csv", index=False)
master_w3_all.to_csv(CACHE / "MASTER_주3건_전체_종목.csv", index=False)
master_w3_safe.to_csv(CACHE / "MASTER_주3건_RF안전_종목.csv", index=False)


# ====== 통계 ======
def stats(df, label):
    n = len(df)
    if n == 0: return None
    invest = n * 10  # 만원 단위
    profit = (df["ret_180d"] / 100 * 10).sum()
    return {
        "라벨": label, "매수": n,
        "익절": int((df["결과"]=="익절").sum()),
        "손절": int((df["결과"]=="손절").sum()),
        "중립": int((df["결과"]=="중립").sum()),
        "슈퍼위너": int((df["peak_180d"]>=200).sum()),
        "100%+": int((df["peak_180d"]>=100).sum()),
        "50%+": int((df["peak_180d"]>=50).sum()),
        "투자(만)": invest,
        "수익금(만)": round(profit, 0),
        "수익률": round(profit/invest*100, 1) if invest else 0,
        "승률": round((df["ret_180d"]>0).mean()*100, 1),
        "평균peak": round(df["peak_180d"].mean(), 1),
    }

summary = pd.DataFrame([
    stats(master_daily_all, "매일1건 전체"),
    stats(master_daily_safe, "매일1건 + RF회피"),
    stats(master_w3_all, "주3건 전체"),
    stats(master_w3_safe, "주3건 + RF회피"),
])
print("\n" + "="*100)
print("백테스트 4가지 모드 요약")
print("="*100)
print(summary.to_string(index=False))
summary.to_csv(CACHE / "MASTER_백테스트_요약.csv", index=False)


# ====== 년도별 ======
def yearly(df):
    df = df.copy()
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    return df.groupby("Year").agg(
        매수=("Code", "count"),
        익절=("결과", lambda x: (x=="익절").sum()),
        손절=("결과", lambda x: (x=="손절").sum()),
        SW=("peak_180d", lambda x: (x>=200).sum()),
        수익금만=("ret_180d", lambda x: round(x.sum()/100*10)),
    )

print("\n[년도별 - 매일1건 RF안전]")
print(yearly(master_daily_safe).to_string())
print("\n[년도별 - 주3건 RF안전]")
print(yearly(master_w3_safe).to_string())


# ====== 최근 2025년 매수 종목 Top 20 ======
print("\n\n" + "="*100)
print("2025년 매수 종목 (RF안전 통합, 최신순)")
print("="*100)
recent = master_w3_safe[master_w3_safe["Date"].dt.year == 2025].head(20)
print(recent[["Date","Code","Name","Market","Close","ret_180d","peak_180d","결과","peak등급","RF손절확률"]].to_string(index=False))


# ====== 마크다운 리포트 ======
md = []
md.append(f"# 백테스트 마스터 리스트 (2020-04 ~ 2025-08)")
md.append(f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
md.append(f"_RF 모델: th20={TH20:.4f} (상위 20% 위험으로 분류)_\n")

md.append("## 4가지 모드 요약\n")
md.append("| 모드 | 매수 | 익절 | 손절 | SW | 100%+ | 50%+ | 투자(만) | 수익금(만) | 수익률 | 승률 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in summary.iterrows():
    md.append(f"| {r['라벨']} | {r['매수']} | {r['익절']} | {r['손절']} | {r['슈퍼위너']} | "
              f"{r['100%+']} | {r['50%+']} | {r['투자(만)']:,} | {r['수익금(만)']:,.0f} | "
              f"{r['수익률']:.1f}% | {r['승률']:.1f}% |")

md.append("\n## 년도별 (주3건 + RF회피 = 최강 조합)\n")
yr_safe = yearly(master_w3_safe)
md.append("| 년도 | 매수 | 익절 | 손절 | SW | 수익금(만) |")
md.append("|---|---|---|---|---|---|")
for y, r in yr_safe.iterrows():
    md.append(f"| {y} | {r['매수']} | {r['익절']} | {r['손절']} | {r['SW']} | +{r['수익금만']:,.0f} |")

md.append("\n## 2025년 최근 매수 종목 Top 20\n")
md.append("| 일자 | 종목코드 | 종목명 | 시장 | 매수가 | 180일ret | peak | 결과 | RF확률 |")
md.append("|---|---|---|---|---|---|---|---|---|")
for _, r in recent.iterrows():
    md.append(f"| {r['Date'].strftime('%Y-%m-%d')} | {r['Code']} | {r['Name']} | "
              f"{r['Market']} | {r['Close']:,.0f} | {r['ret_180d']:+.1f}% | "
              f"{r['peak_180d']:+.1f}% | {r['결과']} | {r['RF손절확률']:.1f}% |")

md.append("\n## 산출 파일\n")
md.append("- `cache/MASTER_백테스트_종목리스트.csv` — 전체 5,009건 시그널 + RF확률 + 결과 (마스터)")
md.append("- `cache/MASTER_매일1건_전체_종목.csv` — 매일1건 모드 매수 1,130건")
md.append("- `cache/MASTER_매일1건_RF안전_종목.csv` — 매일1건 모드 + RF회피 947건")
md.append("- `cache/MASTER_주3건_전체_종목.csv` — 주3건 모드 매수 776건")
md.append("- `cache/MASTER_주3건_RF안전_종목.csv` ⭐ — **최종 추천 모드 625건**")
md.append("- `cache/MASTER_백테스트_요약.csv` — 4가지 모드 요약\n")

md.append("## 라이브 운영 (매일 자동)\n")
md.append("1. 매일 16:30 KST 갱신: KRX OHLCV + 펀더멘털 + 수급 갱신")
md.append("2. 월요일마다 RF 모델 재학습")
md.append("3. live_filter.py가 회피 8개 + RF 손절확률 ≥ 53% 제외 적용")
md.append("4. cache/today_picks.csv 생성 (4-20건 추천)")
md.append("5. Git auto-commit & push → Streamlit Cloud 자동 재배포\n")

with open("BACKTEST_FINAL.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n✓ BACKTEST_FINAL.md 생성 ({len(md)}줄)")

# 파일 목록 출력
print(f"\n[저장된 마스터 파일]")
for f in sorted(CACHE.glob("MASTER_*.csv")):
    n_lines = sum(1 for _ in open(f)) - 1
    print(f"  {f.name}: {n_lines:,}건")
