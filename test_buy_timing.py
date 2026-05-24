"""
매수 시점 3가지 비교 (가장 수익률 좋은 옵션 결정)
==============================================
A. 발견일 당일 3:20 시장가 매수 (= 당일 종가, 백테스트 default)
B. 다음 영업일 (D+1) 시초가 매수
C. 다음 영업일 (D+1) 종가 매수

매도: 일관되게 180거래일 후 종가
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path("cache")
ALLOC = 100_000

# 베스트 픽 로드 (SuperScore weekly_5)
picks = pd.read_csv(CACHE / "MASTER_best_picks_2020-2026.csv")
picks["Date"] = pd.to_datetime(picks["Date"])
print(f"매수 종목: {len(picks):,}건")

# OHLCV
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


# 각 종목에 대해 3가지 시점 가격 계산
print("\n각 종목 매수 시점별 가격 계산...")
results = []
for _, r in picks.iterrows():
    code = r["Code"]
    d0 = r["Date"]
    if code not in OHLCV: continue
    bars = OHLCV[code]

    # 시그널 발생일 (d0) 데이터
    today = bars[bars.index == d0]
    if len(today) == 0:
        # 가장 가까운 영업일
        today = bars[bars.index <= d0].tail(1)
        if len(today) == 0: continue
        d0 = today.index[-1]

    # D+1
    future = bars[bars.index > d0]
    if len(future) < 180: continue
    d1 = future.iloc[0]  # 다음 영업일
    # 180일 후 종가
    sell_close = future.iloc[179]["Close"] if len(future) >= 180 else future.iloc[-1]["Close"]
    sell_date = future.index[179] if len(future) >= 180 else future.index[-1]

    # 3가지 매수 옵션
    buy_A_close_d0 = today.iloc[-1]["Close"]      # A: 당일 종가 (3:20 시장가)
    buy_B_open_d1 = d1["Open"]                     # B: D+1 시가
    buy_C_close_d1 = d1["Close"]                   # C: D+1 종가

    # 수익률
    ret_A = (sell_close/buy_A_close_d0 - 1) * 100
    ret_B = (sell_close/buy_B_open_d1 - 1) * 100
    ret_C = (sell_close/buy_C_close_d1 - 1) * 100

    # peak (180일 최고)
    peak = future["High"].head(180).max()
    peak_pct = (peak/buy_A_close_d0 - 1) * 100

    results.append({
        "Date": d0, "Code": code, "Name": r.get("Name"),
        "buy_A_d0종가": buy_A_close_d0,
        "buy_B_d1시가": buy_B_open_d1,
        "buy_C_d1종가": buy_C_close_d1,
        "매도가180일": sell_close, "매도일": sell_date,
        "ret_A": ret_A, "ret_B": ret_B, "ret_C": ret_C,
        "peak_180d": peak_pct,
    })

df = pd.DataFrame(results)
df["Year"] = df["Date"].dt.year
print(f"\n시뮬 완료: {len(df):,}건")


def stats(df, ret_col, label):
    n = len(df)
    profit = (df[ret_col] / 100 * ALLOC).sum()
    invest = n * ALLOC
    sw = (df["peak_180d"] >= 200).sum()
    w100 = (df["peak_180d"] >= 100).sum()
    w50 = (df["peak_180d"] >= 50).sum()
    loser = (df[ret_col] <= -20).sum()
    print(f"\n[{label}]")
    print(f"  매수 {n}건")
    print(f"  익절(+): {(df[ret_col]>0).sum()} ({(df[ret_col]>0).mean()*100:.1f}%)")
    print(f"  손절(-20%↓): {loser} ({loser/n*100:.1f}%)")
    print(f"  슈퍼위너 (peak ≥200%): {sw} ({sw/n*100:.1f}%)")
    print(f"  100%+ (peak): {w100} ({w100/n*100:.1f}%)")
    print(f"  50%+ (peak): {w50} ({w50/n*100:.1f}%)")
    print(f"  평균 수익: {df[ret_col].mean():+.1f}%")
    print(f"  중앙값:    {df[ret_col].median():+.1f}%")
    print(f"  ★ 누적 투자 {invest/1e4:,.0f}만 → 수익 {profit/1e4:+,.0f}만 ({profit/invest*100:+.1f}%)")
    return {
        "전략": label, "매수": n, "익절%": round((df[ret_col]>0).mean()*100, 1),
        "손절%": round(loser/n*100, 1), "SW%": round(sw/n*100, 1),
        "100+%": round(w100/n*100, 1), "50+%": round(w50/n*100, 1),
        "평균수익%": round(df[ret_col].mean(), 1),
        "투자만": round(invest/1e4), "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }


print("\n" + "="*100)
print("매수 시점 3가지 비교 (전체 1,155건)")
print("="*100)
r_a = stats(df, "ret_A", "A. 당일 종가 (3:20 시장가)")
r_b = stats(df, "ret_B", "B. D+1 시가")
r_c = stats(df, "ret_C", "C. D+1 종가")


# 년도별
print("\n" + "="*100)
print("년도별 수익률 (%)")
print("="*100)
print(f"\n{'Year':<6}{'A 당일종가':>12}{'B D+1시가':>12}{'C D+1종가':>12}{'B-A 차이':>12}{'C-A 차이':>12}")
yr_results = []
for y, g in df.groupby("Year"):
    a = g["ret_A"].sum()/100*ALLOC/1e4
    b = g["ret_B"].sum()/100*ALLOC/1e4
    c = g["ret_C"].sum()/100*ALLOC/1e4
    inv = len(g)*ALLOC/1e4
    pa = a/inv*100; pb = b/inv*100; pc = c/inv*100
    print(f"{int(y):<6}{pa:>+10.1f}% {pb:>+10.1f}% {pc:>+10.1f}% {pb-pa:>+10.2f}p {pc-pa:>+10.2f}p")
    yr_results.append({"year":int(y), "A":pa, "B":pb, "C":pc, "매수":len(g)})

# 누적
total_a = df["ret_A"].sum()/100*ALLOC/1e4
total_b = df["ret_B"].sum()/100*ALLOC/1e4
total_c = df["ret_C"].sum()/100*ALLOC/1e4
inv_t = len(df)*ALLOC/1e4
print(f"{'누적':<6}{total_a/inv_t*100:>+10.1f}% {total_b/inv_t*100:>+10.1f}% {total_c/inv_t*100:>+10.1f}%")

# 저장
df.to_csv(CACHE / "BUY_TIMING_comparison.csv", index=False)
pd.DataFrame([r_a, r_b, r_c]).to_csv(CACHE / "BUY_TIMING_summary.csv", index=False)
pd.DataFrame(yr_results).to_csv(CACHE / "BUY_TIMING_yearly.csv", index=False)
print(f"\n[저장] cache/BUY_TIMING_comparison.csv ({len(df)}건)")
print(f"      cache/BUY_TIMING_summary.csv")
print(f"      cache/BUY_TIMING_yearly.csv")
