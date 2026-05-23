"""
슈퍼위너 포착률 시뮬레이션
=========================
질문: 주에 1개씩 사면 슈퍼위너(200%+)/100%+ 종목을 놓치는 것 아닌가?
      슈퍼위너+100%+ 위주로 최대한 뽑아내는 게 가능한가?

전략 비교:
  E1. 주 1건 - 점수 순
  E2. 주 1건 - 랜덤(Code순)
  E3. 주 1건 - 거래대금 낮은 순
  E4. 자본 풀로 - 시그널 발생 즉시 매수 (자본 10%/종목, 슬롯 부족시 skip)
  E5. 슬롯 N개로 풀 매수 (5/10/20슬롯)

룰:
- 코스피 + 코스닥 동시 스캔
- 회피 6개 자동 제외
- 종목당 자본 10% (=100만원/슬롯, 1000만원 자본)
- 180일 보유 후 청산
- 슈퍼위너(200%+) 포착률, 100%+ 포착률, 최종 자본 추적
"""

import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

# -------- LOAD --------
DF = pd.read_parquet("cache/chart_feats_v1.parquet")
DF["Date"] = pd.to_datetime(DF["Date"])
print(f"[로드] chart_feats: {len(DF):,}건")
print(f"  Market: {DF['Market'].value_counts().to_dict()}")
print(f"  날짜: {DF['Date'].min().date()} ~ {DF['Date'].max().date()}")

with open("cache/ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)
print(f"[로드] OHLCV: {len(OHLCV)}종목")


# -------- 회피 6개 룰 --------
def apply_avoid_6(df):
    """회피 6개 자동 제외 후 후보 반환"""
    d = df.copy()
    # X1: 하락추세+일시반등
    x1 = (d["chart_pattern"] == "pullback_recovery") & (d["slope60"] <= -1) & (d["pos_252_high"] <= -40)
    # X2: KOSPI 떨어지는 칼날
    x2 = (d["Market"] == "KOSPI") & (d["past_120"] <= -20) & (d["pos_252_high"] <= -40)
    # X3: 과열 신고가
    x3 = (d["s12"] >= 80) & (d["new_high_252"] == 1) & (d["past_120"] >= 50)
    # X4: 1년 +100% 과열
    x4 = d["past_240"] >= 100
    # X5: 1년 +150% 극과열
    x5 = d["past_240"] >= 150
    # X6: 거래대금 3,000억 이상
    x6 = d["Amount"] >= 3000e8

    avoid_mask = x1 | x2 | x3 | x4 | x5 | x6
    kept = d[~avoid_mask].copy()
    print(f"[회피적용] {len(d):,}건 -> {len(kept):,}건 (제외 {avoid_mask.sum():,}건)")
    return kept


# -------- peak_180d 계산 --------
def compute_peak_and_ret_180(df_signals):
    """각 시그널의 180일 후 종가/peak를 OHLCV에서 계산"""
    peaks, rets, hold_days_actual = [], [], []
    sell_dates = []
    sell_prices_close180 = []

    for _, row in df_signals.iterrows():
        code = row["Code"]
        d0 = row["Date"]
        close0 = row["Close"]
        if code not in OHLCV:
            peaks.append(np.nan); rets.append(np.nan); hold_days_actual.append(np.nan)
            sell_dates.append(pd.NaT); sell_prices_close180.append(np.nan)
            continue
        bars = OHLCV[code]
        # 시그널 다음 영업일부터 180거래일 보유
        future = bars[bars.index > d0].head(180)
        if len(future) == 0:
            peaks.append(np.nan); rets.append(np.nan); hold_days_actual.append(np.nan)
            sell_dates.append(pd.NaT); sell_prices_close180.append(np.nan)
            continue
        peak = future["High"].max()
        last_close = future["Close"].iloc[-1]
        last_date = future.index[-1]
        peaks.append((peak / close0 - 1) * 100)
        rets.append((last_close / close0 - 1) * 100)
        hold_days_actual.append(len(future))
        sell_dates.append(last_date)
        sell_prices_close180.append(last_close)

    df_signals = df_signals.copy()
    df_signals["peak_180d"] = peaks
    df_signals["ret_180d_actual"] = rets
    df_signals["hold_days_actual"] = hold_days_actual
    df_signals["sell_date"] = sell_dates
    df_signals["sell_close"] = sell_prices_close180
    return df_signals


# -------- 메인 후보 --------
cand = apply_avoid_6(DF)
cand = cand[cand["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
print(f"[필터후] KOSPI+KOSDAQ: {len(cand):,}건")

cand = compute_peak_and_ret_180(cand)
cand = cand.dropna(subset=["peak_180d", "ret_180d_actual"]).copy()
print(f"[180일 결과 가능] {len(cand):,}건")

# 슈퍼위너/100%+ 비율 (필터 후 풀)
total = len(cand)
sw = (cand["peak_180d"] >= 200).sum()
w100 = (cand["peak_180d"] >= 100).sum()
w50 = (cand["peak_180d"] >= 50).sum()
print(f"\n[전체 풀의 슈퍼위너 비율]")
print(f"  슈퍼위너(peak ≥200%): {sw:,}건 ({sw/total*100:.2f}%)")
print(f"  100%+              : {w100:,}건 ({w100/total*100:.2f}%)")
print(f"  50%+               : {w50:,}건 ({w50/total*100:.2f}%)")


# -------- 시뮬레이션 엔진 --------
def simulate_strategy(df, mode, capital0=10_000_000, alloc_pct=0.10,
                     slots=None, weekly=True, sort_by=None, label=""):
    """
    mode: 'weekly_1'  - 주 1건만 매수 (sort_by로 선택)
          'unlimited' - 시그널 발생시 즉시 매수 (자본 부족하면 skip)
          'slots_N'   - 동시 보유 N슬롯 (자본 부족시 skip, slots 지정)

    alloc_pct: 종목당 비중
    sort_by: 'Score desc', 'Amount asc', 'Code', 'Score asc', 'random'
    """
    df = df.sort_values("Date").copy()
    # 주차 라벨
    df["YearWeek"] = df["Date"].dt.strftime("%Y-%U")

    cash = capital0
    holdings = []  # list of dicts: {code, name, buy_date, buy_price, qty, sell_date, sell_close, peak}
    trades = []
    skipped = 0

    # daily processing
    if mode == "weekly_1":
        # group by week, pick first by sort_by
        weeks = df.groupby("YearWeek")
        for wk, group in weeks:
            # 정렬
            if sort_by == "Score desc":
                pick = group.sort_values("Score", ascending=False).iloc[0]
            elif sort_by == "Score asc":
                pick = group.sort_values("Score", ascending=True).iloc[0]
            elif sort_by == "Amount asc":
                pick = group.sort_values("Amount", ascending=True).iloc[0]
            elif sort_by == "random":
                pick = group.sample(1, random_state=42).iloc[0]
            else:
                pick = group.sort_values("Code").iloc[0]
            # 매수 가능?
            alloc = capital0 * alloc_pct
            if cash < alloc:
                skipped += 1
                continue
            buy_price = pick["Close"]
            qty = alloc / buy_price
            cash -= alloc
            trades.append({
                "buy_date": pick["Date"], "code": pick["Code"], "name": pick["Name"],
                "buy_price": buy_price, "qty": qty, "alloc": alloc,
                "sell_date": pick["sell_date"], "sell_close": pick["sell_close"],
                "peak_180d": pick["peak_180d"], "ret_180d": pick["ret_180d_actual"],
            })
            cash += qty * pick["sell_close"]  # 180일 후 매도 (자본 반환)

    elif mode == "unlimited":
        # 시그널 순서대로 매수 시도, 자본 있으면 매수
        # 동시 보유 자본 추적 위해 sell_date 기준 cash 반환
        events = []
        for _, r in df.iterrows():
            events.append(("buy_signal", r["Date"], r))
        events.sort(key=lambda x: x[1])

        # cash flow with date
        cash = capital0
        # sell events tracked: (sell_date, code, value)
        pending_sells = []  # heap-like

        for kind, dt, r in events:
            # 우선 dt 이전의 매도 처리
            still_pending = []
            for sd, vc, val in pending_sells:
                if sd <= dt:
                    cash += val
                else:
                    still_pending.append((sd, vc, val))
            pending_sells = still_pending

            alloc = capital0 * alloc_pct
            if cash < alloc:
                skipped += 1
                continue
            buy_price = r["Close"]
            qty = alloc / buy_price
            cash -= alloc
            sell_value = qty * r["sell_close"]
            pending_sells.append((r["sell_date"], r["Code"], sell_value))
            trades.append({
                "buy_date": r["Date"], "code": r["Code"], "name": r["Name"],
                "buy_price": buy_price, "qty": qty, "alloc": alloc,
                "sell_date": r["sell_date"], "sell_close": r["sell_close"],
                "peak_180d": r["peak_180d"], "ret_180d": r["ret_180d_actual"],
            })
        # 잔여 매도 처리
        for sd, vc, val in pending_sells:
            cash += val

    elif mode.startswith("slots_"):
        slot_n = int(mode.split("_")[1])
        cash = capital0
        events = []
        for _, r in df.iterrows():
            events.append(r)
        events.sort(key=lambda x: x["Date"])
        pending_sells = []  # (sell_date, value)

        for r in events:
            dt = r["Date"]
            still_pending = []
            for sd, val in pending_sells:
                if sd <= dt:
                    cash += val
                else:
                    still_pending.append((sd, val))
            pending_sells = still_pending

            if len(pending_sells) >= slot_n:
                skipped += 1
                continue
            alloc = capital0 * alloc_pct
            if cash < alloc:
                skipped += 1
                continue
            buy_price = r["Close"]
            qty = alloc / buy_price
            cash -= alloc
            sell_value = qty * r["sell_close"]
            pending_sells.append((r["sell_date"], sell_value))
            trades.append({
                "buy_date": r["Date"], "code": r["Code"], "name": r["Name"],
                "buy_price": buy_price, "qty": qty, "alloc": alloc,
                "sell_date": r["sell_date"], "sell_close": r["sell_close"],
                "peak_180d": r["peak_180d"], "ret_180d": r["ret_180d_actual"],
            })
        for sd, val in pending_sells:
            cash += val

    t = pd.DataFrame(trades)
    if len(t) == 0:
        return {"label": label, "n": 0, "skipped": skipped}

    # 통계
    n = len(t)
    sw_caught = (t["peak_180d"] >= 200).sum()
    w100_caught = (t["peak_180d"] >= 100).sum()
    w50_caught = (t["peak_180d"] >= 50).sum()
    avg_ret = t["ret_180d"].mean()
    avg_peak = t["peak_180d"].mean()
    winrate = (t["ret_180d"] > 0).mean() * 100

    # 자본 변화: trades 손익으로 계산
    final_capital = capital0
    # 위 시뮬 안에서 cash로 갱신했지만 weekly_1는 순차이므로 다시 계산
    pnl_total = (t["sell_close"] * t["qty"] - t["alloc"]).sum()
    final_capital = capital0 + pnl_total

    return {
        "label": label,
        "n": n,
        "skipped": skipped,
        "sw_caught": int(sw_caught),
        "sw_rate": sw_caught / n * 100,
        "w100_caught": int(w100_caught),
        "w100_rate": w100_caught / n * 100,
        "w50_caught": int(w50_caught),
        "w50_rate": w50_caught / n * 100,
        "avg_ret_180d": avg_ret,
        "avg_peak_180d": avg_peak,
        "winrate": winrate,
        "final_capital": final_capital,
        "return_pct": (final_capital / 10_000_000 - 1) * 100,
        "trades": t,
    }


# -------- 시뮬 실행 --------
print("\n" + "=" * 80)
print("전체 기간 (Train+Test, 2020-04 ~ 2025-08) 시뮬레이션")
print("=" * 80)

strategies = [
    ("weekly_1", "Score desc", "E1. 주1건 - 점수 높은 순"),
    ("weekly_1", "Score asc",  "E2. 주1건 - 점수 낮은 순"),
    ("weekly_1", "Amount asc", "E3. 주1건 - 거래대금 낮은 순"),
    ("weekly_1", "random",     "E4. 주1건 - 랜덤"),
    ("slots_5",  None,         "E5. 5슬롯 풀가동 (자본 50%)"),
    ("slots_10", None,         "E6. 10슬롯 풀가동 (자본 100%)"),
    ("slots_20", None,         "E7. 20슬롯 풀가동 (자본 200% 가정 = 비중 5%)"),
    ("unlimited", None,        "E8. 무제한 시도 (자본 부족시 skip)"),
]

results = []
for mode, sort_by, label in strategies:
    if mode == "slots_20":
        # 20슬롯이면 비중을 5%로
        r = simulate_strategy(cand, mode, alloc_pct=0.05, sort_by=sort_by, label=label)
    else:
        r = simulate_strategy(cand, mode, alloc_pct=0.10, sort_by=sort_by, label=label)
    results.append(r)

# 출력
print(f"\n{'전략':45s}{'매수':>5s}{'skip':>6s}{'슈퍼위너':>10s}{'100%+':>9s}{'50%+':>8s}{'평균peak':>10s}{'평균ret180':>11s}{'승률':>7s}{'최종자본':>13s}{'수익률':>9s}")
print("-" * 145)
for r in results:
    if r["n"] == 0:
        print(f"{r['label']:45s}{'없음':>5s}")
        continue
    print(f"{r['label']:45s}{r['n']:>5d}{r['skipped']:>6d}{r['sw_caught']:>5d}({r['sw_rate']:>4.1f}%)"
          f"{r['w100_caught']:>4d}({r['w100_rate']:>4.1f}%)"
          f"{r['w50_caught']:>4d}({r['w50_rate']:>4.1f}%)"
          f"{r['avg_peak_180d']:>9.1f}%{r['avg_ret_180d']:>10.1f}%"
          f"{r['winrate']:>6.1f}%{r['final_capital']:>13,.0f}{r['return_pct']:>8.1f}%")


# -------- OOS Test 기간 (2024 이후) --------
print("\n" + "=" * 80)
print("OOS Test (2024-01-01 이후) 시뮬레이션")
print("=" * 80)

cand_oos = cand[cand["Date"] >= "2024-01-01"].copy()
print(f"[OOS 후보] {len(cand_oos):,}건")

total = len(cand_oos)
sw = (cand_oos["peak_180d"] >= 200).sum()
w100 = (cand_oos["peak_180d"] >= 100).sum()
w50 = (cand_oos["peak_180d"] >= 50).sum()
print(f"  슈퍼위너 비율: {sw:,}건 ({sw/total*100:.2f}%)")
print(f"  100%+      : {w100:,}건 ({w100/total*100:.2f}%)")
print(f"  50%+       : {w50:,}건 ({w50/total*100:.2f}%)")

results_oos = []
for mode, sort_by, label in strategies:
    if mode == "slots_20":
        r = simulate_strategy(cand_oos, mode, alloc_pct=0.05, sort_by=sort_by, label=label)
    else:
        r = simulate_strategy(cand_oos, mode, alloc_pct=0.10, sort_by=sort_by, label=label)
    results_oos.append(r)

print(f"\n{'전략':45s}{'매수':>5s}{'skip':>6s}{'슈퍼위너':>10s}{'100%+':>9s}{'50%+':>8s}{'평균peak':>10s}{'평균ret180':>11s}{'승률':>7s}{'최종자본':>13s}{'수익률':>9s}")
print("-" * 145)
for r in results_oos:
    if r["n"] == 0:
        print(f"{r['label']:45s}{'없음':>5s}")
        continue
    print(f"{r['label']:45s}{r['n']:>5d}{r['skipped']:>6d}{r['sw_caught']:>5d}({r['sw_rate']:>4.1f}%)"
          f"{r['w100_caught']:>4d}({r['w100_rate']:>4.1f}%)"
          f"{r['w50_caught']:>4d}({r['w50_rate']:>4.1f}%)"
          f"{r['avg_peak_180d']:>9.1f}%{r['avg_ret_180d']:>10.1f}%"
          f"{r['winrate']:>6.1f}%{r['final_capital']:>13,.0f}{r['return_pct']:>8.1f}%")


# -------- 슈퍼위너 매수 빈도 분석 (주 1건은 몇 주 만에 하나씩 잡나) --------
print("\n" + "=" * 80)
print("슈퍼위너 잡기 빈도 분석")
print("=" * 80)

# 풀에서 슈퍼위너만 추출, 주별 발생 빈도
cand_sw = cand[cand["peak_180d"] >= 200].copy()
cand_sw["YearMonth"] = cand_sw["Date"].dt.strftime("%Y-%m")
print(f"\n[전체기간] 슈퍼위너 {len(cand_sw):,}건 발생")
print(f"  월평균: {len(cand_sw)/65:.1f}건/월 (65개월)")
print(f"  주평균: {len(cand_sw)/280:.1f}건/주 (~280주)")

# 100%+
cand_w100 = cand[cand["peak_180d"] >= 100].copy()
print(f"\n[전체기간] 100%+ {len(cand_w100):,}건 발생")
print(f"  월평균: {len(cand_w100)/65:.1f}건/월")
print(f"  주평균: {len(cand_w100)/280:.1f}건/주")

# 주 1건 매수의 슈퍼위너 포착 확률
print(f"\n[이론적 한계]")
print(f"  주 1건 무작위 선택시 슈퍼위너 포착 확률 = 풀의 SW 비율 = {sw/total*100:.2f}%")
print(f"  주 1건으로 280주 매수 시 기대 슈퍼위너 = 280 × {sw/total*100:.2f}% = {280*sw/total:.1f}개")


# 저장
all_trades = []
for r, (mode, sort_by, label) in zip(results, strategies):
    if "trades" in r:
        t = r["trades"].copy()
        t["strategy"] = label
        all_trades.append(t)
if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv("cache/superwinner_simulation_trades.csv", index=False)
    print("\n[저장] cache/superwinner_simulation_trades.csv")

print("\n[완료]")
