"""
시총 2000 종목 walk-forward OOS 검증
=================================
signals_2000_2020-04_2026-05.parquet 사용.

각 년도별 OOS:
- 2020-2022 학습 → 2023 검증
- 2020-2023 학습 → 2024 검증
- 2020-2024 학습 → 2025 검증
- 2020-2025 학습 → 2026 검증 (미래 일부)

매수 모드: 매일 1건 / 주 3건 (거래대금 낮은 순)
"""

import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

CACHE = Path("cache")
ALLOC = 100_000


print("[1] 데이터 로드")
sigs = pd.read_parquet(CACHE / "signals_2000_2020-04_2026-05.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
print(f"   시그널 (raw): {len(sigs):,}건, {sigs['Date'].min().date()} ~ {sigs['Date'].max().date()}")

# OHLCV
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


# 2) Forward (peak/ret 180일) + Amount + derived 컬럼
print("[2] Forward + derived 추가")
def enrich(df):
    peaks, rets, sell_dates, sell_closes = [], [], [], []
    Amounts, p60, p120, p240, slope60s, pos252, dd60, vol20, vol60_, range60, new_high_252 = [],[],[],[],[],[],[],[],[],[],[]
    days_52w_low, days_52w_high = [], []
    for _, r in df.iterrows():
        code = r["Code"]; d0 = r["Date"]; c0 = r["Close"]
        if code not in OHLCV:
            for lst in [peaks, rets, sell_dates, sell_closes,
                        Amounts, p60, p120, p240, slope60s, pos252, dd60, vol20, vol60_, range60, new_high_252,
                        days_52w_low, days_52w_high]:
                lst.append(np.nan)
            continue
        bars = OHLCV[code]
        # Forward
        future = bars[bars.index > d0].head(180)
        if len(future) >= 1:
            peaks.append((future["High"].max() / c0 - 1) * 100)
            rets.append((future["Close"].iloc[-1] / c0 - 1) * 100)
            sell_dates.append(future.index[-1])
            sell_closes.append(future["Close"].iloc[-1])
        else:
            peaks.append(np.nan); rets.append(np.nan); sell_dates.append(pd.NaT); sell_closes.append(np.nan)

        # Past + derived
        past = bars[bars.index <= d0]
        if len(past) < 252:
            for lst in [Amounts, p60, p120, p240, slope60s, pos252, dd60, vol20, vol60_, range60, new_high_252,
                        days_52w_low, days_52w_high]:
                lst.append(np.nan)
            continue
        latest = past.iloc[-1]
        Amounts.append(float(latest["Close"] * latest["Volume"]))
        p60.append((c0 / past.iloc[-60]["Close"] - 1) * 100)
        p120.append((c0 / past.iloc[-120]["Close"] - 1) * 100)
        p240.append((c0 / past.iloc[-240]["Close"] - 1) * 100)
        last60 = past.tail(60)
        x = np.arange(len(last60))
        slope = np.polyfit(x, last60["Close"].values, 1)[0]
        slope60s.append(slope / last60["Close"].mean() * 100)
        last252 = past.tail(252)
        h252 = last252["High"].max(); l252 = last252["Low"].min()
        pos252.append((c0 / h252 - 1) * 100 if h252>0 else np.nan)
        last60c = past.tail(60); h60 = last60c["High"].max()
        dd60.append((c0 / h60 - 1) * 100 if h60>0 else np.nan)
        rs = past["Close"].pct_change()
        vol20.append(rs.tail(20).std() * 100)
        vol60_.append(rs.tail(60).std() * 100)
        range60.append((h60 - last60c["Low"].min())/last60c["Low"].min()*100 if last60c["Low"].min()>0 else np.nan)
        new_high_252.append(1 if c0 >= h252*0.995 else 0)
        try:
            days_52w_low.append(int((d0 - last252["Low"].idxmin()).days))
            days_52w_high.append(int((d0 - last252["High"].idxmax()).days))
        except Exception:
            days_52w_low.append(np.nan); days_52w_high.append(np.nan)
    df = df.copy()
    df["peak_180d"] = peaks; df["ret_180d"] = rets
    df["sell_date"] = sell_dates; df["sell_close"] = sell_closes
    df["Amount"] = Amounts; df["past_60"] = p60; df["past_120"] = p120; df["past_240"] = p240
    df["slope60"] = slope60s; df["slope120"] = slope60s
    df["pos_252_high"] = pos252; df["drawdown60"] = dd60
    df["new_high_252"] = new_high_252
    df["vol20"] = vol20; df["vol60"] = vol60_
    df["range60_pct"] = range60; df["range120_pct"] = range60
    df["days_since_52w_low"] = days_52w_low; df["days_since_52w_high"] = days_52w_high
    df["pos_60_high"] = pos252; df["pos_120_high"] = pos252; df["pos_240_high"] = pos252
    df["past_5d"] = df["past_60"]/12
    df["past_20"] = df["past_60"]/3
    df["runup60"] = df["past_60"].abs()
    df["candle_pct"] = df.get("candle_pct", 0)
    df["cum_5d_gain"] = df.get("cum_5d_gain", 0)
    df["For_5d"] = 0; df["For_20d"] = 0; df["Inst_5d"] = 0; df["Inst_20d"] = 0
    df["PER_num"] = np.nan; df["PBR_num"] = np.nan; df["외인소진율_num"] = np.nan
    for k in ["pre_5d_max_high_ratio","pre_5d_min_low_ratio","pre_5d_vol_trend",
              "pre_10d_max_high_ratio","pre_10d_drawdown","pre_20d_vol_ratio",
              "gap_up_count_5d","long_red_count_5d","long_red_in_10d","consecutive_red_max"]:
        df[k] = 0
    return df

# 시그널 dedup (날짜+코드 1개로, 가장 높은 점수)
sigs = sigs.sort_values("Score", ascending=False).drop_duplicates(["Date","Code"]).reset_index(drop=True)
print(f"   시그널 (dedup): {len(sigs):,}건")
sigs = enrich(sigs)
sigs = sigs.dropna(subset=["peak_180d", "ret_180d", "Amount"]).copy()
print(f"   forward + derived 가능: {len(sigs):,}건")
sigs["Year"] = sigs["Date"].dt.year
sigs["is_loser"] = (sigs["ret_180d"]<=-20).astype(int)
sigs["is_sw"] = (sigs["peak_180d"]>=200).astype(int)
sigs["is_100"] = (sigs["peak_180d"]>=100).astype(int)
sigs["is_50"] = (sigs["peak_180d"]>=50).astype(int)

sigs.to_parquet(CACHE / "signals_2000_enriched.parquet", index=False)

print(f"\n[년도별 시그널 수]")
yr_stat = sigs.groupby("Year").agg(n=("Code","count"), loser=("is_loser","sum"), sw=("is_sw","sum"))
yr_stat["loser%"] = (yr_stat["loser"]/yr_stat["n"]*100).round(1)
yr_stat["sw%"] = (yr_stat["sw"]/yr_stat["n"]*100).round(1)
print(yr_stat.to_string())


# 3) Walk-Forward OOS
from train_rf_loss_model import FEATURES, prepare_X

forward_windows = [
    {"train_start": 2020, "train_end": 2022, "test_year": 2023},
    {"train_start": 2020, "train_end": 2023, "test_year": 2024},
    {"train_start": 2020, "train_end": 2024, "test_year": 2025},
    {"train_start": 2020, "train_end": 2025, "test_year": 2026},
]

all_results = []
all_picks = []

print("\n" + "="*100)
print("Walk-Forward OOS (시총 2000)")
print("="*100)

for w in forward_windows:
    train = sigs[(sigs["Year"]>=w["train_start"])&(sigs["Year"]<=w["train_end"])].copy()
    test = sigs[sigs["Year"]==w["test_year"]].copy()
    if len(train)<500 or len(test)<50:
        print(f"\nWalk {w['train_start']}-{w['train_end']} → {w['test_year']}: 데이터 부족 skip"); continue
    print(f"\nWalk {w['train_start']}-{w['train_end']} → Test {w['test_year']}")
    print(f"  Train: {len(train):,}건, Test: {len(test):,}건")

    available = [f for f in FEATURES if f in train.columns]
    X_tr, _ = prepare_X(train, features=available)
    X_te, _ = prepare_X(test, features=available)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr, train["is_loser"])
    test["RF확률"] = rf.predict_proba(X_te)[:, 1]
    th20 = np.quantile(rf.predict_proba(X_tr)[:, 1], 0.80)
    test["RF위험"] = (test["RF확률"] >= th20).astype(int)

    # 매일 1건 / 주 3건 모드
    def sim(pool, mode, label):
        if len(pool)==0: return None
        df = pool.sort_values("Amount").copy()
        if mode == "daily_1":
            df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
            n_per = 1
        elif mode == "weekly_3":
            df = df[df["Market"]=="KOSDAQ"].copy()
            df["bucket"] = df["Date"].dt.strftime("%Y-%U")
            n_per = 3
        picks = df.groupby("bucket").head(n_per).drop_duplicates(["Date","Code"])
        if len(picks)==0: return None
        invest = len(picks) * ALLOC
        profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
        return {
            "year": w["test_year"], "필터": label, "모드": mode,
            "매수": len(picks),
            "익절": int((picks["ret_180d"]>0).sum()),
            "손절": int((picks["ret_180d"]<=-20).sum()),
            "SW": int((picks["peak_180d"]>=200).sum()),
            "100+": int((picks["peak_180d"]>=100).sum()),
            "50+": int((picks["peak_180d"]>=50).sum()),
            "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
            "손절률%": round((picks["ret_180d"]<=-20).mean()*100, 1),
            "투자만": invest/1e4,
            "수익만": round(profit/1e4),
            "수익률%": round(profit/invest*100, 1),
        }, picks

    rf_safe = test[test["RF위험"]==0].copy()
    for label, pool in [("회피X", test), ("RF안전", rf_safe)]:
        for mode in ["daily_1", "weekly_3"]:
            r = sim(pool, mode, label)
            if r is None: continue
            res, p = r
            all_results.append(res)
            p_out = p.copy()
            p_out["year"]=w["test_year"]; p_out["필터"]=label; p_out["모드"]=mode
            all_picks.append(p_out)
            print(f"  [{label} {mode}] 매수 {res['매수']}건, SW {res['SW']}, 손절 {res['손절']}, 수익 {res['수익률%']}%")


# 4) 종합 출력
res_df = pd.DataFrame(all_results)
res_df.to_csv(CACHE / "WF2000_results.csv", index=False)
print("\n" + "="*100)
print("종합 (시총 2000)")
print("="*100)
print(res_df.to_string(index=False))

# 누적
for label in ["회피X", "RF안전"]:
    for mode in ["daily_1", "weekly_3"]:
        sub = res_df[(res_df["필터"]==label)&(res_df["모드"]==mode)]
        if len(sub)==0: continue
        print(f"\n[{label} / {mode} 누적]")
        print(f"  매수: {sub['매수'].sum():,}건")
        print(f"  익절: {sub['익절'].sum()} / 손절: {sub['손절'].sum()}")
        print(f"  SW: {sub['SW'].sum()} ({sub['SW'].sum()/sub['매수'].sum()*100:.1f}%)")
        print(f"  100+: {sub['100+'].sum()} ({sub['100+'].sum()/sub['매수'].sum()*100:.1f}%)")
        print(f"  50+: {sub['50+'].sum()} ({sub['50+'].sum()/sub['매수'].sum()*100:.1f}%)")
        inv = sub['투자만'].sum(); prof = sub['수익만'].sum()
        print(f"  투자 {inv:,.0f}만 → 수익 {prof:+,.0f}만 ({prof/inv*100:+.1f}%)")

# 매수 종목 저장
if all_picks:
    picks_df = pd.concat(all_picks, ignore_index=True)
    picks_df.to_csv(CACHE / "WF2000_picks.csv", index=False)
    print(f"\n[저장] cache/WF2000_results.csv + WF2000_picks.csv ({len(picks_df):,}건)")
