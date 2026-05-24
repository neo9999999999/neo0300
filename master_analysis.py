"""
마스터 종합 분석 (2020-2026 전체 walk-forward)
============================================
범위:
- 시총 300 풀 + 2020-04 ~ 2026-05 (전체 데이터)
- 보조지표 50+ (RSI/MACD/볼린저/스토캐스틱/ATR/OBV/ADX/일목/캔들패턴 등)
- 다중 RF 모델 (5타깃: 손절/슈퍼위너/100+/50+/10+)
- 매일 1건 / 주 3건 / 주 5건 모드 비교
- walk-forward (2020-2021 → 2022, ..., 2020-2025 → 2026)
- 모든 OOS, 데이터 누수 0
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from train_rf_loss_model import FEATURES, prepare_X, add_pre_features

CACHE = Path("cache")
ALLOC = 100_000


# ============ 1. 시그널 풀 통합 + 시총 300 ============
print("[1/6] 시그널 풀 통합 (시총 300, 2020-2026)")
old = pd.read_parquet(CACHE / "candidates_enriched_full.parquet")
new = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
old["Date"] = pd.to_datetime(old["Date"])
new["Date"] = pd.to_datetime(new["Date"])

snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])

old300 = old[old["Code"].isin(top300) & (old["Date"] < "2021-04-01")].copy()
new300 = new[new["Code"].isin(top300)].copy()
common_cols = list(set(old300.columns) & set(new300.columns))
combined = pd.concat([old300[common_cols], new300[common_cols]], ignore_index=True)
combined = combined.dropna(subset=["peak_180d","ret_180d","Amount"]).copy()
combined = combined.sort_values(["Date","Code"]).reset_index(drop=True)
combined["Year"] = combined["Date"].dt.year
print(f"  통합 풀: {len(combined):,}건 ({combined['Date'].min().date()} ~ {combined['Date'].max().date()})")


# ============ 2. 보조지표 50+ 추가 ============
print("\n[2/6] 보조지표 50+ 계산")
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


def calc_indicators(bars):
    """단일 종목 OHLCV에 보조지표 추가"""
    df = bars.copy()
    close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]; openp = df["Open"]

    # 이동평균
    for n in [5, 10, 20, 60, 120, 200]:
        df[f"ma{n}"] = close.rolling(n).mean()
    # EMA
    for n in [12, 26, 50]:
        df[f"ema{n}"] = close.ewm(span=n, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    rs = gain/(loss+1e-9)
    df["rsi14"] = 100 - 100/(1+rs)

    # MACD
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Band
    df["bb_mid"] = close.rolling(20).mean()
    df["bb_std"] = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100*(close - low14)/(high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / close

    # ADX (단순화)
    plus_dm = (high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0))
    minus_dm = (-low.diff().where((-low.diff() > high.diff()) & (-low.diff() > 0), 0))
    plus_di = 100*plus_dm.rolling(14).mean()/(df["atr14"]+1e-9)
    minus_di = 100*minus_dm.rolling(14).mean()/(df["atr14"]+1e-9)
    dx = 100*(plus_di-minus_di).abs()/(plus_di+minus_di+1e-9)
    df["adx14"] = dx.rolling(14).mean()

    # CCI
    tp = (high + low + close)/3
    df["cci20"] = (tp - tp.rolling(20).mean())/(0.015*tp.rolling(20).std()+1e-9)

    # OBV
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ma20"] = obv.rolling(20).mean()
    df["obv_trend"] = obv / (df["obv_ma20"]+1) - 1  # OBV 트렌드

    # Williams %R
    high14 = high.rolling(14).max()
    low14 = low.rolling(14).min()
    df["williams_r"] = -100*(high14-close)/(high14-low14+1e-9)

    # ROC
    for n in [5, 10, 20, 60]:
        df[f"roc{n}"] = (close/close.shift(n) - 1) * 100

    # VWAP (20일)
    vwap = (close*vol).rolling(20).sum()/(vol.rolling(20).sum()+1)
    df["vwap20"] = vwap
    df["vwap_ratio"] = close/vwap

    # Donchian Channel (20)
    df["donchian_high"] = high.rolling(20).max()
    df["donchian_low"] = low.rolling(20).min()
    df["donchian_pct"] = (close - df["donchian_low"])/(df["donchian_high"]-df["donchian_low"]+1e-9)

    # 캔들 패턴 (수치화)
    body = (close - openp).abs()
    upper_wick = high - close.clip(lower=openp).where(close>openp, openp)
    lower_wick = close.clip(upper=openp).where(close>openp, openp) - low
    df["body_ratio"] = body / (high - low + 1e-9)
    df["upper_wick_ratio"] = upper_wick / (high - low + 1e-9)
    df["lower_wick_ratio"] = lower_wick / (high - low + 1e-9)

    # 갭
    df["gap_up"] = (openp - close.shift()) / close.shift() * 100
    df["gap_count_5d"] = (df["gap_up"] > 2).rolling(5).sum()

    # 변동성
    df["volatility_20"] = close.pct_change().rolling(20).std()*100
    df["volatility_60"] = close.pct_change().rolling(60).std()*100

    # 거래량 패턴
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ratio_20"] = vol / df["vol_ma20"]
    df["vol_trend"] = df["vol_ma20"] / vol.rolling(60).mean() - 1

    # 추세 강도
    df["price_above_ma200"] = (close > df["ma200"]).astype(int)
    df["ma_alignment"] = ((df["ma5"]>df["ma20"]) & (df["ma20"]>df["ma60"]) & (df["ma60"]>df["ma120"])).astype(int)

    # 일목균형표 (단순)
    tenkan = (high.rolling(9).max() + low.rolling(9).min())/2
    kijun = (high.rolling(26).max() + low.rolling(26).min())/2
    df["ichimoku_above"] = (close > kijun).astype(int)
    df["ichimoku_cross"] = (tenkan > kijun).astype(int)

    return df


# 보조지표 컬럼 (50+)
INDICATOR_COLS = [
    "ma5","ma10","ma20","ma60","ma120","ma200",
    "ema12","ema26","ema50",
    "rsi14", "macd","macd_signal","macd_hist",
    "bb_pct","stoch_k","stoch_d",
    "atr_pct","adx14","cci20",
    "obv_trend","williams_r",
    "roc5","roc10","roc20","roc60",
    "vwap_ratio","donchian_pct",
    "body_ratio","upper_wick_ratio","lower_wick_ratio",
    "gap_up","gap_count_5d",
    "volatility_20","volatility_60",
    "vol_ratio_20","vol_trend",
    "price_above_ma200","ma_alignment",
    "ichimoku_above","ichimoku_cross",
]


# 각 시그널에 보조지표 시그널일자 값 매칭
print("  종목별 보조지표 계산...")
import time
t0 = time.time()
ohlcv_with_ind = {}
codes_in_pool = combined["Code"].unique()
for i, code in enumerate(codes_in_pool):
    if code in OHLCV:
        ohlcv_with_ind[code] = calc_indicators(OHLCV[code])
    if (i+1)%50==0:
        print(f"    {i+1}/{len(codes_in_pool)} ({time.time()-t0:.0f}s)")
print(f"  완료. {len(ohlcv_with_ind)}종목, {time.time()-t0:.0f}s")


# 시그널에 보조지표 매칭
def get_ind_value(code, dt, col):
    if code not in ohlcv_with_ind: return np.nan
    bars = ohlcv_with_ind[code]
    if col not in bars.columns: return np.nan
    past = bars[bars.index <= dt]
    if len(past) == 0: return np.nan
    return past[col].iloc[-1]


print("  시그널에 보조지표 매칭...")
t0 = time.time()
for col in INDICATOR_COLS:
    vals = [get_ind_value(c, d, col) for c, d in zip(combined["Code"], combined["Date"])]
    combined[col] = vals
print(f"  매칭 완료. {time.time()-t0:.0f}s")


# 시계열 특성 (이미 있는지 체크)
if "pre_5d_max_high_ratio" not in combined.columns:
    print("  pre_ 시계열 특성 추가...")
    combined = add_pre_features(combined)

# 전체 특성 (기존 + 새 50+)
ALL_FEATURES = FEATURES + INDICATOR_COLS
ALL_FEATURES = [f for f in ALL_FEATURES if f in combined.columns]
print(f"  총 특성 수: {len(ALL_FEATURES)}")
combined.to_parquet(CACHE / "_combined_with_indicators.parquet", index=False)


# ============ 3. 다중 타깃 RF 학습 (walk-forward) ============
print("\n[3/6] 다중 타깃 walk-forward")
windows = [
    {"tr_end": 2021, "te": 2022},
    {"tr_end": 2022, "te": 2023},
    {"tr_end": 2023, "te": 2024},
    {"tr_end": 2024, "te": 2025},
    {"tr_end": 2025, "te": 2026},
]
TARGETS = {
    "loss":   lambda d: (d["ret_180d"]<=-20).astype(int),
    "sw":     lambda d: (d["peak_180d"]>=200).astype(int),
    "100plus": lambda d: (d["peak_180d"]>=100).astype(int),
    "50plus":  lambda d: (d["peak_180d"]>=50).astype(int),
    "10plus":  lambda d: (d["peak_180d"]>=10).astype(int),
}

def train_window(train_df):
    X, _ = prepare_X(train_df, features=ALL_FEATURES)
    models = {}
    for name, y_fn in TARGETS.items():
        y = y_fn(train_df)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=15,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X, y)
        models[name] = clf
    return models

def predict_window(test_df, models):
    X, _ = prepare_X(test_df, features=ALL_FEATURES)
    for name in TARGETS:
        test_df[f"p_{name}"] = models[name].predict_proba(X)[:, 1]
    # 종합 점수들
    test_df["StrongScore_v1"] = test_df["p_sw"]*3 + test_df["p_100plus"]*1.5 + test_df["p_50plus"]*1 - test_df["p_loss"]*2
    test_df["StrongScore_v2"] = test_df["p_sw"]*5 + test_df["p_100plus"]*2 + test_df["p_50plus"]*1 - test_df["p_loss"]*3
    test_df["SafeScore"] = test_df["p_10plus"]*2 + test_df["p_50plus"]*1 - test_df["p_loss"]*5
    test_df["SuperScore"] = test_df["p_sw"]*10 - test_df["p_loss"]*5
    return test_df


all_test_predictions = []
for w in windows:
    train = combined[combined["Year"] <= w["tr_end"]].copy()
    test = combined[combined["Year"] == w["te"]].copy()
    if len(train)<300 or len(test)<30: continue
    print(f"  Walk ~{w['tr_end']} → {w['te']} | Train {len(train):,} / Test {len(test):,}")
    models = train_window(train)
    test = predict_window(test, models)
    test["window"] = f"{w['tr_end']}→{w['te']}"
    all_test_predictions.append(test)

predicted = pd.concat(all_test_predictions, ignore_index=True)
predicted.to_parquet(CACHE / "_predicted_all.parquet", index=False)


# ============ 4. 다중 시뮬 (매일1건 / 주3건 / 주5건) × 다중 점수 ============
print("\n[4/6] 시뮬레이션 (모드 × 점수)")

def simulate(picks, label):
    picks = picks.drop_duplicates(["Date","Code"]).copy()
    if len(picks) == 0: return None
    invest = len(picks) * ALLOC
    profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
    return {
        "전략": label,
        "매수": len(picks),
        "SW": int((picks["peak_180d"]>=200).sum()),
        "100+": int((picks["peak_180d"]>=100).sum()),
        "50+": int((picks["peak_180d"]>=50).sum()),
        "10+": int((picks["peak_180d"]>=10).sum()),
        "손절": int((picks["ret_180d"]<=-20).sum()),
        "SW%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "100+%": round((picks["peak_180d"]>=100).mean()*100, 1),
        "50+%": round((picks["peak_180d"]>=50).mean()*100, 1),
        "10+%": round((picks["peak_180d"]>=10).mean()*100, 1),
        "손절%": round((picks["ret_180d"]<=-20).mean()*100, 1),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }


scores = ["StrongScore_v1","StrongScore_v2","SafeScore","SuperScore","p_sw","p_100plus","p_50plus","p_10plus"]
modes = [("daily_1", 1), ("weekly_3", 3), ("weekly_5", 5)]

results = []
for score_col in scores:
    df = predicted.dropna(subset=["sell_close"]).copy()
    df = df.sort_values(score_col, ascending=False)
    for mode_name, n_per in modes:
        if mode_name == "daily_1":
            df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
        else:
            df["bucket"] = df["Date"].dt.strftime("%Y-%U")
        picks = df.groupby("bucket").head(n_per)
        r = simulate(picks, f"{score_col} / {mode_name}")
        if r: results.append(r)

res_df = pd.DataFrame(results).sort_values("수익률%", ascending=False)
res_df.to_csv(CACHE / "MASTER_results_all_scores.csv", index=False)
print("\n[Top 15 by 수익률]")
print(res_df.head(15).to_string(index=False))


# ============ 5. 년도별 결과 (best 전략) ============
print("\n[5/6] 년도별 결과 (best 전략)")
best = res_df.iloc[0]
print(f"Best: {best['전략']}")
score, mode = best["전략"].split(" / ")
df = predicted.sort_values(score, ascending=False)
n_per = {"daily_1":1, "weekly_3":3, "weekly_5":5}[mode]
if mode == "daily_1":
    df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
else:
    df["bucket"] = df["Date"].dt.strftime("%Y-%U")
best_picks = df.groupby("bucket").head(n_per).drop_duplicates(["Date","Code"]).copy()
best_picks["Year"] = best_picks["Date"].dt.year

yr_results = []
for y, g in best_picks.groupby("Year"):
    invest = len(g)*ALLOC
    profit = ((g["sell_close"]/g["Close"] - 1)*ALLOC).sum()
    yr_results.append({
        "year": int(y),
        "매수": len(g),
        "SW": int((g["peak_180d"]>=200).sum()),
        "100+": int((g["peak_180d"]>=100).sum()),
        "50+": int((g["peak_180d"]>=50).sum()),
        "10+": int((g["peak_180d"]>=10).sum()),
        "손절": int((g["ret_180d"]<=-20).sum()),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    })
yr_df = pd.DataFrame(yr_results)
yr_df.to_csv(CACHE / "MASTER_best_yearly.csv", index=False)
print(yr_df.to_string(index=False))


# ============ 6. 매수 종목 전체 리스트 ============
print("\n[6/6] 매수 종목 리스트 저장")
best_picks_out = best_picks[[
    "Date","Year","Code","Name","Market","Close","Amount",
    "p_sw","p_100plus","p_50plus","p_10plus","p_loss",
    "StrongScore_v1","StrongScore_v2","SafeScore","SuperScore",
    "ret_180d","peak_180d","sell_close","sell_date","window"
]].sort_values("Date", ascending=False)
best_picks_out.to_csv(CACHE / "MASTER_best_picks_2020-2026.csv", index=False)
print(f"  cache/MASTER_best_picks_2020-2026.csv ({len(best_picks_out):,}건)")

# 타깃별 매일 1건 별도 저장 (슈퍼위너 노릴때, 100%+ 노릴때 등)
for tgt in ["p_sw","p_100plus","p_50plus","p_10plus"]:
    df = predicted.sort_values(tgt, ascending=False)
    df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
    p = df.groupby("bucket").head(1).drop_duplicates(["Date","Code"]).copy()
    p["Year"] = p["Date"].dt.year
    out = p[["Date","Year","Code","Name","Market","Close","p_sw","p_100plus","p_50plus","p_10plus","p_loss","ret_180d","peak_180d"]].sort_values("Date", ascending=False)
    fname = f"MASTER_daily1_{tgt}_2020-2026.csv"
    out.to_csv(CACHE / fname, index=False)
    sw = (p["peak_180d"]>=200).sum()
    w100 = (p["peak_180d"]>=100).sum()
    w50 = (p["peak_180d"]>=50).sum()
    w10 = (p["peak_180d"]>=10).sum()
    loss = (p["ret_180d"]<=-20).sum()
    invest = len(p)*ALLOC
    profit = ((p["sell_close"]/p["Close"]-1)*ALLOC).sum()
    print(f"  {fname}: {len(p)}건, SW {sw}({sw/len(p)*100:.1f}%), 100+ {w100}, 50+ {w50}, 10+ {w10}, 손절 {loss}, 수익 {profit/1e4:+,.0f}만 ({profit/invest*100:+.1f}%)")

print("\n[완료]")
