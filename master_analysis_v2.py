"""
Master Analysis V2 - 보조지표 캐시 사용 + walk-forward
중복 컬럼 처리 fix.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from train_rf_loss_model import FEATURES

CACHE = Path("cache")
ALLOC = 100_000


print("[로드] _combined_with_indicators.parquet")
combined = pd.read_parquet(CACHE / "_combined_with_indicators.parquet")
combined["Date"] = pd.to_datetime(combined["Date"])
combined["Year"] = combined["Date"].dt.year
print(f"  통합 풀: {len(combined):,}건")

# 중복 컬럼 제거
combined = combined.loc[:, ~combined.columns.duplicated()].copy()
print(f"  컬럼 (dedup): {len(combined.columns)}")


INDICATOR_COLS = [
    "rsi14","macd","macd_signal","macd_hist",
    "bb_pct","stoch_k","stoch_d",
    "atr_pct","adx14","cci20",
    "obv_trend","williams_r",
    "roc5","roc10","roc20","roc60",
    "vwap_ratio","donchian_pct",
    "body_ratio",
    "gap_up","gap_count_5d",
    "volatility_20","volatility_60",
    "vol_ratio_20","vol_trend",
    "price_above_ma200","ma_alignment",
    "ichimoku_above","ichimoku_cross",
]

# 기존 FEATURES + 신규 보조지표 (중복 제거)
ALL_FEATURES = list(dict.fromkeys(FEATURES + INDICATOR_COLS))
ALL_FEATURES = [f for f in ALL_FEATURES if f in combined.columns]
print(f"  사용 특성: {len(ALL_FEATURES)}")


def prepare_X(df, features):
    """깨끗하게 prepare (중복 컬럼 안전)"""
    X = df.loc[:, features].copy()
    # 중복 컬럼 처리
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    for f in X.columns:
        if X[f].dtype.kind in "fi":
            try:
                q01, q99 = X[f].quantile(0.001), X[f].quantile(0.999)
                X[f] = X[f].clip(q01, q99)
            except Exception:
                pass
    return X


# Walk-forward
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
    "100plus":lambda d: (d["peak_180d"]>=100).astype(int),
    "50plus": lambda d: (d["peak_180d"]>=50).astype(int),
    "10plus": lambda d: (d["peak_180d"]>=10).astype(int),
}

print("\n[Walk-Forward]")
all_pred = []
for w in windows:
    train = combined[combined["Year"]<=w["tr_end"]].copy()
    test = combined[combined["Year"]==w["te"]].copy()
    if len(train)<300 or len(test)<30: continue
    print(f"  ~{w['tr_end']} → {w['te']} | Train {len(train):,} / Test {len(test):,}")
    X_tr = prepare_X(train, ALL_FEATURES)
    X_te = prepare_X(test, ALL_FEATURES)
    for name, y_fn in TARGETS.items():
        y = y_fn(train)
        clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=15,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_tr, y)
        test[f"p_{name}"] = clf.predict_proba(X_te)[:, 1]
    test["StrongScore_v1"] = test["p_sw"]*3 + test["p_100plus"]*1.5 + test["p_50plus"]*1 - test["p_loss"]*2
    test["StrongScore_v2"] = test["p_sw"]*5 + test["p_100plus"]*2 + test["p_50plus"]*1 - test["p_loss"]*3
    test["SafeScore"] = test["p_10plus"]*2 + test["p_50plus"]*1 - test["p_loss"]*5
    test["SuperScore"] = test["p_sw"]*10 - test["p_loss"]*5
    all_pred.append(test)

predicted = pd.concat(all_pred, ignore_index=True)
predicted.to_parquet(CACHE / "_predicted_all.parquet", index=False)
print(f"\n예측 완료: {len(predicted):,}건")


# 시뮬레이션
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


scores = ["StrongScore_v1","StrongScore_v2","SafeScore","SuperScore",
          "p_sw","p_100plus","p_50plus","p_10plus"]
modes = [("daily_1", 1), ("weekly_3", 3), ("weekly_5", 5)]

print("\n[시뮬레이션 (8 점수 × 3 모드 = 24)]")
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
print("\n[Top 20 by 수익률]")
print(res_df.head(20).to_string(index=False))


# 년도별 best 전략
best = res_df.iloc[0]
print(f"\n[Best: {best['전략']}]")
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
        "year": int(y), "매수": len(g),
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
print("\n[년도별]")
print(yr_df.to_string(index=False))


# 타깃별 매일1건 저장
print("\n[타깃별 매일 1건]")
for tgt in ["p_sw","p_100plus","p_50plus","p_10plus"]:
    df = predicted.sort_values(tgt, ascending=False)
    df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
    p = df.groupby("bucket").head(1).drop_duplicates(["Date","Code"]).copy()
    p["Year"] = p["Date"].dt.year
    out_cols = [c for c in ["Date","Year","Code","Name","Market","Close","Amount",
                            "p_sw","p_100plus","p_50plus","p_10plus","p_loss",
                            "ret_180d","peak_180d","sell_close","sell_date"] if c in p.columns]
    out = p[out_cols].sort_values("Date", ascending=False)
    fname = f"MASTER_daily1_{tgt}_2020-2026.csv"
    out.to_csv(CACHE / fname, index=False)
    sw = (p["peak_180d"]>=200).sum()
    w100 = (p["peak_180d"]>=100).sum()
    w50 = (p["peak_180d"]>=50).sum()
    w10 = (p["peak_180d"]>=10).sum()
    loss = (p["ret_180d"]<=-20).sum()
    invest = len(p)*ALLOC
    profit = ((p["sell_close"]/p["Close"]-1)*ALLOC).sum()
    print(f"  {fname}: {len(p)}건 | SW {sw}({sw/len(p)*100:.1f}%) | 100+ {w100}({w100/len(p)*100:.1f}%) | "
          f"50+ {w50}({w50/len(p)*100:.1f}%) | 10+ {w10}({w10/len(p)*100:.1f}%) | "
          f"손절 {loss}({loss/len(p)*100:.1f}%) | 수익 {profit/1e4:+,.0f}만 ({profit/invest*100:+.1f}%)")

# 베스트 종목 리스트
out_cols = [c for c in ["Date","Year","Code","Name","Market","Close","Amount",
                        "p_sw","p_100plus","p_50plus","p_10plus","p_loss",
                        "StrongScore_v1","StrongScore_v2","SafeScore","SuperScore",
                        "ret_180d","peak_180d","sell_close"] if c in best_picks.columns]
best_picks[out_cols].sort_values("Date", ascending=False).to_csv(CACHE / "MASTER_best_picks_2020-2026.csv", index=False)
print(f"\n[저장] cache/MASTER_best_picks_2020-2026.csv ({len(best_picks):,}건)")

print("\n[완료]")
