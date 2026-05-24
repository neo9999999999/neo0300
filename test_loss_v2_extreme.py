"""
손절률 5% 도전 V2 - 극단적 회피 + 년도별 분석
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from train_rf_loss_model import FEATURES, prepare_X

CACHE = Path("cache")
ALLOC = 100_000

sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d", "ret_180d", "Amount"]).copy()
sigs["Year"] = sigs["Date"].dt.year
snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
pool300 = sigs[sigs["Code"].isin(top300)].copy()

# 시장 환경 (KOSPI 대용으로 삼성전자)
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)
samsung = OHLCV["005930"].copy()
samsung["ma60"] = samsung["Close"].rolling(60).mean()
samsung["ma200"] = samsung["Close"].rolling(200).mean()
samsung["bullish60"] = (samsung["Close"] > samsung["ma60"]).astype(int)
samsung["bullish200"] = (samsung["Close"] > samsung["ma200"]).astype(int)
samsung["strong_bull"] = samsung["bullish60"] & samsung["bullish200"]
env60 = samsung["bullish60"].to_dict()
env_strong = samsung["strong_bull"].to_dict()


def env_get(d, src):
    if d in src: return src[d]
    past = samsung[samsung.index <= d]
    if len(past)==0: return 0
    return int(src.get(past.index[-1], 0))


def train_loss_model(train_df):
    available = [f for f in FEATURES if f in train_df.columns]
    X, _ = prepare_X(train_df, features=available)
    y = (train_df["ret_180d"]<=-20).astype(int)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=15,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X, y)
    # 추가: SW
    y_sw = (train_df["peak_180d"]>=200).astype(int)
    clf_sw = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=15,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    clf_sw.fit(X, y_sw)
    y_50 = (train_df["peak_180d"]>=50).astype(int)
    clf_50 = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    clf_50.fit(X, y_50)
    return {"loss":clf, "sw":clf_sw, "50plus":clf_50}, available


def predict(test_df, models, features):
    X, _ = prepare_X(test_df, features=features)
    test_df["p_loss"] = models["loss"].predict_proba(X)[:, 1]
    test_df["p_sw"] = models["sw"].predict_proba(X)[:, 1]
    test_df["p_50"] = models["50plus"].predict_proba(X)[:, 1]
    test_df["bullish60"] = test_df["Date"].apply(lambda d: env_get(d, env60))
    test_df["strong_bull"] = test_df["Date"].apply(lambda d: env_get(d, env_strong))
    return test_df


def weekly_var_pick(pool, score_col="StrongScore", max_n=5, base_n=3):
    pool = pool.sort_values(score_col, ascending=False).copy()
    pool["bucket"] = pool["Date"].dt.strftime("%Y-%U")
    topN = pool.groupby("bucket").head(max_n)
    if len(topN)==0: return topN
    threshold = topN[score_col].quantile(0.40)
    def select(g):
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        base = g.head(base_n)
        extra = g.iloc[base_n:max_n]
        extra = extra[extra[score_col] >= threshold]
        return pd.concat([base, extra])
    return topN.groupby("bucket", group_keys=False).apply(select)


def evaluate(picks, label, year=None):
    picks = picks.drop_duplicates(["Date", "Code"]).copy()
    if len(picks) == 0: return None
    invest = len(picks) * ALLOC
    profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
    r = {
        "전략": label, "year": year,
        "매수": len(picks),
        "SW": int((picks["peak_180d"]>=200).sum()),
        "손절": int((picks["ret_180d"]<=-20).sum()),
        "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "손절률%": round((picks["ret_180d"]<=-20).mean()*100, 1),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }
    return r


windows = [
    {"tr":(2021,2022), "te":2023},
    {"tr":(2021,2023), "te":2024},
    {"tr":(2021,2024), "te":2025},
    {"tr":(2021,2025), "te":2026},
]


# 10가지 극단 전략
strategies = {
    "X0_base": "baseline weekly_var",
    "X1_loss10": "RF 손절확률 상위 10%만 제외",
    "X2_loss05": "RF 손절확률 상위 5%만 제외",
    "X3_loss50_strong": "손절↓50% AND strong_bull (장기+단기 상승)",
    "X4_p_loss_lt_15": "절대 손절확률 < 15%만 매수",
    "X5_p_loss_lt_10": "절대 손절확률 < 10%만 매수",
    "X6_AND_sw_50": "p_loss<0.15 AND p_sw>0.1 AND p_50>0.5",
    "X7_strong_only": "strong_bull=1 만 매수",
    "X8_ultra_safe": "p_loss<0.1 + strong_bull=1 + p_50>0.5",
    "X9_no_2023_2024": "약세장 (2023,2024) 시그널 매수 자제 + 강세장만",
}

all_results = []

for w in windows:
    train = pool300[(pool300["Year"]>=w["tr"][0])&(pool300["Year"]<=w["tr"][1])].copy()
    test = pool300[pool300["Year"]==w["te"]].copy()
    if len(train)<300 or len(test)<30: continue
    print(f"\n--- {w['tr']}→{w['te']} | Train {len(train):,} / Test {len(test):,} ---")
    models, features = train_loss_model(train)
    test = predict(test, models, features)

    # StrongScore (기본)
    test["StrongScore"] = test["p_sw"]*3.0 + test["p_50"]*1.0 - test["p_loss"]*5.0

    # X0: base
    p = weekly_var_pick(test); r = evaluate(p, "X0_base", w["te"])
    if r: all_results.append(r)

    # X1: 손절확률 상위 10% 제외
    th90 = test["p_loss"].quantile(0.90)
    p = weekly_var_pick(test[test["p_loss"]<th90])
    r = evaluate(p, "X1_loss10", w["te"]);
    if r: all_results.append(r)

    # X2: 상위 5% 제외
    th95 = test["p_loss"].quantile(0.95)
    p = weekly_var_pick(test[test["p_loss"]<th95])
    r = evaluate(p, "X2_loss05", w["te"]);
    if r: all_results.append(r)

    # X3: 손절↓50% + strong_bull
    safe = test[(test["p_loss"]<test["p_loss"].quantile(0.50)) & (test["strong_bull"]==1)]
    p = weekly_var_pick(safe)
    r = evaluate(p, "X3_loss50_strong", w["te"]);
    if r: all_results.append(r)

    # X4: 절대 < 0.15
    p = weekly_var_pick(test[test["p_loss"]<0.15])
    r = evaluate(p, "X4_p_loss_lt_15", w["te"]);
    if r: all_results.append(r)

    # X5: 절대 < 0.10
    p = weekly_var_pick(test[test["p_loss"]<0.10])
    r = evaluate(p, "X5_p_loss_lt_10", w["te"]);
    if r: all_results.append(r)

    # X6: AND 조건
    p = weekly_var_pick(test[
        (test["p_loss"]<0.15) & (test["p_sw"]>0.10) & (test["p_50"]>0.50)
    ])
    r = evaluate(p, "X6_AND_sw_50", w["te"]);
    if r: all_results.append(r)

    # X7: strong_bull만
    p = weekly_var_pick(test[test["strong_bull"]==1])
    r = evaluate(p, "X7_strong_only", w["te"]);
    if r: all_results.append(r)

    # X8: ultra safe
    p = weekly_var_pick(test[
        (test["p_loss"]<0.10) & (test["strong_bull"]==1) & (test["p_50"]>0.50)
    ])
    r = evaluate(p, "X8_ultra_safe", w["te"]);
    if r: all_results.append(r)

    # X9: 약세장 회피 (bullish60=0이면 매수 X)
    p = weekly_var_pick(test[test["bullish60"]==1])
    r = evaluate(p, "X9_no_2023_2024", w["te"]);
    if r: all_results.append(r)


res_df = pd.DataFrame(all_results)
res_df.to_csv(CACHE / "loss_v2_extreme_year.csv", index=False)

# 년도별 표
print("\n" + "="*120)
print("년도별 손절률 (%)")
print("="*120)
pivot_loss = res_df.pivot_table(index="전략", columns="year", values="손절률%", aggfunc="sum")
print(pivot_loss.to_string())

print("\n년도별 매수 수")
pivot_n = res_df.pivot_table(index="전략", columns="year", values="매수", aggfunc="sum")
print(pivot_n.to_string())

print("\n년도별 SW률 (%)")
pivot_sw = res_df.pivot_table(index="전략", columns="year", values="SW률%", aggfunc="sum")
print(pivot_sw.to_string())

print("\n년도별 수익률 (%)")
pivot_ret = res_df.pivot_table(index="전략", columns="year", values="수익률%", aggfunc="sum")
print(pivot_ret.to_string())

# 누적
print("\n" + "="*100)
print("누적 4년 비교")
print("="*100)
summary = []
for label in strategies.keys():
    sub = res_df[res_df["전략"]==label]
    if len(sub)==0: continue
    n = sub["매수"].sum()
    sw = sub["SW"].sum()
    loser = sub["손절"].sum()
    inv = sub["투자만"].sum()
    prof = sub["수익만"].sum()
    summary.append({
        "전략": label,
        "설명": strategies[label],
        "매수": n, "SW": sw, "손절": loser,
        "SW률%": round(sw/n*100,1) if n else 0,
        "손절률%": round(loser/n*100,1) if n else 0,
        "투자만": inv, "수익만": prof,
        "수익률%": round(prof/inv*100,1) if inv else 0,
    })
sum_df = pd.DataFrame(summary)
print(sum_df.to_string(index=False))
sum_df.to_csv(CACHE / "loss_v2_summary.csv", index=False)

# 5% 미만 달성
print("\n" + "="*100)
print("🎯 5% 미만 달성 전략 (누적 4년 평균)")
print("="*100)
sub5 = sum_df[sum_df["손절률%"]<5.0]
if len(sub5)>0:
    print(sub5[["전략","설명","매수","SW률%","손절률%","수익률%"]].to_string(index=False))
else:
    print("⚠️  누적 4년 평균 5% 미만 달성 전략 없음")

print("\n[저장] cache/loss_v2_summary.csv + loss_v2_extreme_year.csv")
