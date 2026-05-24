"""
손절률 < 5% 달성 도전 (시총 300 풀)
=================================
다양한 손절회피 전략 walk-forward OOS:

S1: baseline weekly_var (현재 11.7%)
S2: RF 손절확률 상위 50% 제외 + weekly_var
S3: RF 손절확률 상위 30% 제외 + weekly_var
S4: RF 손절확률 상위 20% 제외 + weekly_var
S5: 시장 환경 필터 (KOSPI 60일 이평 위) + weekly_var
S6: 다중 안전 조건 (p_loss<0.3 AND p_sw>0.05 AND p_50plus>0.4)
S7: top 보수형 - StrongScore + 손절확률 동시 정렬 (w_sw=7, w_loss=10)
S8: 2단계 = 손절확률 ↓ 50%만 → 그 안에서 weekly_var
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from train_rf_loss_model import FEATURES, prepare_X

CACHE = Path("cache")
ALLOC = 100_000


# 시그널 + top300 풀
sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d", "ret_180d", "Amount"]).copy()
sigs["Year"] = sigs["Date"].dt.year

snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
pool300 = sigs[sigs["Code"].isin(top300)].copy()
print(f"시총 300 풀 시그널: {len(pool300):,}")


# KOSPI 지수 (참조용 - 매일 시장 상승추세 여부)
# 삼성전자 200일 이평 위인지로 시장 환경 근사
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)

samsung = OHLCV["005930"].copy()
samsung["ma60"] = samsung["Close"].rolling(60).mean()
samsung["bullish"] = (samsung["Close"] > samsung["ma60"]).astype(int)

# 매일 시장 환경 dict
market_env = samsung["bullish"].to_dict()


def get_market_bullish(d):
    """그 일자에서 시장 상승추세 여부"""
    if d in market_env:
        return market_env[d]
    # 가장 가까운 이전 일자
    past = samsung[samsung.index <= d]
    if len(past) == 0: return 0
    return int(past["bullish"].iloc[-1])


def train_all_models(train_df):
    available = [f for f in FEATURES if f in train_df.columns]
    X, _ = prepare_X(train_df, features=available)
    models = {}
    for tgt_name, y in [
        ("loss", (train_df["ret_180d"]<=-20).astype(int)),
        ("sw", (train_df["peak_180d"]>=200).astype(int)),
        ("100plus", (train_df["peak_180d"]>=100).astype(int)),
        ("50plus", (train_df["peak_180d"]>=50).astype(int)),
    ]:
        clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X, y)
        models[tgt_name] = clf
    return models, available


def predict(test_df, models, features):
    X, _ = prepare_X(test_df, features=features)
    test_df["p_loss"] = models["loss"].predict_proba(X)[:, 1]
    test_df["p_sw"] = models["sw"].predict_proba(X)[:, 1]
    test_df["p_100plus"] = models["100plus"].predict_proba(X)[:, 1]
    test_df["p_50plus"] = models["50plus"].predict_proba(X)[:, 1]
    test_df["StrongScore"] = test_df["p_sw"]*3.0 + test_df["p_100plus"]*1.5 + test_df["p_50plus"]*1.0 - test_df["p_loss"]*2.0
    # 시장 환경
    test_df["market_bullish"] = test_df["Date"].apply(get_market_bullish)
    return test_df


def weekly_var_pick(pool, score_col="StrongScore", max_n=5, base_n=3, extra_threshold=None):
    """주별 score↑ TOP3 + 임계값 통과시 4-5등 추가"""
    pool = pool.sort_values(score_col, ascending=False).copy()
    pool["bucket"] = pool["Date"].dt.strftime("%Y-%U")
    topN = pool.groupby("bucket").head(max_n)
    if extra_threshold is None:
        extra_threshold = topN[score_col].quantile(0.40)
    def select(g):
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        base = g.head(base_n)
        extra = g.iloc[base_n:max_n]
        extra = extra[extra[score_col] >= extra_threshold]
        return pd.concat([base, extra])
    return topN.groupby("bucket", group_keys=False).apply(select)


def evaluate(picks, label):
    picks = picks.drop_duplicates(["Date", "Code"]).copy()
    if len(picks) == 0:
        return None
    invest = len(picks) * ALLOC
    profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
    return {
        "전략": label,
        "매수": len(picks),
        "SW": int((picks["peak_180d"]>=200).sum()),
        "100+": int((picks["peak_180d"]>=100).sum()),
        "50+": int((picks["peak_180d"]>=50).sum()),
        "손절": int((picks["ret_180d"]<=-20).sum()),
        "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "100+%": round((picks["peak_180d"]>=100).mean()*100, 1),
        "50+%": round((picks["peak_180d"]>=50).mean()*100, 1),
        "손절률%": round((picks["ret_180d"]<=-20).mean()*100, 1),
        "투자(만)": invest/1e4,
        "수익(만)": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }


# Walk-forward 검증
windows = [
    {"tr":(2021,2022), "te":2023},
    {"tr":(2021,2023), "te":2024},
    {"tr":(2021,2024), "te":2025},
    {"tr":(2021,2025), "te":2026},
]

all_picks = {s: [] for s in ["S1","S2","S3","S4","S5","S6","S7","S8"]}

for w in windows:
    train = pool300[(pool300["Year"]>=w["tr"][0])&(pool300["Year"]<=w["tr"][1])].copy()
    test = pool300[pool300["Year"]==w["te"]].copy()
    if len(train)<300 or len(test)<30: continue
    print(f"\n--- {w['tr']} → {w['te']} | Train {len(train):,} / Test {len(test):,} ---")
    models, features = train_all_models(train)
    test = predict(test, models, features)

    # ====== 전략 시뮬 ======

    # S1: baseline
    s1 = weekly_var_pick(test); all_picks["S1"].append(s1)

    # S2: 손절확률 상위 50% 제외 (median 이상 제외)
    th50 = test["p_loss"].median()
    safe50 = test[test["p_loss"] < th50].copy()
    s2 = weekly_var_pick(safe50); all_picks["S2"].append(s2)

    # S3: 손절확률 상위 30% 제외 (70 분위 이상 제외)
    th70 = test["p_loss"].quantile(0.70)
    safe70 = test[test["p_loss"] < th70].copy()
    s3 = weekly_var_pick(safe70); all_picks["S3"].append(s3)

    # S4: 손절확률 상위 20% 제외 (80 분위 이상 제외)
    th80 = test["p_loss"].quantile(0.80)
    safe80 = test[test["p_loss"] < th80].copy()
    s4 = weekly_var_pick(safe80); all_picks["S4"].append(s4)

    # S5: 시장 환경 (bullish=1만)
    bullish_test = test[test["market_bullish"]==1].copy()
    s5 = weekly_var_pick(bullish_test); all_picks["S5"].append(s5)

    # S6: 다중 안전 조건
    safe_multi = test[
        (test["p_loss"] < test["p_loss"].quantile(0.50))
        & (test["p_sw"] > test["p_sw"].quantile(0.50))
        & (test["p_50plus"] > test["p_50plus"].quantile(0.50))
    ].copy()
    s6 = weekly_var_pick(safe_multi); all_picks["S6"].append(s6)

    # S7: 보수형 StrongScore (w_sw=7, w_loss=10)
    test["S7_score"] = test["p_sw"]*7.0 + test["p_100plus"]*1.5 + test["p_50plus"]*1.0 - test["p_loss"]*10.0
    s7 = weekly_var_pick(test, score_col="S7_score"); all_picks["S7"].append(s7)

    # S8: 2단계 - 손절확률 ↓ 50% + bullish 시장
    safe_bullish = test[(test["p_loss"] < test["p_loss"].quantile(0.50)) & (test["market_bullish"]==1)].copy()
    s8 = weekly_var_pick(safe_bullish); all_picks["S8"].append(s8)


# 누적 집계
print("\n" + "="*120)
print("8가지 손절회피 전략 누적 비교 (시총 300, 4년 walk-forward)")
print("="*120)

results = []
strategies = {
    "S1_baseline": "weekly_var 그대로",
    "S2_loss50": "손절확률 상위 50% 제외",
    "S3_loss30": "손절확률 상위 30% 제외",
    "S4_loss20": "손절확률 상위 20% 제외",
    "S5_bullish": "시장 상승추세일 때만",
    "S6_multi": "다중 안전 (p_loss↓ AND p_sw↑ AND p_50↑)",
    "S7_conservative": "보수형 점수 (w_sw=7, w_loss=10)",
    "S8_safe_bullish": "S2 + 시장 상승추세 동시",
}
for code, picks_list in all_picks.items():
    if not picks_list: continue
    combined = pd.concat(picks_list, ignore_index=True)
    label = list(strategies.keys())[list(all_picks.keys()).index(code)]
    r = evaluate(combined, label)
    if r:
        r["설명"] = strategies[label]
        results.append(r)

res_df = pd.DataFrame(results)
print(res_df[["전략", "설명", "매수", "SW", "SW률%", "100+%", "50+%", "손절", "손절률%", "투자(만)", "수익(만)", "수익률%"]].to_string(index=False))
res_df.to_csv(CACHE / "loss_avoidance_strategies.csv", index=False)

# 손절률 < 5% 달성 여부
print("\n" + "="*100)
print("🎯 손절률 < 5% 달성 전략")
print("="*100)
sub5 = res_df[res_df["손절률%"] < 5.0]
if len(sub5) > 0:
    print(sub5[["전략", "설명", "매수", "SW률%", "손절률%", "수익률%"]].to_string(index=False))
else:
    print("⚠️  5% 미만 달성 전략 없음 - 더 강한 회피 필요")

print(f"\n[저장] cache/loss_avoidance_strategies.csv")
