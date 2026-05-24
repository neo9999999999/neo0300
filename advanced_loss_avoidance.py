"""
고도화 손절회피 분석 — 주3건 모드 집중
=====================================

방법:
1. 손실 종목 (ret_180d <= -20%) vs 비손실 종목 비교
2. 결정트리로 자동 패턴 마이닝 (AND 조합 발굴)
3. 2-3 변수 AND 조합 그리드 서치 (수동 보완)
4. 시그널 발생 직전 N일 가격/거래량 시계열 패턴
5. KOSDAQ 지수 상태 (시장 환경) 조건
6. 발굴된 모든 룰을 합집합 + AND 회피로 손절률 ↓

목표: 주3건 (코스닥) 풀의 손절률 21.6% → 15% 이하
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

CACHE = Path("cache")

cand = pd.read_parquet(CACHE / "candidates_enriched.parquet")
cand["Date"] = pd.to_datetime(cand["Date"])
cand = cand.dropna(subset=["peak_180d", "sell_close", "ret_180d"]).copy()

# 주3건 풀: KOSDAQ만, 거래대금 낮은 순 주 3건
kosdaq_pool = cand[cand["Market"] == "KOSDAQ"].sort_values("Amount").copy()
kosdaq_pool["bucket"] = kosdaq_pool["Date"].dt.strftime("%Y-%U")
weekly3 = kosdaq_pool.groupby("bucket").head(3).sort_values("Date").reset_index(drop=True)
print(f"주3건 풀 (KOSDAQ): {len(weekly3):,}건")

LOSER_TH = -20
SW_TH = 200

# 손실/비손실
weekly3["is_loser"] = (weekly3["ret_180d"] <= LOSER_TH).astype(int)
weekly3["is_sw"] = (weekly3["peak_180d"] >= SW_TH).astype(int)
weekly3["is_w50"] = (weekly3["peak_180d"] >= 50).astype(int)
print(f"  손실(<=-20%): {weekly3['is_loser'].sum()}건 ({weekly3['is_loser'].mean()*100:.1f}%)")
print(f"  슈퍼위너: {weekly3['is_sw'].sum()}건 ({weekly3['is_sw'].mean()*100:.1f}%)")

# 직전 N일 시계열 특성 추가
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


def add_pre_signal_features(df):
    """매수 시그널 발생 전 N일의 가격/거래량 행동 패턴"""
    rows = {
        "pre_5d_max_high_ratio": [],   # 직전 5일 최고가 / 시그널일 종가
        "pre_5d_min_low_ratio": [],    # 직전 5일 최저가 / 시그널일 종가
        "pre_5d_vol_trend": [],         # 직전 5일 거래량 추세 (선형 기울기)
        "pre_10d_max_high_ratio": [],
        "pre_10d_drawdown": [],         # 직전 10일 내 고점 대비 현재 위치
        "pre_20d_vol_ratio": [],        # 직전 20일 vs 직전 60일 거래량 비율
        "gap_up_count_5d": [],          # 직전 5일 갭상승 카운트
        "long_red_count_5d": [],        # 직전 5일 음봉 카운트
        "long_red_in_10d": [],          # 직전 10일 큰 음봉(-3%↓) 카운트
        "consecutive_red_max": [],      # 직전 10일 연속 음봉 최대
    }
    for _, r in df.iterrows():
        code = r["Code"]; d0 = r["Date"]; close0 = r["Close"]
        if code not in OHLCV:
            for k in rows: rows[k].append(np.nan)
            continue
        past = OHLCV[code][OHLCV[code].index < d0].tail(30)
        if len(past) < 10:
            for k in rows: rows[k].append(np.nan)
            continue
        # 5일
        p5 = past.tail(5)
        rows["pre_5d_max_high_ratio"].append(p5["High"].max() / close0)
        rows["pre_5d_min_low_ratio"].append(p5["Low"].min() / close0)
        rows["pre_5d_vol_trend"].append(
            np.polyfit(range(len(p5)), p5["Volume"], 1)[0] / (p5["Volume"].mean() + 1) if p5["Volume"].mean() > 0 else 0
        )
        # 10일
        p10 = past.tail(10)
        rows["pre_10d_max_high_ratio"].append(p10["High"].max() / close0)
        rows["pre_10d_drawdown"].append((close0 - p10["High"].max()) / p10["High"].max() * 100)
        # 20일 vol vs 60일
        p20 = past.tail(20); p60 = past.tail(60) if len(past) >= 60 else past
        rows["pre_20d_vol_ratio"].append(p20["Volume"].mean() / (p60["Volume"].mean() + 1))
        # 갭상승
        opens = p5["Open"].values; closes_prev = p5["Close"].shift(1).values
        gap_up = ((opens[1:] / closes_prev[1:]) > 1.02).sum() if len(opens) > 1 else 0
        rows["gap_up_count_5d"].append(gap_up)
        # 음봉
        red5 = (p5["Close"] < p5["Open"]).sum()
        rows["long_red_count_5d"].append(red5)
        long_red10 = ((p10["Close"] / p10["Open"] - 1) <= -0.03).sum()
        rows["long_red_in_10d"].append(long_red10)
        # 연속 음봉
        red_series = (p10["Close"] < p10["Open"]).astype(int).values
        max_streak = 0; cur = 0
        for v in red_series:
            if v: cur += 1; max_streak = max(max_streak, cur)
            else: cur = 0
        rows["consecutive_red_max"].append(max_streak)
    for k, v in rows.items():
        df[k] = v
    return df


print("\n[직전 가격행동 시계열 특성 추가 중...]")
weekly3 = add_pre_signal_features(weekly3)
print(f"  완료. pre_* 특성 추가됨.")


# 특성 변수 정의
features = [
    "Score", "Amount", "vol_ratio", "candle_pct", "cum_5d_gain",
    "rs_ratio", "ma3", "ma5", "ma10",
    "pos_60_high", "pos_120_high", "pos_240_high", "pos_252_high",
    "past_5d", "past_20", "past_60", "past_120", "past_240",
    "slope60", "slope120", "range60_pct", "range120_pct",
    "drawdown60", "runup60", "vol20", "vol60",
    "days_since_52w_low", "days_since_52w_high",
    "For_5d", "Inst_5d", "For_20d", "Inst_20d",
    "PER_num", "PBR_num", "외인소진율_num",
    # 신규 시계열
    "pre_5d_max_high_ratio", "pre_5d_min_low_ratio", "pre_5d_vol_trend",
    "pre_10d_max_high_ratio", "pre_10d_drawdown", "pre_20d_vol_ratio",
    "gap_up_count_5d", "long_red_count_5d", "long_red_in_10d", "consecutive_red_max",
]
features = [f for f in features if f in weekly3.columns]
X = weekly3[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
# 극단치 clipping
for f in features:
    if X[f].dtype.kind in 'fi':
        q01, q99 = X[f].quantile(0.001), X[f].quantile(0.999)
        X[f] = X[f].clip(q01, q99)
y_loser = weekly3["is_loser"]
y_sw = weekly3["is_sw"]
print(f"\n특성 수: {len(features)}")


# =========== 1. 결정트리 (해석가능) ============
print("\n" + "="*100)
print("[1] 결정트리 — 손절 종목 자동 패턴 마이닝")
print("="*100)

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight='balanced', random_state=42)
dt.fit(X, y_loser)
print(export_text(dt, feature_names=features, max_depth=4))

# leaf 별 손절률
def get_leaves_stats(tree, X, y_loser, y_sw, features, df):
    """각 leaf에 해당하는 종목의 손절률/SW률 측정"""
    leaf_ids = tree.apply(X)
    df = df.copy()
    df["leaf"] = leaf_ids
    stats = df.groupby("leaf").agg(
        n=("Code", "count"),
        loser_rate=("is_loser", "mean"),
        sw_rate=("is_sw", "mean"),
        w50_rate=("is_w50", "mean"),
        avg_ret=("ret_180d", "mean"),
        avg_peak=("peak_180d", "mean"),
    ).reset_index()
    return stats.sort_values("loser_rate", ascending=False)


leaf_stats = get_leaves_stats(dt, X, y_loser, y_sw, features, weekly3)
print("\n[Leaf별 통계 (손절률 높은 순)]")
print(leaf_stats.to_string(index=False))


# =========== 2. 2-변수 AND 조합 그리드 ============
print("\n" + "="*100)
print("[2] 2-변수 AND 조합 그리드 (손절률 높은 패턴)")
print("="*100)

def find_and_combos(df, features, top_k=20):
    """2변수 AND 조합 (각 변수에 cutoff)"""
    n_total = len(df)
    base_loser = df["is_loser"].mean() * 100
    candidates = []
    # 각 변수마다 cutoff 양쪽 (≤10% q, ≥90% q)
    cuts = {}
    for f in features:
        if f not in df.columns: continue
        s = df[f].dropna()
        if len(s) < 100: continue
        cuts[f] = {
            "low5": s.quantile(0.05), "low10": s.quantile(0.10), "low20": s.quantile(0.20),
            "high80": s.quantile(0.80), "high90": s.quantile(0.90), "high95": s.quantile(0.95),
        }
    feat_list = list(cuts.keys())
    print(f"  변수 {len(feat_list)}개, 조합 탐색 중...")
    pair_count = 0
    for i, f1 in enumerate(feat_list):
        for f2 in feat_list[i+1:]:
            for d1, op1, lbl1 in [
                ("low10", "≤", "low10"), ("low20", "≤", "low20"),
                ("high80", "≥", "high80"), ("high90", "≥", "high90"),
            ]:
                for d2, op2, lbl2 in [
                    ("low10", "≤", "low10"), ("low20", "≤", "low20"),
                    ("high80", "≥", "high80"), ("high90", "≥", "high90"),
                ]:
                    c1 = cuts[f1][d1]; c2 = cuts[f2][d2]
                    if op1 == "≤": m1 = df[f1] <= c1
                    else: m1 = df[f1] >= c1
                    if op2 == "≤": m2 = df[f2] <= c2
                    else: m2 = df[f2] >= c2
                    m = (m1 & m2).fillna(False)
                    n = m.sum()
                    if n < 30 or n > 0.20 * n_total: continue
                    loser_rate = df[m]["is_loser"].mean() * 100
                    sw_rate = df[m]["is_sw"].mean() * 100
                    if loser_rate < base_loser + 10: continue  # 평균보다 +10%p 이상 손절률
                    # score = 손절률 - SW 손실(가중 ×2)
                    sc = loser_rate - 2 * sw_rate
                    candidates.append({
                        "rule": f"{f1} {op1} {c1:.2f} AND {f2} {op2} {c2:.2f}",
                        "f1": f1, "f2": f2,
                        "n": int(n), "loser_rate": loser_rate, "sw_rate": sw_rate,
                        "score": sc,
                    })
                    pair_count += 1
    if not candidates:
        return pd.DataFrame()
    res = pd.DataFrame(candidates).sort_values("score", ascending=False).head(top_k)
    return res

top_and = find_and_combos(weekly3, features, top_k=25)
print(f"\n[Top 25 AND 조합 (손절률 최고)]")
print(top_and.to_string(index=False))


# =========== 3. 손절회피 룰 합집합 적용 ============
print("\n" + "="*100)
print("[3] 발굴된 룰 합집합 회피")
print("="*100)

# 결정트리 leaf 중 손절률 >= 35% 인 것만 회피
high_risk_leaves = leaf_stats[leaf_stats["loser_rate"] >= 0.35]
print(f"\n결정트리 고위험 leaf: {len(high_risk_leaves)}개")
print(high_risk_leaves.to_string(index=False))
# 트리 leaf의 종목 마스킹
leaf_ids = dt.apply(X)
weekly3["_leaf"] = leaf_ids
tree_excl_mask = weekly3["_leaf"].isin(high_risk_leaves["leaf"].values)

# AND 조합 중 손절률 >= 40% 이상만 회피
strong_and = top_and[top_and["loser_rate"] >= 40].head(10)
print(f"\nAND 조합 강한 손절패턴: {len(strong_and)}개")
print(strong_and.to_string(index=False))

# 합집합
combined_excl = tree_excl_mask.copy()
for _, r in strong_and.iterrows():
    # 룰 파싱
    parts = r["rule"].split(" AND ")
    masks = []
    for p in parts:
        items = p.split()
        f = items[0]; op = items[1]; val = float(items[2])
        if op == "≤": masks.append(weekly3[f] <= val)
        else: masks.append(weekly3[f] >= val)
    m = masks[0] & masks[1]
    combined_excl |= m.fillna(False)

print(f"\n합집합 회피 적용: 제외 {combined_excl.sum()}건 / 잔여 {(~combined_excl).sum()}건")

# 결과 비교
def show(df, label):
    if len(df) == 0:
        print(f"[{label}] 비어있음")
        return
    invest = len(df) * 100_000
    profit = ((df["sell_close"] / df["Close"] - 1) * 100_000).sum()
    print(f"\n[{label}] {len(df):,}건")
    print(f"  손절(≤-20%): {df['is_loser'].sum()}건 ({df['is_loser'].mean()*100:.1f}%)")
    print(f"  SW(≥200%):  {df['is_sw'].sum()}건 ({df['is_sw'].mean()*100:.1f}%)")
    print(f"  100%+:      {(df['peak_180d']>=100).sum()}건 ({(df['peak_180d']>=100).mean()*100:.1f}%)")
    print(f"  50%+:       {df['is_w50'].sum()}건 ({df['is_w50'].mean()*100:.1f}%)")
    print(f"  평균peak: {df['peak_180d'].mean():.1f}%   평균ret: {df['ret_180d'].mean():+.1f}%")
    print(f"  ★ 투자 {invest/1e4:,.0f}만 → 수익 {profit/1e4:+,.0f}만 ({profit/invest*100:+.1f}%)")
    return df

show(weekly3, "초기 주3건 (KOSDAQ 거래대금↓)")
show(weekly3[~combined_excl], "고도화 회피 적용 후")


# 4. 더 공격적 회피 - 손절률 30% 이상 leaf만 회피
print("\n" + "="*100)
print("[4] 매우 공격적 (손절률 30%↑ leaf + AND조합 35%↑ 합집합)")
print("="*100)

high_risk_30 = leaf_stats[leaf_stats["loser_rate"] >= 0.30]
print(f"고위험 leaf (≥30%): {len(high_risk_30)}개")
strong_and_35 = top_and[top_and["loser_rate"] >= 35].head(15)

combined2 = weekly3["_leaf"].isin(high_risk_30["leaf"].values)
for _, r in strong_and_35.iterrows():
    parts = r["rule"].split(" AND ")
    masks = []
    for p in parts:
        items = p.split()
        f = items[0]; op = items[1]; val = float(items[2])
        if op == "≤": masks.append(weekly3[f] <= val)
        else: masks.append(weekly3[f] >= val)
    m = masks[0] & masks[1]
    combined2 |= m.fillna(False)

print(f"\n공격적 회피: 제외 {combined2.sum()}건 / 잔여 {(~combined2).sum()}건")
show(weekly3[~combined2], "공격적 회피 적용 후")


# 5. RF로 손절 확률 → 상위 confidence 회피
print("\n" + "="*100)
print("[5] RandomForest 손절확률 모델 — top quantile 회피")
print("="*100)

rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X, y_loser)
weekly3["loser_prob"] = rf.predict_proba(X)[:, 1]

# 변수 중요도
fi = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
print("\n[변수 중요도 Top 15]")
print(fi.head(15).to_string(index=False))

# 손절확률 분위별
print("\n[손절확률 분위별 풀 결과]")
weekly3["prob_decile"] = pd.qcut(weekly3["loser_prob"], 10, labels=False)
print(weekly3.groupby("prob_decile").agg(
    n=("Code", "count"),
    loser_rate=("is_loser", "mean"),
    sw_rate=("is_sw", "mean"),
    avg_peak=("peak_180d", "mean"),
    avg_ret=("ret_180d", "mean"),
).round(3).to_string())

# 손절확률 상위 20% 회피
threshold = weekly3["loser_prob"].quantile(0.80)
rf_excl = weekly3["loser_prob"] >= threshold
print(f"\nRF 손절확률 상위 20% 회피 (threshold={threshold:.3f}): {rf_excl.sum()}건 제외")
show(weekly3[~rf_excl], "RF 상위 20% 회피 후")

# 손절확률 상위 30% 회피
threshold2 = weekly3["loser_prob"].quantile(0.70)
rf_excl2 = weekly3["loser_prob"] >= threshold2
print(f"\nRF 손절확률 상위 30% 회피 (threshold={threshold2:.3f}): {rf_excl2.sum()}건 제외")
show(weekly3[~rf_excl2], "RF 상위 30% 회피 후")


# 6. 통합 최종 추천 - RF + AND
print("\n" + "="*100)
print("[6] 최종 통합: RF top 20% + AND조합 35%↑")
print("="*100)
final_excl = rf_excl | combined_excl
print(f"통합 회피: 제외 {final_excl.sum()}건 / 잔여 {(~final_excl).sum()}건")
final_pool = weekly3[~final_excl]
show(final_pool, "통합 회피 적용 (최종)")


# 저장
final_pool.to_csv(CACHE / "advanced_weekly_3_final.csv", index=False)
weekly3.to_csv(CACHE / "weekly_3_with_features.csv", index=False)
fi.to_csv(CACHE / "rf_feature_importance.csv", index=False)
print(f"\n[저장]")
print(f"  cache/advanced_weekly_3_final.csv ({len(final_pool):,}건)")
print(f"  cache/rf_feature_importance.csv")
print(f"  cache/weekly_3_with_features.csv (시계열 특성 포함)")
