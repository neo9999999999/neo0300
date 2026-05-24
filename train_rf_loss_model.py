"""
RF 손절예측 모델 학습 + 저장
==========================
- 전체 시그널 풀(5,009건)로 학습
- target: ret_180d <= -20% (손절)
- 출력: cache/rf_loss_model.pkl (모델), cache/rf_features.json (특성 목록)
- 매일 daily_update.py에서 재학습 가능 (간단)
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

CACHE = Path("cache")

cand = pd.read_parquet(CACHE / "candidates_enriched.parquet")
cand["Date"] = pd.to_datetime(cand["Date"])
cand = cand.dropna(subset=["peak_180d", "sell_close", "ret_180d"]).copy()

with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


def add_pre_features(df):
    """시그널 발생 전 시계열 특성"""
    rows = {k: [] for k in [
        "pre_5d_max_high_ratio","pre_5d_min_low_ratio","pre_5d_vol_trend",
        "pre_10d_max_high_ratio","pre_10d_drawdown","pre_20d_vol_ratio",
        "gap_up_count_5d","long_red_count_5d","long_red_in_10d","consecutive_red_max",
    ]}
    for _, r in df.iterrows():
        code = r["Code"]; d0 = r["Date"]; c0 = r["Close"]
        if code not in OHLCV:
            for k in rows: rows[k].append(np.nan); continue
        past = OHLCV[code][OHLCV[code].index < d0].tail(30)
        if len(past) < 10:
            for k in rows: rows[k].append(np.nan); continue
        p5 = past.tail(5); p10 = past.tail(10); p20 = past.tail(20)
        p60 = past.tail(60) if len(past)>=60 else past
        rows["pre_5d_max_high_ratio"].append(p5["High"].max()/c0)
        rows["pre_5d_min_low_ratio"].append(p5["Low"].min()/c0)
        rows["pre_5d_vol_trend"].append(
            np.polyfit(range(len(p5)), p5["Volume"], 1)[0] / (p5["Volume"].mean()+1) if p5["Volume"].mean()>0 else 0)
        rows["pre_10d_max_high_ratio"].append(p10["High"].max()/c0)
        hi_max = p10["High"].max()
        rows["pre_10d_drawdown"].append((c0 - hi_max)/hi_max*100 if hi_max>0 else 0)
        rows["pre_20d_vol_ratio"].append(p20["Volume"].mean()/(p60["Volume"].mean()+1))
        opens = p5["Open"].values; closes_prev = p5["Close"].shift(1).values
        gap_up = ((opens[1:]/closes_prev[1:])>1.02).sum() if len(opens)>1 else 0
        rows["gap_up_count_5d"].append(gap_up)
        rows["long_red_count_5d"].append((p5["Close"]<p5["Open"]).sum())
        rows["long_red_in_10d"].append(((p10["Close"]/p10["Open"]-1)<=-0.03).sum())
        red_s = (p10["Close"]<p10["Open"]).astype(int).values
        ms = 0; cur = 0
        for v in red_s:
            if v: cur+=1; ms=max(ms,cur)
            else: cur=0
        rows["consecutive_red_max"].append(ms)
    for k,v in rows.items(): df[k] = v
    return df


FEATURES = [
    "Score","Amount","vol_ratio","candle_pct","cum_5d_gain","rs_ratio",
    "ma3","ma5","ma10","pos_60_high","pos_120_high","pos_240_high","pos_252_high",
    "past_5d","past_20","past_60","past_120","past_240",
    "slope60","slope120","range60_pct","range120_pct","drawdown60","runup60",
    "vol20","vol60","days_since_52w_low","days_since_52w_high",
    "For_5d","Inst_5d","For_20d","Inst_20d","PER_num","PBR_num","외인소진율_num",
    "pre_5d_max_high_ratio","pre_5d_min_low_ratio","pre_5d_vol_trend",
    "pre_10d_max_high_ratio","pre_10d_drawdown","pre_20d_vol_ratio",
    "gap_up_count_5d","long_red_count_5d","long_red_in_10d","consecutive_red_max",
]


def prepare_X(df, features=None):
    """모델 학습/예측용 X 생성 (NaN/inf 처리)."""
    if features is None:
        features = [f for f in FEATURES if f in df.columns]
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    for f in features:
        if X[f].dtype.kind in "fi":
            try:
                q01, q99 = X[f].quantile(0.001), X[f].quantile(0.999)
                X[f] = X[f].clip(q01, q99)
            except Exception:
                pass
    return X, features


def main():
    print("[1] 시계열 특성 추가...")
    cand2 = add_pre_features(cand)
    cand2["is_loser"] = (cand2["ret_180d"] <= -20).astype(int)

    print("[2] X 준비...")
    X, features = prepare_X(cand2)
    print(f"   특성 {len(features)}개, 샘플 {len(X):,}")

    print("[3] RF 학습 (n=200, max_depth=6)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, cand2["is_loser"])

    # 임계값 (상위 20%, 30%) 산출
    probs = rf.predict_proba(X)[:, 1]
    th20 = float(np.quantile(probs, 0.80))
    th30 = float(np.quantile(probs, 0.70))
    print(f"   임계값: th20={th20:.4f}, th30={th30:.4f}")

    # 저장
    model_path = CACHE / "rf_loss_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    meta = {
        "features": features,
        "th20": th20,
        "th30": th30,
        "n_samples": int(len(X)),
        "n_loser": int(cand2["is_loser"].sum()),
        "auc_in_sample": None,
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    # 변수 중요도
    fi = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    meta["top_features"] = fi.head(10).to_dict(orient="records")
    with open(CACHE / "rf_features.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    fi.to_csv(CACHE / "rf_feature_importance.csv", index=False)

    print(f"\n✓ {model_path}")
    print(f"✓ cache/rf_features.json")
    print(f"\nTop 5 변수 중요도:")
    print(fi.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
