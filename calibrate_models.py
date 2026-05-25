"""
모델 확률 보정 (Calibrated Classifier)
====================================
RF 출력을 isotonic regression으로 보정 → 실제 비율에 가깝게.
walk-forward로 학습 + 보정.
"""

import warnings
warnings.filterwarnings("ignore")
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from train_rf_loss_model import FEATURES, prepare_X

CACHE = Path("cache")

# 시그널 풀 (시총 300)
sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d","ret_180d","Amount"]).copy()
snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
pool = sigs[sigs["Code"].isin(top300)].copy()
print(f"풀: {len(pool):,}건")

available = [f for f in FEATURES if f in pool.columns]
X, _ = prepare_X(pool, features=available)

targets = {
    "loss": (pool["ret_180d"]<=-20).astype(int),
    "sw": (pool["peak_180d"]>=200).astype(int),
    "100plus": (pool["peak_180d"]>=100).astype(int),
    "50plus": (pool["peak_180d"]>=50).astype(int),
}

print("\n[Calibrated 학습 (isotonic, 5-fold)]")
models = {}
for name, y in targets.items():
    pos_rate = y.mean() * 100
    base = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                    class_weight=None, random_state=42, n_jobs=-1)
    # class_weight=None: balanced 안 함 (확률 over-confidence 줄임)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5, n_jobs=-1)
    clf.fit(X, y)
    models[name] = clf
    probs = clf.predict_proba(X)[:, 1]
    print(f"  [{name}] 실제 비율 {pos_rate:.1f}% / 평균 예측 {probs.mean()*100:.1f}%")

# 회귀 (그대로)
reg = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=15,
                              random_state=42, n_jobs=-1)
reg.fit(X, np.clip(pool["peak_180d"].values, -50, 500))
models["peak_reg"] = reg

# 저장
with open(CACHE / "strong_buy_models.pkl", "wb") as f:
    pickle.dump({
        "models": models,
        "features": available,
        "trained_at": pd.Timestamp.now().isoformat(),
        "calibrated": True,
        "method": "isotonic",
    }, f)
print(f"\n[저장] cache/strong_buy_models.pkl (calibrated)")

meta = {
    "features": available,
    "n_samples": len(pool),
    "trained_at": pd.Timestamp.now().isoformat(),
    "calibrated": True,
}
with open(CACHE / "strong_buy_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("[저장] cache/strong_buy_meta.json")
