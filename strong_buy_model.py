"""
필수 매수 (Strong Buy) 모델 — 시총 2000 학습 기반
==========================================
3가지 모델 학습:
1. RF 손절예측 (ret_180d ≤ -20%) — 낮을수록 좋음
2. RF 슈퍼위너 예측 (peak_180d ≥ 200%) — 높을수록 좋음
3. RF 50%+ 예측 (peak_180d ≥ 50%) — 높을수록 좋음
4. 회귀 모델로 peak_180d 예측 — 예상 수익률

종합 점수 = SW확률 × 3 + 50+확률 × 1 - 손절확률 × 2

walk-forward로 각 모델 학습.
산출:
- cache/strong_buy_models.pkl (3 RF + 1 regressor)
- cache/strong_buy_meta.json
"""

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error

CACHE = Path("cache")
from train_rf_loss_model import FEATURES, prepare_X


def train_models(train_df, label="2000"):
    """3 classifier + 1 regressor 학습"""
    available = [f for f in FEATURES if f in train_df.columns]
    X, _ = prepare_X(train_df, features=available)

    targets = {
        "loss": (train_df["ret_180d"] <= -20).astype(int),
        "sw": (train_df["peak_180d"] >= 200).astype(int),
        "100plus": (train_df["peak_180d"] >= 100).astype(int),
        "50plus": (train_df["peak_180d"] >= 50).astype(int),
    }

    models = {}
    for tgt_name, y in targets.items():
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X, y)
        models[tgt_name] = clf
        print(f"  [{tgt_name}] positive: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    # 회귀: peak_180d 예측 (큰 값 cap)
    y_reg = np.clip(train_df["peak_180d"].values, -50, 500)
    reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=15,
                                  random_state=42, n_jobs=-1)
    reg.fit(X, y_reg)
    models["peak_reg"] = reg

    return models, available


def predict_all(test_df, models, features):
    X, _ = prepare_X(test_df, features=features)
    probs = {}
    for name in ["loss", "sw", "100plus", "50plus"]:
        probs[f"p_{name}"] = models[name].predict_proba(X)[:, 1]
    probs["peak_pred"] = models["peak_reg"].predict(X)
    for k, v in probs.items():
        test_df[k] = v
    # 종합 점수 = SW × 3 + 50+ × 1 - 손절 × 2
    test_df["StrongScore"] = (
        test_df["p_sw"] * 3.0
        + test_df["p_50plus"] * 1.0
        + test_df["p_100plus"] * 1.5
        - test_df["p_loss"] * 2.0
    )
    return test_df


def main():
    # 시총 2000 풀 사용 (signals_2000_enriched.parquet)
    p = CACHE / "signals_2000_enriched.parquet"
    if not p.exists():
        # fallback to existing
        p = CACHE / "candidates_enriched_full.parquet"
        print(f"[경고] signals_2000_enriched 없음. fallback: {p}")
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["peak_180d", "ret_180d"]).copy()
    df["Year"] = df["Date"].dt.year
    print(f"[전체] {len(df):,}건")

    # Walk-forward 학습 (각 forward year)
    windows = [
        {"tr": (2020,2022), "te": 2023},
        {"tr": (2020,2023), "te": 2024},
        {"tr": (2020,2024), "te": 2025},
        {"tr": (2020,2025), "te": 2026},
    ]
    all_test = []
    for w in windows:
        train = df[(df["Year"]>=w["tr"][0])&(df["Year"]<=w["tr"][1])].copy()
        test = df[df["Year"]==w["te"]].copy()
        if len(train)<500 or len(test)<50: continue
        print(f"\n=== {w['tr'][0]}-{w['tr'][1]} → {w['te']} ===")
        print(f"  Train: {len(train):,}, Test: {len(test):,}")
        models, features = train_models(train)
        test = predict_all(test, models, features)
        test["window"] = f"{w['tr'][0]}-{w['tr'][1]}→{w['te']}"
        all_test.append(test)

    # 최종 모델: 전체 데이터로 학습 (라이브용)
    print("\n=== 최종 모델 (전체 학습) ===")
    final_models, final_features = train_models(df)

    # 모델 저장
    with open(CACHE / "strong_buy_models.pkl", "wb") as f:
        pickle.dump({
            "models": final_models,
            "features": final_features,
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_samples": len(df),
        }, f)
    print(f"  cache/strong_buy_models.pkl 저장")

    # 메타
    meta = {
        "features": final_features,
        "n_samples": len(df),
        "trained_at": pd.Timestamp.now().isoformat(),
        "targets": ["loss(≤-20%)", "sw(≥200%)", "100plus(≥100%)", "50plus(≥50%)", "peak_reg"],
        "score_formula": "p_sw*3 + p_50plus*1 + p_100plus*1.5 - p_loss*2",
    }
    with open(CACHE / "strong_buy_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  cache/strong_buy_meta.json")

    # walk-forward 결과 합본
    if all_test:
        combined = pd.concat(all_test, ignore_index=True)
        combined.to_parquet(CACHE / "strong_buy_wf_test.parquet", index=False)

        # 매수 시뮬: 매일 거래대금 낮은 3건 → StrongScore 1등만 "필수 매수"
        print("\n" + "="*100)
        print("필수 매수 시뮬레이션 (walk-forward OOS)")
        print("="*100)

        combined["bucket_day"] = combined["Date"].dt.strftime("%Y-%m-%d")
        # 매일 거래대금↓ 3건
        top3 = combined.sort_values("Amount").groupby("bucket_day").head(3).copy()
        # StrongScore 가장 높은 1건 = 필수 매수
        must_buy = top3.sort_values("StrongScore", ascending=False).groupby("bucket_day").head(1).copy()

        for label, picks in [("매일 3건 (거래대금↓)", top3), ("필수 매수 1건 (StrongScore↑)", must_buy)]:
            n = len(picks)
            sw = (picks["peak_180d"]>=200).sum()
            w100 = (picks["peak_180d"]>=100).sum()
            w50 = (picks["peak_180d"]>=50).sum()
            loser = (picks["ret_180d"]<=-20).sum()
            invest = n * 10
            profit = ((picks["sell_close"]/picks["Close"] - 1) * 10).sum()
            print(f"\n  [{label}] {n}건")
            print(f"    슈퍼위너: {sw} ({sw/n*100:.1f}%)")
            print(f"    100%+: {w100} ({w100/n*100:.1f}%)")
            print(f"    50%+: {w50} ({w50/n*100:.1f}%)")
            print(f"    손절: {loser} ({loser/n*100:.1f}%)")
            print(f"    투자 {invest:,.0f}만 → 수익 {profit:+,.0f}만 ({profit/invest*100:+.1f}%)")
            print(f"    예상 peak (모델): {picks['peak_pred'].mean():.1f}% / 실제: {picks['peak_180d'].mean():.1f}%")

        # MAE 측정
        mae = (combined["peak_pred"] - combined["peak_180d"]).abs().mean()
        print(f"\n  peak 예측 MAE: {mae:.1f}%")

    print("\n[완료]")


if __name__ == "__main__":
    main()
