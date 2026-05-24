"""
실시간 필터링 모듈 - 매일 자동 갱신
=================================

매일 16:30 KST 갱신 흐름:
1. daily_update.py → KRX/KIS로 OHLCV 캐시 갱신
2. precompute_enriched.py → 4 프리셋 chart_feats 재계산
3. collect_supply_demand_daily.py → 신규 종목 수급 추가
4. collect_fundamentals_daily.py → 현재 시점 PER/PBR 갱신
5. live_filter.py → 회피 6+v2 적용 + V/S/A/B 등급 → 추천 리스트

추천 리스트 출력:
- cache/today_picks.csv   - 오늘의 추천
- cache/today_picks.json  - app.py에서 로드용
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")


def load_rf_model():
    """RF 손절예측 모델 로드. 없으면 None."""
    model_path = CACHE / "rf_loss_model.pkl"
    meta_path = CACHE / "rf_features.json"
    if not (model_path.exists() and meta_path.exists()):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def add_pre_features_one(df, ohlcv_dict):
    """라이브 시그널에 시계열 특성 추가."""
    rows = {k: [] for k in [
        "pre_5d_max_high_ratio","pre_5d_min_low_ratio","pre_5d_vol_trend",
        "pre_10d_max_high_ratio","pre_10d_drawdown","pre_20d_vol_ratio",
        "gap_up_count_5d","long_red_count_5d","long_red_in_10d","consecutive_red_max",
    ]}
    def append_nans():
        for k in rows:
            rows[k].append(np.nan)

    for _, r in df.iterrows():
        code = str(r["Code"])
        d0 = r["Date"]; c0 = r["Close"]
        if code not in ohlcv_dict:
            code_padded = code.zfill(6) if code.isdigit() else code
            if code_padded not in ohlcv_dict:
                append_nans()
                continue
            code = code_padded
        bars = ohlcv_dict[code]
        past = bars[bars.index < d0].tail(30)
        if len(past) < 10:
            append_nans()
            continue
        p5 = past.tail(5); p10 = past.tail(10); p20 = past.tail(20)
        p60 = past.tail(60) if len(past)>=60 else past
        rows["pre_5d_max_high_ratio"].append(p5["High"].max()/c0)
        rows["pre_5d_min_low_ratio"].append(p5["Low"].min()/c0)
        rows["pre_5d_vol_trend"].append(
            np.polyfit(range(len(p5)), p5["Volume"], 1)[0]/(p5["Volume"].mean()+1) if p5["Volume"].mean()>0 else 0)
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


def apply_rf_filter(df, sd=None, cur=None, threshold_key="th20"):
    """RF 손절예측 모델 적용 — 손절확률 ≥ threshold면 제외."""
    model, meta = load_rf_model()
    if model is None:
        print("[RF] 모델 없음 — skip")
        df["_RF손절확률"] = np.nan
        df["_RF위험"] = 0
        return df

    # OHLCV 로드 후 시계열 특성 추가
    ohlcv_path = sorted(CACHE.glob("ohlcv_*.pkl"))[-1]
    with open(ohlcv_path, "rb") as f:
        ohlcv_dict = pickle.load(f)
    df = add_pre_features_one(df, ohlcv_dict)

    # 외인20일 / 기관20일 매칭
    if sd is not None and not sd.empty:
        sd_dict = {}
        for code in sd["Code"].unique():
            sub = sd[sd["Code"] == code].sort_values("Date")
            sd_dict[code] = sub.set_index("Date")[["Foreign_NetBuy","Inst_NetBuy"]]
        for col, days in [("For_5d", 5), ("Inst_5d", 5), ("For_20d", 20), ("Inst_20d", 20)]:
            vals = []
            for code, dt in zip(df["Code"], df["Date"]):
                if code in sd_dict:
                    sub = sd_dict[code]
                    past = sub[sub.index <= dt].tail(days)
                    src_col = "Foreign_NetBuy" if col.startswith("For") else "Inst_NetBuy"
                    vals.append(past[src_col].sum() if len(past) else np.nan)
                else:
                    vals.append(np.nan)
            df[col] = vals

    # 펀더멘털 매칭
    if cur is not None and not cur.empty:
        cur_idx = cur.set_index("Code")[["PER_num","PBR_num","외인소진율_num"]]
        for c in ["PER_num","PBR_num","외인소진율_num"]:
            df[c] = df["Code"].map(cur_idx[c])

    # 학습 시 사용된 특성 추출
    features = meta["features"]
    X = df.reindex(columns=features).copy().replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    # 학습 임계값
    th = meta.get(threshold_key, 0.53)
    probs = model.predict_proba(X)[:, 1]
    df["_RF손절확률"] = probs
    df["_RF위험"] = (probs >= th).astype(int)
    return df


def load_data():
    """기본 데이터 로드."""
    feats = pd.read_parquet(CACHE / "chart_feats_v1.parquet")
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats = feats[feats["Market"].isin(["KOSPI", "KOSDAQ"])].copy()

    sd_path = CACHE / "supply_demand.parquet"
    sd = pd.read_parquet(sd_path) if sd_path.exists() else pd.DataFrame()
    if not sd.empty:
        sd["Date"] = pd.to_datetime(sd["Date"])

    cur_path = CACHE / "fundamentals_current.parquet"
    cur = pd.read_parquet(cur_path) if cur_path.exists() else pd.DataFrame()
    return feats, sd, cur


def apply_avoid_full(df, sd=None, cur=None, mode="v2"):
    """회피 6 + 분석 결과 기반 보강

    mode:
      v2 - 회피6 + X7(외인20일 하위10%) + X13(강하락+52주高-50%↓) (SW 손실 최소)
      v3 - v2 + 강화 P9(외인+기관 동시 순매수) — 100%+/50%+ 농도 ↑
      v4 - v2 + 강화 P2(slope60≥0.5 상승추세) — SW 농도 ↑
    """
    d = df.copy()

    # 회피 6개 (기존)
    x1 = (d["chart_pattern"] == "pullback_recovery") & (d["slope60"] <= -1) & (d["pos_252_high"] <= -40)
    x2 = (d["Market"] == "KOSPI") & (d["past_120"] <= -20) & (d["pos_252_high"] <= -40)
    x3 = (d["s12"] >= 80) & (d["new_high_252"] == 1) & (d["past_120"] >= 50)
    x4 = d["past_240"] >= 100
    x5 = d["past_240"] >= 150
    x6 = d["Amount"] >= 3000e8
    mask = x1 | x2 | x3 | x4 | x5 | x6

    # X13: 강하락추세+52주高-50%↓
    mask |= (d["slope60"] <= -2) & (d["pos_252_high"] <= -50)

    # 수급 매칭
    if sd is not None and not sd.empty:
        sd_dict = {}
        for code in sd["Code"].unique():
            sub = sd[sd["Code"] == code].sort_values("Date")
            sd_dict[code] = sub.set_index("Date")[["Foreign_NetBuy", "Inst_NetBuy"]]
        for_20, inst_20 = [], []
        for code, dt in zip(d["Code"], d["Date"]):
            if code in sd_dict:
                sub = sd_dict[code]
                past20 = sub[sub.index <= dt].tail(20)
                for_20.append(past20["Foreign_NetBuy"].sum() if len(past20) else np.nan)
                inst_20.append(past20["Inst_NetBuy"].sum() if len(past20) else np.nan)
            else:
                for_20.append(np.nan); inst_20.append(np.nan)
        d["_for_20d"] = for_20
        d["_inst_20d"] = inst_20
        # X7: 외인 20일 누적 하위 10%
        valid = d["_for_20d"].notna()
        if valid.sum() > 100:
            q10 = d.loc[valid, "_for_20d"].quantile(0.10)
            mask |= (d["_for_20d"] < q10).fillna(False)

    # 펀더멘털
    if cur is not None and not cur.empty:
        cur_idx = cur.set_index("Code")[["PER_num", "PBR_num", "시총_num"]]
        d["_PER"] = d["Code"].map(cur_idx["PER_num"])
        d["_PBR"] = d["Code"].map(cur_idx["PBR_num"])
        d["_시총"] = d["Code"].map(cur_idx["시총_num"])

    out = d[~mask].copy()

    # 강화(포함) 룰
    if mode == "v3":
        # P9: 외인+기관 동시 20일 누적 순매수
        if "_for_20d" in out.columns:
            out = out[(out["_for_20d"] > 0) & (out["_inst_20d"] > 0)]
    elif mode == "v4":
        # P2: slope60 ≥ 0.5
        out = out[out["slope60"] >= 0.5]

    return out


def build_today_picks(top_n=20, use_rf=True, rf_threshold="th20"):
    """오늘 발생한 시그널 중 최종 추천.

    파이프라인:
      1. chart_feats 마지막 일자 시그널 로드
      2. apply_avoid_full() - 회피 8개 + (X7 외인매도 + X13 강하락)
      3. RF 손절예측 모델로 위험 종목 제외 (rf_threshold='th20' 상위 20%)
      4. 동일 종목 중복 제거 + 거래대금 낮은 순 TOP N
    """
    feats, sd, cur = load_data()
    last_date = feats["Date"].max()
    today = feats[feats["Date"] == last_date].copy()
    print(f"[기준일] {last_date.date()} 시그널 {len(today)}건")

    # 1) 회피 룰 적용
    filtered = apply_avoid_full(today, sd, cur)
    print(f"[회피 8 적용 후] {len(filtered)}건")

    # 2) RF 손절예측
    if use_rf and len(filtered) > 0:
        filtered = apply_rf_filter(filtered, sd, cur, threshold_key=rf_threshold)
        n_before = len(filtered)
        filtered_safe = filtered[filtered["_RF위험"] == 0].copy()
        print(f"[RF 손절회피 ({rf_threshold}) 적용 후] {n_before} → {len(filtered_safe)}건 "
              f"(위험 {(filtered['_RF위험']==1).sum()}건 제외)")
        filtered = filtered_safe

    if len(filtered) == 0:
        print("[알림] 오늘은 추천 종목 없음")
        # 빈 결과 저장
        with open(CACHE / "today_picks.json", "w", encoding="utf-8") as f:
            json.dump({"updated_at": datetime.now().isoformat(),
                       "base_date": last_date.strftime("%Y-%m-%d"),
                       "n_picks": 0, "picks": []}, f, ensure_ascii=False, indent=2)
        pd.DataFrame().to_csv(CACHE / "today_picks.csv", index=False)
        return pd.DataFrame()

    # 3) 동일 종목 중복 제거 + 거래대금 낮은 순
    filtered = filtered.sort_values("Score", ascending=False).drop_duplicates("Code")
    filtered = filtered.sort_values("Amount").head(top_n)

    out_cols = ["Date", "Code", "Name", "Market", "Close", "Amount", "Score",
                "chart_pattern", "past_60", "past_120", "pos_252_high",
                "slope60", "drawdown60"]
    extra = [c for c in ["_for_20d", "_inst_20d", "_PER", "_PBR",
                         "_RF손절확률", "_RF위험"] if c in filtered.columns]
    out_cols.extend(extra)
    out_cols = [c for c in out_cols if c in filtered.columns]

    picks = filtered[out_cols].copy()
    picks["기준일"] = picks["Date"].dt.strftime("%Y-%m-%d")
    picks.to_csv(CACHE / "today_picks.csv", index=False)

    picks_dict = picks.to_dict(orient="records")
    with open(CACHE / "today_picks.json", "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(),
            "base_date": last_date.strftime("%Y-%m-%d"),
            "n_picks": len(picks),
            "use_rf": use_rf,
            "rf_threshold": rf_threshold,
            "picks": picks_dict,
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"[저장] cache/today_picks.csv ({len(picks)}건)")
    return picks


if __name__ == "__main__":
    picks = build_today_picks(top_n=20)
    if len(picks):
        print("\n[오늘의 추천]")
        print(picks[["Code", "Name", "Close", "Amount", "Score"]].to_string(index=False))
