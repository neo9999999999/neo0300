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
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")


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


def build_today_picks(top_n=20):
    """오늘 발생한 시그널 중 최종 추천."""
    feats, sd, cur = load_data()
    # 최신 영업일 (chart_feats 마지막 날짜)
    last_date = feats["Date"].max()
    today = feats[feats["Date"] == last_date].copy()
    print(f"[기준일] {last_date.date()} 시그널 {len(today)}건")

    filtered = apply_avoid_full(today, sd, cur)
    print(f"[필터후] {len(filtered)}건")

    # 동일 종목 중복 제거 (preset마다 발생 → 최고 점수 1개만)
    filtered = filtered.sort_values("Score", ascending=False).drop_duplicates("Code")

    # 거래대금 낮은 순 정렬 (소형주 농도 ↑)
    filtered = filtered.sort_values("Amount").head(top_n)

    # 출력 컬럼
    out_cols = ["Date", "Code", "Name", "Market", "Close", "Amount", "Score",
                "chart_pattern", "past_60", "past_120", "pos_252_high",
                "slope60", "drawdown60"]
    extra = [c for c in ["_for_20d", "_inst_20d", "_PER", "_PBR"] if c in filtered.columns]
    out_cols.extend(extra)
    out_cols = [c for c in out_cols if c in filtered.columns]

    picks = filtered[out_cols].copy()
    picks["기준일"] = picks["Date"].dt.strftime("%Y-%m-%d")
    picks.to_csv(CACHE / "today_picks.csv", index=False)

    # JSON (app.py용)
    picks_dict = picks.to_dict(orient="records")
    with open(CACHE / "today_picks.json", "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(),
            "base_date": last_date.strftime("%Y-%m-%d"),
            "n_picks": len(picks),
            "picks": picks_dict,
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"[저장] cache/today_picks.csv ({len(picks)}건)")
    return picks


if __name__ == "__main__":
    picks = build_today_picks(top_n=20)
    if len(picks):
        print("\n[오늘의 추천]")
        print(picks[["Code", "Name", "Close", "Amount", "Score"]].to_string(index=False))
