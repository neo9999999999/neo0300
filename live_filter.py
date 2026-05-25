"""
실시간 필터링 - 시총 300 + SuperScore 통합 (최종판)
=================================================
매일 16:30 KST 갱신:
1. chart_feats 시그널 풀에서 시총 300 필터
2. 회피 6 적용
3. RF 4분류기 + peak 회귀로 확률 계산
4. SuperScore = p_sw×5 + p_100×2 + p_50×1 - p_loss×3
5. 등급 부여 (★ 강력매수 / ○ 추천 / - 관망 / ⚠️ 손절위험)
6. SuperScore TOP 5 + 가능성 태그 + 예상수익률

산출: cache/today_picks.csv, week_picks.csv, month_picks.csv
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")


def load_data():
    # 신규 시그널 풀 우선 (signals_2000_enriched.parquet, 2021-04 ~ 2026-05)
    p_new = CACHE / "signals_2000_enriched.parquet"
    p_old = CACHE / "chart_feats_v1.parquet"
    if p_new.exists():
        feats = pd.read_parquet(p_new)
    else:
        feats = pd.read_parquet(p_old)
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats = feats[feats["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
    sd_path = CACHE / "supply_demand.parquet"
    sd = pd.read_parquet(sd_path) if sd_path.exists() else pd.DataFrame()
    if not sd.empty:
        sd["Date"] = pd.to_datetime(sd["Date"])
    cur_path = CACHE / "fundamentals_current.parquet"
    cur = pd.read_parquet(cur_path) if cur_path.exists() else pd.DataFrame()
    snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
    return feats, sd, cur, snap


def apply_avoid_6(df):
    """회피 6 (기본, 누락 컬럼 안전 처리)"""
    d = df.copy()
    mask = pd.Series(False, index=d.index)
    if "chart_pattern" in d.columns and "slope60" in d.columns and "pos_252_high" in d.columns:
        mask |= (d["chart_pattern"] == "pullback_recovery") & (d["slope60"] <= -1) & (d["pos_252_high"] <= -40)
    if "past_120" in d.columns and "pos_252_high" in d.columns:
        mask |= (d["Market"] == "KOSPI") & (d["past_120"] <= -20) & (d["pos_252_high"] <= -40)
    if "s12" in d.columns and "new_high_252" in d.columns and "past_120" in d.columns:
        mask |= (d["s12"] >= 80) & (d["new_high_252"] == 1) & (d["past_120"] >= 50)
    if "past_240" in d.columns:
        mask |= d["past_240"] >= 100
        mask |= d["past_240"] >= 150
    if "Amount" in d.columns:
        mask |= d["Amount"] >= 3000e8
    return d[~mask.fillna(False)].copy()


def filter_top300(df, snap):
    top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
    return df[df["Code"].isin(top300)].copy()


def load_strong_buy_models():
    p = CACHE / "strong_buy_models.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def add_pre_features_one(df, ohlcv_dict):
    """라이브 시그널에 시계열 특성 추가."""
    rows = {k: [] for k in [
        "pre_5d_max_high_ratio","pre_5d_min_low_ratio","pre_5d_vol_trend",
        "pre_10d_max_high_ratio","pre_10d_drawdown","pre_20d_vol_ratio",
        "gap_up_count_5d","long_red_count_5d","long_red_in_10d","consecutive_red_max",
    ]}

    def nans():
        for k in rows: rows[k].append(np.nan)

    for _, r in df.iterrows():
        code = str(r["Code"])
        d0 = r["Date"]; c0 = r["Close"]
        if code not in ohlcv_dict:
            code_pad = code.zfill(6) if code.isdigit() else code
            if code_pad not in ohlcv_dict:
                nans(); continue
            code = code_pad
        bars = ohlcv_dict[code]
        past = bars[bars.index < d0].tail(30)
        if len(past) < 10:
            nans(); continue
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


def compute_supabilities(df, sd, cur):
    """SuperScore + 4 RF 확률 + 등급 부여"""
    sb = load_strong_buy_models()
    if sb is None:
        print("[경고] strong_buy_models.pkl 없음")
        df["StrongScore"] = np.nan
        df["SuperScore"] = np.nan
        df["등급"] = "- 미정"
        df["가능성태그"] = ""
        df["예상수익률"] = ""
        return df

    # OHLCV 로드 + 시계열 특성
    ohlcv_path = sorted(CACHE.glob("ohlcv_*.pkl"))[-1]
    with open(ohlcv_path, "rb") as f:
        ohlcv_dict = pickle.load(f)
    df = add_pre_features_one(df, ohlcv_dict)

    # 수급 매칭
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

    # 펀더멘털
    if cur is not None and not cur.empty:
        cur_idx = cur.set_index("Code")[["PER_num","PBR_num","외인소진율_num"]]
        for c in ["PER_num","PBR_num","외인소진율_num"]:
            df[c] = df["Code"].map(cur_idx[c])

    # RF 예측 (4 분류기 + peak 회귀)
    features = sb["features"]
    X = df.reindex(columns=features).copy().replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    models = sb["models"]

    df["p_loss"] = models["loss"].predict_proba(X)[:, 1]
    df["p_sw"] = models["sw"].predict_proba(X)[:, 1]
    df["p_100plus"] = models["100plus"].predict_proba(X)[:, 1]
    df["p_50plus"] = models["50plus"].predict_proba(X)[:, 1]
    df["예상peak%"] = models["peak_reg"].predict(X).round(1)

    # 확률 % 표시
    df["슈퍼위너확률%"] = (df["p_sw"]*100).round(1)
    df["100%+확률"] = (df["p_100plus"]*100).round(1)
    df["50%+확률"] = (df["p_50plus"]*100).round(1)
    df["손절확률%"] = (df["p_loss"]*100).round(1)

    # 두 점수 (StrongScore + SuperScore)
    df["StrongScore"] = (df["p_sw"]*3 + df["p_100plus"]*1.5 + df["p_50plus"]*1 - df["p_loss"]*2).round(2)
    df["SuperScore"] = (df["p_sw"]*5 + df["p_100plus"]*2 + df["p_50plus"]*1 - df["p_loss"]*3).round(2)

    # 등급 부여 (당일 내 StrongScore 분위)
    df["_score_pct"] = df.groupby(df["Date"].dt.strftime("%Y-%m-%d"))["StrongScore"].rank(pct=True)
    grades = []
    for _, r in df.iterrows():
        if r["p_loss"] >= 0.55:
            grades.append("⚠️ 손절위험")
        elif r["_score_pct"] >= 0.80:
            grades.append("★ 강력매수")
        elif r["_score_pct"] >= 0.60:
            grades.append("○ 추천")
        else:
            grades.append("- 관망")
    df["등급"] = grades

    # 가능성 태그
    tags = []
    for _, r in df.iterrows():
        t = []
        if r["p_sw"] >= 0.20: t.append("🏆 슈퍼위너 강력후보")
        elif r["p_sw"] >= 0.10: t.append("⭐ 슈퍼위너후보")
        if r["p_100plus"] >= 0.30: t.append("💯 100%+ 가능")
        if r["p_50plus"] >= 0.50: t.append("📈 50%+ 가능")
        if 0.40 <= r["p_loss"] < 0.55: t.append("🔻 손절 주의")
        tags.append(" / ".join(t) if t else "")
    df["가능성태그"] = tags

    # 예상 수익률 카테고리
    cats = []
    for v in df["예상peak%"]:
        if v >= 100: cats.append(f"+{v:.0f}% (대박)")
        elif v >= 50: cats.append(f"+{v:.0f}% (대상승)")
        elif v >= 20: cats.append(f"+{v:.0f}% (상승)")
        elif v >= 0: cats.append(f"+{v:.0f}% (보합)")
        else: cats.append(f"{v:.0f}% (약세)")
    df["예상수익률"] = cats

    return df


def build_picks(top_n_today=5, top_n_week=5):
    """오늘/이번주/이번달 추천 빌드 - SuperScore 기반"""
    feats, sd, cur, snap = load_data()
    last_date = feats["Date"].max()

    # 시총 300 필터
    feats = filter_top300(feats, snap)
    print(f"[시총 300 필터] {len(feats):,}건")

    # 회피 6
    feats = apply_avoid_6(feats)
    print(f"[회피 6 적용] {len(feats):,}건")

    # 기간별 시그널 추출
    today = feats[feats["Date"] == last_date].copy()
    week_start = last_date - pd.Timedelta(days=last_date.weekday())
    month_start = last_date.replace(day=1)
    this_week = feats[(feats["Date"] >= week_start) & (feats["Date"] <= last_date)].copy()
    this_month = feats[(feats["Date"] >= month_start) & (feats["Date"] <= last_date)].copy()

    print(f"[기준일 {last_date.date()}] 오늘 {len(today)}, 이번주 {len(this_week)}, 이번달 {len(this_month)}")

    # SuperScore 계산
    if len(this_month) > 0:
        this_month = compute_supabilities(this_month, sd, cur)
        # 중복 제거 (Date+Code)
        this_month = this_month.sort_values("SuperScore", ascending=False).drop_duplicates(["Date","Code"])

    # 오늘 (그날 시그널 전부 + 정렬)
    today_picks = this_month[this_month["Date"] == last_date].copy()
    today_picks = today_picks.sort_values("SuperScore", ascending=False)

    # 이번 주 TOP (월~금 누적 - 모두 표시)
    week_picks = this_month[(this_month["Date"] >= week_start)].copy()
    week_picks = week_picks.sort_values("SuperScore", ascending=False)

    # 지난 주 (월~금)
    last_week_start = week_start - pd.Timedelta(days=7)
    last_week_end = week_start - pd.Timedelta(days=1)
    # 지난주 시그널은 별도 로드 필요할 수 있음 - 같은 풀 안에서 추출
    last_week_pool = feats[(feats["Date"] >= last_week_start) & (feats["Date"] <= last_week_end)].copy()
    if len(last_week_pool) > 0:
        last_week_pool = compute_supabilities(last_week_pool, sd, cur)
        last_week_pool = last_week_pool.sort_values("SuperScore", ascending=False).drop_duplicates(["Date","Code"])
    else:
        last_week_pool = pd.DataFrame()

    # 이번 달 TOP 20
    month_picks = this_month.sort_values("SuperScore", ascending=False).head(20)

    # 출력 컬럼
    cols = ["Date","등급","가능성태그","예상수익률","Code","Name","Market","Close","Amount",
            "SuperScore","StrongScore","예상peak%",
            "슈퍼위너확률%","100%+확률","50%+확률","손절확률%",
            "Score","chart_pattern","past_60","past_120","pos_252_high","slope60"]

    def save_picks(df, fname):
        if len(df) == 0:
            pd.DataFrame().to_csv(CACHE / fname, index=False); return
        avail = [c for c in cols if c in df.columns]
        out = df[avail].copy()
        out["기준일"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(CACHE / fname, index=False)

    save_picks(today_picks, "today_picks.csv")
    save_picks(week_picks, "week_picks.csv")
    save_picks(last_week_pool, "last_week_picks.csv")
    save_picks(month_picks, "month_picks.csv")

    # JSON
    summary = {
        "updated_at": datetime.now().isoformat(),
        "base_date": last_date.strftime("%Y-%m-%d"),
        "week_start": week_start.strftime("%Y-%m-%d"),
        "month_start": month_start.strftime("%Y-%m-%d"),
        "today": {"n": len(today_picks), "picks": today_picks.head(50).to_dict(orient="records")},
        "week": {"n": len(week_picks), "picks": week_picks.head(50).to_dict(orient="records")},
        "last_week": {"n": len(last_week_pool), "picks": last_week_pool.head(50).to_dict(orient="records") if len(last_week_pool)>0 else []},
        "month": {"n": len(month_picks), "picks": month_picks.to_dict(orient="records")},
    }
    with open(CACHE / "today_picks.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 콘솔 출력
    print(f"\n[오늘 TOP {len(today_picks)}건]")
    if len(today_picks) > 0:
        print(today_picks[["Code","Name","SuperScore","등급","예상peak%","가능성태그"]].to_string(index=False))

    print(f"\n[이번 주 TOP {len(week_picks)}건]")
    if len(week_picks) > 0:
        print(week_picks[["Date","Code","Name","SuperScore","등급","예상peak%"]].to_string(index=False))

    return today_picks, week_picks, month_picks


if __name__ == "__main__":
    build_picks(top_n_today=5, top_n_week=5)
