"""
모든 매수 종목 리스트업 (최신순, 2020-04 ~ 2025-08)
==================================================
- 매일1건 모드 매수 종목 전체
- 주3건 모드 매수 종목 전체
- RF 손절회피 적용 + 미적용 둘 다
- 2025년 가장 최신 → 2020년 순
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

CACHE = Path("cache")

cand = pd.read_parquet(CACHE / "candidates_enriched.parquet")
cand["Date"] = pd.to_datetime(cand["Date"])
cand = cand.dropna(subset=["peak_180d", "sell_close", "ret_180d"]).copy()

with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


# ===== 시계열 특성 =====
def add_pre_features(df):
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


print("[1/3] 시계열 특성 추가...")
cand = add_pre_features(cand)
cand["is_loser"] = (cand["ret_180d"] <= -20).astype(int)
cand["is_sw"] = (cand["peak_180d"] >= 200).astype(int)

# ===== RF 손절예측 =====
features = [
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
features = [f for f in features if f in cand.columns]
X = cand[features].copy().replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
for f in features:
    if X[f].dtype.kind in 'fi':
        q01, q99 = X[f].quantile(0.001), X[f].quantile(0.999)
        X[f] = X[f].clip(q01, q99)

print("[2/3] RF 손절확률 학습 + 예측...")
rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X, cand["is_loser"])
cand["loser_prob"] = rf.predict_proba(X)[:, 1]
# RF 상위 20% = 위험
rf_th20 = cand["loser_prob"].quantile(0.80)
cand["RF_위험"] = (cand["loser_prob"] >= rf_th20).astype(int)
print(f"   RF 위험 임계값: {rf_th20:.3f} (상위 20%)")

# ===== 매수 종목 추출 =====
print("[3/3] 매수 종목 추출...")
# 매일1건 (회피6 적용된 풀, 거래대금↓ 1건/일)
daily = cand.sort_values("Amount").copy()
daily["bucket"] = daily["Date"].dt.strftime("%Y-%m-%d")
daily_picks = daily.groupby("bucket").head(1).copy()
daily_picks["모드"] = "매일1건"

# 주3건 (KOSDAQ만, 거래대금↓ 3건/주)
kosdaq = cand[cand["Market"]=="KOSDAQ"].sort_values("Amount").copy()
kosdaq["bucket"] = kosdaq["Date"].dt.strftime("%Y-%U")
weekly3_picks = kosdaq.groupby("bucket").head(3).copy()
weekly3_picks["모드"] = "주3건"

# RF 회피 적용 버전 (RF 위험 0인 것만)
daily_safe = daily_picks[daily_picks["RF_위험"]==0].copy()
weekly3_safe = weekly3_picks[weekly3_picks["RF_위험"]==0].copy()


# ===== 출력 정리 =====
def format_out(df, modo_label):
    cols = ["Date","Code","Name","Market","Close","sell_date","sell_close",
            "ret_180d","peak_180d","Score","Amount","chart_pattern",
            "past_60","past_120","past_240","pos_252_high","slope60","drawdown60",
            "For_20d","Inst_20d","PER_num","PBR_num","외인소진율_num",
            "loser_prob","RF_위험"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out["모드"] = modo_label
    # 분류
    out["결과"] = "중립"
    out.loc[out["ret_180d"]<=-20, "결과"] = "손절"
    out.loc[out["ret_180d"]>0, "결과"] = "익절"
    out.loc[out["peak_180d"]>=50, "결과(peak)"] = "50%+"
    out.loc[out["peak_180d"]>=100, "결과(peak)"] = "100%+"
    out.loc[out["peak_180d"]>=200, "결과(peak)"] = "슈퍼위너"
    out["결과(peak)"] = out.get("결과(peak)", "").fillna("일반")
    out["수익률(180일)"] = out["ret_180d"].round(1)
    out["고점도달(peak)"] = out["peak_180d"].round(1)
    out["RF손절확률"] = (out["loser_prob"]*100).round(1)
    return out


# ===== 4가지 리스트 =====
list_daily_all = format_out(daily_picks, "매일1건(전체)").sort_values("Date", ascending=False)
list_daily_safe = format_out(daily_safe, "매일1건(RF안전)").sort_values("Date", ascending=False)
list_w3_all = format_out(weekly3_picks, "주3건(전체)").sort_values("Date", ascending=False)
list_w3_safe = format_out(weekly3_safe, "주3건(RF안전)").sort_values("Date", ascending=False)

# 합본 (모든 시그널 풀)
list_all_signals = format_out(cand, "시그널전체풀").sort_values("Date", ascending=False)

# 저장
list_daily_all.to_csv(CACHE/"LIST_매일1건_전체_최신순.csv", index=False)
list_daily_safe.to_csv(CACHE/"LIST_매일1건_RF안전_최신순.csv", index=False)
list_w3_all.to_csv(CACHE/"LIST_주3건_전체_최신순.csv", index=False)
list_w3_safe.to_csv(CACHE/"LIST_주3건_RF안전_최신순.csv", index=False)
list_all_signals.to_csv(CACHE/"LIST_전체시그널풀_최신순.csv", index=False)


# ===== 요약 통계 =====
print('\n'+'='*80)
print('생성된 리스트 요약 (최신순 정렬)')
print('='*80)
def stat(df, label):
    if len(df)==0: print(f'[{label}] 비어있음'); return
    print(f'\n[{label}] {len(df):,}건')
    print(f'  기간: {df["Date"].max().date()} ~ {df["Date"].min().date()}')
    print(f'  최근 5건: ')
    print(df.head(5)[["Date","Code","Name","Market","Close","수익률(180일)","고점도달(peak)","결과","RF손절확률"]].to_string(index=False))
    print(f'  익절: {(df["결과"]=="익절").sum()}건  손절: {(df["결과"]=="손절").sum()}건  중립: {(df["결과"]=="중립").sum()}건')
    print(f'  슈퍼위너: {(df["peak_180d"]>=200).sum()}건  100%+: {(df["peak_180d"]>=100).sum()}건  50%+: {(df["peak_180d"]>=50).sum()}건')

stat(list_daily_all, "매일1건 전체 매수 (회피6 적용)")
stat(list_daily_safe, "매일1건 RF안전 매수 (RF 손절확률↓80% 제외)")
stat(list_w3_all, "주3건 전체 매수")
stat(list_w3_safe, "주3건 RF안전 매수")

# 년도별 요약
print('\n\n'+'='*80)
print('년도별 매수 수 비교')
print('='*80)
for label, df in [("매일1건 전체", list_daily_all), ("매일1건 RF안전", list_daily_safe),
                   ("주3건 전체", list_w3_all), ("주3건 RF안전", list_w3_safe)]:
    df_ = df.copy()
    df_["Year"] = df_["Date"].dt.year
    yr = df_.groupby("Year").agg(
        매수=("Code", "count"),
        익절=("결과", lambda x: (x=="익절").sum()),
        손절=("결과", lambda x: (x=="손절").sum()),
        SW=("peak_180d", lambda x: (x>=200).sum()),
    )
    yr["수익금만원"] = df_.groupby("Year")["ret_180d"].sum().div(100).mul(10).round(0).astype(int)
    print(f"\n[{label}]")
    print(yr.to_string())

print(f"\n[저장] cache/LIST_*.csv 5개")
for f in CACHE.glob("LIST_*.csv"):
    print(f"  - {f.name}: {sum(1 for _ in open(f))-1:,}건")
