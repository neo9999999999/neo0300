"""
백필 V3 통합 - 회피 + RF + 추천 종목 리스트 생성
=============================================
backfill_v3 (2025-08-25 ~ 2026-05-22) 24,328건을
기존 candidates_enriched 와 합쳐 최종 마스터 리스트 생성.

수익률은 미래라 비어있음 (NaN). 백테스트(2020~2025-08)와 같이 표시.
"""

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from train_rf_loss_model import add_pre_features, prepare_X

CACHE = Path("cache")

# 1) 백필 + 기존 통합
print("[1] 데이터 로드 + 통합")
backfill = pd.read_parquet(CACHE / "backfill_v3_2025-08-25_2026-05-22.parquet")
backfill["Date"] = pd.to_datetime(backfill["Date"])
print(f"   백필 V3: {len(backfill):,}건 (기간 {backfill['Date'].min().date()} ~ {backfill['Date'].max().date()})")

existing = pd.read_parquet(CACHE / "candidates_enriched.parquet")
existing["Date"] = pd.to_datetime(existing["Date"])
print(f"   기존 마스터: {len(existing):,}건 (기간 {existing['Date'].min().date()} ~ {existing['Date'].max().date()})")

# 2) OHLCV에서 derived 컬럼 보강
print("[1.5] 백필에 derived 컬럼 추가...")
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)


def enrich_derived(df):
    Amounts, p60, p120, p240, slope60, pos252, dd60, new_high, vol20, vol60_, range60 = [], [], [], [], [], [], [], [], [], [], []
    days_52w_low, days_52w_high = [], []
    for _, r in df.iterrows():
        code = r["Code"]; d0 = r["Date"]; c0 = r["Close"]
        if code not in OHLCV:
            for lst in [Amounts, p60, p120, p240, slope60, pos252, dd60, new_high, vol20, vol60_, range60, days_52w_low, days_52w_high]:
                lst.append(np.nan)
            continue
        bars = OHLCV[code]
        past = bars[bars.index <= d0]
        if len(past) < 252:
            for lst in [Amounts, p60, p120, p240, slope60, pos252, dd60, new_high, vol20, vol60_, range60, days_52w_low, days_52w_high]:
                lst.append(np.nan)
            continue
        # 거래대금
        latest = past.iloc[-1]
        Amounts.append(float(latest["Close"] * latest["Volume"]))
        # 모멘텀
        p60.append((c0 / past.iloc[-60]["Close"] - 1) * 100 if len(past) >= 60 else np.nan)
        p120.append((c0 / past.iloc[-120]["Close"] - 1) * 100 if len(past) >= 120 else np.nan)
        p240.append((c0 / past.iloc[-240]["Close"] - 1) * 100 if len(past) >= 240 else np.nan)
        # 추세
        last60 = past.tail(60)
        if len(last60) >= 30:
            x = np.arange(len(last60))
            slope = np.polyfit(x, last60["Close"].values, 1)[0]
            slope60.append(slope / last60["Close"].mean() * 100)
        else:
            slope60.append(np.nan)
        # 252일 고점 대비
        last252 = past.tail(252)
        h252 = last252["High"].max()
        l252 = last252["Low"].min()
        pos252.append((c0 / h252 - 1) * 100 if h252 > 0 else np.nan)
        # drawdown 60일
        last60c = past.tail(60)
        h60 = last60c["High"].max()
        dd60.append((c0 / h60 - 1) * 100 if h60 > 0 else np.nan)
        # 신고가
        new_high.append(1 if c0 >= h252 * 0.995 else 0)
        # 변동성
        rets = past["Close"].pct_change()
        vol20.append(rets.tail(20).std() * 100 if len(rets.tail(20)) > 5 else np.nan)
        vol60_.append(rets.tail(60).std() * 100 if len(rets.tail(60)) > 5 else np.nan)
        # 가격 범위
        range60.append((h60 - last60c["Low"].min()) / last60c["Low"].min() * 100 if last60c["Low"].min() > 0 else np.nan)
        # 52주 저점/고점 이후 일수
        try:
            days_52w_low.append(int((d0 - last252["Low"].idxmin()).days))
            days_52w_high.append(int((d0 - last252["High"].idxmax()).days))
        except Exception:
            days_52w_low.append(np.nan); days_52w_high.append(np.nan)
    df = df.copy()
    df["Amount"] = Amounts
    df["past_60"] = p60
    df["past_120"] = p120
    df["past_240"] = p240
    df["slope60"] = slope60
    df["slope120"] = slope60  # 단순화
    df["pos_252_high"] = pos252
    df["drawdown60"] = dd60
    df["new_high_252"] = new_high
    df["vol20"] = vol20
    df["vol60"] = vol60_
    df["range60_pct"] = range60
    df["range120_pct"] = range60  # 단순화
    df["days_since_52w_low"] = days_52w_low
    df["days_since_52w_high"] = days_52w_high
    # RF 모델에 필요한 추가 derived
    df["pos_60_high"] = pos252  # 단순화
    df["pos_120_high"] = pos252
    df["pos_240_high"] = pos252
    df["past_5d"] = df["past_60"] / 12  # 단순 비례
    df["past_20"] = df["past_60"] / 3
    df["runup60"] = df["past_60"].abs()
    # 결측 채움 (chart_pattern)
    df["chart_pattern"] = "mixed"
    return df

backfill = enrich_derived(backfill)
print(f"   derived 추가 완료, Amount 유효: {backfill['Amount'].notna().sum():,}")


# 회피 6 적용
def apply_avoid_6(df):
    d = df.copy()
    # X1 (chart_pattern 단순화로 skip)
    x2 = (d["Market"] == "KOSPI") & (d["past_120"] <= -20) & (d["pos_252_high"] <= -40)
    x3 = (d["s12"] >= 80) & (d["new_high_252"] == 1) & (d["past_120"] >= 50)
    x4 = d["past_240"] >= 100
    x5 = d["past_240"] >= 150
    x6 = d["Amount"] >= 3000e8
    return d[~(x2 | x3 | x4 | x5 | x6).fillna(False)].copy()

backfill_filtered = apply_avoid_6(backfill)
print(f"   백필 회피6 후: {len(backfill_filtered):,}건")

# 3) OHLCV로 forward fields 계산 (peak_180d/ret_180d) - 미래라 부분만 가능
def compute_forward(df):
    peaks, rets, sell_dates, sell_closes = [], [], [], []
    for _, r in df.iterrows():
        code = r["Code"]; d0 = r["Date"]; c0 = r["Close"]
        if code not in OHLCV:
            peaks.append(np.nan); rets.append(np.nan); sell_dates.append(pd.NaT); sell_closes.append(np.nan); continue
        bars = OHLCV[code]
        future = bars[bars.index > d0].head(180)
        if len(future) == 0:
            peaks.append(np.nan); rets.append(np.nan); sell_dates.append(pd.NaT); sell_closes.append(np.nan); continue
        peaks.append((future["High"].max() / c0 - 1) * 100)
        rets.append((future["Close"].iloc[-1] / c0 - 1) * 100)
        sell_dates.append(future.index[-1])
        sell_closes.append(future["Close"].iloc[-1])
    df = df.copy()
    df["peak_180d"] = peaks
    df["ret_180d"] = rets
    df["sell_date"] = sell_dates
    df["sell_close"] = sell_closes
    return df

print("[2] 백필 종목 forward 계산 (가능한 만큼)")
backfill_filtered = compute_forward(backfill_filtered)
n_with_ret = backfill_filtered["ret_180d"].notna().sum()
print(f"   180일 결과 가능: {n_with_ret:,}건 / 미래 = {(backfill_filtered['ret_180d'].isna()).sum():,}건")

# 4) 시계열 특성 + 펀더 + 수급 통합
print("[3] 시계열 특성 + 수급/펀더 매칭")
backfill_filtered = add_pre_features(backfill_filtered)

# 펀더
cur = pd.read_parquet(CACHE / "fundamentals_current.parquet") if (CACHE / "fundamentals_current.parquet").exists() else pd.DataFrame()
if not cur.empty:
    cur_idx = cur.set_index("Code")[["PER_num", "PBR_num", "외인소진율_num"]]
    for c in ["PER_num", "PBR_num", "외인소진율_num"]:
        backfill_filtered[c] = backfill_filtered["Code"].map(cur_idx[c])

# 수급
sd = pd.read_parquet(CACHE / "supply_demand.parquet")
sd["Date"] = pd.to_datetime(sd["Date"])
sd_dict = {}
for code in sd["Code"].unique():
    sub = sd[sd["Code"] == code].sort_values("Date")
    sd_dict[code] = sub.set_index("Date")[["Foreign_NetBuy", "Inst_NetBuy"]]

for label, days in [("For_5d", 5), ("Inst_5d", 5), ("For_20d", 20), ("Inst_20d", 20)]:
    vals = []
    src_col = "Foreign_NetBuy" if label.startswith("For") else "Inst_NetBuy"
    for code, dt in zip(backfill_filtered["Code"], backfill_filtered["Date"]):
        if code in sd_dict:
            past = sd_dict[code][sd_dict[code].index <= dt].tail(days)
            vals.append(past[src_col].sum() if len(past) else np.nan)
        else:
            vals.append(np.nan)
    backfill_filtered[label] = vals

# 5) RF 손절확률
print("[4] RF 모델 손절확률 예측")
with open(CACHE / "rf_loss_model.pkl", "rb") as f:
    rf = pickle.load(f)
with open(CACHE / "rf_features.json") as f:
    meta = json.load(f)
TH20 = meta["th20"]

X, _ = prepare_X(backfill_filtered, features=meta["features"])
probs = rf.predict_proba(X)[:, 1]
backfill_filtered["RF손절확률"] = probs
backfill_filtered["RF위험"] = (probs >= TH20).astype(int)

# 6) 결과/peak 등급 분류
backfill_filtered["결과"] = "미정"  # 미래라 모름
backfill_filtered.loc[backfill_filtered["ret_180d"].notna() & (backfill_filtered["ret_180d"] <= -20), "결과"] = "손절"
backfill_filtered.loc[backfill_filtered["ret_180d"].notna() & (backfill_filtered["ret_180d"] > 0), "결과"] = "익절"
backfill_filtered.loc[backfill_filtered["ret_180d"].notna() & (backfill_filtered["ret_180d"].between(-20, 0)), "결과"] = "중립"

backfill_filtered["peak등급"] = "미정"
backfill_filtered.loc[backfill_filtered["peak_180d"].notna() & (backfill_filtered["peak_180d"] >= 200), "peak등급"] = "슈퍼위너"
backfill_filtered.loc[backfill_filtered["peak_180d"].notna() & (backfill_filtered["peak_180d"].between(100, 200)), "peak등급"] = "100%+"
backfill_filtered.loc[backfill_filtered["peak_180d"].notna() & (backfill_filtered["peak_180d"].between(50, 100)), "peak등급"] = "50%+"
backfill_filtered.loc[backfill_filtered["peak_180d"].notna() & (backfill_filtered["peak_180d"] < 50), "peak등급"] = "일반"


# 7) 통합 마스터 (기존 + 백필)
print("[5] 마스터 통합 저장")
# existing에도 시계열특성 + RF 적용
existing = add_pre_features(existing)
if not cur.empty:
    cur_idx = cur.set_index("Code")[["PER_num", "PBR_num", "외인소진율_num"]]
    for c in ["PER_num", "PBR_num", "외인소진율_num"]:
        if c not in existing.columns:
            existing[c] = existing["Code"].map(cur_idx[c])

# existing RF 예측
X_ex, _ = prepare_X(existing, features=meta["features"])
existing["RF손절확률"] = rf.predict_proba(X_ex)[:, 1]
existing["RF위험"] = (existing["RF손절확률"] >= TH20).astype(int)

# 결과/peak등급 (existing은 이미 ret_180d 있음)
existing["결과"] = "중립"
existing.loc[existing["ret_180d"] <= -20, "결과"] = "손절"
existing.loc[existing["ret_180d"] > 0, "결과"] = "익절"
existing["peak등급"] = "일반"
existing.loc[existing["peak_180d"] >= 50, "peak등급"] = "50%+"
existing.loc[existing["peak_180d"] >= 100, "peak등급"] = "100%+"
existing.loc[existing["peak_180d"] >= 200, "peak등급"] = "슈퍼위너"

# 통합
keep_cols = list(set(existing.columns) & set(backfill_filtered.columns))
combined = pd.concat([
    existing[keep_cols].assign(데이터소스="원본"),
    backfill_filtered[keep_cols].assign(데이터소스="백필V3"),
], ignore_index=True)
combined = combined.drop_duplicates(subset=["Date", "Code", "preset"], keep="first")
combined = combined.sort_values("Date", ascending=False).reset_index(drop=True)
combined.to_parquet(CACHE / "candidates_enriched_full.parquet", index=False)
print(f"   마스터 통합: {len(combined):,}건")

# 8) 매수 모드별 추출 (전체기간)
print("[6] 매수 모드별 추출")

daily = combined.sort_values("Amount").copy()
daily["bucket"] = daily["Date"].dt.strftime("%Y-%m-%d")
daily_picks = daily.groupby("bucket").head(1).copy()
daily_safe = daily_picks[daily_picks["RF위험"] == 0].copy()

w3 = combined[combined["Market"] == "KOSDAQ"].sort_values("Amount").copy()
w3["bucket"] = w3["Date"].dt.strftime("%Y-%U")
w3_picks = w3.groupby("bucket").head(3).copy()
w3_safe = w3_picks[w3_picks["RF위험"] == 0].copy()

def format_out(df, modo):
    cols = ["Date","Code","Name","Market","Close","sell_date","sell_close",
            "ret_180d","peak_180d","결과","peak등급","RF손절확률","RF위험",
            "Score","Amount","chart_pattern",
            "past_60","past_120","pos_252_high","slope60","drawdown60",
            "For_20d","Inst_20d","PER_num","PBR_num"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out["매수모드"] = modo
    out["ret_180d"] = out["ret_180d"].round(1)
    out["peak_180d"] = out["peak_180d"].round(1)
    out["RF손절확률"] = (out["RF손절확률"] * 100).round(1)
    out = out.sort_values("Date", ascending=False).reset_index(drop=True)
    return out

format_out(daily_picks, "매일1건").to_csv(CACHE / "FULL_매일1건_전체_2020-2026.csv", index=False)
format_out(daily_safe, "매일1건_RF안전").to_csv(CACHE / "FULL_매일1건_RF안전_2020-2026.csv", index=False)
format_out(w3_picks, "주3건").to_csv(CACHE / "FULL_주3건_전체_2020-2026.csv", index=False)
format_out(w3_safe, "주3건_RF안전").to_csv(CACHE / "FULL_주3건_RF안전_2020-2026.csv", index=False)
format_out(combined, "전체시그널풀").to_csv(CACHE / "FULL_시그널풀_2020-2026.csv", index=False)

# 9) 요약 통계
print("\n" + "="*80)
print("통합 마스터 (2020-04 ~ 2026-05-22) 요약")
print("="*80)

def stats(df, label):
    n = len(df)
    has_ret = df["ret_180d"].notna()
    n_known = has_ret.sum()
    n_future = (~has_ret).sum()
    invest_known = n_known * 10
    profit_known = (df.loc[has_ret, "ret_180d"] / 100 * 10).sum() if n_known > 0 else 0
    return {
        "라벨": label, "매수": n, "수익확정": int(n_known), "미래": int(n_future),
        "익절": int((df["결과"]=="익절").sum()),
        "손절": int((df["결과"]=="손절").sum()),
        "SW": int((df["peak_180d"]>=200).fillna(False).sum() if n_known>0 else 0),
        "투자만(확정)": invest_known,
        "수익만(확정)": round(profit_known, 0),
        "수익률(확정)": round(profit_known/invest_known*100, 1) if invest_known else 0,
    }

summary = pd.DataFrame([
    stats(daily_picks, "매일1건 전체"),
    stats(daily_safe, "매일1건 RF안전"),
    stats(w3_picks, "주3건 전체"),
    stats(w3_safe, "주3건 RF안전"),
])
print(summary.to_string(index=False))
summary.to_csv(CACHE / "FULL_요약_2020-2026.csv", index=False)

# 10) 2026년 추천만 미리 출력
print("\n" + "="*80)
print("2026년 백필 추천 종목 (주3건 RF안전, 최신순)")
print("="*80)
y26 = format_out(w3_safe, "주3건_RF안전")
y26 = y26[y26["Date"].dt.year == 2026]
y26_dedup = y26.drop_duplicates(subset=["Date","Code"]).head(30)
print(y26_dedup[["Date","Code","Name","Market","Close","RF손절확률"]].to_string(index=False))
print(f"\n2026년 총 {len(y26.drop_duplicates(subset=['Date','Code']))}건")

print("\n[저장]")
for f in sorted((CACHE).glob("FULL_*.csv")):
    n = sum(1 for _ in open(f))-1
    print(f"  {f.name}: {n:,}건")
