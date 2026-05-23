"""
슈퍼위너 발굴 + 회피 보강 분석
=============================

1. 시그널 풀에 수급+펀더멘털 매칭
2. 슈퍼위너(peak_180d≥200%) / 100%+ / 50%+ vs 비슈퍼위너 변수 차이
3. 회피 보강 룰 발굴 (X7~)
4. OOS 검증 (Train 2020-2023 / Test 2024-2026)
5. 전체 종목 리스트 출력

룰:
- 코스피+코스닥 동시
- 회피 6개 적용
- 주 1건 + 자본 10% + 180일 보유
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

CACHE = Path("cache")

# -------- LOAD --------
DF = pd.read_parquet(CACHE / "chart_feats_v1.parquet")
DF["Date"] = pd.to_datetime(DF["Date"])
DF = DF[DF["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
print(f"[로드] chart_feats: {len(DF):,}건")

with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)

# 수급
SD_PATH = CACHE / "supply_demand.parquet"
if SD_PATH.exists():
    SD = pd.read_parquet(SD_PATH)
    SD["Date"] = pd.to_datetime(SD["Date"])
    print(f"[로드] supply_demand: {len(SD):,}건, {SD['Code'].nunique()}종목")
else:
    SD = pd.DataFrame()
    print("[경고] supply_demand 없음")

# 펀더멘털
CUR_PATH = CACHE / "fundamentals_current.parquet"
if CUR_PATH.exists():
    CUR = pd.read_parquet(CUR_PATH)
    print(f"[로드] fundamentals_current: {len(CUR)}종목")
else:
    CUR = pd.DataFrame()

ANN_PATH = CACHE / "fundamentals_annual.parquet"
QUA_PATH = CACHE / "fundamentals_quarter.parquet"
ANN = pd.read_parquet(ANN_PATH) if ANN_PATH.exists() else pd.DataFrame()
QUA = pd.read_parquet(QUA_PATH) if QUA_PATH.exists() else pd.DataFrame()
print(f"[로드] fund annual: {len(ANN):,}건, quarter: {len(QUA):,}건")


# -------- 회피 6개 --------
def apply_avoid_6(df):
    d = df.copy()
    x1 = (d["chart_pattern"] == "pullback_recovery") & (d["slope60"] <= -1) & (d["pos_252_high"] <= -40)
    x2 = (d["Market"] == "KOSPI") & (d["past_120"] <= -20) & (d["pos_252_high"] <= -40)
    x3 = (d["s12"] >= 80) & (d["new_high_252"] == 1) & (d["past_120"] >= 50)
    x4 = d["past_240"] >= 100
    x5 = d["past_240"] >= 150
    x6 = d["Amount"] >= 3000e8
    return d[~(x1 | x2 | x3 | x4 | x5 | x6)].copy()


# -------- peak_180d 계산 --------
def compute_peak(df_signals):
    peaks, rets, sell_dates, sell_closes = [], [], [], []
    for _, row in df_signals.iterrows():
        code, d0, close0 = row["Code"], row["Date"], row["Close"]
        if code not in OHLCV:
            peaks.append(np.nan); rets.append(np.nan); sell_dates.append(pd.NaT); sell_closes.append(np.nan); continue
        future = OHLCV[code][OHLCV[code].index > d0].head(180)
        if len(future) == 0:
            peaks.append(np.nan); rets.append(np.nan); sell_dates.append(pd.NaT); sell_closes.append(np.nan); continue
        peaks.append((future["High"].max() / close0 - 1) * 100)
        rets.append((future["Close"].iloc[-1] / close0 - 1) * 100)
        sell_dates.append(future.index[-1])
        sell_closes.append(future["Close"].iloc[-1])
    out = df_signals.copy()
    out["peak_180d"] = peaks
    out["ret_180d"] = rets
    out["sell_date"] = sell_dates
    out["sell_close"] = sell_closes
    return out


cand = apply_avoid_6(DF)
print(f"\n[회피 후] {len(cand):,}건")
cand = compute_peak(cand)
cand = cand.dropna(subset=["peak_180d"]).copy()
print(f"[180일 결과] {len(cand):,}건")


# -------- 수급 매칭 (시그널일 D-5~D 누적 외국인/기관 순매수) --------
def merge_supply_demand(df):
    if SD.empty:
        return df.assign(For_5d=np.nan, Inst_5d=np.nan, For_20d=np.nan, Inst_20d=np.nan)

    # SD를 (Code, Date) 인덱스로 변환
    sd_dict = {}
    for code in SD["Code"].unique():
        sub = SD[SD["Code"] == code].sort_values("Date")
        sd_dict[code] = sub.set_index("Date")[["Foreign_NetBuy", "Inst_NetBuy"]]

    results = []
    for _, row in df.iterrows():
        code, d0 = row["Code"], row["Date"]
        rec = {}
        if code in sd_dict:
            sub = sd_dict[code]
            # D-5~D (시그널일 포함 5거래일)
            past5 = sub[sub.index <= d0].tail(5)
            past20 = sub[sub.index <= d0].tail(20)
            rec["For_5d"] = past5["Foreign_NetBuy"].sum() if len(past5) else np.nan
            rec["Inst_5d"] = past5["Inst_NetBuy"].sum() if len(past5) else np.nan
            rec["For_20d"] = past20["Foreign_NetBuy"].sum() if len(past20) else np.nan
            rec["Inst_20d"] = past20["Inst_NetBuy"].sum() if len(past20) else np.nan
        else:
            rec = {"For_5d": np.nan, "Inst_5d": np.nan, "For_20d": np.nan, "Inst_20d": np.nan}
        results.append(rec)
    sd_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, sd_df], axis=1)


cand = merge_supply_demand(cand)
print(f"[수급 매칭] For_5d 유효 {cand['For_5d'].notna().sum():,}건")


# -------- 펀더멘털 매칭 (현재값만) --------
def merge_fundamentals(df):
    if CUR.empty:
        return df

    f_cols = ["PER_num", "PBR_num", "EPS_num", "BPS_num", "시총_num", "외인소진율_num",
              "52주 최고_num", "52주 최저_num"]
    f_avail = [c for c in f_cols if c in CUR.columns]
    cur_idx = CUR.set_index("Code")[f_avail]
    out = df.copy()
    for c in f_avail:
        out[c] = out["Code"].map(cur_idx[c])
    return out


cand = merge_fundamentals(cand)


# -------- 슈퍼위너 vs 비슈퍼위너 변수 차이 --------
print("\n" + "=" * 80)
print("슈퍼위너(peak≥200%) vs 나머지 변수 차이")
print("=" * 80)

cand["is_sw"] = cand["peak_180d"] >= 200
cand["is_w100"] = cand["peak_180d"] >= 100
cand["is_w50"] = cand["peak_180d"] >= 50

numeric_cols = [
    "Score", "Amount", "vol_ratio", "candle_pct", "cum_5d_gain",
    "rs_ratio", "past_5d", "past_20", "past_60", "past_120", "past_240",
    "slope60", "slope120", "range60_pct", "drawdown60", "runup60", "vol20", "vol60",
    "pos_60_high", "pos_120_high", "pos_240_high", "pos_252_high",
    "days_since_52w_low", "days_since_52w_high",
    "For_5d", "Inst_5d", "For_20d", "Inst_20d",
    "PER_num", "PBR_num", "시총_num", "외인소진율_num",
]

results_diff = []
for col in numeric_cols:
    if col not in cand.columns:
        continue
    sw_mean = cand[cand["is_sw"]][col].mean()
    norm_mean = cand[~cand["is_sw"]][col].mean()
    sw_med = cand[cand["is_sw"]][col].median()
    norm_med = cand[~cand["is_sw"]][col].median()
    diff_pct = (sw_mean - norm_mean) / abs(norm_mean) * 100 if norm_mean != 0 else 0
    results_diff.append({
        "col": col,
        "SW평균": sw_mean,
        "일반평균": norm_mean,
        "차이%": diff_pct,
        "SW중앙": sw_med,
        "일반중앙": norm_med,
    })

diff_df = pd.DataFrame(results_diff).sort_values("차이%", key=lambda x: x.abs(), ascending=False)
print(diff_df.to_string(index=False))

diff_df.to_csv(CACHE / "sw_diff_analysis.csv", index=False)


# -------- 회피 룰 X7~X10 발굴 --------
print("\n" + "=" * 80)
print("회피 보강 룰 발굴 (200% peak 부재 + 손실 큰 구간 탐색)")
print("=" * 80)

# 손실 종목 (180일 안에 -20% 이상 손절될 종목): max drawdown < -20%
# 단순화: ret_180d <= -20%
loser = cand[cand["ret_180d"] <= -20].copy()
neutral = cand[(cand["ret_180d"] > -20) & (cand["ret_180d"] < 50)].copy()
print(f"\n루저(-20%↓): {len(loser):,}건 ({len(loser)/len(cand)*100:.1f}%)")
print(f"슈퍼위너(200%↑): {cand['is_sw'].sum():,}건 ({cand['is_sw'].mean()*100:.1f}%)")

# 후보 회피 룰 - 단변량 (루저 비율 ↑, 슈퍼위너 비율 ↓)
def evaluate_rule(mask, name):
    excluded = cand[mask]
    remained = cand[~mask]
    if len(excluded) == 0 or len(remained) == 0:
        return None
    return {
        "rule": name,
        "n_excluded": len(excluded),
        "exc_loser_rate": (excluded["ret_180d"] <= -20).mean() * 100,
        "exc_sw_rate": (excluded["peak_180d"] >= 200).mean() * 100,
        "rem_n": len(remained),
        "rem_loser_rate": (remained["ret_180d"] <= -20).mean() * 100,
        "rem_sw_rate": (remained["peak_180d"] >= 200).mean() * 100,
        "rem_w100_rate": (remained["peak_180d"] >= 100).mean() * 100,
        "rem_avg_peak": remained["peak_180d"].mean(),
    }


rule_results = []

# 수급 룰
if "Foreign_NetBuy" in SD.columns:
    rule_results.append(evaluate_rule(
        cand["For_20d"] < cand["For_20d"].quantile(0.10),
        "X7. 외국인 20일 누적순매수 하위10% (집중 매도)"))
    rule_results.append(evaluate_rule(
        cand["Inst_20d"] < cand["Inst_20d"].quantile(0.10),
        "X8. 기관 20일 누적순매수 하위10%"))
    rule_results.append(evaluate_rule(
        (cand["For_20d"] < 0) & (cand["Inst_20d"] < 0),
        "X9. 외인+기관 동시 20일 누적 순매도"))

# 펀더멘털 룰
if "PER_num" in cand.columns:
    rule_results.append(evaluate_rule(
        cand["PER_num"] > 50,
        "X10. PER > 50 (고평가)"))
    rule_results.append(evaluate_rule(
        cand["PER_num"] <= 0,
        "X11. PER ≤ 0 (적자)"))
    rule_results.append(evaluate_rule(
        cand["PBR_num"] > 5,
        "X12. PBR > 5"))

# 차트 추가 룰
rule_results.append(evaluate_rule(
    (cand["slope60"] <= -2) & (cand["pos_252_high"] <= -50),
    "X13. 강하락추세+52주고점 -50%↓"))
rule_results.append(evaluate_rule(
    cand["drawdown60"] <= -30,
    "X14. 60일 drawdown -30%↓"))
rule_results.append(evaluate_rule(
    cand["vol60"] > cand["vol60"].quantile(0.90),
    "X15. 60일 변동성 상위10% (극변동)"))
rule_results.append(evaluate_rule(
    cand["range60_pct"] > cand["range60_pct"].quantile(0.90),
    "X16. 60일 가격range 상위10%"))
rule_results.append(evaluate_rule(
    cand["past_60"] >= 80,
    "X17. 60일 +80%↑ (단기과열)"))
rule_results.append(evaluate_rule(
    cand["past_60"] <= -30,
    "X18. 60일 -30%↓ (낙폭과대)"))

rule_results = [r for r in rule_results if r]
rule_df = pd.DataFrame(rule_results)
print("\n[단변량 회피 후보 평가]")
print(rule_df.to_string(index=False))

rule_df.to_csv(CACHE / "avoid_rule_candidates.csv", index=False)


# -------- 슈퍼위너 강화 룰 발굴 --------
print("\n" + "=" * 80)
print("슈퍼위너 농도 강화 룰 (유지하면 SW 농도 ↑)")
print("=" * 80)

# 슈퍼위너만 가지는 특징 추출
sw = cand[cand["is_sw"]]
nw = cand[~cand["is_sw"]]

include_rules = []

def eval_include(mask, name):
    sub = cand[mask]
    if len(sub) < 30:
        return None
    return {
        "rule": name,
        "n": len(sub),
        "포착비율": len(sub) / len(cand) * 100,
        "SW농도": (sub["peak_180d"] >= 200).mean() * 100,
        "100%+농도": (sub["peak_180d"] >= 100).mean() * 100,
        "50%+농도": (sub["peak_180d"] >= 50).mean() * 100,
        "평균peak": sub["peak_180d"].mean(),
        "평균ret": sub["ret_180d"].mean(),
    }


# 차트 변수 단변량 cutoff 탐색
include_rules.append(eval_include(cand["past_60"].between(-10, 30), "P1. 60일 -10~+30% (안정 추세)"))
include_rules.append(eval_include(cand["slope60"] >= 0.5, "P2. slope60 ≥ 0.5 (상승추세)"))
include_rules.append(eval_include(
    (cand["slope60"] >= 0.5) & (cand["past_240"].between(-20, 60)),
    "P3. 상승추세+1년 -20~+60% (과열X)"))
include_rules.append(eval_include(cand["pos_252_high"].between(-30, -10), "P4. 52주高 -30~-10% (눌림목)"))
include_rules.append(eval_include(cand["near_52w_low"] == 1, "P5. 52주저점 근접"))
include_rules.append(eval_include(
    (cand["pos_252_high"] >= -20) & (cand["past_60"].between(-15, 25)),
    "P6. 52주高 -20%↑ + 안정 60일"))

if "For_20d" in cand.columns and cand["For_20d"].notna().sum() > 100:
    include_rules.append(eval_include(
        cand["For_20d"] > cand["For_20d"].quantile(0.75),
        "P7. 외국인 20일 누적순매수 상위25%"))
    include_rules.append(eval_include(
        cand["Inst_20d"] > cand["Inst_20d"].quantile(0.75),
        "P8. 기관 20일 누적순매수 상위25%"))
    include_rules.append(eval_include(
        (cand["For_20d"] > 0) & (cand["Inst_20d"] > 0),
        "P9. 외인+기관 동시 20일 순매수"))

if "PER_num" in cand.columns:
    include_rules.append(eval_include(
        cand["PER_num"].between(0, 20),
        "P10. PER 0~20 (저평가)"))
    include_rules.append(eval_include(
        cand["PBR_num"].between(0, 2),
        "P11. PBR 0~2 (자산저평가)"))

include_rules = [r for r in include_rules if r]
inc_df = pd.DataFrame(include_rules).sort_values("SW농도", ascending=False)
print(inc_df.to_string(index=False))
inc_df.to_csv(CACHE / "include_rule_candidates.csv", index=False)


# -------- 저장 --------
print("\n[전체 candidate 저장]")
cand.to_parquet(CACHE / "candidates_enriched.parquet", index=False)
print(f"  → cache/candidates_enriched.parquet ({len(cand):,}건)")

print("\n[완료]")
