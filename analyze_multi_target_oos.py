"""
analyze_multi_target_oos.py
===========================

Multi-target OOS pattern analysis with optional fundamentals overlay.

Targets analyzed (per buy date in chart_feats_v1.parquet):
- peak_60d / peak_90d / peak_120d / peak_180d:  max close-return within N trading days
- hit_10/hit_20/hit_30/hit_50:                  whether peak_180d >= +X%

Train: 2020-04-03 .. 2023-12-31
Test : 2024-01-01 .. 2025-08-22

For each target, we mine:
- univariate threshold rules on numeric features
- pairwise / triple AND combos on binarised features

Keep rules with Train hit-rate >= 70% and decent support, then validate on Test.

Outputs:
- cache/multi_target_results.csv
- MULTI_TARGET_PATTERNS.md
"""
from __future__ import annotations

import warnings
import pickle
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/neo/Desktop/jongga_picker")
CACHE = BASE / "cache"
FEATS = CACHE / "chart_feats_v1.parquet"
OHLCV = CACHE / "ohlcv_2020-01-01_2026-05-21.pkl"

TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-08-22")

# Mining params
MIN_N_TRAIN = 80
MIN_N_TEST = 25

# Per-target adaptive thresholds (Train hit-rate min, Test hit-rate min, lift min Train, lift min Test).
# Calibrated so that "good" = "noticeably above base rate AND meaningful absolute level".
TARGETS = {
    "hit_10": 10.0,    # base ~71%, ask 85%+ Train, 80%+ Test
    "hit_20": 20.0,    # base ~58%, ask 75%+ Train, 70%+ Test
    "hit_30": 30.0,    # base ~48%, ask 65%+ Train, 60%+ Test
    "hit_50": 50.0,    # base ~33%, ask 50%+ Train, 45%+ Test
}
TARGET_THRESHOLDS = {
    # (train_hit_min, test_hit_min, train_lift_min, test_lift_min)
    "hit_10": (0.85, 0.80, 1.20, 1.10),
    "hit_20": (0.75, 0.70, 1.30, 1.15),
    "hit_30": (0.65, 0.60, 1.35, 1.20),
    "hit_50": (0.50, 0.45, 1.50, 1.25),
}

NUMERIC_FEATS = [
    "s1", "s2", "s3", "s5", "s6", "s8", "s10", "s12",
    "vol_ratio", "candle_pct", "cum_5d_gain", "upper_wick_ratio", "rs_ratio", "past_5d",
    "ChangeRatio", "Amount", "Score",
    "pos_60_high", "pos_120_high", "pos_240_high", "pos_252_high", "pos_252_low",
    "past_20", "past_60", "past_120", "past_240",
    "slope60", "slope120", "range60_pct", "range120_pct",
    "drawdown60", "runup60", "vol20", "vol60", "vol_trend",
    "days_since_52w_high", "days_since_52w_low",
]


# ---------------------------------------------------------------------------
# Step 1: Compute peak returns from OHLCV
# ---------------------------------------------------------------------------

def compute_peaks(df: pd.DataFrame, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each (Date, Code), compute peak_60d/90d/120d/180d using trading days.
    Uses Close prices only (not High) — i.e. would have to "see" the peak close
    to sell that day. Calls with a deduplicated (Date,Code) view, then merges.
    """
    pairs = df[["Date", "Code", "Close"]].drop_duplicates(["Date", "Code"]).copy()
    pairs["Date"] = pd.to_datetime(pairs["Date"])

    peaks = {n: [] for n in [60, 90, 120, 180]}
    final_180 = []   # close after exactly 180 trading days (recompute to verify)

    for _, row in pairs.iterrows():
        code = row["Code"]
        buy_date = row["Date"]
        if code not in ohlcv:
            for n in [60, 90, 120, 180]:
                peaks[n].append(np.nan)
            final_180.append(np.nan)
            continue
        o = ohlcv[code]
        idx = o.index.searchsorted(buy_date)
        if idx >= len(o) or o.index[idx] != buy_date:
            # buy_date not in OHLCV (e.g. data lag); use nearest forward
            if idx >= len(o):
                for n in [60, 90, 120, 180]:
                    peaks[n].append(np.nan)
                final_180.append(np.nan)
                continue
        buy_close = float(o.iloc[idx]["Close"])
        if buy_close <= 0:
            for n in [60, 90, 120, 180]:
                peaks[n].append(np.nan)
            final_180.append(np.nan)
            continue
        for n in [60, 90, 120, 180]:
            window = o["Close"].iloc[idx + 1 : idx + 1 + n]
            if len(window) > 0:
                peak = float(window.max())
                peaks[n].append((peak / buy_close - 1) * 100)
            else:
                peaks[n].append(np.nan)
        # 180-trading-day exact final
        if idx + 180 < len(o):
            final_180.append((float(o.iloc[idx + 180]["Close"]) / buy_close - 1) * 100)
        else:
            final_180.append(np.nan)

    pairs["peak_60d"] = peaks[60]
    pairs["peak_90d"] = peaks[90]
    pairs["peak_120d"] = peaks[120]
    pairs["peak_180d"] = peaks[180]
    pairs["final_180d_check"] = final_180
    return pairs.drop(columns=["Close"])


# ---------------------------------------------------------------------------
# Step 2: Build binaries (same definitions as analyze_oos_patterns.py)
# ---------------------------------------------------------------------------

def build_binaries(df: pd.DataFrame) -> Dict[str, pd.Series]:
    bin_: Dict[str, pd.Series] = {}
    for s in ["s1", "s2", "s3", "s5", "s6", "s8", "s10"]:
        if s in df.columns:
            bin_[f"{s}>=70"] = df[s] >= 70
            bin_[f"{s}>=90"] = df[s] >= 90
    if "s4" in df.columns:
        bin_["s4>=75"] = df["s4"] >= 75
    if "s12" in df.columns:
        bin_["s12>=80"] = df["s12"] >= 80
    for b in ["is_first_pullback", "cup_and_handle_detected",
              "inverse_hns_detected", "gap_support_detected"]:
        if b in df.columns:
            bin_[b] = df[b].astype(bool)
    for b in ["new_high_60", "new_high_120", "new_high_240", "new_high_252", "near_52w_low"]:
        if b in df.columns:
            bin_[b] = (df[b].fillna(0).astype(float) == 1)
    if "Market" in df.columns:
        bin_["KOSDAQ"] = df["Market"] == "KOSDAQ"
        bin_["KOSPI"] = df["Market"] == "KOSPI"
    if "chart_pattern" in df.columns:
        for label in ["new_high_60", "new_high_120", "new_high_240",
                      "box_breakout", "V_recovery", "pullback_recovery",
                      "persistent_uptrend", "downtrend", "sideways", "mixed"]:
            bin_[f"chart={label}"] = df["chart_pattern"] == label
    if "past_60" in df.columns:
        bin_["past_60>=30"] = pd.to_numeric(df["past_60"], errors="coerce") >= 30
        bin_["past_60<=-15"] = pd.to_numeric(df["past_60"], errors="coerce") <= -15
    if "past_120" in df.columns:
        bin_["past_120>=50"] = pd.to_numeric(df["past_120"], errors="coerce") >= 50
        bin_["past_120<=-20"] = pd.to_numeric(df["past_120"], errors="coerce") <= -20
    if "slope60" in df.columns:
        bin_["slope60>=1"] = pd.to_numeric(df["slope60"], errors="coerce") >= 1.0
        bin_["slope60<=-1"] = pd.to_numeric(df["slope60"], errors="coerce") <= -1.0
    if "pos_252_high" in df.columns:
        bin_["pos252_top10"] = pd.to_numeric(df["pos_252_high"], errors="coerce") >= -10
        bin_["pos252_far"] = pd.to_numeric(df["pos_252_high"], errors="coerce") <= -40
    if "rs_ratio" in df.columns:
        bin_["rs>=1.1"] = pd.to_numeric(df["rs_ratio"], errors="coerce") >= 1.1
        bin_["rs<=0.95"] = pd.to_numeric(df["rs_ratio"], errors="coerce") <= 0.95
    if "Marcap_bn" in df.columns:
        m = pd.to_numeric(df["Marcap_bn"], errors="coerce")
        bin_["marcap<=200bn"]    = m <= 200       # 2천억 이하
        bin_["marcap200~500bn"]  = (m > 200) & (m <= 500)
        bin_["marcap500~3000bn"] = (m > 500) & (m <= 3000)
        bin_["marcap>3000bn"]    = m > 3000       # 3조 이상
        bin_["marcap>10000bn"]   = m > 10000      # 10조 이상
    if "Marcap_bn" in df.columns and "Amount" in df.columns:
        amt = pd.to_numeric(df["Amount"], errors="coerce")
        mbn = pd.to_numeric(df["Marcap_bn"], errors="coerce")
        # 회전율 proxy: 거래대금/시가총액
        turnover = amt / (mbn * 1e9)
        bin_["turnover_hot>=5pct"] = turnover >= 0.05    # 회전율 5%+ (활발한 거래)
        bin_["turnover_low<=1pct"] = turnover <= 0.01
    if "Close" in df.columns:
        c = pd.to_numeric(df["Close"], errors="coerce")
        bin_["price_low<=5000won"] = c <= 5000           # 저가주 (PBR/PER 정보 없을 때 보조)
        bin_["price_high>=50000won"] = c >= 50000
    # Coerce
    for k, s in list(bin_.items()):
        bin_[k] = s.fillna(False).astype(bool)
    return bin_


# ---------------------------------------------------------------------------
# Step 3: Mine univariate + combo rules for one target
# ---------------------------------------------------------------------------

def mine_univariate(train: pd.DataFrame, target_col: str) -> pd.DataFrame:
    base = train[target_col].mean()
    th_train_hit, _, th_train_lift, _ = TARGET_THRESHOLDS[target_col]
    out = []
    for feat in NUMERIC_FEATS:
        if feat not in train.columns:
            continue
        s = pd.to_numeric(train[feat], errors="coerce")
        if s.notna().sum() < 200:
            continue
        qs = np.linspace(0.05, 0.95, 19)
        cutoffs = sorted(set(np.round(s.dropna().quantile(qs).values, 4).tolist()))
        if feat.startswith("s") and len(feat) <= 3 and feat[1:].isdigit():
            for v in [50, 70, 80, 90, 95]:
                cutoffs.append(float(v))
        cutoffs = sorted(set(cutoffs))
        for cut in cutoffs:
            for direction in ("ge", "le"):
                mask = (s >= cut) if direction == "ge" else (s <= cut)
                n = int(mask.sum())
                if n < MIN_N_TRAIN:
                    continue
                rate = train.loc[mask, target_col].mean()
                if rate < th_train_hit:
                    continue
                lift = rate / base if base > 0 else np.nan
                if lift < th_train_lift:
                    continue
                rule_repr = f"{feat} {'>=' if direction=='ge' else '<='} {cut:g}"
                out.append({
                    "rule": rule_repr, "type": "uni", "feat": feat,
                    "direction": direction, "cut": float(cut),
                    "train_n": n, "train_hit": rate, "train_lift": lift,
                })
    return pd.DataFrame(out)


def mine_combos(train: pd.DataFrame, target_col: str, pairs: bool = True, triples: bool = True) -> pd.DataFrame:
    base = train[target_col].mean()
    th_train_hit, _, th_train_lift, _ = TARGET_THRESHOLDS[target_col]
    bin_ = build_binaries(train)
    keys = list(bin_.keys())
    out = []
    if pairs:
        for a, b in combinations(keys, 2):
            mask = bin_[a] & bin_[b]
            n = int(mask.sum())
            if n < MIN_N_TRAIN:
                continue
            rate = train.loc[mask, target_col].mean()
            if rate < th_train_hit:
                continue
            lift = rate / base if base > 0 else np.nan
            if lift < th_train_lift:
                continue
            out.append({
                "rule": f"{a} AND {b}", "type": "combo2",
                "train_n": n, "train_hit": rate, "train_lift": lift,
            })
    if triples:
        for a, b, c in combinations(keys, 3):
            mask = bin_[a] & bin_[b] & bin_[c]
            n = int(mask.sum())
            if n < MIN_N_TRAIN:
                continue
            rate = train.loc[mask, target_col].mean()
            if rate < th_train_hit:
                continue
            lift = rate / base if base > 0 else np.nan
            if lift < th_train_lift:
                continue
            out.append({
                "rule": f"{a} AND {b} AND {c}", "type": "combo3",
                "train_n": n, "train_hit": rate, "train_lift": lift,
            })
    return pd.DataFrame(out)


def rule_to_mask(rule_row: pd.Series, df: pd.DataFrame, bin_: Dict[str, pd.Series]) -> pd.Series:
    rtype = rule_row["type"]
    if rtype == "uni":
        feat = rule_row["feat"]
        direction = rule_row["direction"]
        cut = float(rule_row["cut"])
        s = pd.to_numeric(df[feat], errors="coerce")
        return (s >= cut) if direction == "ge" else (s <= cut)
    terms = [t.strip() for t in rule_row["rule"].split(" AND ")]
    mask = pd.Series(True, index=df.index)
    for t in terms:
        if t not in bin_:
            return pd.Series(False, index=df.index)
        mask &= bin_[t]
    return mask


def evaluate_on_test(rules: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.DataFrame:
    base = test[target_col].mean()
    _, th_test_hit, _, th_test_lift = TARGET_THRESHOLDS[target_col]
    bin_te = build_binaries(test)
    rows = []
    for _, r in rules.iterrows():
        try:
            mask = rule_to_mask(r, test, bin_te)
        except Exception:
            mask = pd.Series(False, index=test.index)
        n = int(mask.sum())
        if n < MIN_N_TEST:
            rec = r.to_dict()
            rec.update({
                "test_n": n,
                "test_hit": np.nan,
                "test_lift": np.nan,
                "test_mean_peak": np.nan,
                "passes_oos": False,
            })
            rows.append(rec)
            continue
        sub = test[mask]
        rate = sub[target_col].mean()
        lift = rate / base if base > 0 else np.nan
        rec = r.to_dict()
        rec.update({
            "test_n": n,
            "test_hit": rate,
            "test_lift": lift,
            "test_mean_peak": sub.get("peak_180d", pd.Series(dtype=float)).mean(),
            "passes_oos": (rate >= th_test_hit) and (lift >= th_test_lift),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fundamentals overlay (current Marcap from fdr — proxy)
# ---------------------------------------------------------------------------

def add_marcap(df: pd.DataFrame) -> pd.DataFrame:
    """Approx historical marcap at buy date = Close * Stocks (current shares outstanding).
    Stocks count is fairly stable over time, so this is a reasonable proxy and avoids
    the strong survivorship bias of using current Marcap (which inflates "winners").
    """
    try:
        import FinanceDataReader as fdr
        listing = fdr.StockListing("KRX")
        if "Stocks" not in listing.columns:
            return df
        m = listing[["Code", "Stocks"]].copy()
        df = df.merge(m, on="Code", how="left")
        df["Marcap_bn"] = df["Close"] * df["Stocks"] / 1e9
        return df
    except Exception as e:
        print("[warn] fdr marcap merge failed:", e)
        return df


# ---------------------------------------------------------------------------
# Per-target end-to-end
# ---------------------------------------------------------------------------

def analyze_target(train: pd.DataFrame, test: pd.DataFrame, target_col: str,
                    include_triples: bool = True) -> pd.DataFrame:
    print(f"\n=== {target_col} ===")
    base_tr = train[target_col].mean()
    base_te = test[target_col].mean()
    print(f"  base hit-rate  train={base_tr:.3f}  test={base_te:.3f}")
    uni = mine_univariate(train, target_col)
    combos = mine_combos(train, target_col, pairs=True, triples=include_triples)
    all_rules = pd.concat([uni, combos], ignore_index=True)
    print(f"  Train rules passing threshold: {len(all_rules)}")
    if all_rules.empty:
        return pd.DataFrame()
    evaluated = evaluate_on_test(all_rules, test, target_col)
    evaluated["target"] = target_col
    return evaluated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/5] Loading chart_feats...")
    df = pd.read_parquet(FEATS)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  rows={len(df)}  codes={df['Code'].nunique()}  dates={df['Date'].nunique()}")

    print("[2/5] Loading OHLCV...")
    with open(OHLCV, "rb") as f:
        ohlcv = pickle.load(f)
    print(f"  ohlcv tickers={len(ohlcv)}")

    print("[3/5] Computing peak returns (60/90/120/180d trading)...")
    peaks = compute_peaks(df, ohlcv)
    print(f"  peaks computed for {len(peaks)} (Date,Code) pairs")
    print(f"  NaN peak_180d: {peaks['peak_180d'].isna().sum()}")
    # Merge back
    df = df.merge(peaks, on=["Date", "Code"], how="left")

    # Build hit columns based on peak_180d (within 180 trading days)
    for tgt, thr in TARGETS.items():
        df[tgt] = (df["peak_180d"] >= thr).astype(int)

    # Drop rows where peak_180d is NaN (no future data)
    pre_n = len(df)
    df = df[df["peak_180d"].notna()].copy()
    print(f"  dropped {pre_n - len(df)} rows with missing peak_180d")

    # Add marcap
    print("[3.5] Adding marcap from fdr...")
    df = add_marcap(df)
    if "Marcap_bn" in df.columns:
        print(f"  marcap available for {df['Marcap_bn'].notna().sum()} / {len(df)} rows")

    # Split
    train = df[df["Date"] <= TRAIN_END].copy()
    test = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].copy()
    print(f"[4/5] Train rows: {len(train)} | Test rows: {len(test)}")

    # Print baseline hit rates
    print("\nBase hit rates (whole, train, test):")
    for tgt in TARGETS:
        print(f"  {tgt:>7}: whole={df[tgt].mean():.3f}  train={train[tgt].mean():.3f}  test={test[tgt].mean():.3f}")

    # Run per-target analysis
    print("\n[5/5] Mining patterns per target...")
    all_results = []
    for tgt in TARGETS:
        res = analyze_target(train, test, tgt, include_triples=True)
        if not res.empty:
            all_results.append(res)
    full = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if full.empty:
        print("[!] No patterns found.")
        return
    full = full.sort_values(["target", "test_hit", "test_n"], ascending=[True, False, False])
    full.to_csv(CACHE / "multi_target_results.csv", index=False)
    print(f"Saved {len(full)} rules -> cache/multi_target_results.csv")

    # Marcap analysis
    marcap_findings = analyze_marcap_effect(train, test)

    # Build markdown report
    write_report(full, train, test, marcap_findings)


def analyze_marcap_effect(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if "Marcap_bn" not in train.columns:
        return pd.DataFrame()
    out = []
    bins = [
        ("micro <= 100bn (1000억)",        lambda d: d["Marcap_bn"] <= 100),
        ("small 100~200bn",                lambda d: (d["Marcap_bn"] > 100) & (d["Marcap_bn"] <= 200)),
        ("small-mid 200~500bn",            lambda d: (d["Marcap_bn"] > 200) & (d["Marcap_bn"] <= 500)),
        ("mid 500~3000bn (3조)",            lambda d: (d["Marcap_bn"] > 500) & (d["Marcap_bn"] <= 3000)),
        ("large 3000~10000bn (10조)",      lambda d: (d["Marcap_bn"] > 3000) & (d["Marcap_bn"] <= 10000)),
        ("mega > 10000bn",                  lambda d: d["Marcap_bn"] > 10000),
    ]
    for tgt in TARGETS:
        for label, fn in bins:
            for split_name, d in [("train", train), ("test", test)]:
                m = fn(d)
                n = int(m.sum())
                if n < 30:
                    continue
                rate = d.loc[m, tgt].mean()
                mean_peak = d.loc[m, "peak_180d"].mean() if "peak_180d" in d.columns else np.nan
                out.append({
                    "target": tgt, "bin": label, "split": split_name,
                    "n": n, "hit_rate": rate, "mean_peak180": mean_peak,
                })
    return pd.DataFrame(out)


def write_report(full: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                 marcap_findings: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# 다중 수익 타깃 OOS 패턴 분석\n")
    lines.append("**peak_180d** = 매수일 종가 대비 180 영업일 동안의 최고 종가 수익률\n")
    lines.append(f"- Train: 2020-04-03 .. 2023-12-31 ({len(train)} rows)\n")
    lines.append(f"- Test : 2024-01-01 .. 2025-08-22 ({len(test)} rows)\n")
    lines.append("- 통과 기준 (타깃별 적응형):\n")
    for tgt, thr in TARGETS.items():
        th = TARGET_THRESHOLDS[tgt]
        lines.append(f"  - **{tgt}** (peak ≥ +{thr:g}%): "
                     f"Train hit ≥ {th[0]:.0%} & lift ≥ {th[2]}, "
                     f"Test hit ≥ {th[1]:.0%} & lift ≥ {th[3]}")
    lines.append(f"- Train n ≥ {MIN_N_TRAIN}, Test n ≥ {MIN_N_TEST}\n")
    lines.append("- 시가총액(`marcap*`)은 매수일 종가 × 현재 발행주식수로 근사한 **매수 시점 추정치**임 (생존편향 완화).\n")
    lines.append("\n## 기준 적중률 (base rates)\n")
    lines.append("| Target | Whole | Train | Test |\n|---|---|---|---|")
    whole = pd.concat([train, test])
    for tgt, thr in TARGETS.items():
        lines.append(f"| {tgt} (peak ≥ +{thr:g}%) | "
                     f"{whole[tgt].mean():.3f} | {train[tgt].mean():.3f} | {test[tgt].mean():.3f} |")
    lines.append("")

    # Per-target top patterns (only OOS-passing) + deduplicated
    for tgt, thr in TARGETS.items():
        all_pass = full[(full["target"] == tgt) & full["passes_oos"]].copy()
        # Sort by test_hit desc, then n
        all_pass = all_pass.sort_values(["test_hit", "test_n"], ascending=[False, False])
        lines.append(f"\n## 🎯 +{thr:g}% 도달 베스트 패턴 (OOS 통과: {len(all_pass)}개)\n")
        if all_pass.empty:
            lines.append("_OOS 검증을 통과한 패턴이 없음._\n")
            continue
        # Show top 15
        sub = all_pass.head(15)
        lines.append("| 패턴 | Type | Train n | Train hit | Train lift | Test n | Test hit | Test lift | Mean peak180 |\n"
                     "|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['rule']} | {r['type']} | "
                         f"{int(r['train_n'])} | {r['train_hit']:.3f} | {r['train_lift']:.2f} | "
                         f"{int(r['test_n'])} | {r['test_hit']:.3f} | {r['test_lift']:.2f} | "
                         f"{r['test_mean_peak']:.2f}% |")
        # Deduplicate by overlap: pick rules whose test_n is high AND test_hit is high
        # Greedy: pick top rule, drop rules whose phrase is contained or contains-it
        dedup = []
        seen = set()
        for _, r in all_pass.iterrows():
            terms = frozenset(t.strip() for t in str(r["rule"]).split(" AND "))
            redundant = False
            for s in seen:
                if terms.issubset(s) or s.issubset(terms):
                    redundant = True; break
            if redundant:
                continue
            dedup.append(r); seen.add(terms)
            if len(dedup) >= 8: break
        if dedup:
            lines.append(f"\n**핵심 비중복 패턴 (Top 8):**\n")
            lines.append("| 패턴 | Train n / hit | Test n / hit | Test lift | Mean peak180 |\n"
                         "|---|---|---|---:|---:|")
            for r in dedup:
                lines.append(f"| `{r['rule']}` | "
                             f"{int(r['train_n'])} / {r['train_hit']:.1%} | "
                             f"{int(r['test_n'])} / {r['test_hit']:.1%} | "
                             f"{r['test_lift']:.2f} | {r['test_mean_peak']:.1f}% |")

    # Marcap effect
    if not marcap_findings.empty:
        lines.append("\n## 📊 시가총액 효과 (매수 시점 추정치, 십억원)\n")
        lines.append("매수일 종가 × 현재 발행주식수로 추정. 절대값은 부정확하지만 상대 순서는 신뢰 가능.\n")
        for tgt, thr in TARGETS.items():
            lines.append(f"\n### +{thr:g}% 도달 (peak 기준)\n")
            lines.append("| Marcap bin | Split | N | Hit rate | Mean peak180 |\n|---|---|---:|---:|---:|")
            sub = marcap_findings[marcap_findings["target"] == tgt]
            for _, r in sub.iterrows():
                mp = f"{r['mean_peak180']:.1f}%" if pd.notna(r.get('mean_peak180')) else "-"
                lines.append(f"| {r['bin']} | {r['split']} | {int(r['n'])} | {r['hit_rate']:.3f} | {mp} |")

    # Final strategy recs
    lines.append("\n## 🏆 권장 실전 전략 (다양한 타깃별)\n")
    style_labels = {
        "hit_10": "안전형 (+10%) — 작은 익절·높은 빈도",
        "hit_20": "균형형 (+20%) — 적당한 익절·높은 적중",
        "hit_30": "추세형 (+30%) — 큰 익절·여전히 70%+",
        "hit_50": "폭발형 (+50%+) — 큰 익절·신중한 진입",
    }
    for tgt, thr in TARGETS.items():
        sub = full[(full["target"] == tgt) & full["passes_oos"]].copy()
        base_te = test[tgt].mean()
        lines.append(f"\n### {style_labels[tgt]}")
        lines.append(f"- **Base rate**: 테스트 구간에서 무작위로 매수 시 {base_te:.1%} 적중")
        if sub.empty:
            lines.append(f"- _OOS 통과 패턴 없음. base rate({base_te:.1%})를 유지._")
            continue
        # Pick best for each of two priorities: max test_hit (precision), max test_n with test_hit>=threshold (volume)
        best_precision = sub.sort_values(["test_hit", "test_n"], ascending=[False, False]).iloc[0]
        best_volume = sub.sort_values(["test_n", "test_hit"], ascending=[False, False]).iloc[0]
        lines.append(f"- **최고 적중률**: `{best_precision['rule']}`")
        lines.append(f"  - Train: n={int(best_precision['train_n'])}, hit={best_precision['train_hit']:.1%}")
        lines.append(f"  - Test : n={int(best_precision['test_n'])}, hit={best_precision['test_hit']:.1%} "
                     f"(base 대비 +{(best_precision['test_hit']-base_te)*100:.1f}pp), mean peak180={best_precision['test_mean_peak']:.1f}%")
        if best_volume["rule"] != best_precision["rule"]:
            lines.append(f"- **최대 빈도**: `{best_volume['rule']}`")
            lines.append(f"  - Train: n={int(best_volume['train_n'])}, hit={best_volume['train_hit']:.1%}")
            lines.append(f"  - Test : n={int(best_volume['test_n'])}, hit={best_volume['test_hit']:.1%} "
                         f"(base 대비 +{(best_volume['test_hit']-base_te)*100:.1f}pp), mean peak180={best_volume['test_mean_peak']:.1f}%")
    lines.append("\n## ⚠️ 해석 주의사항\n")
    lines.append("- `peak_180d`는 180 영업일 내 종가 기준 최고 수익률. 실전에서는 해당 종가에 매도해야 하므로, "
                 "실시간 추적/익절 룰이 필수.\n")
    lines.append("- chart_feats 데이터는 9개 프리셋(default/box_breakout/pullback 등) 시그널이 발생한 종목/일자만 포함. "
                 "따라서 '시그널 발생 종목 안에서 어떤 패턴이 추가 우위가 있는가'를 답함.\n")
    lines.append("- `marcap*` 빈은 매수일 시점 추정치 (현재 발행주식수 × 매수일 종가). "
                 "공모/분할/증자 등으로 인한 오차가 있을 수 있음.\n")
    lines.append("- PER/PBR/EPS 등 본격 펀더멘털은 pykrx KRX API 형식 변경으로 본 분석에서 미반영. "
                 "필요 시 별도 DART API 연동 필요.\n")
    lines.append("- 수급(외국인/기관 순매수)은 KIS API 호출량 한계로 본 분석에서 미반영.\n")

    out = BASE / "MULTI_TARGET_PATTERNS.md"
    out.write_text("\n".join(lines))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
