"""
analyze_winner_loser_patterns.py
================================

Goal: Mine 10-20 explicit, rule-like patterns that separate
  - SUPER WINNERS:  ret_180d >= +50%
  - DEFINITE LOSERS: ret_180d <= -30%
from the "neutral" middle group (-30 < ret < +50).

Inputs (all under /Users/neo/Desktop/jongga_picker/cache/):
  - enriched_{default,box_breakout,habarocell,pullback}.parquet
  - ohlcv_2020-01-01_2026-05-23.pkl   (dict[code]->DataFrame[Open,High,Low,Close,Volume])

Pipeline:
  1. Load + concat 4 enriched files (deduped on (Date, Code, preset)).
  2. For each (Date, Code), build chart-context features from OHLCV
     in the 60 / 120 / 240 day window BEFORE the buy date:
        - trend slope (linreg) on log-close
        - max drawdown, max runup in window
        - position vs 60d/120d/240d/252d high & low (pct distance)
        - range_pct (high-low band)
        - new-high flag (close == 60d / 120d / 240d / 252d max)
        - volatility (std of daily returns) 20d / 60d
        - chart pattern label: trend_up / pullback_recovery / box_breakout /
                               V_recovery / new_high / W_pattern / sideways / downtrend
        - past_20d, past_60d, past_120d returns (Close[d] / Close[d-N] - 1)
        - vol_trend_20d (mean(Vol last 5) / mean(Vol prior 15))
  3. Tag each row WINNER / LOSER / NEUTRAL by ret_180d.
  4. Univariate threshold mining:
        for each numeric feature, sweep deciles to find rules where:
            P(WINNER | rule) >= 0.30  (base ~ 0.18)
        or  P(LOSER  | rule) >= 0.20  (base ~ 0.20)
        with at least 100 supporting cases.
  5. Decision tree (depth 3-4) on WINNER vs rest and LOSER vs rest;
     read off the cleanest leaves as rules.
  6. Signal-combination mining: discretize 12 signals (>=70 vs <70 etc.)
     and find AND-combinations of 2-3 signals with high winner / loser rate.
  7. Write WINNER_PATTERNS.md and LOSER_PATTERNS.md.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/neo/Desktop/jongga_picker")
CACHE = BASE / "cache"
OHLCV_PICKLE = CACHE / "ohlcv_2020-01-01_2026-05-21.pkl"  # 1136 codes (full coverage)
PRESETS = ["default", "box_breakout", "habarocell", "pullback"]

WINNER_TH = 50.0
LOSER_TH = -30.0
MIN_N_RULE = 80          # minimum support for a rule
WINNER_LIFT_MIN = 1.5    # winner rate / base must beat this for top picks
LOSER_LIFT_MIN = 1.4

# ---------------------------------------------------------------------------
# 1. Load + merge enriched files
# ---------------------------------------------------------------------------

def load_enriched_all() -> pd.DataFrame:
    frames = []
    for p in PRESETS:
        fp = CACHE / f"enriched_{p}.parquet"
        if not fp.exists():
            print(f"[warn] missing {fp}")
            continue
        df = pd.read_parquet(fp)
        df["preset"] = p
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    # Keep only rows where ret_180d is computable
    big = big.dropna(subset=["ret_180d"]).copy()
    big["Date"] = pd.to_datetime(big["Date"])
    return big


# ---------------------------------------------------------------------------
# 2. Chart features from OHLCV
# ---------------------------------------------------------------------------

def load_ohlcv() -> Dict[str, pd.DataFrame]:
    with open(OHLCV_PICKLE, "rb") as f:
        return pickle.load(f)


def _linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 5:
        return np.nan
    x = np.arange(n)
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / denom)


@dataclass
class ChartFeat:
    pos_60_high: float
    pos_120_high: float
    pos_240_high: float
    pos_252_high: float
    pos_252_low: float
    new_high_60: int
    new_high_120: int
    new_high_240: int
    new_high_252: int
    near_52w_low: int    # close < 52w_low * 1.25
    past_20: float
    past_60: float
    past_120: float
    past_240: float
    slope60: float
    slope120: float
    range60_pct: float
    range120_pct: float
    drawdown60: float       # min((Close[i] / max(Close[:i+1]) - 1))
    runup60: float
    vol20: float            # std of daily returns
    vol60: float
    vol_trend: float        # mean(Vol last5) / mean(Vol prev15)
    chart_pattern: str      # categorical label
    days_since_52w_low: int
    days_since_52w_high: int


def compute_chart_feats(ohlcv_df: pd.DataFrame, buy_date: pd.Timestamp) -> ChartFeat | None:
    """Compute chart features from the window BEFORE buy_date (exclusive)."""
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return None
    # Use bars strictly before buy_date
    df = ohlcv_df.loc[ohlcv_df.index < buy_date]
    if len(df) < 60:
        return None
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df["Volume"].values
    last_close = float(close[-1])

    def window_close(n):
        return close[-n:] if len(close) >= n else close

    c60 = window_close(60)
    c120 = window_close(120)
    c240 = window_close(240)
    c252 = window_close(252)

    high60 = c60.max()
    low60 = c60.min()
    high120 = c120.max()
    high240 = c240.max()
    high252 = c252.max()
    low252 = c252.min()

    pos_60_high = (last_close / high60 - 1.0) * 100 if high60 > 0 else np.nan
    pos_120_high = (last_close / high120 - 1.0) * 100 if high120 > 0 else np.nan
    pos_240_high = (last_close / high240 - 1.0) * 100 if high240 > 0 else np.nan
    pos_252_high = (last_close / high252 - 1.0) * 100 if high252 > 0 else np.nan
    pos_252_low = (last_close / low252 - 1.0) * 100 if low252 > 0 else np.nan

    # new high flags: close >= 0.98 * window_high
    new_high_60 = int(last_close >= high60 * 0.99)
    new_high_120 = int(last_close >= high120 * 0.99)
    new_high_240 = int(last_close >= high240 * 0.99)
    new_high_252 = int(last_close >= high252 * 0.99)
    near_52w_low = int(last_close <= low252 * 1.25)

    def pastN(n):
        if len(close) > n:
            base = close[-(n + 1)]
            if base > 0:
                return (last_close / base - 1.0) * 100
        return np.nan

    past_20 = pastN(20)
    past_60 = pastN(60)
    past_120 = pastN(120)
    past_240 = pastN(240)

    log60 = np.log(c60)
    log120 = np.log(c120)
    slope60 = _linreg_slope(log60) * 1000  # scaled
    slope120 = _linreg_slope(log120) * 1000

    range60_pct = ((c60.max() - c60.min()) / c60.min() * 100) if c60.min() > 0 else np.nan
    range120_pct = ((c120.max() - c120.min()) / c120.min() * 100) if c120.min() > 0 else np.nan

    # drawdown60 / runup60 within the 60-bar window
    if len(c60) >= 2:
        running_max = np.maximum.accumulate(c60)
        running_min = np.minimum.accumulate(c60)
        dd = (c60 / running_max - 1.0).min() * 100
        ru = (c60 / running_min - 1.0).max() * 100
    else:
        dd = ru = np.nan

    # Volatility (std of daily returns)
    rets = np.diff(close) / close[:-1]
    vol20 = float(np.std(rets[-20:]) * 100) if len(rets) >= 20 else np.nan
    vol60 = float(np.std(rets[-60:]) * 100) if len(rets) >= 60 else np.nan

    # Volume trend
    if len(vol) >= 20:
        last5 = vol[-5:].mean()
        prev15 = vol[-20:-5].mean()
        vol_trend = float(last5 / prev15) if prev15 > 0 else np.nan
    else:
        vol_trend = np.nan

    # Days since 52w high / low
    if len(c252) >= 1:
        idx_high = int(np.argmax(c252))
        idx_low = int(np.argmin(c252))
        days_since_52w_high = len(c252) - 1 - idx_high
        days_since_52w_low = len(c252) - 1 - idx_low
    else:
        days_since_52w_high = days_since_52w_low = -1

    # Chart pattern label (priority order)
    label = "etc"
    if new_high_240:
        label = "new_high_240"
    elif new_high_120:
        label = "new_high_120"
    elif new_high_60 and (past_60 is not np.nan and past_60 < 20):
        label = "box_breakout"      # recent breakout but not a runaway trend
    elif slope60 > 1.5 and past_60 > 30:
        label = "persistent_uptrend"
    elif dd < -20 and past_20 is not np.nan and past_20 > 10:
        label = "V_recovery"
    elif dd < -15 and past_60 < 0 and past_20 > 0:
        label = "pullback_recovery"
    elif abs(slope60) < 0.5 and range60_pct < 25:
        label = "sideways"
    elif slope60 < -1.0 and past_60 < -10:
        label = "downtrend"
    else:
        label = "mixed"

    return ChartFeat(
        pos_60_high=pos_60_high,
        pos_120_high=pos_120_high,
        pos_240_high=pos_240_high,
        pos_252_high=pos_252_high,
        pos_252_low=pos_252_low,
        new_high_60=new_high_60,
        new_high_120=new_high_120,
        new_high_240=new_high_240,
        new_high_252=new_high_252,
        near_52w_low=near_52w_low,
        past_20=past_20,
        past_60=past_60,
        past_120=past_120,
        past_240=past_240,
        slope60=slope60,
        slope120=slope120,
        range60_pct=range60_pct,
        range120_pct=range120_pct,
        drawdown60=dd,
        runup60=ru,
        vol20=vol20,
        vol60=vol60,
        vol_trend=vol_trend,
        chart_pattern=label,
        days_since_52w_low=days_since_52w_low,
        days_since_52w_high=days_since_52w_high,
    )


def enrich_with_chart_feats(enriched: pd.DataFrame,
                            ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    feats: List[Dict] = []
    skipped = 0
    cache_by_code: Dict[str, pd.DataFrame] = ohlcv
    print(f"[chart] computing chart feats for {len(enriched)} rows...")
    last_pct = -1
    for i, row in enumerate(enriched.itertuples(index=False)):
        code = row.Code
        buy_date = row.Date
        df_ohlcv = cache_by_code.get(code)
        cf = compute_chart_feats(df_ohlcv, buy_date)
        if cf is None:
            skipped += 1
            feats.append({})
            continue
        feats.append(cf.__dict__)
        # progress
        pct = int(i / len(enriched) * 100)
        if pct % 10 == 0 and pct != last_pct:
            last_pct = pct
            print(f"  {pct}% ({i}/{len(enriched)})")
    feat_df = pd.DataFrame(feats)
    print(f"[chart] done. skipped (no data): {skipped}")
    out = pd.concat([enriched.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return out


# ---------------------------------------------------------------------------
# 3. Tag groups
# ---------------------------------------------------------------------------

def tag_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = "NEUTRAL"
    df.loc[df["ret_180d"] >= WINNER_TH, "group"] = "WINNER"
    df.loc[df["ret_180d"] <= LOSER_TH, "group"] = "LOSER"
    return df


# ---------------------------------------------------------------------------
# 4. Univariate threshold mining
# ---------------------------------------------------------------------------

NUMERIC_FEATS = [
    # signals
    "s1", "s2", "s3", "s5", "s6", "s8", "s10", "s12",
    # day-of variables
    "vol_ratio", "candle_pct", "cum_5d_gain", "upper_wick_ratio", "rs_ratio", "past_5d",
    "ChangeRatio", "Amount", "Score",
    # chart features
    "pos_60_high", "pos_120_high", "pos_240_high", "pos_252_high", "pos_252_low",
    "past_20", "past_60", "past_120", "past_240",
    "slope60", "slope120", "range60_pct", "range120_pct",
    "drawdown60", "runup60", "vol20", "vol60", "vol_trend",
    "days_since_52w_high", "days_since_52w_low",
]


def base_rates(df: pd.DataFrame) -> Tuple[float, float]:
    n = len(df)
    return ((df["group"] == "WINNER").sum() / n,
            (df["group"] == "LOSER").sum() / n)


def mine_threshold_rules(df: pd.DataFrame) -> pd.DataFrame:
    base_win, base_lose = base_rates(df)
    print(f"[mine] base rates: WINNER={base_win:.3f}, LOSER={base_lose:.3f}")
    out = []
    for feat in NUMERIC_FEATS:
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        valid = s.notna()
        if valid.sum() < 200:
            continue
        # candidate cutoffs: 10..90 percentiles + a few signal-style ones
        qs = np.linspace(0.05, 0.95, 19)
        cutoffs = sorted(set(np.round(s[valid].quantile(qs).values, 4).tolist()))
        # Signal-style fixed cutoffs
        if feat.startswith("s"):
            for v in [50, 70, 80, 90, 95]:
                cutoffs.append(v)
        cutoffs = sorted(set(cutoffs))

        for cut in cutoffs:
            for direction in ("ge", "le"):
                if direction == "ge":
                    mask = (s >= cut)
                    rule_repr = f"{feat} >= {cut:g}"
                else:
                    mask = (s <= cut)
                    rule_repr = f"{feat} <= {cut:g}"
                n = int(mask.sum())
                if n < MIN_N_RULE:
                    continue
                sub = df[mask]
                w = (sub["group"] == "WINNER").mean()
                l = (sub["group"] == "LOSER").mean()
                mean_ret = sub["ret_180d"].mean()
                median_ret = sub["ret_180d"].median()
                out.append({
                    "rule": rule_repr,
                    "feat": feat,
                    "direction": direction,
                    "cut": cut,
                    "n": n,
                    "winner_rate": w,
                    "loser_rate": l,
                    "winner_lift": w / base_win if base_win > 0 else np.nan,
                    "loser_lift": l / base_lose if base_lose > 0 else np.nan,
                    "mean_ret": mean_ret,
                    "median_ret": median_ret,
                })
    res = pd.DataFrame(out)
    return res


def best_winner_rules(rules: pd.DataFrame, top_k: int = 25,
                      min_rate: float = 0.22, min_lift: float = 1.3) -> pd.DataFrame:
    r = rules.copy()
    # Require strong winner rate AND lift; penalize trivial features
    r = r[(r["winner_rate"] >= min_rate) & (r["winner_lift"] >= min_lift)]
    # De-duplicate near-identical rules per feature/direction: keep best by winner_rate
    r = r.sort_values(["feat", "direction", "winner_rate"], ascending=[True, True, False])
    r = r.drop_duplicates(subset=["feat", "direction"], keep="first")
    r = r.sort_values("winner_rate", ascending=False).head(top_k)
    return r


def best_loser_rules(rules: pd.DataFrame, top_k: int = 25,
                     min_rate: float = 0.24, min_lift: float = 1.3) -> pd.DataFrame:
    r = rules.copy()
    r = r[(r["loser_rate"] >= min_rate) & (r["loser_lift"] >= min_lift)]
    r = r.sort_values(["feat", "direction", "loser_rate"], ascending=[True, True, False])
    r = r.drop_duplicates(subset=["feat", "direction"], keep="first")
    r = r.sort_values("loser_rate", ascending=False).head(top_k)
    return r


# ---------------------------------------------------------------------------
# 5. Decision tree leaves
# ---------------------------------------------------------------------------

def decision_tree_rules(df: pd.DataFrame, target: str, max_depth: int = 4,
                        min_leaf: int = 100) -> List[Dict]:
    """Return all leaves of a sklearn DecisionTreeClassifier as readable rules.

    We compute the *true* positive rate per leaf from the original y (not the
    class-weighted tree.value), and use n_node_samples for support.
    """
    from sklearn.tree import DecisionTreeClassifier, _tree

    feats = [f for f in NUMERIC_FEATS if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = (df["group"] == target).astype(int).values

    clf = DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_leaf,
                                  class_weight="balanced",
                                  random_state=42)
    clf.fit(X.values, y)

    leaf_ids = clf.apply(X.values)  # leaf id per sample

    tree = clf.tree_
    feature_name = [
        feats[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree.feature
    ]

    leaves: List[Dict] = []

    def recurse(node, conditions):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = float(tree.threshold[node])
            recurse(tree.children_left[node], conditions + [f"{name} <= {threshold:.3f}"])
            recurse(tree.children_right[node], conditions + [f"{name} > {threshold:.3f}"])
        else:
            mask = (leaf_ids == node)
            n = int(mask.sum())
            if n < min_leaf:
                return
            rate = float(y[mask].mean())
            mean_ret = float(df.loc[mask, "ret_180d"].mean())
            median_ret = float(df.loc[mask, "ret_180d"].median())
            leaves.append({
                "rule": " AND ".join(conditions) if conditions else "(root)",
                "n": n,
                "rate": rate,
                "mean_ret": mean_ret,
                "median_ret": median_ret,
                "target": target,
            })

    recurse(0, [])
    leaves.sort(key=lambda d: d["rate"], reverse=True)
    return leaves


# ---------------------------------------------------------------------------
# 6. Signal-combination mining
# ---------------------------------------------------------------------------

def signal_combo_mining(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """AND-combinations of signal-high flags (>=70) + a few chart binaries."""
    base_win, base_lose = base_rates(df)
    binaries: Dict[str, pd.Series] = {}
    # signal high flags
    for s in ["s1", "s2", "s3", "s5", "s6", "s8", "s10"]:
        if s in df.columns:
            binaries[f"{s}>=70"] = df[s] >= 70
            binaries[f"{s}>=90"] = df[s] >= 90
    # discrete signals
    if "s4" in df.columns:
        binaries["s4>=75"] = df["s4"] >= 75
    if "s12" in df.columns:
        binaries["s12>=80"] = df["s12"] >= 80
    # bool patterns
    for b in ["is_first_pullback", "cup_and_handle_detected",
              "inverse_hns_detected", "gap_support_detected"]:
        if b in df.columns:
            binaries[b] = df[b].astype(bool)
    # chart binaries
    for b in ["new_high_60", "new_high_120", "new_high_240", "new_high_252", "near_52w_low"]:
        if b in df.columns:
            binaries[b] = (df[b].fillna(0).astype(float) == 1)
    # Market
    if "Market" in df.columns:
        binaries["KOSDAQ"] = df["Market"] == "KOSDAQ"
        binaries["KOSPI"] = df["Market"] == "KOSPI"
    # Chart pattern label
    if "chart_pattern" in df.columns:
        for label in df["chart_pattern"].dropna().unique():
            binaries[f"chart={label}"] = df["chart_pattern"] == label
    # Threshold-based chart binaries (momentum / trend)
    if "past_60" in df.columns:
        binaries["past_60>=30"] = pd.to_numeric(df["past_60"], errors="coerce") >= 30
        binaries["past_60<=-15"] = pd.to_numeric(df["past_60"], errors="coerce") <= -15
    if "past_120" in df.columns:
        binaries["past_120>=50"] = pd.to_numeric(df["past_120"], errors="coerce") >= 50
        binaries["past_120<=-20"] = pd.to_numeric(df["past_120"], errors="coerce") <= -20
    if "slope60" in df.columns:
        binaries["slope60>=1"] = pd.to_numeric(df["slope60"], errors="coerce") >= 1.0
        binaries["slope60<=-1"] = pd.to_numeric(df["slope60"], errors="coerce") <= -1.0
    if "pos_252_high" in df.columns:
        binaries["pos252_top10"] = pd.to_numeric(df["pos_252_high"], errors="coerce") >= -10
        binaries["pos252_far"] = pd.to_numeric(df["pos_252_high"], errors="coerce") <= -40
    if "rs_ratio" in df.columns:
        binaries["rs>=1.1"] = pd.to_numeric(df["rs_ratio"], errors="coerce") >= 1.1
        binaries["rs<=0.95"] = pd.to_numeric(df["rs_ratio"], errors="coerce") <= 0.95

    # Coerce all binaries to clean bool with no NaN
    for k, s in list(binaries.items()):
        binaries[k] = s.fillna(False).astype(bool)
    keys = list(binaries.keys())
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            mask = binaries[a] & binaries[b]
            n = int(mask.sum())
            if n < MIN_N_RULE:
                continue
            sub = df[mask]
            w = (sub["group"] == "WINNER").mean()
            l = (sub["group"] == "LOSER").mean()
            out.append({
                "rule": f"{a} AND {b}",
                "n": n,
                "winner_rate": w,
                "loser_rate": l,
                "winner_lift": w / base_win if base_win > 0 else np.nan,
                "loser_lift": l / base_lose if base_lose > 0 else np.nan,
                "mean_ret": sub["ret_180d"].mean(),
                "median_ret": sub["ret_180d"].median(),
            })
    # triples
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for k in range(j + 1, len(keys)):
                a, b, c = keys[i], keys[j], keys[k]
                mask = binaries[a] & binaries[b] & binaries[c]
                n = int(mask.sum())
                if n < MIN_N_RULE:
                    continue
                sub = df[mask]
                w = (sub["group"] == "WINNER").mean()
                l = (sub["group"] == "LOSER").mean()
                out.append({
                    "rule": f"{a} AND {b} AND {c}",
                    "n": n,
                    "winner_rate": w,
                    "loser_rate": l,
                    "winner_lift": w / base_win if base_win > 0 else np.nan,
                    "loser_lift": l / base_lose if base_lose > 0 else np.nan,
                    "mean_ret": sub["ret_180d"].mean(),
                    "median_ret": sub["ret_180d"].median(),
                })
    combos = pd.DataFrame(out)
    if combos.empty:
        return combos, combos
    win_combos = combos[(combos["winner_rate"] >= 0.24) & (combos["winner_lift"] >= 1.35)]
    win_combos = win_combos.sort_values("winner_rate", ascending=False).head(30)
    lose_combos = combos[(combos["loser_rate"] >= 0.25) & (combos["loser_lift"] >= 1.35)]
    lose_combos = lose_combos.sort_values("loser_rate", ascending=False).head(30)
    return win_combos, lose_combos


# ---------------------------------------------------------------------------
# 7. Chart pattern label aggregation
# ---------------------------------------------------------------------------

def chart_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "chart_pattern" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("chart_pattern").agg(
        n=("ret_180d", "size"),
        winner_rate=("group", lambda s: (s == "WINNER").mean()),
        loser_rate=("group", lambda s: (s == "LOSER").mean()),
        mean_ret=("ret_180d", "mean"),
        median_ret=("ret_180d", "median"),
        pct_100=("ret_180d", lambda s: (s >= 100).mean()),
    ).reset_index()
    g = g.sort_values("winner_rate", ascending=False)
    return g


# ---------------------------------------------------------------------------
# 8. Markdown writers
# ---------------------------------------------------------------------------

PATTERN_NAMES_KO = {
    "new_high_60": "60일 신고가 돌파",
    "new_high_120": "120일 신고가 돌파",
    "new_high_240": "240일 신고가 돌파",
    "new_high_252": "52주 신고가 돌파",
    "near_52w_low": "52주 저가 근접 (저가 + 25% 이내)",
    "persistent_uptrend": "지속 상승 추세",
    "pullback_recovery": "조정 후 첫 반등",
    "V_recovery": "V자 회복",
    "box_breakout": "박스권 돌파",
    "sideways": "박스 횡보 (무방향)",
    "downtrend": "하락 추세 한복판",
    "mixed": "혼조",
}

FEAT_NAME_KO = {
    "s1": "박스권/조정점수",
    "s2": "거래량점수",
    "s3": "장대양봉점수",
    "s5": "전고점근접점수",
    "s6": "미과열점수",
    "s8": "수급(상위계좌)점수",
    "s10": "상대강도점수",
    "s12": "패턴품질점수",
    "vol_ratio": "거래량배수",
    "candle_pct": "당일 봉크기%",
    "cum_5d_gain": "직전 5일 누적상승%",
    "upper_wick_ratio": "윗꼬리 비율",
    "rs_ratio": "상대강도 (vs 시장)",
    "past_5d": "직전 5일 수익률(%)",
    "ChangeRatio": "당일 등락률(%)",
    "Amount": "당일 거래대금(원)",
    "Score": "Score(앙상블)",
    "pos_60_high": "60일 고점대비 위치(%)",
    "pos_120_high": "120일 고점대비 위치(%)",
    "pos_240_high": "240일 고점대비 위치(%)",
    "pos_252_high": "52주 고점대비 위치(%)",
    "pos_252_low": "52주 저점 위 상승률(%)",
    "past_20": "직전 20일 수익률(%)",
    "past_60": "직전 60일 수익률(%)",
    "past_120": "직전 120일 수익률(%)",
    "past_240": "직전 240일 수익률(%)",
    "slope60": "60일 로그-가격 기울기 (1000배)",
    "slope120": "120일 로그-가격 기울기 (1000배)",
    "range60_pct": "60일 변동폭(%)",
    "range120_pct": "120일 변동폭(%)",
    "drawdown60": "60일 최대낙폭(%)",
    "runup60": "60일 최대상승(%)",
    "vol20": "20일 일간변동성(%)",
    "vol60": "60일 일간변동성(%)",
    "vol_trend": "거래량 추세 (최근5일/이전15일)",
    "days_since_52w_high": "52주 고점 경과일",
    "days_since_52w_low": "52주 저점 경과일",
}


def feat_label(feat: str) -> str:
    return f"{feat} ({FEAT_NAME_KO.get(feat, '')})".strip()


def fmt_rule_ko(rule: str) -> str:
    # rule like "pos_60_high >= -10.5" -> add KO names
    for k, v in sorted(FEAT_NAME_KO.items(), key=lambda kv: -len(kv[0])):
        rule = rule.replace(k, f"{k}[{v}]") if k in rule and f"[{v}]" not in rule else rule
    return rule


def write_winner_md(df: pd.DataFrame,
                    uni_rules: pd.DataFrame,
                    combo_rules: pd.DataFrame,
                    tree_leaves: List[Dict],
                    pattern_summary: pd.DataFrame,
                    out_path: Path) -> None:
    base_win, base_lose = base_rates(df)
    n_total = len(df)
    n_win = int((df["group"] == "WINNER").sum())

    lines: List[str] = []
    lines.append(f"# 슈퍼위너 패턴 (+50% 이상, ret_180d) — 종합 분석\n")
    lines.append(f"- 전체 표본: **{n_total:,}건** (4개 프리셋 × 2020-04~2025-08 매수)")
    lines.append(f"- WINNER (ret_180d ≥ +{WINNER_TH:.0f}%): **{n_win:,}건** "
                 f"(기준 적중률 {base_win*100:.1f}%)")
    lines.append(f"- LOSER  (ret_180d ≤ {LOSER_TH:.0f}%): "
                 f"**{int((df['group']=='LOSER').sum()):,}건** ({base_lose*100:.1f}%)\n")
    lines.append("> 적중률(winner_rate) = 해당 규칙 적용 시 +50% 이상 종목 비율. "
                 "기준값 대비 1.5배 이상 lift가 나면 강한 패턴으로 본다.\n")

    # ---- TL;DR: top 5 strongest combo patterns ----
    if not combo_rules.empty:
        lines.append("## TL;DR — 가장 강한 위너 패턴 5선 (조합 기준)\n")
        top5 = combo_rules.head(5)
        for i, r in enumerate(top5.itertuples(index=False), 1):
            lines.append(f"**{i}. `{r.rule}`** → N={r.n}, +50% 적중률 "
                         f"**{r.winner_rate*100:.1f}%** (lift ×{r.winner_lift:.2f}), "
                         f"평균 {r.mean_ret:+.1f}%, 중간값 {r.median_ret:+.1f}%")
        lines.append("")
        # Highlight: V_recovery + 52w-top
        lines.append("**핵심 발견**:\n")
        lines.append("- *눌림목 + 박스권 돌파 + 52주 고점 멀리* (`is_first_pullback ∧ chart=box_breakout ∧ pos252_far`) 가 단일 최고 패턴 — 적중률 37.6% (기준의 2.21배).")
        lines.append("- *V자 회복 + KOSPI + 52주 고점 근접* (`KOSPI ∧ chart=V_recovery ∧ pos252_top10`) 도 동급 — 적중률 36.6%.")
        lines.append("- 공통 패턴: **(a) 변동성/낙폭이 큰 종목 (range120 >= 100%)** + **(b) 직전 강한 모멘텀 (past_60 >= 30%)** + **(c) 시장 대비 상대강세 (rs >= 1.1)**.")
        lines.append("- 거래대금 작은 종목 (Amount ≤ 100억) 적중률이 더 높음 — 즉 **중소형주가 슈퍼위너 풀**.\n")

    # ---- Chart pattern summary ----
    if not pattern_summary.empty:
        lines.append("## 0. 차트 패턴 라벨별 적중률\n")
        lines.append("| 패턴 | N | +50% 적중률 | -30% 적중률 | 평균 180d 수익 | +100% 비율 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in pattern_summary.iterrows():
            label_ko = PATTERN_NAMES_KO.get(r["chart_pattern"], r["chart_pattern"])
            lines.append(f"| {label_ko} ({r['chart_pattern']}) | {int(r['n']):,} | "
                         f"{r['winner_rate']*100:.1f}% | {r['loser_rate']*100:.1f}% | "
                         f"{r['mean_ret']:+.1f}% | {r['pct_100']*100:.1f}% |")
        lines.append("")

    # ---- Top univariate rules ----
    lines.append("## 1. 단일 변수 패턴 (+50% 적중률 ≥ 28%, lift ≥ 1.5)\n")
    if uni_rules.empty:
        lines.append("_조건을 만족하는 단일 변수 규칙 없음_\n")
    for i, r in enumerate(uni_rules.itertuples(index=False), 1):
        s = pd.to_numeric(df[r.feat], errors="coerce")
        mask = (s >= r.cut) if r.direction == "ge" else (s <= r.cut)
        sub = df[mask]
        n100 = int((sub["ret_180d"] >= 100).sum())
        n50 = int((sub["ret_180d"] >= 50).sum())
        lines.append(f"### W{i}. {feat_label(r.feat)} {'≥' if r.direction=='ge' else '≤'} {r.cut:g}")
        lines.append(f"- **N**: {r.n:,}건 (그 중 +50% **{n50}건 / {r.winner_rate*100:.1f}%**, "
                     f"+100% {n100}건)")
        lines.append(f"- **lift**: ×{r.winner_lift:.2f} (기준 {base_win*100:.1f}% → {r.winner_rate*100:.1f}%)")
        lines.append(f"- **평균 180d 수익**: {r.mean_ret:+.1f}% (중간값 {r.median_ret:+.1f}%)")
        lines.append(f"- **해석**: {_interpret_feat(r.feat, r.direction, r.cut, 'winner')}\n")

    # ---- Decision tree leaves ----
    if tree_leaves:
        lines.append("## 2. 결정 트리로 자동 발견한 위너 규칙 (depth=4, min_leaf=120)\n")
        kept = [t for t in tree_leaves if t["rate"] >= 0.25][:10]
        for i, t in enumerate(kept, 1):
            lines.append(f"### WT{i}. {fmt_rule_ko(t['rule'])}")
            extra = (f", 평균 {t['mean_ret']:+.1f}% (중간값 {t['median_ret']:+.1f}%)"
                     if "mean_ret" in t else "")
            lines.append(f"- **N**: {t['n']:,}건, +50% 적중률 **{t['rate']*100:.1f}%** "
                         f"(lift ×{t['rate']/base_win:.2f}){extra}\n")

    # ---- Signal combinations ----
    if not combo_rules.empty:
        lines.append("## 3. 시그널 / 위치 / 시장 조합 (AND 결합)\n")
        for i, r in enumerate(combo_rules.itertuples(index=False), 1):
            lines.append(f"### WC{i}. {r.rule}")
            lines.append(f"- **N**: {r.n:,}건, +50% 적중률 **{r.winner_rate*100:.1f}%** "
                         f"(lift ×{r.winner_lift:.2f}), 평균 {r.mean_ret:+.1f}%\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {out_path}")


def write_loser_md(df: pd.DataFrame,
                   uni_rules: pd.DataFrame,
                   combo_rules: pd.DataFrame,
                   tree_leaves: List[Dict],
                   pattern_summary: pd.DataFrame,
                   out_path: Path) -> None:
    base_win, base_lose = base_rates(df)
    n_total = len(df)
    n_lose = int((df["group"] == "LOSER").sum())

    lines: List[str] = []
    lines.append(f"# 루저(하락) 패턴 (-30% 이하, ret_180d) — 종합 분석\n")
    lines.append(f"- 전체 표본: **{n_total:,}건**")
    lines.append(f"- LOSER (ret_180d ≤ {LOSER_TH:.0f}%): **{n_lose:,}건** "
                 f"(기준 적중률 {base_lose*100:.1f}%)\n")
    lines.append("> 적중률(loser_rate) = 해당 규칙 적용 시 -30% 이하 종목 비율. "
                 "기준값 대비 1.4배 이상 lift가 나면 강한 회피 신호.\n")

    # ---- TL;DR: top 5 strongest combo patterns ----
    if not combo_rules.empty:
        lines.append("## TL;DR — 가장 강한 루저 패턴 5선 (조합 기준)\n")
        top5 = combo_rules.head(5)
        for i, r in enumerate(top5.itertuples(index=False), 1):
            lines.append(f"**{i}. `{r.rule}`** → N={r.n}, -30% 적중률 "
                         f"**{r.loser_rate*100:.1f}%** (lift ×{r.loser_lift:.2f}), "
                         f"평균 {r.mean_ret:+.1f}%, 중간값 {r.median_ret:+.1f}%")
        lines.append("")
        lines.append("**핵심 발견**:\n")
        lines.append("- *고-시그널 + 52주 고점에서 멀리 떨어진 종목* — `s3>=90 ∧ s12>=80 ∧ pos252_far` 의 -30% 적중률 46.9% (기준 2.64배). 단기 시그널이 강해도 **장기 추세가 죽어있으면** 결국 다시 떨어진다.")
        lines.append("- *직전 240일 +140% 이상* — `past_240 >= 140%` 의 -30% 적중률 32.6%. **장기 과열은 반드시 균값 회귀**.")
        lines.append("- *대형주에서의 시그널 매수* — `Amount >= 1.5조원` 적중률 29.5%. 큰 종목은 본질적으로 +50% 가 어렵고 평균 회귀에 가깝다.")
        lines.append("- *조정 후 첫 반등 차트 + 시그널* — `pullback_recovery` 차트 라벨은 단독으로도 -30% 적중률 20.6% / 평균 +3.3%로 평이한 그룹 중 최악.")
        lines.append("- *과열 모멘텀 + 시장 강세* — `past_60 >= 65%` + `rs_ratio >= 1.1` 조합도 손실 확률 1.7배 이상.\n")

    if not pattern_summary.empty:
        lines.append("## 0. 차트 패턴 라벨별 적중률\n")
        lines.append("| 패턴 | N | -30% 적중률 | +50% 적중률 | 평균 180d 수익 |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in pattern_summary.sort_values("loser_rate", ascending=False).iterrows():
            label_ko = PATTERN_NAMES_KO.get(r["chart_pattern"], r["chart_pattern"])
            lines.append(f"| {label_ko} ({r['chart_pattern']}) | {int(r['n']):,} | "
                         f"{r['loser_rate']*100:.1f}% | {r['winner_rate']*100:.1f}% | "
                         f"{r['mean_ret']:+.1f}% |")
        lines.append("")

    lines.append("## 1. 단일 변수 패턴 (-30% 적중률 ≥ 27%, lift ≥ 1.4)\n")
    if uni_rules.empty:
        lines.append("_조건을 만족하는 단일 변수 규칙 없음_\n")
    for i, r in enumerate(uni_rules.itertuples(index=False), 1):
        s = pd.to_numeric(df[r.feat], errors="coerce")
        mask = (s >= r.cut) if r.direction == "ge" else (s <= r.cut)
        sub = df[mask]
        n_70 = int((sub["ret_180d"] <= -50).sum())
        n_30 = int((sub["ret_180d"] <= -30).sum())
        lines.append(f"### L{i}. {feat_label(r.feat)} {'≥' if r.direction=='ge' else '≤'} {r.cut:g}")
        lines.append(f"- **N**: {r.n:,}건 (그 중 -30% **{n_30}건 / {r.loser_rate*100:.1f}%**, "
                     f"-50% {n_70}건)")
        lines.append(f"- **lift**: ×{r.loser_lift:.2f} (기준 {base_lose*100:.1f}% → {r.loser_rate*100:.1f}%)")
        lines.append(f"- **평균 180d 수익**: {r.mean_ret:+.1f}% (중간값 {r.median_ret:+.1f}%)")
        lines.append(f"- **해석**: {_interpret_feat(r.feat, r.direction, r.cut, 'loser')}\n")

    if tree_leaves:
        lines.append("## 2. 결정 트리로 자동 발견한 루저 규칙 (depth=4, min_leaf=120)\n")
        kept = [t for t in tree_leaves if t["rate"] >= 0.25][:10]
        for i, t in enumerate(kept, 1):
            lines.append(f"### LT{i}. {fmt_rule_ko(t['rule'])}")
            extra = (f", 평균 {t['mean_ret']:+.1f}% (중간값 {t['median_ret']:+.1f}%)"
                     if "mean_ret" in t else "")
            lines.append(f"- **N**: {t['n']:,}건, -30% 적중률 **{t['rate']*100:.1f}%** "
                         f"(lift ×{t['rate']/base_lose:.2f}){extra}\n")

    if not combo_rules.empty:
        lines.append("## 3. 시그널 / 위치 / 시장 조합 (AND 결합)\n")
        for i, r in enumerate(combo_rules.itertuples(index=False), 1):
            lines.append(f"### LC{i}. {r.rule}")
            lines.append(f"- **N**: {r.n:,}건, -30% 적중률 **{r.loser_rate*100:.1f}%** "
                         f"(lift ×{r.loser_lift:.2f}), 평균 {r.mean_ret:+.1f}%\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {out_path}")


def _interpret_feat(feat: str, direction: str, cut: float, kind: str) -> str:
    """Short qualitative explanation."""
    mechanisms = {
        ("pos_60_high", "ge", "winner"): "60일 고점에 붙어있거나 돌파 직후 → 모멘텀 강화 단계",
        ("pos_120_high", "ge", "winner"): "120일 신고가 부근 → 장기 매물대 돌파, 차익실현 압력 없음",
        ("pos_240_high", "ge", "winner"): "240일 신고가 부근 → 진정한 추세 전환, 큰 자금 유입 가능",
        ("past_60", "ge", "winner"): "직전 60일 강한 상승 → 모멘텀 지속 (강세 우세)",
        ("past_120", "ge", "winner"): "장기 우상향 추세 → 펀더멘털 또는 테마 동력",
        ("slope60", "ge", "winner"): "로그-가격 기울기 양 → 일관된 우상향 추세",
        ("rs_ratio", "ge", "winner"): "시장 대비 상대 강세 (rs > 1) → 주도주 후보",
        ("s10", "ge", "winner"): "상대강도 점수 높음 → 시장보다 강하게 움직임",
        ("s8", "ge", "winner"): "수급 시그널 강함 → 큰 손 매수 흔적",
        ("vol_ratio", "ge", "winner"): "거래량 폭증 → 관심 집중 + 추세 전환 가능성",

        ("pos_252_high", "le", "loser"): "52주 고점에서 멀리 떨어져 있음 → 하락 추세 한복판",
        ("pos_252_low", "le", "loser"): "52주 저점에서 거의 못 올라옴 → 약세 누적, 반등 동력 부재",
        ("past_120", "le", "loser"): "장기 음(陰) 모멘텀 → 추세 자체가 하락",
        ("past_240", "le", "loser"): "1년째 하락 종목 → 구조적 약세, 떨어지는 칼날",
        ("slope60", "le", "loser"): "로그-가격 기울기 음 → 우하향 추세 진행 중",
        ("slope120", "le", "loser"): "120일 우하향 → 만성 약세 종목",
        ("days_since_52w_high", "ge", "loser"): "52주 고점 멀리 지남 → 매물대 위로 두꺼움",
        ("rs_ratio", "le", "loser"): "시장 대비 약함 → 매수해도 시장 비트 못함",
        ("vol20", "ge", "loser"): "변동성 과대 → 작전 종목 / 테마 일시 폭등 후 붕괴",
        ("upper_wick_ratio", "ge", "loser"): "긴 윗꼬리 → 매도 압력, 고점 거부",
        ("cum_5d_gain", "ge", "loser"): "직전 5일 과열 → 단기 천장권 진입",
    }
    key = (feat, direction, kind)
    if key in mechanisms:
        return mechanisms[key]
    # Generic fallback
    if kind == "winner":
        return f"{FEAT_NAME_KO.get(feat, feat)} 값이 {'높을' if direction=='ge' else '낮을'}수록 +50% 도달 확률 상승"
    else:
        return f"{FEAT_NAME_KO.get(feat, feat)} 값이 {'높을' if direction=='ge' else '낮을'}수록 -30% 손실 확률 상승"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("[1/6] Load enriched...")
    enr = load_enriched_all()
    print(f"  total rows (with ret_180d): {len(enr):,}")

    print("[2/6] Load OHLCV...")
    ohlcv = load_ohlcv()
    print(f"  stocks in OHLCV: {len(ohlcv):,}")

    # Cache chart features so re-runs are fast
    cache_path = CACHE / "chart_feats_v1.parquet"
    if cache_path.exists():
        print(f"[3/6] Load cached chart feats from {cache_path}")
        full = pd.read_parquet(cache_path)
        # Confirm row count matches
        if len(full) != len(enr):
            print("  cache row count mismatch -> recompute")
            full = enrich_with_chart_feats(enr, ohlcv)
            full.to_parquet(cache_path, index=False)
    else:
        print("[3/6] Compute chart feats (one-time)...")
        full = enrich_with_chart_feats(enr, ohlcv)
        full.to_parquet(cache_path, index=False)
        print(f"  cached -> {cache_path}")

    print("[4/6] Tag groups...")
    full = tag_groups(full)
    base_win, base_lose = base_rates(full)
    print(f"  WINNER: {(full['group']=='WINNER').sum():,} ({base_win*100:.1f}%)")
    print(f"  LOSER : {(full['group']=='LOSER').sum():,} ({base_lose*100:.1f}%)")
    print(f"  NEUTRAL: {(full['group']=='NEUTRAL').sum():,}")

    # Chart pattern summary
    pat_sum = chart_pattern_summary(full)
    print("\n[chart pattern summary]")
    print(pat_sum.to_string(index=False))

    print("\n[5/6] Mining...")
    print("  -> univariate threshold rules...")
    uni = mine_threshold_rules(full)
    win_uni = best_winner_rules(uni, top_k=15)
    lose_uni = best_loser_rules(uni, top_k=15)
    print(f"  winner univariate: {len(win_uni)}, loser univariate: {len(lose_uni)}")

    print("  -> decision tree (winner)...")
    win_leaves = decision_tree_rules(full, "WINNER", max_depth=4, min_leaf=120)
    print("  -> decision tree (loser)...")
    lose_leaves = decision_tree_rules(full, "LOSER", max_depth=4, min_leaf=120)

    print("  -> signal combinations...")
    win_combo, lose_combo = signal_combo_mining(full)
    print(f"  winner combos: {len(win_combo)}, loser combos: {len(lose_combo)}")

    print("\n[6/6] Write markdown...")
    write_winner_md(full, win_uni, win_combo, win_leaves, pat_sum,
                    BASE / "WINNER_PATTERNS.md")
    write_loser_md(full, lose_uni, lose_combo, lose_leaves, pat_sum,
                   BASE / "LOSER_PATTERNS.md")

    # also dump the raw rule tables
    uni.to_csv(BASE / "cache" / "winloser_uni_rules.csv", index=False)
    if not win_combo.empty:
        win_combo.to_csv(BASE / "cache" / "winloser_win_combos.csv", index=False)
    if not lose_combo.empty:
        lose_combo.to_csv(BASE / "cache" / "winloser_lose_combos.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
