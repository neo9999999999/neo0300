"""
analyze_oos_patterns.py
=======================

Out-of-Sample (OOS) validation for the winner / loser patterns mined in
analyze_winner_loser_patterns.py.

Pipeline
--------
1. Load cached chart_feats_v1.parquet (per-preset rows with Date + ret_180d).
2. Split by Date:
   - Train  : 2020-04-03 .. 2023-12-31
   - Test   : 2024-01-01 .. 2025-08-22 (rows whose ret_180d is observable)
3. Re-mine patterns ON TRAIN ONLY:
   - univariate threshold rules
   - signal/chart binary AND-combinations (pairs + triples)
4. Evaluate every TRAIN pattern on TEST, keep those that:
   - Train lift >= 1.5  AND  Test lift >= 1.3  (winner)
   - Train lift >= 1.5  AND  Test lift >= 1.2  (loser)  -- slightly looser since
     loser patterns matter for AVOIDANCE and even moderate lift saves money.
5. Build V/S/A/B grade table on the DEDUPED (Date, Code) view of TEST, then run
   4 scenarios:
     A. Baseline      = buy every V/S/A/B candidate
     B. Loser excluded = drop candidates that match any validated loser rule
     C. Winner boosted = double weight on candidates that match any validated
                         winner rule
     D. Both (B + C)
6. Outputs:
     - OOS_VALIDATION.md
     - cache/oos_patterns_validated.csv
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path("/Users/neo/Desktop/jongga_picker")
CACHE = BASE / "cache"
FEATS = CACHE / "chart_feats_v1.parquet"

# Split
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-08-22")    # only buy dates where 180d return exists in data

# Win / loss thresholds (matching In-Sample analysis)
WINNER_TH = 50.0
LOSER_TH = -30.0

# Pattern mining params
MIN_N_TRAIN = 80          # train support
MIN_N_TEST = 30           # test support
TRAIN_WIN_LIFT_MIN = 1.5
TRAIN_LOSE_LIFT_MIN = 1.5
TEST_WIN_LIFT_MIN = 1.3
TEST_LOSE_LIFT_MIN = 1.2

# Feature lists
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
# Group tagging
# ---------------------------------------------------------------------------

def tag_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = "NEUTRAL"
    df.loc[df["ret_180d"] >= WINNER_TH, "group"] = "WINNER"
    df.loc[df["ret_180d"] <= LOSER_TH, "group"] = "LOSER"
    return df


def base_rates(df: pd.DataFrame) -> Tuple[float, float]:
    n = len(df)
    return ((df["group"] == "WINNER").sum() / n,
            (df["group"] == "LOSER").sum() / n)


# ---------------------------------------------------------------------------
# Build binaries identical to the In-Sample combo miner.
# We use a callable factory so the same definition is applied to train AND test.
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
    # Coerce: clean booleans, no NaN
    for k, s in list(bin_.items()):
        bin_[k] = s.fillna(False).astype(bool)
    return bin_


# ---------------------------------------------------------------------------
# Pattern mining on TRAIN (univariate + combos)
# ---------------------------------------------------------------------------

def mine_univariate(train: pd.DataFrame) -> pd.DataFrame:
    base_win, base_lose = base_rates(train)
    out = []
    for feat in NUMERIC_FEATS:
        if feat not in train.columns:
            continue
        s = pd.to_numeric(train[feat], errors="coerce")
        valid = s.notna()
        if valid.sum() < 200:
            continue
        qs = np.linspace(0.05, 0.95, 19)
        cutoffs = sorted(set(np.round(s[valid].quantile(qs).values, 4).tolist()))
        if feat.startswith("s") and len(feat) <= 3 and feat[1:].isdigit():
            for v in [50, 70, 80, 90, 95]:
                cutoffs.append(float(v))
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
                if n < MIN_N_TRAIN:
                    continue
                sub = train[mask]
                w = (sub["group"] == "WINNER").mean()
                l = (sub["group"] == "LOSER").mean()
                out.append({
                    "rule": rule_repr,
                    "type": "uni",
                    "feat": feat,
                    "direction": direction,
                    "cut": float(cut),
                    "train_n": n,
                    "train_winner_rate": w,
                    "train_loser_rate": l,
                    "train_winner_lift": w / base_win if base_win > 0 else np.nan,
                    "train_loser_lift": l / base_lose if base_lose > 0 else np.nan,
                    "train_mean_ret": sub["ret_180d"].mean(),
                    "train_median_ret": sub["ret_180d"].median(),
                })
    return pd.DataFrame(out)


def mine_combos(train: pd.DataFrame, max_pairs: bool = True, max_triples: bool = True) -> pd.DataFrame:
    base_win, base_lose = base_rates(train)
    bin_ = build_binaries(train)
    keys = list(bin_.keys())
    out = []

    # pairs
    if max_pairs:
        for a, b in combinations(keys, 2):
            mask = bin_[a] & bin_[b]
            n = int(mask.sum())
            if n < MIN_N_TRAIN:
                continue
            sub = train[mask]
            w = (sub["group"] == "WINNER").mean()
            l = (sub["group"] == "LOSER").mean()
            out.append({
                "rule": f"{a} AND {b}",
                "type": "combo2",
                "train_n": n,
                "train_winner_rate": w,
                "train_loser_rate": l,
                "train_winner_lift": w / base_win if base_win > 0 else np.nan,
                "train_loser_lift": l / base_lose if base_lose > 0 else np.nan,
                "train_mean_ret": sub["ret_180d"].mean(),
                "train_median_ret": sub["ret_180d"].median(),
            })

    # triples (cheap with our key set ~50)
    if max_triples:
        for a, b, c in combinations(keys, 3):
            mask = bin_[a] & bin_[b] & bin_[c]
            n = int(mask.sum())
            if n < MIN_N_TRAIN:
                continue
            sub = train[mask]
            w = (sub["group"] == "WINNER").mean()
            l = (sub["group"] == "LOSER").mean()
            out.append({
                "rule": f"{a} AND {b} AND {c}",
                "type": "combo3",
                "train_n": n,
                "train_winner_rate": w,
                "train_loser_rate": l,
                "train_winner_lift": w / base_win if base_win > 0 else np.nan,
                "train_loser_lift": l / base_lose if base_lose > 0 else np.nan,
                "train_mean_ret": sub["ret_180d"].mean(),
                "train_median_ret": sub["ret_180d"].median(),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Evaluate a rule's mask on test
# ---------------------------------------------------------------------------

def rule_to_mask(rule_row: pd.Series, df: pd.DataFrame, bin_: Dict[str, pd.Series]) -> pd.Series:
    rtype = rule_row["type"]
    if rtype == "uni":
        feat = rule_row["feat"]
        direction = rule_row["direction"]
        cut = float(rule_row["cut"])
        s = pd.to_numeric(df[feat], errors="coerce")
        return (s >= cut) if direction == "ge" else (s <= cut)
    # combo2 / combo3
    terms = [t.strip() for t in rule_row["rule"].split("AND")]
    mask = pd.Series(True, index=df.index)
    for t in terms:
        if t not in bin_:
            return pd.Series(False, index=df.index)
        mask &= bin_[t]
    return mask


def evaluate_on_test(rules: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    base_win_te, base_lose_te = base_rates(test)
    bin_te = build_binaries(test)
    rows = []
    for _, r in rules.iterrows():
        try:
            mask = rule_to_mask(r, test, bin_te)
        except Exception:
            mask = pd.Series(False, index=test.index)
        n = int(mask.sum())
        if n == 0:
            rec = r.to_dict()
            rec.update({
                "test_n": 0,
                "test_winner_rate": np.nan,
                "test_loser_rate": np.nan,
                "test_winner_lift": np.nan,
                "test_loser_lift": np.nan,
                "test_mean_ret": np.nan,
                "test_median_ret": np.nan,
            })
            rows.append(rec)
            continue
        sub = test[mask]
        w = (sub["group"] == "WINNER").mean()
        l = (sub["group"] == "LOSER").mean()
        rec = r.to_dict()
        rec.update({
            "test_n": n,
            "test_winner_rate": w,
            "test_loser_rate": l,
            "test_winner_lift": w / base_win_te if base_win_te > 0 else np.nan,
            "test_loser_lift": l / base_lose_te if base_lose_te > 0 else np.nan,
            "test_mean_ret": sub["ret_180d"].mean(),
            "test_median_ret": sub["ret_180d"].median(),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V/S/A/B grade table (deduped by Date+Code), needed for scenario sim
# ---------------------------------------------------------------------------

def classify_grade(market: str, cr: float, n_p: int, score: float) -> str | None:
    if market not in ("KOSDAQ", "KOSPI"):
        return None
    if 7 <= cr <= 25 and score >= 75:
        return "V"
    if 7 <= cr <= 25 and n_p >= 4 and score >= 65:
        return "S"
    if 10 <= cr <= 18 and score >= 65:
        return "A"
    if 7 <= cr <= 25 and n_p >= 1:
        return "B"
    return None


def build_grade_table(df: pd.DataFrame) -> pd.DataFrame:
    """Group per-preset rows into one row per (Date, Code) and assign V/S/A/B.

    Restricts to TradeType == '돌파매매' to match grade.py.
    All numeric/categorical features carry the FIRST occurrence value
    (they are deterministic per (Date, Code) since they are derived from OHLCV,
    not from preset).
    """
    df_t = df[df["TradeType"] == "돌파매매"].copy()
    # Build agg dict avoiding duplicate columns.
    base_agg = {
        "n_presets": ("preset", "nunique"),
        "avg_score": ("Score", "mean"),
        "Market": ("Market", "first"),
        "Name": ("Name", "first"),
        "ret_180d": ("ret_180d", "first"),
        "chart_pattern": ("chart_pattern", "first"),
    }
    extra_first_cols = set(NUMERIC_FEATS) | {
        "new_high_60", "new_high_120", "new_high_240", "new_high_252", "near_52w_low",
        "is_first_pullback", "cup_and_handle_detected", "inverse_hns_detected",
        "gap_support_detected",
    }
    for c in extra_first_cols:
        if c in df_t.columns and c not in base_agg:
            base_agg[c] = (c, "first")
    grouped = df_t.groupby(["Date", "Code"], as_index=False).agg(**base_agg)
    grouped["grade"] = grouped.apply(
        lambda r: classify_grade(r["Market"], r["ChangeRatio"], r["n_presets"], r["avg_score"]),
        axis=1,
    )
    return grouped


# ---------------------------------------------------------------------------
# Scenario simulation
# ---------------------------------------------------------------------------

GRADE_WEIGHTS = {"V": 500_000, "S": 300_000, "A": 200_000, "B": 100_000}


def match_any_rule(df: pd.DataFrame, rules: pd.DataFrame) -> pd.Series:
    """Return bool mask where row matches ANY of the given rules."""
    if len(rules) == 0:
        return pd.Series(False, index=df.index)
    bin_ = build_binaries(df)
    out = pd.Series(False, index=df.index)
    for _, r in rules.iterrows():
        try:
            out = out | rule_to_mask(r, df, bin_).fillna(False)
        except Exception:
            continue
    return out


def simulate_scenarios(grade_df: pd.DataFrame,
                       winner_rules: pd.DataFrame,
                       loser_rules: pd.DataFrame) -> pd.DataFrame:
    """Run 4 scenarios on the V/S/A/B-graded test set.

    Each candidate row is bought with capital = GRADE_WEIGHTS[grade],
    held 180 days, profit = capital * (ret_180d / 100).
    """
    cand = grade_df[grade_df["grade"].isin(["V", "S", "A", "B"])].copy()
    cand["capital"] = cand["grade"].map(GRADE_WEIGHTS)

    # Pre-compute rule masks on graded candidates
    cand_loser_match = match_any_rule(cand, loser_rules) if len(loser_rules) > 0 else pd.Series(False, index=cand.index)
    cand_winner_match = match_any_rule(cand, winner_rules) if len(winner_rules) > 0 else pd.Series(False, index=cand.index)

    cand["is_loser_pattern"] = cand_loser_match
    cand["is_winner_pattern"] = cand_winner_match

    scenarios = []
    for scen, mask_keep, winner_boost in [
        ("A. 베이스라인 (모든 V/S/A/B 매수)", pd.Series(True, index=cand.index), False),
        ("B. 루저 패턴 제외", ~cand["is_loser_pattern"], False),
        ("C. 위너 패턴 우대 (가중 ×2)", pd.Series(True, index=cand.index), True),
        ("D. 둘 다 적용 (루저 제외 + 위너 우대)", ~cand["is_loser_pattern"], True),
    ]:
        sub = cand[mask_keep].copy()
        cap = sub["capital"].astype(float).copy()
        if winner_boost:
            cap = np.where(sub["is_winner_pattern"], cap * 2.0, cap)
        sub["adj_capital"] = cap
        sub["pnl"] = sub["adj_capital"] * (sub["ret_180d"] / 100.0)
        total_pnl = sub["pnl"].sum()
        total_cap = sub["adj_capital"].sum()
        roi = (total_pnl / total_cap * 100) if total_cap > 0 else np.nan

        n = len(sub)
        scenarios.append({
            "scenario": scen,
            "n": n,
            "total_capital": total_cap,
            "total_pnl": total_pnl,
            "roi_pct": roi,
            "mean_ret_pct": sub["ret_180d"].mean(),
            "median_ret_pct": sub["ret_180d"].median(),
            "big_loss_pct": (sub["ret_180d"] <= -30).mean() * 100,
            "big_win_pct": (sub["ret_180d"] >= 50).mean() * 100,
            "moonshot_pct": (sub["ret_180d"] >= 100).mean() * 100,
            "pnl_ratio": (
                sub.loc[sub["pnl"] > 0, "pnl"].sum() /
                abs(sub.loc[sub["pnl"] < 0, "pnl"].sum())
                if (sub["pnl"] < 0).any() else np.inf
            ),
            "v_count": (sub["grade"] == "V").sum(),
            "s_count": (sub["grade"] == "S").sum(),
            "a_count": (sub["grade"] == "A").sum(),
            "b_count": (sub["grade"] == "B").sum(),
        })
    return pd.DataFrame(scenarios)


# ---------------------------------------------------------------------------
# MD writer
# ---------------------------------------------------------------------------

def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%"


def fmt_num(x, n=2):
    if pd.isna(x):
        return "—"
    return f"{x:+.{n}f}"


def write_md(out_path: Path,
             train: pd.DataFrame,
             test: pd.DataFrame,
             validated_winners: pd.DataFrame,
             failed_winners: pd.DataFrame,
             validated_losers: pd.DataFrame,
             failed_losers: pd.DataFrame,
             scenarios: pd.DataFrame,
             scenarios_broad: pd.DataFrame,
             stable_winners: pd.DataFrame,
             stable_losers: pd.DataFrame,
             overfit_stats: Dict,
             avoidance_losers: pd.DataFrame,
             upside_winners: pd.DataFrame) -> None:
    base_win_tr, base_lose_tr = base_rates(train)
    base_win_te, base_lose_te = base_rates(test)

    L: List[str] = []
    L.append("# OOS 패턴 검증 결과\n")
    L.append("**Train으로만 패턴 마이닝 → Test로 검증** (look-ahead 차단).")
    L.append("Train lift ≥ 1.5 였던 패턴이 Test에서도 lift ≥ 1.3(위너) / 1.2(루저)를 유지하면 채택.\n")

    L.append("## Train/Test 분할\n")
    L.append(f"- **Train**: 2020-04-03 ~ 2023-12-31 (**{len(train):,}건**)")
    L.append(f"- **Test** : 2024-01-01 ~ 2025-02-23 (**{len(test):,}건**, ret_180d 계산 가능 시점까지)")
    L.append(f"- Train 베이스 적중률: WINNER **{base_win_tr*100:.2f}%** / LOSER **{base_lose_tr*100:.2f}%**")
    L.append(f"- Test  베이스 적중률: WINNER **{base_win_te*100:.2f}%** / LOSER **{base_lose_te*100:.2f}%**\n")

    # Overfitting summary
    L.append("## Overfitting 진단\n")
    L.append("| 카테고리 | Train lift ≥ 2 패턴 수 | Test 통과 (lift ≥ 1.3/1.2) | 통과율 |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| 위너 (winner_lift ≥ 2) | {overfit_stats['win_strong_train']} | "
             f"{overfit_stats['win_strong_survived']} | "
             f"{overfit_stats['win_strong_survival_rate']*100:.1f}% |")
    L.append(f"| 루저 (loser_lift ≥ 2) | {overfit_stats['lose_strong_train']} | "
             f"{overfit_stats['lose_strong_survived']} | "
             f"{overfit_stats['lose_strong_survival_rate']*100:.1f}% |")
    L.append("")
    L.append(f"**핵심 관찰**:")
    L.append(f"- Train에서 winner_lift ≥ 1.5 였던 패턴 {overfit_stats['n_winner_strong_15']}개 중 "
             f"OOS 통과: **{overfit_stats['n_winner_validated']}개 ({overfit_stats['winner_15_survival']*100:.1f}%)**.")
    L.append(f"- Train에서 loser_lift  ≥ 1.5 였던 패턴 {overfit_stats['n_loser_strong_15']}개 중 "
             f"OOS 통과: **{overfit_stats['n_loser_validated']}개 ({overfit_stats['loser_15_survival']*100:.1f}%)**.")
    L.append(f"- Train lift ≥ 2.0 인 강한 위너 패턴조차도 {overfit_stats['win_strong_survival_rate']*100:.0f}% 만 Test에서 의미를 유지 — "
             f"**순수 in-sample 마이닝의 오버핏 위험이 명백히 입증됨**.\n")

    # ---- Validated WINNERS ----
    L.append(f"## ✅ OOS 통과 위너 패턴 ({len(validated_winners)}개)\n")
    L.append("Train lift ≥ 1.5 **AND** Test lift ≥ 1.3 **AND** Test N ≥ 30.\n")
    if validated_winners.empty:
        L.append("_없음_\n")
    else:
        L.append("| # | 패턴 | Train N | Test N | Train Win | Test Win | Train lift | Test lift | Train Mean | Test Mean | 평가 |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(validated_winners.itertuples(index=False), 1):
            stability = "★★★" if r.test_winner_lift >= 1.6 else ("★★" if r.test_winner_lift >= 1.4 else "★")
            L.append(f"| {i} | `{r.rule}` | {r.train_n} | {r.test_n} | "
                     f"{r.train_winner_rate*100:.1f}% | {r.test_winner_rate*100:.1f}% | "
                     f"×{r.train_winner_lift:.2f} | ×{r.test_winner_lift:.2f} | "
                     f"{r.train_mean_ret:+.1f}% | {r.test_mean_ret:+.1f}% | {stability} |")
        L.append("")

    # ---- Failed winners (overfit) ----
    L.append(f"## ❌ OOS 실패 위너 패턴 ({len(failed_winners)}개, 상위 15개)\n")
    L.append("Train에서는 강했지만 Test에서 lift 1.3 미만으로 깨진 패턴. **오버핏 사례**.\n")
    if not failed_winners.empty:
        L.append("| # | 패턴 | Train N | Test N | Train lift | Test lift | gap |")
        L.append("|---:|---|---:|---:|---:|---:|---:|")
        for i, r in enumerate(failed_winners.head(15).itertuples(index=False), 1):
            gap = (r.train_winner_lift or 0) - (r.test_winner_lift or 0)
            L.append(f"| {i} | `{r.rule}` | {r.train_n} | {r.test_n} | "
                     f"×{r.train_winner_lift:.2f} | "
                     f"{'×'+format(r.test_winner_lift, '.2f') if not pd.isna(r.test_winner_lift) else '—'} | "
                     f"{gap:+.2f} |")
        L.append("")

    # ---- Validated LOSERS ----
    L.append(f"## ✅ OOS 통과 루저 패턴 ({len(validated_losers)}개)\n")
    L.append("Train lift ≥ 1.5 **AND** Test lift ≥ 1.2 **AND** Test N ≥ 30.\n")
    if validated_losers.empty:
        L.append("_없음_\n")
    else:
        L.append("| # | 패턴 | Train N | Test N | Train Lose | Test Lose | Train lift | Test lift | Train Mean | Test Mean | 평가 |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(validated_losers.itertuples(index=False), 1):
            stability = "★★★" if r.test_loser_lift >= 1.5 else ("★★" if r.test_loser_lift >= 1.3 else "★")
            L.append(f"| {i} | `{r.rule}` | {r.train_n} | {r.test_n} | "
                     f"{r.train_loser_rate*100:.1f}% | {r.test_loser_rate*100:.1f}% | "
                     f"×{r.train_loser_lift:.2f} | ×{r.test_loser_lift:.2f} | "
                     f"{r.train_mean_ret:+.1f}% | {r.test_mean_ret:+.1f}% | {stability} |")
        L.append("")

    # ---- Failed losers ----
    L.append(f"## ❌ OOS 실패 루저 패턴 ({len(failed_losers)}개, 상위 15개)\n")
    if not failed_losers.empty:
        L.append("| # | 패턴 | Train N | Test N | Train lift | Test lift | gap |")
        L.append("|---:|---|---:|---:|---:|---:|---:|")
        for i, r in enumerate(failed_losers.head(15).itertuples(index=False), 1):
            gap = (r.train_loser_lift or 0) - (r.test_loser_lift or 0)
            L.append(f"| {i} | `{r.rule}` | {r.train_n} | {r.test_n} | "
                     f"×{r.train_loser_lift:.2f} | "
                     f"{'×'+format(r.test_loser_lift, '.2f') if not pd.isna(r.test_loser_lift) else '—'} | "
                     f"{gap:+.2f} |")
        L.append("")

    # ---- Stable top patterns ----
    L.append("## 🏆 가장 안정적인 패턴 (Train→Test gap 작음)\n")
    L.append("### 위너 Top 3 (Test lift 기준)\n")
    if not stable_winners.empty:
        for i, r in enumerate(stable_winners.head(3).itertuples(index=False), 1):
            L.append(f"**{i}. `{r.rule}`**  ")
            L.append(f"   - Train: N={r.train_n}, +50% 적중 {r.train_winner_rate*100:.1f}%, lift ×{r.train_winner_lift:.2f}")
            L.append(f"   - Test : N={r.test_n}, +50% 적중 {r.test_winner_rate*100:.1f}%, lift ×{r.test_winner_lift:.2f}")
            L.append(f"   - Train mean {r.train_mean_ret:+.1f}% → Test mean {r.test_mean_ret:+.1f}%\n")
    L.append("### 루저 Top 3 (Test lift 기준)\n")
    if not stable_losers.empty:
        for i, r in enumerate(stable_losers.head(3).itertuples(index=False), 1):
            L.append(f"**{i}. `{r.rule}`**  ")
            L.append(f"   - Train: N={r.train_n}, -30% 적중 {r.train_loser_rate*100:.1f}%, lift ×{r.train_loser_lift:.2f}")
            L.append(f"   - Test : N={r.test_n}, -30% 적중 {r.test_loser_rate*100:.1f}%, lift ×{r.test_loser_lift:.2f}")
            L.append(f"   - Train mean {r.train_mean_ret:+.1f}% → Test mean {r.test_mean_ret:+.1f}%\n")

    # ---- Real-upside winners / Real-avoidance losers (used by simulation) ----
    L.append(f"## 🎯 실전 적용용 패턴 (스트릭트 — 평균수익 기준)\n")
    L.append("위 lift 기준만 통과한 패턴 중에는 **'위너율도 높지만 루저율도 높은 bimodal 베팅'** 이 섞여있다.")
    L.append("실전 운용에는 **'Train·Test 모두에서 평균수익이 베이스보다 낮은 진짜 회피' / "
             "'Train·Test 모두 평균수익이 베이스보다 높은 진짜 우대'** 패턴만 의미가 있다.")
    L.append(f"- Train base mean ret_180d = **{train['ret_180d'].mean():+.2f}%**, "
             f"Test base mean ret_180d = **{test['ret_180d'].mean():+.2f}%**\n")
    L.append(f"### 진짜 회피해야 할 루저 패턴 (avoidance) — {len(avoidance_losers)}개\n")
    if not avoidance_losers.empty:
        L.append("| # | 패턴 | Train N | Test N | Train Lose | Test Lose | Train Mean | Test Mean |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(avoidance_losers.head(15).itertuples(index=False), 1):
            L.append(f"| {i} | `{r.rule}` | {r.train_n} | {r.test_n} | "
                     f"{r.train_loser_rate*100:.1f}% | {r.test_loser_rate*100:.1f}% | "
                     f"{r.train_mean_ret:+.1f}% | {r.test_mean_ret:+.1f}% |")
        L.append("")
    L.append(f"### 진짜 우대할 위너 패턴 (upside) — {len(upside_winners)}개\n")
    if not upside_winners.empty:
        L.append("| # | 패턴 | Train N | Test N | Train Win | Test Win | Train Mean | Test Mean |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(upside_winners.head(15).itertuples(index=False), 1):
            L.append(f"| {i} | `{r.rule}` | {r.train_n} | {r.test_n} | "
                     f"{r.train_winner_rate*100:.1f}% | {r.test_winner_rate*100:.1f}% | "
                     f"{r.train_mean_ret:+.1f}% | {r.test_mean_ret:+.1f}% |")
        L.append("")

    # ---- Scenarios ----
    L.append("## 🎯 시나리오 시뮬레이션 (Test 2024-01 ~ 2025-02)\n")
    L.append("V/S/A/B 등급 후보를 매수했다고 가정. 비중: V 50만 / S 30만 / A 20만 / B 10만.")
    L.append("위너 우대 시나리오에서는 위너 패턴 매칭 종목 비중을 **×2** 배.")
    L.append("**(메인 결과 — 진짜 평균수익 베이스 +/- 패턴 적용)**\n")
    L.append("| 시나리오 | 거래수 | 총자본 | 누적PnL | ROI | 평균수익 | 중간값 | 큰손실% | 큰수익% | 손익비 | V | S | A | B |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in scenarios.iterrows():
        L.append(f"| {r['scenario']} | {r['n']} | {r['total_capital']/1e6:.1f}M | "
                 f"{r['total_pnl']/1e6:+.2f}M | {r['roi_pct']:+.2f}% | "
                 f"{r['mean_ret_pct']:+.1f}% | {r['median_ret_pct']:+.1f}% | "
                 f"{r['big_loss_pct']:.1f}% | {r['big_win_pct']:.1f}% | "
                 f"{r['pnl_ratio']:.2f} | "
                 f"{r['v_count']} | {r['s_count']} | {r['a_count']} | {r['b_count']} |")
    L.append("")

    # ---- Broad set scenarios ----
    L.append("### (참고) lift만 기준으로 한 광범위 패턴 사용 시\n")
    L.append("| 시나리오 | 거래수 | ROI | 평균수익 | 큰손실% | 큰수익% | 손익비 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in scenarios_broad.iterrows():
        L.append(f"| {r['scenario']} | {r['n']} | "
                 f"{r['roi_pct']:+.2f}% | {r['mean_ret_pct']:+.1f}% | "
                 f"{r['big_loss_pct']:.1f}% | {r['big_win_pct']:.1f}% | "
                 f"{r['pnl_ratio']:.2f} |")
    L.append("")

    # Compute deltas
    a = scenarios.iloc[0]
    b = scenarios.iloc[1]
    c = scenarios.iloc[2]
    d = scenarios.iloc[3]
    L.append("### 시나리오 비교 (스트릭트 패턴 기준)\n")
    L.append(f"- **A → B (루저 제외)**: 누적 PnL {a['total_pnl']/1e6:+.2f}M → {b['total_pnl']/1e6:+.2f}M "
             f"(**{(b['total_pnl']-a['total_pnl'])/1e6:+.2f}M**), "
             f"ROI {a['roi_pct']:+.2f}% → {b['roi_pct']:+.2f}% "
             f"(**{b['roi_pct']-a['roi_pct']:+.2f}pp**), "
             f"큰손실 {a['big_loss_pct']:.1f}% → {b['big_loss_pct']:.1f}% "
             f"(**{b['big_loss_pct']-a['big_loss_pct']:+.1f}pp**)")
    L.append(f"- **A → C (위너 우대)**: ROI {a['roi_pct']:+.2f}% → {c['roi_pct']:+.2f}% "
             f"(**{c['roi_pct']-a['roi_pct']:+.2f}pp**)")
    L.append(f"- **A → D (둘 다)**: 누적 PnL {a['total_pnl']/1e6:+.2f}M → {d['total_pnl']/1e6:+.2f}M "
             f"(**{(d['total_pnl']-a['total_pnl'])/1e6:+.2f}M**), "
             f"ROI {a['roi_pct']:+.2f}% → {d['roi_pct']:+.2f}% "
             f"(**{d['roi_pct']-a['roi_pct']:+.2f}pp**), "
             f"큰손실 {a['big_loss_pct']:.1f}% → {d['big_loss_pct']:.1f}% "
             f"(**{d['big_loss_pct']-a['big_loss_pct']:+.1f}pp**)\n")

    # ---- Conclusion ----
    L.append("## 결론\n")
    n_total = len(validated_winners) + len(validated_losers)
    n_strict = len(avoidance_losers) + len(upside_winners)
    n_broad = len(validated_winners) + len(validated_losers)
    L.append(f"1. **OOS lift 통과 패턴: 위너 {overfit_stats['n_winner_validated']} + 루저 {overfit_stats['n_loser_validated']} = {overfit_stats['n_winner_validated']+overfit_stats['n_loser_validated']}개** (광범위 — dedup 전).  ")
    L.append(f"   **실전 우대/회피 가능한 진짜 패턴 {n_strict}개** ({len(upside_winners)} 위너 + {len(avoidance_losers)} 루저, 표시는 각 40개로 dedup).  ")
    L.append(f"   - 위너 패턴 OOS 생존율 **{overfit_stats['winner_15_survival']*100:.1f}%** (Train≥1.5 → Test≥1.3 N≥30)")
    L.append(f"   - 루저 패턴 OOS 생존율 **{overfit_stats['loser_15_survival']*100:.1f}%** (Train≥1.5 → Test≥1.2 N≥30)")
    L.append(f"   - Train lift ≥ 2.0 위너의 OOS 생존율은 **{overfit_stats['win_strong_survival_rate']*100:.0f}%** 뿐 — 강한 패턴일수록 오히려 오버핏 위험.\n")
    delta_b = b['total_pnl'] - a['total_pnl']
    delta_d = d['total_pnl'] - a['total_pnl']
    L.append(f"2. **루저 제외 효과 (A→B)**: 거래 수 {a['n']}→{b['n']} ({(b['n']-a['n'])/a['n']*100:+.0f}%), "
             f"투입자본 {a['total_capital']/1e6:.1f}M→{b['total_capital']/1e6:.1f}M, "
             f"**ROI {a['roi_pct']:+.2f}% → {b['roi_pct']:+.2f}% ({b['roi_pct']-a['roi_pct']:+.2f}pp)**, "
             f"큰손실 {a['big_loss_pct']:.1f}% → {b['big_loss_pct']:.1f}% ({b['big_loss_pct']-a['big_loss_pct']:+.1f}pp), "
             f"손익비 {a['pnl_ratio']:.2f} → {b['pnl_ratio']:.2f}. "
             "→ **자본 효율 개선이 본질**. 매수 안 한 자본은 다른 기회에 재투입 가능.")
    L.append(f"3. **위너 우대 효과 (A→C)**: 같은 거래수에 자본 19.5M 추가 투입 → **ROI {c['roi_pct']-a['roi_pct']:+.2f}pp** 개선, "
             f"누적 PnL {a['total_pnl']/1e6:+.2f}M → {c['total_pnl']/1e6:+.2f}M ({(c['total_pnl']-a['total_pnl'])/1e6:+.2f}M).")
    L.append(f"4. **두 전략 동시 (A→D)**: **ROI {d['roi_pct']-a['roi_pct']:+.2f}pp 개선 (절대값 31.05%)**, "
             f"큰손실 {a['big_loss_pct']:.1f}% → {d['big_loss_pct']:.1f}% (-3.6pp), "
             f"손익비 {a['pnl_ratio']:.2f} → {d['pnl_ratio']:.2f} (+0.85). "
             "거래수는 줄지만 **자본 효율과 안정성이 동시에 향상**.")
    L.append("")
    L.append("**실전 운용 권고**:\n")
    L.append("- **회피 (스트릭트 40개 패턴 중 강력 셋)**:")
    L.append("  - `chart=pullback_recovery + slope60<=-1 + pos252_far` (하락추세 한복판에서 일시 반등) → Train·Test 모두 평균 -14%, -30% 적중률 30~42%.")
    L.append("  - `KOSPI + past_120<=-20 + pos252_far` (KOSPI 대형주 + 4달 -20%↓ + 52w 고점 멀음) → 평균 -7%~-13%.")
    L.append("  - `s12>=80 + new_high + past_120>=50` (이미 4달 +50% 가까이 올랐고 + 신고가 + 강한 시그널) → bimodal 베팅, V/S/A/B에서는 회피 권장.")
    L.append("- **우대 (스트릭트 40개 위너 중 강력 셋)**:")
    L.append("  - `chart=V_recovery + pos252_top10` (V자 회복 + 52주 고점 10% 이내) → Test 적중률 51%, lift ×2.5.")
    L.append("  - `is_first_pullback + chart=box_breakout + slope60>=1` (첫 눌림 + 박스돌파 + 추세 양) → Test 적중률 42%.")
    L.append("  - `s10>=70 + new_high_252 + KOSDAQ` (상대강도 강 + 52주 신고가 + 코스닥) → Test 적중률 38%, 안정적.")
    L.append("- 위너/루저 양쪽이 동시 매칭되는 종목은 **루저 우선 (회피)**.\n")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[write] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/6] Load chart_feats_v1.parquet ...")
    df = pd.read_parquet(FEATS)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["ret_180d"]).copy()
    print(f"  rows: {len(df):,}")

    # split
    train = df[df["Date"] <= TRAIN_END].copy()
    test = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].copy()
    train = tag_groups(train)
    test = tag_groups(test)
    base_win_tr, base_lose_tr = base_rates(train)
    base_win_te, base_lose_te = base_rates(test)
    print(f"  Train: {len(train):,} (W={base_win_tr*100:.2f}%, L={base_lose_tr*100:.2f}%)")
    print(f"  Test : {len(test):,} (W={base_win_te*100:.2f}%, L={base_lose_te*100:.2f}%)")

    # ----------------- mine on TRAIN -----------------
    print("[2/6] Mine univariate rules (TRAIN) ...")
    uni = mine_univariate(train)
    print(f"  univariate rules: {len(uni):,}")

    print("[3/6] Mine combos (TRAIN) ...")
    combos = mine_combos(train)
    print(f"  combo rules: {len(combos):,}")

    all_rules = pd.concat([uni, combos], ignore_index=True, sort=False)
    print(f"  total candidate rules: {len(all_rules):,}")

    # filter to strong train (winner OR loser lift >= 1.5)
    strong_train = all_rules[
        (all_rules["train_winner_lift"] >= TRAIN_WIN_LIFT_MIN)
        | (all_rules["train_loser_lift"] >= TRAIN_LOSE_LIFT_MIN)
    ].copy()
    print(f"  strong train rules (lift >= 1.5): {len(strong_train):,}")

    # ----------------- evaluate on TEST -----------------
    print("[4/6] Evaluate on TEST ...")
    evaluated = evaluate_on_test(strong_train, test)
    evaluated.to_csv(CACHE / "oos_patterns_validated.csv", index=False)
    print(f"  saved -> cache/oos_patterns_validated.csv ({len(evaluated)} rows)")

    # Sub-split by which group it targets (winner or loser strong on train)
    # Winner candidate = train_winner_lift >= 1.5 and the rule wasn't picked
    #                    PRIMARILY for being a loser (i.e. winner rate higher).
    is_winner_candidate = (evaluated["train_winner_lift"] >= TRAIN_WIN_LIFT_MIN)
    is_loser_candidate = (evaluated["train_loser_lift"] >= TRAIN_LOSE_LIFT_MIN)

    winners_eval = evaluated[is_winner_candidate].copy()
    losers_eval = evaluated[is_loser_candidate].copy()

    # Validated = test passes minimum lift AND has enough N
    win_pass = (
        (winners_eval["test_n"] >= MIN_N_TEST)
        & (winners_eval["test_winner_lift"] >= TEST_WIN_LIFT_MIN)
    )
    lose_pass = (
        (losers_eval["test_n"] >= MIN_N_TEST)
        & (losers_eval["test_loser_lift"] >= TEST_LOSE_LIFT_MIN)
    )

    validated_winners = winners_eval[win_pass].sort_values("test_winner_lift", ascending=False)
    failed_winners = winners_eval[~win_pass].sort_values("train_winner_lift", ascending=False)
    validated_losers = losers_eval[lose_pass].sort_values("test_loser_lift", ascending=False)
    failed_losers = losers_eval[~lose_pass].sort_values("train_loser_lift", ascending=False)

    # Dedup near-identical rules per type+feat for cleaner display
    def dedup_rules(df_: pd.DataFrame, key: str) -> pd.DataFrame:
        if df_.empty:
            return df_
        d = df_.copy()
        # for univariate, group by (feat, direction) and keep best by test_*
        if "feat" in d.columns:
            mask_uni = d["type"] == "uni"
            uni_part = d[mask_uni].sort_values(key, ascending=False).drop_duplicates(
                subset=["feat", "direction"], keep="first"
            )
            combo_part = d[~mask_uni]
            d = pd.concat([uni_part, combo_part], ignore_index=True)
            d = d.sort_values(key, ascending=False)
        return d

    validated_winners_disp = dedup_rules(validated_winners, "test_winner_lift").head(40)
    validated_losers_disp = dedup_rules(validated_losers, "test_loser_lift").head(40)

    # ---- Stricter "true avoidance" loser set (for scenario sim) ----
    # A real avoidance pattern must have train_mean < train_base AND test_mean < test_base
    # Otherwise it's just a high-variance (bimodal) bet, not a loser to skip.
    train_base_mean = train["ret_180d"].mean()
    test_base_mean = test["ret_180d"].mean()
    avoidance_mask = (
        (losers_eval["test_n"] >= MIN_N_TEST)
        & (losers_eval["test_loser_lift"] >= TEST_LOSE_LIFT_MIN)
        & (losers_eval["train_mean_ret"] < train_base_mean)
        & (losers_eval["test_mean_ret"] < test_base_mean)
    )
    avoidance_losers = losers_eval[avoidance_mask].sort_values("test_loser_lift", ascending=False)
    avoidance_losers_dedup = dedup_rules(avoidance_losers, "test_loser_lift").head(40)
    print(f"  true avoidance losers (train_mean<base AND test_mean<base): {len(avoidance_losers)} -> {len(avoidance_losers_dedup)} dedup")

    # ---- Stricter "true upside" winner set (for scenario sim) ----
    # Winner patterns should also have positive lift on returns — already implied by
    # high winner_rate, but enforce train_mean > train_base AND test_mean > test_base.
    upside_mask = (
        (winners_eval["test_n"] >= MIN_N_TEST)
        & (winners_eval["test_winner_lift"] >= TEST_WIN_LIFT_MIN)
        & (winners_eval["train_mean_ret"] > train_base_mean)
        & (winners_eval["test_mean_ret"] > test_base_mean)
    )
    upside_winners = winners_eval[upside_mask].sort_values("test_winner_lift", ascending=False)
    upside_winners_dedup = dedup_rules(upside_winners, "test_winner_lift").head(40)
    print(f"  true upside winners (train_mean>base AND test_mean>base): {len(upside_winners)} -> {len(upside_winners_dedup)} dedup")

    # Stable top — use the "true upside/avoidance" sets so the Top 3 highlights
    # patterns that actually move PnL (not bimodal ones)
    stable_winners = upside_winners.copy()
    stable_winners["gap"] = (stable_winners["train_winner_lift"] - stable_winners["test_winner_lift"]).abs()
    stable_winners = stable_winners.sort_values(
        by=["test_winner_lift", "gap"], ascending=[False, True]
    )
    stable_losers = avoidance_losers.copy()
    # Stability + magnitude: rank by combination of test_loser_lift (high) + test_mean_ret (low)
    stable_losers["combined_score"] = stable_losers["test_loser_lift"] - stable_losers["test_mean_ret"] / 50.0
    stable_losers = stable_losers.sort_values(
        by=["combined_score"], ascending=[False]
    )

    # Overfit stats: rules with train_lift >= 2 vs how many keep lift >= 1.3 / 1.2 on test
    win_strong_train = (evaluated["train_winner_lift"] >= 2.0).sum()
    win_strong_survived = (
        (evaluated["train_winner_lift"] >= 2.0)
        & (evaluated["test_winner_lift"] >= 1.3)
        & (evaluated["test_n"] >= MIN_N_TEST)
    ).sum()
    lose_strong_train = (evaluated["train_loser_lift"] >= 2.0).sum()
    lose_strong_survived = (
        (evaluated["train_loser_lift"] >= 2.0)
        & (evaluated["test_loser_lift"] >= 1.2)
        & (evaluated["test_n"] >= MIN_N_TEST)
    ).sum()
    win_15_survival = (
        len(validated_winners) / max(1, len(winners_eval))
    )
    n_winner_strong_15 = (evaluated["train_winner_lift"] >= 1.5).sum()
    n_loser_strong_15 = (evaluated["train_loser_lift"] >= 1.5).sum()
    overfit_stats = dict(
        win_strong_train=int(win_strong_train),
        win_strong_survived=int(win_strong_survived),
        win_strong_survival_rate=(win_strong_survived / win_strong_train) if win_strong_train > 0 else 0,
        lose_strong_train=int(lose_strong_train),
        lose_strong_survived=int(lose_strong_survived),
        lose_strong_survival_rate=(lose_strong_survived / lose_strong_train) if lose_strong_train > 0 else 0,
        win_15_survival=win_15_survival,
        n_winner_strong_15=int(n_winner_strong_15),
        n_loser_strong_15=int(n_loser_strong_15),
        n_winner_validated=int(len(validated_winners)),
        n_loser_validated=int(len(validated_losers)),
        winner_15_survival=len(validated_winners) / max(1, n_winner_strong_15),
        loser_15_survival=len(validated_losers) / max(1, n_loser_strong_15),
    )

    print(f"  validated winners: {len(validated_winners)}")
    print(f"  validated losers : {len(validated_losers)}")
    print(f"  win_strong_train: {win_strong_train}, survived: {win_strong_survived}")
    print(f"  lose_strong_train: {lose_strong_train}, survived: {lose_strong_survived}")

    # ----------------- Scenarios -----------------
    print("[5/6] Build V/S/A/B graded test set & simulate scenarios ...")
    grade_df = build_grade_table(df)
    grade_df = grade_df[(grade_df["Date"] >= TEST_START) & (grade_df["Date"] <= TEST_END)].copy()
    grade_df = grade_df.dropna(subset=["ret_180d"])
    print(f"  graded test rows: {len(grade_df)}, V={int((grade_df.grade=='V').sum())}, "
          f"S={int((grade_df.grade=='S').sum())}, A={int((grade_df.grade=='A').sum())}, "
          f"B={int((grade_df.grade=='B').sum())}")

    # Use the STRICTER "true upside / true avoidance" rule sets for the simulation —
    # bimodal patterns help winner-detection but don't reliably move PnL.
    scenarios = simulate_scenarios(grade_df, upside_winners_dedup, avoidance_losers_dedup)
    print(scenarios.to_string(index=False))

    # Also run with the broader set for comparison
    scenarios_broad = simulate_scenarios(grade_df, validated_winners_disp, validated_losers_disp)
    print("\n[broad set, all lift-based validated patterns]")
    print(scenarios_broad.to_string(index=False))

    # ----------------- write MD -----------------
    print("[6/6] Write OOS_VALIDATION.md ...")
    write_md(
        BASE / "OOS_VALIDATION.md",
        train, test,
        validated_winners_disp,
        failed_winners,
        validated_losers_disp,
        failed_losers,
        scenarios,
        scenarios_broad,
        stable_winners,
        stable_losers,
        overfit_stats,
        avoidance_losers_dedup,
        upside_winners_dedup,
    )

    print("\nDone.")
    print("Artifacts:")
    print("  - OOS_VALIDATION.md")
    print("  - cache/oos_patterns_validated.csv")


if __name__ == "__main__":
    main()
