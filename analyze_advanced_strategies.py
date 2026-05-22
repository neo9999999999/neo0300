"""
Advanced Strategy Exploration — V/S/A/B 를 넘어서는 전략 탐색.

[테스트할 전략들]
1. 익절/손절 조합 최적화 — 익절 +30/50/100% × 트레일링 -20/-30%
2. 다른 보유 기간 — 210d / 240d / 365d
3. 변동성 적응형 비중 — 시장 변동성에 따른 비중 조정
4. 시장 환경별 전략 — 강세장/약세장 자동 감지 후 다른 전략
5. 눌림목매매 (TradeType=눌림목매매) 통계
6. 대시세 초입 (TradeType=대시세 초입) 통계

[베이스라인]
- 현재 V/S/A/B 시스템 (180일 보유, 손절/익절 없음)
- 6년 누적: +6,318만원 (연 ROI +50.1%)

[메트릭]
- 6년 누적 손익
- 손익비 (winning trades total / losing trades total)
- 큰손실률 (return < -30%)
- 큰수익률 (return > +100%)
- 자본효율 (연 ROI: 총손익 / 최대자본 / 6년)
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import json
from pathlib import Path
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np

CACHE = Path("cache")
PRESETS_4 = ["default", "box_breakout", "habarocell", "pullback"]
WEIGHTS = {"V": 500_000, "S": 300_000, "A": 200_000, "B": 100_000}

print("OHLCV 로딩 중...", end=" ", flush=True)
with open(CACHE / "ohlcv_2020-01-01_2026-05-21.pkl", "rb") as f:
    OHLCV = pickle.load(f)
print(f"{len(OHLCV)}종목")


# ──────────────────────────────────────────────────────────
# 시장 레짐 계산 (강세장/약세장)
# ──────────────────────────────────────────────────────────
def build_market_regime():
    """시장 평균 60일 수익률 기반 레짐 시계열.

    Returns:
        DataFrame[Date] with columns:
            ret60: 시장 평균 60일 수익률
            vol60: 시장 평균 60일 변동성 (일일수익률 std × √252)
            regime: "BULL" / "BEAR" / "NEUTRAL"
            vol_regime: "HIGH_VOL" / "LOW_VOL" / "NORMAL"
    """
    # 대형주 200개 (전체 평균과 거의 일치하면서 빠름)
    sample_codes = list(OHLCV.keys())[:200]
    closes = pd.DataFrame({k: OHLCV[k]["Close"] for k in sample_codes if "Close" in OHLCV[k].columns})
    daily_ret = closes.pct_change()
    ret60 = closes.pct_change(60).mean(axis=1)
    vol60 = daily_ret.rolling(60).std().mean(axis=1) * np.sqrt(252)
    regime = pd.cut(ret60, bins=[-np.inf, -0.05, 0.05, np.inf], labels=["BEAR", "NEUTRAL", "BULL"])
    # vol 분위수로 분류
    vol_q33, vol_q66 = vol60.quantile(0.33), vol60.quantile(0.66)
    vol_regime = pd.cut(vol60, bins=[-np.inf, vol_q33, vol_q66, np.inf], labels=["LOW_VOL", "NORMAL", "HIGH_VOL"])
    df = pd.DataFrame({"ret60": ret60, "vol60": vol60, "regime": regime, "vol_regime": vol_regime})
    return df


print("시장 레짐 계산 중...", end=" ", flush=True)
MARKET_REGIME = build_market_regime()
print(f"{MARKET_REGIME.dropna().shape[0]}일")
print(f"  BULL: {(MARKET_REGIME['regime']=='BULL').sum()}, NEUTRAL: {(MARKET_REGIME['regime']=='NEUTRAL').sum()}, BEAR: {(MARKET_REGIME['regime']=='BEAR').sum()}")


# ──────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────
def load_subset(trade_types=None):
    if trade_types is None:
        trade_types = ["돌파매매"]
    frames = []
    for p in PRESETS_4:
        df = pd.read_parquet(CACHE / f"enriched_{p}.parquet")
        df = df[df["TradeType"].isin(trade_types)].copy()
        df["preset"] = p
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_consensus(df):
    return df.groupby(["Date", "Code"]).agg(
        n_presets=("preset", "nunique"),
        avg_score=("Score", "mean"),
        Name=("Name", "first"),
        Market=("Market", "first"),
        ChangeRatio=("ChangeRatio", "first"),
        Close=("Close", "first"),
        s5=("s5", "first"),
        TradeType=("TradeType", "first"),
    ).reset_index()


def classify_grade(row):
    if row["Market"] != "KOSDAQ": return None
    if row["n_presets"] < 1: return None
    cr = row["ChangeRatio"]
    if 7 <= cr <= 25 and row["avg_score"] >= 75: return "V"
    if 7 <= cr <= 25 and row["n_presets"] == 4 and row["avg_score"] >= 65: return "S"
    if 10 <= cr <= 18 and row["avg_score"] >= 65: return "A"
    if 7 <= cr <= 25: return "B"
    return None


def get_picks(trade_types=None, daily_top_n=1):
    """V/S/A/B 등급 매수 후보 picks 생성."""
    df = load_subset(trade_types=trade_types)
    cons = build_consensus(df)
    cons["grade"] = cons.apply(classify_grade, axis=1)
    grade_priority = {"V": 4, "S": 3, "A": 2, "B": 1}
    cons["_pri"] = cons["grade"].map(grade_priority).fillna(0)
    picks = cons[cons["grade"].notna()] \
        .sort_values(["Date", "_pri", "avg_score"], ascending=[True, False, False]) \
        .groupby("Date").head(daily_top_n).reset_index(drop=True)
    return picks


# ──────────────────────────────────────────────────────────
# 시뮬레이션 (익절/손절/트레일링 지원)
# ──────────────────────────────────────────────────────────
def simulate_trade(code, buy_date, buy_close, hold_days,
                   take_profit=None, trailing_stop=None, hard_stop=None):
    """단일 거래 시뮬레이션.

    Args:
        take_profit: 익절 (예: 0.5 = +50%) — 도달 시 즉시 청산
        trailing_stop: 트레일링 손절 (예: 0.2 = 고점 대비 -20%)
        hard_stop: 하드 손절 (예: 0.3 = 매수가 대비 -30%)

    Returns:
        dict: ret_pct, sell_date, sell_close, exit_reason
    """
    if code not in OHLCV:
        return None
    df = OHLCV[code]
    future = df[df.index > buy_date].head(hold_days)
    if len(future) == 0:
        return None

    peak = buy_close
    for dt, row in future.iterrows():
        # 일중 고가/저가 기반 청산
        high, low, close = float(row["High"]), float(row["Low"]), float(row["Close"])
        peak = max(peak, high)

        # 1. 익절 (high 가 익절가 터치)
        if take_profit is not None:
            tp_price = buy_close * (1 + take_profit)
            if high >= tp_price:
                ret = take_profit * 100
                return {
                    "ret_pct": ret, "sell_date": dt, "sell_close": tp_price,
                    "exit_reason": "TP",
                }
        # 2. 트레일링 손절 (low가 트레일링 손절가 터치)
        if trailing_stop is not None:
            ts_price = peak * (1 - trailing_stop)
            # 트레일링은 매수 직후엔 발동 안되게: 최소 1% 익은 다음부터
            if peak >= buy_close * 1.05 and low <= ts_price:
                ret = (ts_price - buy_close) / buy_close * 100
                return {
                    "ret_pct": ret, "sell_date": dt, "sell_close": ts_price,
                    "exit_reason": "TS",
                }
        # 3. 하드 손절
        if hard_stop is not None:
            hs_price = buy_close * (1 - hard_stop)
            if low <= hs_price:
                ret = -hard_stop * 100
                return {
                    "ret_pct": ret, "sell_date": dt, "sell_close": hs_price,
                    "exit_reason": "HS",
                }

    # 만기 청산
    last = future.iloc[-1]
    sell_close = float(last["Close"])
    ret_pct = (sell_close - buy_close) / buy_close * 100
    return {
        "ret_pct": ret_pct, "sell_date": future.index[-1], "sell_close": sell_close,
        "exit_reason": "EXP",
    }


def simulate(picks_df, hold_days, weight_fn=None,
             take_profit=None, trailing_stop=None, hard_stop=None,
             skip_bear=False, regime_weight_scale=None):
    """포지션 시뮬레이션.

    Args:
        weight_fn: callable(row) -> 비중(원). 기본은 WEIGHTS[grade].
        skip_bear: True 이면 BEAR 레짐에서 매수 안함
        regime_weight_scale: dict like {"BULL": 1.2, "BEAR": 0.5, ...} — 비중 스케일
    """
    trades = []
    for _, r in picks_df.iterrows():
        grade = r["grade"]
        if grade is None: continue
        buy_date = r["Date"]
        buy_close = r["Close"]
        code = r["Code"]

        # 레짐 필터
        if skip_bear or regime_weight_scale is not None:
            regime_date = buy_date
            # 최근 가용 레짐
            if regime_date in MARKET_REGIME.index:
                cur_regime = MARKET_REGIME.loc[regime_date, "regime"]
            else:
                # 가장 가까운 과거
                past = MARKET_REGIME.index[MARKET_REGIME.index <= regime_date]
                if len(past) == 0:
                    cur_regime = "NEUTRAL"
                else:
                    cur_regime = MARKET_REGIME.loc[past[-1], "regime"]
            if skip_bear and cur_regime == "BEAR":
                continue
            scale = 1.0
            if regime_weight_scale is not None:
                scale = regime_weight_scale.get(str(cur_regime), 1.0)
        else:
            cur_regime = None
            scale = 1.0

        # 비중
        if weight_fn is not None:
            base_weight = weight_fn(r)
        else:
            base_weight = WEIGHTS[grade]
        weight = base_weight * scale

        # 시뮬
        result = simulate_trade(
            code, buy_date, buy_close, hold_days,
            take_profit=take_profit, trailing_stop=trailing_stop, hard_stop=hard_stop,
        )
        if result is None:
            continue
        n_shares = weight / buy_close if buy_close > 0 else 0
        pnl = n_shares * (result["sell_close"] - buy_close)
        trades.append({
            "buy_date": buy_date,
            "sell_date": result["sell_date"],
            "grade": grade,
            "code": code,
            "name": r.get("Name", ""),
            "buy_close": buy_close,
            "sell_close": result["sell_close"],
            "weight": weight,
            "ret_pct": result["ret_pct"],
            "pnl": pnl,
            "exit_reason": result["exit_reason"],
            "regime": str(cur_regime) if cur_regime is not None else "",
        })
    return pd.DataFrame(trades)


def compute_capital_required(trades):
    """매일 동시보유 가치 → 최대 자본 계산."""
    if trades.empty:
        return {"max_capital": 0, "avg_capital": 0, "p95_capital": 0}
    start = trades["buy_date"].min()
    end = trades["sell_date"].max()
    dates = pd.date_range(start, end, freq="D")
    capital_series = []
    for d in dates:
        active = trades[(trades["buy_date"] <= d) & (trades["sell_date"] > d)]
        cap = active["weight"].sum()
        capital_series.append(cap)
    capital_series = pd.Series(capital_series, index=dates)
    return {
        "max_capital": float(capital_series.max()),
        "avg_capital": float(capital_series.mean()),
        "p95_capital": float(capital_series.quantile(0.95)),
    }


def metrics(trades, label=""):
    """모든 메트릭 계산."""
    if trades.empty:
        return None
    cap = compute_capital_required(trades)
    n = len(trades)
    total_pnl = trades["pnl"].sum()
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_sum = wins["pnl"].sum() if len(wins) else 0.0
    loss_sum = abs(losses["pnl"].sum()) if len(losses) else 1.0
    pf = win_sum / loss_sum if loss_sum > 0 else float("inf")
    big_loss_pct = (trades["ret_pct"] < -30).mean() * 100
    big_gain_pct = (trades["ret_pct"] > 100).mean() * 100
    win_rate = (trades["pnl"] > 0).mean() * 100
    avg_ret = trades["ret_pct"].mean()
    median_ret = trades["ret_pct"].median()
    years = 6.0
    roi_annual = (total_pnl / cap["max_capital"]) * 100 / years if cap["max_capital"] > 0 else 0
    roi_avg_cap = (total_pnl / cap["avg_capital"]) * 100 / years if cap["avg_capital"] > 0 else 0
    return {
        "label": label,
        "n_trades": n,
        "total_pnl": total_pnl,
        "max_capital": cap["max_capital"],
        "avg_capital": cap["avg_capital"],
        "roi_annual": roi_annual,         # %/yr on max capital
        "roi_avg_cap": roi_avg_cap,       # %/yr on avg capital
        "profit_factor": pf,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "median_ret": median_ret,
        "big_loss_pct": big_loss_pct,
        "big_gain_pct": big_gain_pct,
    }


def fmt(p):
    if abs(p) >= 1e8: return f"{p/1e8:+,.2f}억"
    if abs(p) >= 1e4: return f"{p/1e4:+,.0f}만"
    return f"{p:+,.0f}원"


def fmt_cap(p):
    if abs(p) >= 1e8: return f"{p/1e8:.2f}억"
    if abs(p) >= 1e4: return f"{p/1e4:,.0f}만"
    return f"{p:,.0f}원"


def print_metrics(m, indent=""):
    if m is None:
        print(f"{indent}(거래 없음)")
        return
    print(f"{indent}거래수      : {m['n_trades']:>4}건")
    print(f"{indent}6년 누적 손익: {fmt(m['total_pnl'])}")
    print(f"{indent}최대 자본    : {fmt_cap(m['max_capital'])}  (평균 {fmt_cap(m['avg_capital'])})")
    print(f"{indent}연 ROI       : {m['roi_annual']:+.1f}% (최대자본기준) · {m['roi_avg_cap']:+.1f}% (평균자본)")
    print(f"{indent}손익비       : {m['profit_factor']:.2f}")
    print(f"{indent}승률         : {m['win_rate']:.1f}%")
    print(f"{indent}평균/중간수익: {m['avg_ret']:+.1f}% / {m['median_ret']:+.1f}%")
    print(f"{indent}큰손실률     : {m['big_loss_pct']:.1f}% (return<-30%)")
    print(f"{indent}큰수익률     : {m['big_gain_pct']:.1f}% (return>+100%)")


# ──────────────────────────────────────────────────────────
# 메인 분석
# ──────────────────────────────────────────────────────────
ALL_RESULTS = []  # 모든 전략 결과 누적


def run_baseline():
    print("\n" + "=" * 100)
    print("【0】 베이스라인: V/S/A/B 180일 보유 (손절/익절 없음)")
    print("=" * 100)
    picks = get_picks()
    trades = simulate(picks, hold_days=180)
    m = metrics(trades, label="Baseline V/S/A/B 180d")
    print_metrics(m, indent="  ")
    ALL_RESULTS.append(m)
    return picks, m


def run_exit_strategy(picks):
    print("\n" + "=" * 100)
    print("【1】 익절/손절/트레일링 조합 최적화 (180일 보유)")
    print("=" * 100)
    # 매트릭스: 익절(없음/50/100/200) × 트레일링(없음/20/30) × 하드손절(없음/25)
    combos = []
    for tp in [None, 0.5, 1.0, 2.0]:
        for ts in [None, 0.20, 0.30]:
            for hs in [None, 0.25]:
                # 모두 없는 건 베이스라인이므로 skip
                if tp is None and ts is None and hs is None:
                    continue
                combos.append((tp, ts, hs))

    print(f"\n{len(combos)} 조합 테스트 중...")
    rows = []
    for tp, ts, hs in combos:
        label = f"TP{int(tp*100) if tp else '--'} TS{int(ts*100) if ts else '--'} HS{int(hs*100) if hs else '--'} 180d"
        trades = simulate(picks, hold_days=180, take_profit=tp, trailing_stop=ts, hard_stop=hs)
        m = metrics(trades, label=label)
        if m is None: continue
        rows.append(m)
    rows.sort(key=lambda x: x["total_pnl"], reverse=True)
    print(f"\n{'전략':<45} {'손익':>12} {'연ROI':>8} {'손익비':>6} {'큰손실':>7} {'큰수익':>7} {'승률':>6}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label']:<45} {fmt(r['total_pnl']):>12} "
              f"{r['roi_annual']:>+7.1f}% {r['profit_factor']:>6.2f} "
              f"{r['big_loss_pct']:>6.1f}% {r['big_gain_pct']:>6.1f}% {r['win_rate']:>5.1f}%")
        ALL_RESULTS.append(r)
    return rows


def run_hold_days(picks):
    print("\n" + "=" * 100)
    print("【2】 다른 보유 기간 (180/210/240/365일)")
    print("=" * 100)
    rows = []
    for hold in [180, 210, 240, 365]:
        trades = simulate(picks, hold_days=hold)
        m = metrics(trades, label=f"V/S/A/B {hold}d")
        if m: rows.append(m)
    print(f"\n{'전략':<25} {'손익':>12} {'연ROI':>8} {'손익비':>6} {'큰손실':>7} {'큰수익':>7} {'승률':>6}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label']:<25} {fmt(r['total_pnl']):>12} "
              f"{r['roi_annual']:>+7.1f}% {r['profit_factor']:>6.2f} "
              f"{r['big_loss_pct']:>6.1f}% {r['big_gain_pct']:>6.1f}% {r['win_rate']:>5.1f}%")
        ALL_RESULTS.append(r)

    # 365일에 익절 +200% 트레일링 30% 조합
    print("\n[보조] 365일 보유 + 익절 +200% + 트레일링 -30%")
    trades = simulate(picks, hold_days=365, take_profit=2.0, trailing_stop=0.30)
    m = metrics(trades, label="V/S/A/B 365d TP200 TS30")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)
    return rows


def run_regime_adaptive(picks):
    print("\n" + "=" * 100)
    print("【3】 시장 환경별 전략 — BULL/NEUTRAL/BEAR")
    print("=" * 100)

    # 각 레짐별 단순 통계
    print("\n3-1) 레짐별 단순 통계 (180일 보유, 익절/손절 없음)")
    trades_full = simulate(picks, hold_days=180)
    # 레짐 라벨 추가
    def label_regime(buy_date):
        past = MARKET_REGIME.index[MARKET_REGIME.index <= buy_date]
        if len(past) == 0: return "NA"
        return str(MARKET_REGIME.loc[past[-1], "regime"])
    trades_full["regime"] = trades_full["buy_date"].apply(label_regime)
    for reg in ["BULL", "NEUTRAL", "BEAR"]:
        sub = trades_full[trades_full["regime"] == reg]
        if len(sub) == 0: continue
        avg = sub["ret_pct"].mean()
        wr = (sub["ret_pct"] > 0).mean() * 100
        big_loss = (sub["ret_pct"] < -30).mean() * 100
        big_gain = (sub["ret_pct"] > 100).mean() * 100
        pnl = sub["pnl"].sum()
        print(f"  {reg:8s}: {len(sub):>4}건 · 평균 {avg:+.1f}% · 승률 {wr:.0f}% · "
              f"큰손실 {big_loss:.0f}% · 큰수익 {big_gain:.0f}% · 손익 {fmt(pnl)}")

    # BEAR 스킵 전략
    print("\n3-2) BEAR 레짐 스킵 (180일 보유)")
    trades = simulate(picks, hold_days=180, skip_bear=True)
    m = metrics(trades, label="V/S/A/B 180d skip-BEAR")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    # 레짐별 비중 조정
    print("\n3-3) 레짐별 비중 스케일 (BULL 1.5x, NEUTRAL 1.0x, BEAR 0.3x)")
    trades = simulate(picks, hold_days=180,
                      regime_weight_scale={"BULL": 1.5, "NEUTRAL": 1.0, "BEAR": 0.3})
    m = metrics(trades, label="V/S/A/B 180d regime-scale(1.5/1.0/0.3)")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    # 적극형
    print("\n3-4) 레짐별 비중 스케일 (BULL 2.0x, NEUTRAL 1.0x, BEAR 0.0x = 매수안함)")
    trades = simulate(picks, hold_days=180,
                      regime_weight_scale={"BULL": 2.0, "NEUTRAL": 1.0, "BEAR": 0.0})
    m = metrics(trades, label="V/S/A/B 180d regime-scale(2.0/1.0/0.0)")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    # 레짐 + 익절/트레일링 조합
    print("\n3-5) 레짐 스케일 + 익절 +100% + 트레일링 -30%")
    trades = simulate(picks, hold_days=180,
                      regime_weight_scale={"BULL": 1.5, "NEUTRAL": 1.0, "BEAR": 0.3},
                      take_profit=1.0, trailing_stop=0.30)
    m = metrics(trades, label="V/S/A/B 180d regime+TP100+TS30")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)


def run_vol_adaptive(picks):
    print("\n" + "=" * 100)
    print("【4】 변동성 적응형 비중 — VIX 같은 변동성에 따른 비중")
    print("=" * 100)
    # 변동성 레짐 정의 후, LOW_VOL 시 비중 증가, HIGH_VOL 시 감소

    def vol_scale_fn(row):
        buy_date = row["Date"]
        past = MARKET_REGIME.index[MARKET_REGIME.index <= buy_date]
        if len(past) == 0:
            return WEIGHTS[row["grade"]]
        vol_reg = str(MARKET_REGIME.loc[past[-1], "vol_regime"])
        base = WEIGHTS[row["grade"]]
        scale = {"LOW_VOL": 1.5, "NORMAL": 1.0, "HIGH_VOL": 0.5}.get(vol_reg, 1.0)
        return base * scale

    print("\n4-1) 변동성 스케일 (LOW 1.5x, NORMAL 1.0x, HIGH 0.5x) (180일 보유)")
    trades = simulate(picks, hold_days=180, weight_fn=vol_scale_fn)
    m = metrics(trades, label="V/S/A/B 180d vol-scale(1.5/1.0/0.5)")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    # 변동성별 통계
    print("\n4-2) 변동성 레짐별 단순 수익 (180일 보유)")
    trades_full = simulate(picks, hold_days=180)
    def label_vol(buy_date):
        past = MARKET_REGIME.index[MARKET_REGIME.index <= buy_date]
        if len(past) == 0: return "NA"
        return str(MARKET_REGIME.loc[past[-1], "vol_regime"])
    trades_full["vol_regime"] = trades_full["buy_date"].apply(label_vol)
    for reg in ["LOW_VOL", "NORMAL", "HIGH_VOL"]:
        sub = trades_full[trades_full["vol_regime"] == reg]
        if len(sub) == 0: continue
        avg = sub["ret_pct"].mean()
        wr = (sub["ret_pct"] > 0).mean() * 100
        big_loss = (sub["ret_pct"] < -30).mean() * 100
        big_gain = (sub["ret_pct"] > 100).mean() * 100
        pnl = sub["pnl"].sum()
        print(f"  {reg:9s}: {len(sub):>4}건 · 평균 {avg:+.1f}% · 승률 {wr:.0f}% · "
              f"큰손실 {big_loss:.0f}% · 큰수익 {big_gain:.0f}% · 손익 {fmt(pnl)}")


def run_trade_type():
    print("\n" + "=" * 100)
    print("【5/6】 눌림목매매 / 대시세 초입 통계")
    print("=" * 100)

    # 5) 눌림목매매
    print("\n5-1) 눌림목매매 단독 (V/S/A/B 등급 적용)")
    picks_p = get_picks(trade_types=["눌림목매매"])
    print(f"  후보 수: {len(picks_p)}건 (등급 분포: {picks_p['grade'].value_counts().to_dict()})")
    trades = simulate(picks_p, hold_days=180)
    m = metrics(trades, label="눌림목매매 V/S/A/B 180d")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    print("\n5-2) 눌림목매매 단독 + 익절 +50% + 트레일링 -20% (180일)")
    trades = simulate(picks_p, hold_days=180, take_profit=0.5, trailing_stop=0.20)
    m = metrics(trades, label="눌림목매매 180d TP50 TS20")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    # 6) 대시세 초입
    print("\n6-1) 대시세 초입 단독 (V/S/A/B 등급 적용)")
    picks_d = get_picks(trade_types=["대시세 초입"])
    print(f"  후보 수: {len(picks_d)}건 (등급 분포: {picks_d['grade'].value_counts().to_dict()})")
    trades = simulate(picks_d, hold_days=180)
    m = metrics(trades, label="대시세초입 V/S/A/B 180d")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    print("\n6-2) 대시세 초입 + 365일 보유 + 익절 +200% + 트레일링 -30%")
    trades = simulate(picks_d, hold_days=365, take_profit=2.0, trailing_stop=0.30)
    m = metrics(trades, label="대시세초입 365d TP200 TS30")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)

    # 통합: 돌파+눌림목+대시세
    print("\n[보조] 돌파매매 + 눌림목매매 + 대시세 초입 통합 (180일)")
    picks_all = get_picks(trade_types=["돌파매매", "눌림목매매", "대시세 초입"])
    print(f"  후보 수: {len(picks_all)}건")
    trades = simulate(picks_all, hold_days=180)
    m = metrics(trades, label="3타입통합 V/S/A/B 180d")
    print_metrics(m, indent="  ")
    if m: ALL_RESULTS.append(m)


def run_combo_search(picks):
    """베스트 후보 5~6가지 조합을 365일에 익절/트레일링 다양화."""
    print("\n" + "=" * 100)
    print("【7】 추가 콤보 탐색 — 365일 + 익절/트레일링 + 레짐")
    print("=" * 100)
    combos = [
        # (hold, tp, ts, hs, regime_scale, label)
        (240, 1.0, 0.30, None, None, "240d TP100 TS30"),
        (240, 2.0, 0.30, None, None, "240d TP200 TS30"),
        (365, 1.0, 0.30, None, None, "365d TP100 TS30"),
        (365, 2.0, 0.30, None, None, "365d TP200 TS30"),
        (365, None, 0.30, None, None, "365d TS30 only"),
        (240, 1.0, 0.30, None, {"BULL": 1.5, "NEUTRAL": 1.0, "BEAR": 0.3}, "240d TP100 TS30 +regime"),
        (240, 2.0, 0.30, None, {"BULL": 1.5, "NEUTRAL": 1.0, "BEAR": 0.3}, "240d TP200 TS30 +regime"),
        (180, 1.0, 0.30, None, {"BULL": 2.0, "NEUTRAL": 1.0, "BEAR": 0.0}, "180d TP100 TS30 +regime(2/1/0)"),
    ]
    rows = []
    for hold, tp, ts, hs, rs, label in combos:
        trades = simulate(picks, hold_days=hold, take_profit=tp, trailing_stop=ts,
                          hard_stop=hs, regime_weight_scale=rs)
        m = metrics(trades, label=label)
        if m: rows.append(m); ALL_RESULTS.append(m)
    print(f"\n{'전략':<40} {'손익':>12} {'연ROI':>8} {'손익비':>6} {'큰손실':>7} {'큰수익':>7} {'승률':>6}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label']:<40} {fmt(r['total_pnl']):>12} "
              f"{r['roi_annual']:>+7.1f}% {r['profit_factor']:>6.2f} "
              f"{r['big_loss_pct']:>6.1f}% {r['big_gain_pct']:>6.1f}% {r['win_rate']:>5.1f}%")


def write_md_report(baseline_m):
    """모든 결과 정렬 후 TOP 3 + 전체 표 MD로 저장."""
    # 자본효율(연ROI) 기준 정렬
    valid = [r for r in ALL_RESULTS if r is not None]
    by_pnl = sorted(valid, key=lambda x: x["total_pnl"], reverse=True)
    by_roi = sorted(valid, key=lambda x: x["roi_annual"], reverse=True)

    lines = []
    lines.append("# ADVANCED STRATEGIES — V/S/A/B 를 넘어서는 전략 탐색\n")
    lines.append("**분석일: 2026-05-23** · 데이터: 4 프리셋 enriched parquet (2020-04-03 ~ 2026-05-21)\n")
    lines.append("## 베이스라인\n")
    lines.append(f"- **V/S/A/B 180일 보유** (손절/익절 없음): "
                 f"6년 누적 {fmt(baseline_m['total_pnl'])}, 연ROI {baseline_m['roi_annual']:+.1f}%, "
                 f"손익비 {baseline_m['profit_factor']:.2f}, 큰손실 {baseline_m['big_loss_pct']:.1f}%, "
                 f"큰수익 {baseline_m['big_gain_pct']:.1f}%\n")
    lines.append("---\n")

    lines.append("## TOP 3 by 총 누적 손익\n")
    for i, r in enumerate(by_pnl[:3], 1):
        better = "✅ 더 나음" if r["total_pnl"] > baseline_m["total_pnl"] else "❌ 더 못함"
        lines.append(f"### #{i}. {r['label']} — {better}\n")
        lines.append(f"- 6년 누적 손익: **{fmt(r['total_pnl'])}** "
                     f"(베이스라인 대비 {(r['total_pnl']-baseline_m['total_pnl'])/abs(baseline_m['total_pnl'])*100:+.1f}%)")
        lines.append(f"- 연 ROI: **{r['roi_annual']:+.1f}%** (최대자본 {fmt_cap(r['max_capital'])})")
        lines.append(f"- 손익비 (PF): {r['profit_factor']:.2f} · 승률: {r['win_rate']:.1f}%")
        lines.append(f"- 큰손실률: {r['big_loss_pct']:.1f}% · 큰수익률: {r['big_gain_pct']:.1f}%")
        lines.append(f"- 거래수: {r['n_trades']}건 · 평균수익: {r['avg_ret']:+.1f}% · 중간값: {r['median_ret']:+.1f}%\n")
    lines.append("---\n")

    lines.append("## TOP 3 by 연 ROI (자본효율)\n")
    for i, r in enumerate(by_roi[:3], 1):
        better = "✅ 더 나음" if r["roi_annual"] > baseline_m["roi_annual"] else "❌ 더 못함"
        lines.append(f"### #{i}. {r['label']} — {better}\n")
        lines.append(f"- 연 ROI: **{r['roi_annual']:+.1f}%** "
                     f"(베이스라인 {baseline_m['roi_annual']:+.1f}% 대비 {r['roi_annual']-baseline_m['roi_annual']:+.1f}%p)")
        lines.append(f"- 6년 누적 손익: {fmt(r['total_pnl'])} (최대자본 {fmt_cap(r['max_capital'])})")
        lines.append(f"- 손익비 (PF): {r['profit_factor']:.2f} · 승률: {r['win_rate']:.1f}%")
        lines.append(f"- 큰손실률: {r['big_loss_pct']:.1f}% · 큰수익률: {r['big_gain_pct']:.1f}%")
        lines.append(f"- 거래수: {r['n_trades']}건\n")
    lines.append("---\n")

    # 전체 결과 표
    lines.append("## 전체 결과 표 (총 손익순)\n")
    lines.append("| 전략 | 6년 누적손익 | 연 ROI | 손익비 | 큰손실 | 큰수익 | 승률 | 거래수 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in by_pnl:
        lines.append(f"| {r['label']} | {fmt(r['total_pnl'])} | {r['roi_annual']:+.1f}% | "
                     f"{r['profit_factor']:.2f} | {r['big_loss_pct']:.1f}% | "
                     f"{r['big_gain_pct']:.1f}% | {r['win_rate']:.1f}% | {r['n_trades']} |")
    lines.append("\n---\n")

    lines.append("## 결론\n")
    best = by_pnl[0]
    if best["total_pnl"] > baseline_m["total_pnl"]:
        improvement = (best["total_pnl"] - baseline_m["total_pnl"]) / abs(baseline_m["total_pnl"]) * 100
        lines.append(f"**베스트 전략 — `{best['label']}`** 가 베이스라인 대비 {improvement:+.1f}% 더 나음.\n")
        lines.append(f"- 6년 누적 손익: {fmt(best['total_pnl'])} vs 베이스라인 {fmt(baseline_m['total_pnl'])}")
        lines.append(f"- 연 ROI: {best['roi_annual']:+.1f}% vs {baseline_m['roi_annual']:+.1f}%")
        lines.append(f"- 큰손실률: {best['big_loss_pct']:.1f}% vs {baseline_m['big_loss_pct']:.1f}%")
        lines.append(f"- 큰수익률: {best['big_gain_pct']:.1f}% vs {baseline_m['big_gain_pct']:.1f}%\n")
    else:
        lines.append("**베이스라인 V/S/A/B 180일이 여전히 최고** — 손절/익절을 추가해도 큰 개선 없음.\n")

    Path("ADVANCED_STRATEGIES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nADVANCED_STRATEGIES.md 저장 완료.")


def main():
    picks, baseline_m = run_baseline()
    run_exit_strategy(picks)
    run_hold_days(picks)
    run_regime_adaptive(picks)
    run_vol_adaptive(picks)
    run_trade_type()
    run_combo_search(picks)
    write_md_report(baseline_m)

    # 최종 요약 (콘솔)
    print("\n" + "=" * 100)
    print("🏆 최종 TOP 3 (총 누적 손익순)")
    print("=" * 100)
    valid = [r for r in ALL_RESULTS if r is not None]
    by_pnl = sorted(valid, key=lambda x: x["total_pnl"], reverse=True)[:3]
    for i, r in enumerate(by_pnl, 1):
        print(f"\n#{i}. {r['label']}")
        print(f"   6년 누적 손익: {fmt(r['total_pnl'])}")
        print(f"   연 ROI: {r['roi_annual']:+.1f}% (최대자본 {fmt_cap(r['max_capital'])})")
        print(f"   손익비: {r['profit_factor']:.2f} · 승률: {r['win_rate']:.0f}%")
        print(f"   큰손실: {r['big_loss_pct']:.0f}% · 큰수익: {r['big_gain_pct']:.0f}%")


if __name__ == "__main__":
    main()
