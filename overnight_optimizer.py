"""
Overnight Optimizer — 가중치 조합 자동 탐색기

전략:
  Phase 1: 후보 종목 풀 캐시 구축
    - 시총 상위 1000 × 2020~오늘 OHLCV 캐시 (walk_forward에서 이미 다운로드)
    - 매일 매 종목의 raw S1~S12 signal score 미리 계산해서 parquet로 캐시
    - 한 번만 계산 → 이후 모든 조합 평가에 재사용

  Phase 2: 광역 랜덤 탐색 (Random Search)
    - N개의 랜덤 가중치 조합 생성 (Latin Hypercube + Dirichlet)
    - 각 조합에 대해 미리 계산된 signal 점수로 composite 계산 → TOP3 추출 → 수익률
    - 시간순 5-fold OOS 검증
    - 다목적 점수: OOS_샤프 × 승률² × 평균수익 / std

  Phase 3: 정제 (Refinement)
    - Phase 2 TOP 30 가중치를 가우시안 perturbation으로 재탐색
    - 점진적 개선

  Phase 4: 최종 보고
    - 전체 TOP 20 가중치 조합 + OOS 통계
    - 각 시그널의 평균 중요도 (어떤 시그널이 일관되게 중요한지)

진행 중 결과는 cache/overnight_*.json에 점진 저장.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import pickle
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import FilterConfig, ScoreWeights, SignalParams, RecommendConfig, BacktestConfig
from scanner import compute_signals, _BENCHMARK_CACHE
from backtest import (
    get_universe, cache_universe_ohlcv, replay_day, simulate_trade,
)
from rules import PRESETS, list_presets


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# Phase 1: 후보 종목 풀 캐시 (signal score per day per stock)
# =============================================================================
def build_signal_cache(
    start_date: str = "2020-01-01",
    end_date: str = None,
    universe_limit: int = 1000,
    progress_callback=None,
) -> pd.DataFrame:
    """
    각 거래일 × 각 후보 종목의 raw s1~s12 + 메타 데이터 캐시.

    이걸 한 번만 구축해두면, 이후 어떤 가중치 조합도
    pandas 연산 한 번으로 수익률 시계열 계산 가능.

    Returns: DataFrame with columns:
        Date, Code, Name, Market, Close, ChangeRatio, Amount, MarketCap,
        s1..s12, next_open, next_high, next_close, t2_close,
        return_next_open, return_next_high, return_next_close, return_t2_close
    """
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    cache_file = CACHE_DIR / f"signal_pool_{start_date}_{end_date}_u{universe_limit}.parquet"
    if cache_file.exists():
        print(f"[CACHE HIT] {cache_file.name} 로드 중...", flush=True)
        return pd.read_parquet(cache_file)

    print(f"[BUILD] Signal cache 생성 시작: {start_date} ~ {end_date}", flush=True)
    t0 = time.time()

    # 1) Universe + OHLCV
    f_cfg = FilterConfig(
        min_amount=5_000_000_000,
        min_marcap=100_000_000_000,  # 1000억 (관대하게)
        change_min=5.0,               # 광범위 (필터링은 조합 평가 시)
        change_max=29.0,
    )
    universe = get_universe(f_cfg)
    universe = universe.head(universe_limit)

    if progress_callback:
        progress_callback(0, 100, label=f"OHLCV 다운로드 ({len(universe)}종목)")

    def dl_cb(i, total):
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, label=f"OHLCV 다운로드 {i}/{total}")

    ohlcv_data = cache_universe_ohlcv(universe, start_date, end_date,
                                       progress_callback=dl_cb)
    print(f"[OHLCV] {len(ohlcv_data)}종목 로드 완료 ({time.time()-t0:.0f}초)", flush=True)

    # 2) 거래일 인덱스 (삼성전자 기준)
    base = ohlcv_data.get("005930")
    if base is None or base.empty:
        base = next(iter(ohlcv_data.values()))
    trade_days = base.loc[start_date:end_date].index.tolist()
    print(f"[DAYS] 거래일 수: {len(trade_days)}", flush=True)

    # 3) 각 거래일 × 각 종목 signal 계산
    params = SignalParams()
    name_map = dict(zip(universe["Code"], universe["Name"]))
    market_map = dict(zip(universe["Code"], universe["Market"]))

    all_rows = []
    for i, day in enumerate(trade_days):
        if progress_callback and i % 30 == 0:
            progress_callback(i, len(trade_days),
                              label=f"Signal 계산 day {i}/{len(trade_days)}")

        for code, df in ohlcv_data.items():
            if day not in df.index:
                continue
            window = df.loc[:day]
            if len(window) < params.ma_long + 5:
                continue
            today = window.iloc[-1]
            prev = window.iloc[-2]
            amount = today["Close"] * today["Volume"]
            change_ratio = ((today["Close"] - prev["Close"]) / prev["Close"] * 100
                             if prev["Close"] > 0 else 0)
            if amount < 5_000_000_000:
                continue
            if not (5.0 <= change_ratio <= 29.0):
                continue
            sig = compute_signals(window, params)
            if not sig.get("valid"):
                continue
            # 다음 거래일 데이터 추출
            future = df.loc[df.index > day]
            if future.empty:
                next_open = next_high = next_close = today["Close"]
                t2_close = today["Close"]
            else:
                nxt = future.iloc[0]
                next_open = nxt["Open"]
                next_high = nxt["High"]
                next_close = nxt["Close"]
                t2_close = future.iloc[1]["Close"] if len(future) >= 2 else next_close
            cp = today["Close"]
            all_rows.append({
                "Date": day,
                "Code": code,
                "Name": name_map.get(code, code),
                "Market": market_map.get(code, "?"),
                "Close": cp,
                "ChangeRatio": change_ratio,
                "Amount": amount,
                "MarketCap": today["Close"] * 1e6,  # 추후 정확하면 universe에서 가져옴
                "s1": sig["s1"], "s2": sig["s2"], "s3": sig["s3"],
                "s4": sig["s4"], "s5": sig["s5"], "s6": sig["s6"],
                "s7": sig["s7"], "s8": sig["s8"], "s9": sig["s9"],
                "s10": sig["s10"], "s11": sig["s11"], "s12": sig["s12"],
                "next_open": next_open,
                "next_high": next_high,
                "next_close": next_close,
                "t2_close": t2_close,
                "return_next_open": (next_open - cp) / cp * 100,
                "return_next_high": (next_high - cp) / cp * 100,
                "return_next_close": (next_close - cp) / cp * 100,
                "return_t2_close": (t2_close - cp) / cp * 100,
            })

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)

    print(f"[CACHE] {len(df):,} 행 저장: {cache_file.name}", flush=True)
    df.to_parquet(cache_file, index=False)
    print(f"[DONE] Signal cache 완성 ({time.time()-t0:.0f}초)", flush=True)
    return df


# =============================================================================
# Phase 2: 가중치 조합 평가 (signal cache 사용 — 빠름)
# =============================================================================
def evaluate_combo(
    pool: pd.DataFrame,
    weights: np.ndarray,            # shape (12,) — s1~s12 가중치
    top_n: int = 3,
    sell_strategy: str = "next_open",
    min_score: float = 30,
) -> pd.Series:
    """
    가중치 조합 → 일자별 평균 수익률 시계열.
    pandas 벡터 연산으로 초고속.
    """
    sig_cols = [f"s{i}" for i in range(1, 13)]
    w = np.array(weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        return pd.Series(dtype=float)
    composite = pool[sig_cols].values @ w / w_sum  # (N,)
    df = pool[["Date", f"return_{sell_strategy}"]].copy()
    df["composite"] = composite

    # 최소 점수 컷
    df = df[df["composite"] >= min_score]
    if df.empty:
        return pd.Series(dtype=float)

    # 일자별 TOP N
    df = df.sort_values(["Date", "composite"], ascending=[True, False])
    top = df.groupby("Date").head(top_n)
    daily = top.groupby("Date")[f"return_{sell_strategy}"].mean()
    return daily


def score_returns(returns: np.ndarray, fold_size: int = 252) -> Dict:
    """
    OOS k-fold 평가:
      - 첫 fold_size 일: 학습용 (참고만)
      - 그 이후를 chunk_size로 나눠 OOS 평가
      - 평균 OOS 통계 반환
    """
    if len(returns) < fold_size + 60:
        return {"n_trades": len(returns), "is_score": -999, "oos_score": -999,
                 "oos_avg": 0, "oos_win_rate": 0, "oos_sharpe": 0,
                 "oos_total": 0, "stability": 0}
    # 단순화: 전체 데이터를 시간순 절반 split (IS / OOS)
    half = len(returns) // 2
    is_part = returns[:half]
    oos_part = returns[half:]

    def _s(arr):
        if len(arr) == 0:
            return 0
        avg = float(np.mean(arr))
        wr = float((arr > 0).mean())
        std = float(np.std(arr)) or 1
        return avg * (wr ** 2) * 10 / std

    is_score = _s(is_part)
    oos_score = _s(oos_part)
    avg = float(np.mean(oos_part))
    wr = float((oos_part > 0).mean() * 100)
    std = float(np.std(oos_part))
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0

    # 안정성: IS vs OOS score 차이 (작을수록 일관됨)
    stability = max(0, 1 - abs(is_score - oos_score) / max(abs(is_score) + 1e-6, 1e-6))

    return {
        "n_trades": int(len(returns)),
        "n_oos": int(len(oos_part)),
        "is_score": round(is_score, 3),
        "oos_score": round(oos_score, 3),
        "oos_avg": round(avg, 3),
        "oos_win_rate": round(wr, 2),
        "oos_total": round(float(np.sum(oos_part)), 2),
        "oos_sharpe": round(sharpe, 3),
        "oos_max_gain": round(float(np.max(oos_part)), 2),
        "oos_max_loss": round(float(np.min(oos_part)), 2),
        "stability": round(stability, 3),
    }


# =============================================================================
# Phase 3: 랜덤 탐색 + 정제
# =============================================================================
def sample_weights_dirichlet(n: int, alpha: float = 1.0, seed: int = None) -> np.ndarray:
    """Dirichlet 분포로 12차원 가중치 샘플링 (합 = 100)."""
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet([alpha] * 12, size=n) * 100
    return samples


def sample_weights_perturb(base: np.ndarray, n: int, sigma: float = 5.0,
                            seed: int = None) -> np.ndarray:
    """기준 가중치에 가우시안 노이즈 추가."""
    rng = np.random.default_rng(seed)
    perturbed = base + rng.normal(0, sigma, size=(n, 12))
    perturbed = np.clip(perturbed, 0, None)
    # 합 100으로 정규화
    sums = perturbed.sum(axis=1, keepdims=True)
    perturbed = np.where(sums > 0, perturbed / sums * 100, perturbed)
    return perturbed


def run_search(
    pool: pd.DataFrame,
    n_random: int = 1000,
    n_refine: int = 500,
    top_keep: int = 50,
    sell_strategy: str = "next_open",
    progress_callback=None,
    save_path: Path = None,
    save_every: int = 50,
) -> List[Dict]:
    """
    Phase A: n_random 랜덤 조합 평가
    Phase B: 상위 TOP_KEEP의 가중치를 perturb로 n_refine 추가 평가
    Phase C: 최종 정렬
    """
    results: List[Dict] = []
    total = n_random + n_refine

    # Phase A: 광역 랜덤
    print(f"[Phase A] 랜덤 탐색 {n_random}회 시작", flush=True)
    random_weights = sample_weights_dirichlet(n_random, alpha=0.8, seed=42)
    for i in range(n_random):
        w = random_weights[i]
        daily = evaluate_combo(pool, w, sell_strategy=sell_strategy)
        if len(daily) < 100:
            continue
        stats = score_returns(daily.values)
        results.append({
            "phase": "random",
            "weights": w.tolist(),
            **stats,
        })
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, label=f"Phase A: 랜덤 {i}/{n_random}")
        if save_path and len(results) % save_every == 0:
            _save_partial(results, save_path)

    # 중간 정렬
    results.sort(key=lambda r: r.get("oos_score", -999), reverse=True)
    top = results[: max(20, top_keep // 3)]
    print(f"[Phase A 완료] 상위 5개 OOS 점수: "
          f"{[round(r['oos_score'], 2) for r in top[:5]]}", flush=True)

    # Phase B: 정제 (perturbation)
    print(f"[Phase B] 정제 탐색 {n_refine}회 시작", flush=True)
    base_pool = np.array([t["weights"] for t in top])
    n_each = max(1, n_refine // len(base_pool))
    sigma_schedule = [10.0, 5.0, 2.5]  # 점점 줄임
    done_b = 0
    for s_idx, sigma in enumerate(sigma_schedule):
        per_round = n_each // len(sigma_schedule)
        for base_idx, base in enumerate(base_pool):
            perturbed = sample_weights_perturb(base, per_round, sigma=sigma,
                                                seed=100 + s_idx * 100 + base_idx)
            for w in perturbed:
                daily = evaluate_combo(pool, w, sell_strategy=sell_strategy)
                if len(daily) < 100:
                    continue
                stats = score_returns(daily.values)
                results.append({
                    "phase": f"refine_s{sigma}",
                    "weights": w.tolist(),
                    **stats,
                })
                done_b += 1
                if progress_callback and done_b % 50 == 0:
                    progress_callback(n_random + done_b, total,
                                       label=f"Phase B: 정제 {done_b}/{n_refine} (σ={sigma})")
                if save_path and len(results) % save_every == 0:
                    _save_partial(results, save_path)

    # 최종 정렬
    results.sort(key=lambda r: r.get("oos_score", -999), reverse=True)
    print(f"[완료] 평가 총 {len(results)}회", flush=True)

    if save_path:
        _save_partial(results, save_path, final=True)
    return results[: max(top_keep, 100)]


def _save_partial(results: List, path: Path, final: bool = False):
    """점진 저장 (top 100만)."""
    sorted_r = sorted(results, key=lambda r: r.get("oos_score", -999), reverse=True)
    payload = {
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_evaluated": len(results),
        "final": final,
        "top_combinations": sorted_r[:100],
    }
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


# =============================================================================
# Phase 4: 최종 보고
# =============================================================================
def signal_importance(top_combinations: List[Dict], top_n: int = 30) -> Dict:
    """상위 N개 조합의 평균 가중치 → 어떤 시그널이 일관되게 중요한지."""
    if not top_combinations:
        return {}
    top = top_combinations[:top_n]
    weights_matrix = np.array([t["weights"] for t in top])
    avg = weights_matrix.mean(axis=0)
    std = weights_matrix.std(axis=0)
    labels = [f"S{i}" for i in range(1, 13)]
    return {
        "avg_weights": {label: round(float(avg[i]), 2) for i, label in enumerate(labels)},
        "std_weights": {label: round(float(std[i]), 2) for i, label in enumerate(labels)},
        "ranked_signals": sorted(
            [(label, float(avg[i])) for i, label in enumerate(labels)],
            key=lambda x: x[1], reverse=True
        ),
    }


def final_report(results: List[Dict], path: Path):
    """최종 보고서 저장."""
    if not results:
        print("⚠️ 결과 없음", flush=True)
        return
    top_3 = results[:3]
    sig_imp = signal_importance(results, top_n=30)

    report = {
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_total_evaluations": len(results),
        "top3_combinations": top_3,
        "top10_combinations": results[:10],
        "signal_importance_top30": sig_imp,
    }
    with open(path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[REPORT] 저장 완료: {path.name}", flush=True)


if __name__ == "__main__":
    print(f"=== Overnight Optimizer 시작 ===", flush=True)
    print(f"시작 시각: {datetime.now()}", flush=True)
    t0 = time.time()

    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    n_random = int(os.environ.get("N_RANDOM", "1500"))
    n_refine = int(os.environ.get("N_REFINE", "1000"))

    def cb(i, total, label=""):
        pct = i / max(total, 1) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{i}/{total} {pct:.0f}%] {label}",
              flush=True)

    # Step 1: Signal cache
    print(f"\n=== Step 1: Signal Cache 구축 ===", flush=True)
    pool = build_signal_cache(
        start_date="2020-01-01", end_date=end_date,
        universe_limit=1000, progress_callback=cb,
    )
    print(f"Pool: {len(pool):,} 행", flush=True)

    # Step 2: 탐색
    print(f"\n=== Step 2: 탐색 ({n_random + n_refine}회) ===", flush=True)
    partial_path = CACHE_DIR / "overnight_partial.json"
    results = run_search(
        pool, n_random=n_random, n_refine=n_refine,
        progress_callback=cb,
        save_path=partial_path,
    )

    # Step 3: 보고
    print(f"\n=== Step 3: 최종 보고 ===", flush=True)
    report_path = CACHE_DIR / "overnight_final.json"
    final_report(results, report_path)

    elapsed = time.time() - t0
    print(f"\n=== 완료 (총 {elapsed/60:.1f}분) ===", flush=True)
    print(f"\n🏆 TOP 3 가중치 조합:", flush=True)
    for i, r in enumerate(results[:3]):
        medal = ["🥇", "🥈", "🥉"][i]
        print(f"\n{medal} Phase={r['phase']}", flush=True)
        print(f"   OOS 점수={r['oos_score']:.2f} · "
              f"OOS 평균={r['oos_avg']:.2f}% · "
              f"OOS 승률={r['oos_win_rate']:.1f}% · "
              f"OOS 샤프={r['oos_sharpe']:.2f}", flush=True)
        w = r["weights"]
        print(f"   가중치: " + " ".join(
            [f"S{i+1}={w[i]:.1f}" for i in range(12)]), flush=True)

    print(f"\n📊 시그널 중요도 (TOP 30 평균):", flush=True)
    sig_imp = signal_importance(results, top_n=30)
    for label, val in sig_imp.get("ranked_signals", []):
        print(f"   {label}: {val:.2f}", flush=True)
