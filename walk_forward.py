"""
Walk-Forward Out-Of-Sample (OOS) 검증

핵심 원칙:
  - 각 "결정 시점(decision point)"에서:
    1) 그 시점 이전 데이터로만 9개 프리셋을 평가 (In-Sample)
    2) TOP 3 프리셋 선정
    3) 다음 기간(test chunk)에 적용 → OOS 수익률 기록
  - 미래 데이터를 절대 사용하지 않음 (look-ahead bias 제거)
  - 모든 결정 시점의 OOS 결과를 누적 → 진짜 OOS 성과

이렇게 하면 "과거 데이터 다 보고 사후적으로 좋은 거 골랐다" (overfitting) 가 아니라,
"그 시점에 진짜로 선택했다면 그 다음 결과가 어땠다" (genuine OOS) 가 됨.

워크플로:
  Step 1: 9개 프리셋을 전체 기간(2020~오늘) 백테스트 (OHLCV 다운로드 캐시)
          → 각 프리셋별 일자별 수익률 시계열 확보
  Step 2: 결정 시점들을 정의 (예: 분기마다)
  Step 3: 각 결정 시점에서:
          - 이전 데이터 기반으로 9개 프리셋 점수 산출
          - TOP 3 선정
          - 다음 청크에 그 TOP 3 적용 → OOS 수익률 기록
  Step 4: 누적 OOS 결과 분석
          - Consensus TOP 3: 결정 시점들에서 가장 자주 TOP 3에 든 프리셋
          - 프리셋별 OOS 통계
          - 안정성 (TOP 1이 얼마나 자주 바뀌었는지)
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import FilterConfig, ScoreWeights, SignalParams, RecommendConfig, BacktestConfig
from backtest import run_backtest
from rules import PRESETS, list_presets


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# 헬퍼: 프리셋 → 설정 객체
# =============================================================================
def _build_configs_for_preset(preset_key: str):
    p = PRESETS[preset_key]
    f_cfg = FilterConfig(
        min_amount=p["filter"]["min_amount_eok"] * 100_000_000,
        min_marcap=p["filter"]["min_marcap_eok"] * 100_000_000,
        change_min=p["filter"]["change_min"],
        change_max=p["filter"]["change_max"],
    )
    w = p["weights"]
    weights = ScoreWeights(
        s1_box_breakout=w.get("s1", 18),
        s2_volume_surge=w.get("s2", 18),
        s3_long_candle=w.get("s3", 8),
        s4_ma_alignment=w.get("s4", 8),
        s5_near_high=w.get("s5", 10),
        s6_no_overheating=w.get("s6", 4),
        s7_pullback_setup=w.get("s7", 5),
        s8_demand_continuity=w.get("s8", 4),
        s9_longterm_ma_breakout=w.get("s9", 7),
        s10_relative_strength=w.get("s10", 7),
        s11_gap_ma_confluence=w.get("s11", 5),
        s12_pattern_quality=w.get("s12", 6),
        bonus_watchlist=w.get("bonus", 5),
    )
    rec_cfg = RecommendConfig(top_n=3, min_score=p["min_score"])
    return f_cfg, weights, rec_cfg


# =============================================================================
# Step 1: 전체 기간 백테스트 (9개 프리셋, OHLCV 캐싱 공유)
# =============================================================================
def run_all_presets_full(
    start_date: str = "2020-01-01",
    end_date: str = None,
    universe_limit: int = 1000,
    sell_strategy: str = "next_open",
    progress_callback=None,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    9개 프리셋을 동일 기간에 모두 백테스트.
    OHLCV는 한 번만 다운로드 후 캐시 공유 → 9번 다운로드 ❌
    """
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    cache_file = CACHE_DIR / f"wf_full_{start_date}_{end_date}_u{universe_limit}.pkl"
    if cache_file.exists() and not force_refresh:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    params = SignalParams()
    results_by_preset: Dict[str, pd.DataFrame] = {}
    preset_keys = list_presets()
    total = len(preset_keys)

    for i, key in enumerate(preset_keys):
        if progress_callback:
            progress_callback(i, total, label=f"[{i+1}/{total}] {PRESETS[key]['name']} 백테스트")

        f_cfg, weights, rec_cfg = _build_configs_for_preset(key)
        bt_cfg = BacktestConfig(
            start_date=start_date, end_date=end_date, sell_strategy=sell_strategy,
        )
        try:
            trades = run_backtest(
                start_date, end_date,
                f_cfg, weights, params, rec_cfg, bt_cfg,
                universe_limit=universe_limit,
            )
            results_by_preset[key] = trades
        except Exception as e:
            results_by_preset[key] = pd.DataFrame()

    with open(cache_file, "wb") as f:
        pickle.dump(results_by_preset, f)
    return results_by_preset


# =============================================================================
# 헬퍼: 수익률 시계열 → 점수 (Sharpe-유사)
# =============================================================================
def _score_returns(returns: np.ndarray, min_trades: int = 10) -> float:
    if len(returns) < min_trades:
        return -999.0
    avg = np.mean(returns)
    win_rate = (returns > 0).mean()
    std = np.std(returns) or 1
    # 평균수익 × 승률² × 10 / 표준편차
    return float(avg * (win_rate ** 2) * 10 / std)


def _stats_from_returns(returns: np.ndarray) -> Dict:
    if len(returns) == 0:
        return {}
    return {
        "n_trades": int(len(returns)),
        "avg_return": float(np.mean(returns)),
        "win_rate": float((returns > 0).mean() * 100),
        "total_return": float(np.sum(returns)),
        "std": float(np.std(returns)),
        "sharpe_annual": float(np.mean(returns) / np.std(returns) * np.sqrt(252))
                          if np.std(returns) > 0 else 0,
        "max_gain": float(np.max(returns)),
        "max_loss": float(np.min(returns)),
    }


# =============================================================================
# Step 2~3: Walk-Forward 검증
# =============================================================================
def walk_forward_validation(
    start_date: str = "2020-01-01",
    end_date: str = None,
    train_min_days: int = 252,        # 최초 학습 기간 (1년)
    test_chunk_days: int = 63,        # OOS 청크 (약 분기)
    universe_limit: int = 1000,
    sell_strategy: str = "next_open",
    progress_callback=None,
    force_refresh: bool = False,
) -> Dict:
    """
    Walk-Forward OOS 검증.

    Returns:
        {
            "consensus_top3": [{preset_key, preset_name, votes, oos_*}, ...],
            "decision_points": [...],
            "per_preset_oos": {preset_key: stats},
            "period": (start, end),
            "n_decision_points": int,
            "stability_score": float,    # 0~1 (1=매번 동일 TOP1, 0=계속 바뀜)
            "computed_at": str,
        }
    """
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    cache_file = CACHE_DIR / f"wf_oos_{start_date}_{end_date}_u{universe_limit}_t{train_min_days}_c{test_chunk_days}.json"
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass

    # Step 1: 전체 기간 백테스트 (캐시 활용)
    if progress_callback:
        progress_callback(0, 100, label="Step 1/3: 9개 프리셋 전체 백테스트 (OHLCV 다운로드 포함)")

    all_trades = run_all_presets_full(
        start_date, end_date, universe_limit,
        sell_strategy=sell_strategy,
        progress_callback=progress_callback,
        force_refresh=force_refresh,
    )

    # Step 2: 프리셋별 일자별 수익률 시계열 구축
    daily_by_preset: Dict[str, pd.Series] = {}
    for key, trades in all_trades.items():
        if trades is None or trades.empty or "return_pct" not in trades.columns:
            continue
        df = trades.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        daily = df.groupby("Date")["return_pct"].mean().sort_index()
        if len(daily) > 0:
            daily_by_preset[key] = daily

    if not daily_by_preset:
        result = {"error": "백테스트 결과가 비어있음. 캐시/데이터 확인 필요."}
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result

    # Step 3: Walk-forward
    if progress_callback:
        progress_callback(95, 100, label="Step 2/3: Walk-Forward 결정 시점 분석")

    # 모든 결정 시점에 사용할 거래일 집합
    all_dates = sorted(set().union(*[set(s.index) for s in daily_by_preset.values()]))

    if len(all_dates) < train_min_days + test_chunk_days:
        return {"error": f"데이터 부족: {len(all_dates)}일 < 필요 {train_min_days + test_chunk_days}일"}

    decision_points: List[Dict] = []
    oos_returns_by_preset: Dict[str, List[float]] = {k: [] for k in daily_by_preset}

    dp_idx = train_min_days
    while dp_idx + test_chunk_days <= len(all_dates):
        train_dates = all_dates[:dp_idx]
        test_dates = all_dates[dp_idx : dp_idx + test_chunk_days]
        train_start = pd.Timestamp(train_dates[0])
        train_end = pd.Timestamp(train_dates[-1])
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])

        # IS 점수
        is_scores = {}
        for key, daily in daily_by_preset.items():
            is_returns = daily.loc[train_start:train_end].values
            is_scores[key] = _score_returns(is_returns)

        ranked = sorted(is_scores.items(), key=lambda x: x[1], reverse=True)
        top3_at_dp = [k for k, s in ranked[:3] if s > -100]

        # OOS 적용
        dp_oos_perf = {}
        for key in top3_at_dp:
            oos_slice = daily_by_preset[key].loc[test_start:test_end]
            oos_values = oos_slice.values
            if len(oos_values) > 0:
                oos_returns_by_preset[key].extend(oos_values.tolist())
                dp_oos_perf[key] = {
                    "avg": float(np.mean(oos_values)),
                    "n": len(oos_values),
                    "win_rate": float((oos_values > 0).mean() * 100),
                }

        decision_points.append({
            "decision_date": train_end.strftime("%Y-%m-%d"),
            "test_period": f"{test_start:%Y-%m-%d} ~ {test_end:%Y-%m-%d}",
            "top1": top3_at_dp[0] if len(top3_at_dp) > 0 else None,
            "top1_name": PRESETS.get(top3_at_dp[0], {}).get("name", "?") if len(top3_at_dp) > 0 else "?",
            "top3": top3_at_dp,
            "top3_names": [PRESETS[k]["name"] for k in top3_at_dp],
            "top3_is_scores": [{"key": k, "name": PRESETS[k]["name"], "is_score": round(is_scores[k], 3)}
                                for k in top3_at_dp],
            "oos_chunk_perf": dp_oos_perf,
        })

        dp_idx += test_chunk_days

    # Step 4: 집계
    if progress_callback:
        progress_callback(99, 100, label="Step 3/3: 합의 TOP 3 집계")

    # 투표 카운트
    vote_count = {}
    top1_count = {}
    for dp in decision_points:
        for preset in dp["top3"]:
            vote_count[preset] = vote_count.get(preset, 0) + 1
        if dp["top1"]:
            top1_count[dp["top1"]] = top1_count.get(dp["top1"], 0) + 1

    # 프리셋별 OOS 통계
    per_preset_oos = {}
    for key, returns_list in oos_returns_by_preset.items():
        if not returns_list:
            continue
        arr = np.array(returns_list)
        stats = _stats_from_returns(arr)
        per_preset_oos[key] = {
            "preset_key": key,
            "preset_name": PRESETS[key]["name"],
            "preset_desc": PRESETS[key]["desc"],
            "votes_in_top3": int(vote_count.get(key, 0)),
            "votes_as_top1": int(top1_count.get(key, 0)),
            **stats,
        }

    # Consensus TOP 3: 투표수 × OOS Sharpe 종합
    def _consensus_score(d: Dict) -> float:
        # 투표수 가중 (정규화) + OOS Sharpe
        n_dp = max(len(decision_points), 1)
        vote_pct = d["votes_in_top3"] / n_dp
        sharpe = d.get("sharpe_annual", 0)
        return vote_pct * 100 + sharpe * 10

    consensus_top3 = sorted(per_preset_oos.values(),
                             key=_consensus_score, reverse=True)[:3]

    # 안정성 (TOP1이 얼마나 자주 바뀌는가)
    if decision_points:
        unique_top1 = set(dp["top1"] for dp in decision_points if dp["top1"])
        stability_score = round(1.0 - (len(unique_top1) - 1) / max(len(decision_points), 1), 3)
    else:
        stability_score = 0

    result = {
        "consensus_top3": consensus_top3,
        "decision_points": decision_points,
        "per_preset_oos": per_preset_oos,
        "period": (start_date, end_date),
        "n_decision_points": len(decision_points),
        "train_min_days": train_min_days,
        "test_chunk_days": test_chunk_days,
        "universe_limit": universe_limit,
        "stability_score": stability_score,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "methodology": (
            "Walk-Forward OOS: 각 결정 시점에서 그 이전 데이터로만 TOP 3 선정 → "
            "다음 청크에 적용 → 누적 OOS 수익률. Look-ahead bias 없음."
        ),
    }

    with open(cache_file, "w") as f:
        json.dump(result, f, default=str, ensure_ascii=False, indent=2)
    return result


def get_cached_walk_forward() -> Optional[Dict]:
    """가장 최근 캐시된 walk-forward 결과."""
    files = list(CACHE_DIR.glob("wf_oos_*.json"))
    if not files:
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception:
        return None
