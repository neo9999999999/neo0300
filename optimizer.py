"""
자동 최적화 모듈

각 프리셋(default, conservative, aggressive, box_breakout, habarocell,
haseunghoon, pullback, mega_trend, master_guide)을 동일한 기간에
백테스트하고, 가장 좋은 성과(샤프비율 × 승률)를 보인 프리셋을 자동 선정.

결과를 캐싱하여 다음번엔 즉시 반환.
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
from backtest import run_backtest, summarize_backtest
from rules import PRESETS, list_presets


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _score_preset(stats: Dict) -> float:
    """
    프리셋 종합 점수 (높을수록 좋음):
      평균수익률 × 승률 / 표준편차
    """
    if not stats:
        return -999
    avg = stats.get("평균 수익률(%)", 0) or 0
    win_rate = (stats.get("승률(%)", 50) or 50) / 100
    std = stats.get("표준편차", 5) or 5
    n = stats.get("총 추천 건수", 0) or 0
    if n < 10:
        return -100  # 표본 부족 시 무시
    # 평균수익 × (승률^2) / std — 승률을 강하게 반영
    return round(avg * (win_rate ** 2) * 10 / max(std, 1), 3)


def run_full_backtest(
    start_date: str = "2020-01-01",
    end_date: str = None,
    universe_limit: int = 1000,
    sell_strategy: str = "next_open",
    progress_callback=None,
    force_refresh: bool = False,
) -> Dict:
    """
    9개 프리셋 모두 전체 기간 백테스트 후 TOP 3 추출.

    Returns: optimize_preset과 동일 구조 + 'top3' 필드
    """
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    result = optimize_preset(
        start_date=start_date, end_date=end_date,
        universe_limit=universe_limit, sell_strategy=sell_strategy,
        progress_callback=progress_callback, force_refresh=force_refresh,
    )
    # TOP 3 추출
    valid = [r for r in result["all_results"] if r["score"] > -100]
    top3 = sorted(valid, key=lambda r: r["score"], reverse=True)[:3]
    result["top3"] = top3
    return result


def optimize_preset(
    start_date: str = None,
    end_date: str = None,
    universe_limit: int = 1000,
    sell_strategy: str = "next_open",
    progress_callback=None,
    force_refresh: bool = False,
) -> Dict:
    """
    9개 프리셋을 모두 백테스트하고 최고 성과 프리셋 선정.

    Returns:
        {
            'best_preset': str,         # 예: 'haseunghoon'
            'best_score': float,
            'all_results': List[Dict],  # 모든 프리셋 결과
            'period': (start, end),
        }
    """
    # 기본 기간: 최근 1년
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    cache_key = f"opt_{start_date}_{end_date}_u{universe_limit}_{sell_strategy}.json"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists() and not force_refresh:
        with open(cache_file) as f:
            return json.load(f)

    all_results = []
    preset_keys = list_presets()
    total_steps = len(preset_keys)

    for i, key in enumerate(preset_keys):
        p = PRESETS[key]
        if progress_callback:
            progress_callback(i, total_steps, label=f"{p['name']} 백테스트 중...")

        # 프리셋의 필터/가중치 구성
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
        params = SignalParams()
        rec_cfg = RecommendConfig(top_n=3, min_score=p["min_score"])
        bt_cfg = BacktestConfig(start_date=start_date, end_date=end_date,
                                sell_strategy=sell_strategy)

        try:
            results = run_backtest(
                start_date, end_date,
                f_cfg, weights, params, rec_cfg, bt_cfg,
                universe_limit=universe_limit,
            )
            stats = summarize_backtest(results)
            score = _score_preset(stats)
            all_results.append({
                "preset_key": key,
                "preset_name": p["name"],
                "score": score,
                "stats": stats,
            })
        except Exception as e:
            all_results.append({
                "preset_key": key,
                "preset_name": p["name"],
                "score": -999,
                "stats": {},
                "error": str(e),
            })

    # 최고 성과 프리셋 선정
    valid_results = [r for r in all_results if r["score"] > -100]
    if not valid_results:
        best = {"preset_key": "default", "preset_name": "기본 (균형)",
                "score": 0, "stats": {}}
    else:
        best = max(valid_results, key=lambda r: r["score"])

    output = {
        "best_preset": best["preset_key"],
        "best_preset_name": best["preset_name"],
        "best_score": best["score"],
        "best_stats": best.get("stats", {}),
        "all_results": all_results,
        "period": (start_date, end_date),
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(cache_file, "w") as f:
        json.dump(output, f, default=str, ensure_ascii=False, indent=2)
    return output


def get_cached_optimization() -> Optional[Dict]:
    """가장 최근 캐시된 최적화 결과 반환."""
    files = list(CACHE_DIR.glob("opt_*.json"))
    if not files:
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception:
        return None
