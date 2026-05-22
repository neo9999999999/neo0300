"""
히스토리 스캔 모듈 — 년/월별 일자별 추천 복원

특정 년/월을 선택하면 그 기간의 매 거래일에 대해 종가매수 추천을
재현하여 보여줌. 백테스트 모듈을 활용.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import calendar
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import FilterConfig, ScoreWeights, SignalParams, RecommendConfig, BacktestConfig
from backtest import run_backtest, summarize_backtest


def get_month_date_range(year: int, month: int) -> tuple:
    """선택한 년/월의 시작/종료 일자 반환."""
    start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end = f"{year:04d}-{month:02d}-{last_day:02d}"
    return start, end


def scan_historical_month(
    year: int,
    month: int,
    filter_cfg: FilterConfig,
    weights: ScoreWeights,
    params: SignalParams,
    recommend_cfg: RecommendConfig,
    sell_strategy: str = "next_open",
    universe_limit: Optional[int] = 300,
    progress_callback=None,
) -> Dict:
    """단일 년/월 스캔 (하위 호환용)."""
    start, end = get_month_date_range(year, month)
    bt_cfg = BacktestConfig(start_date=start, end_date=end, sell_strategy=sell_strategy)
    picks = run_backtest(
        start, end,
        filter_cfg, weights, params, recommend_cfg, bt_cfg,
        universe_limit=universe_limit,
        progress_callback=progress_callback,
    )
    return {
        "recommendations": picks,
        "stats": summarize_backtest(picks) if not picks.empty else {},
        "period": (start, end),
    }


def scan_historical_period(
    years: List[int],
    months: List[int],
    filter_cfg: FilterConfig,
    weights: ScoreWeights,
    params: SignalParams,
    recommend_cfg: RecommendConfig,
    sell_strategy: str = "next_open",
    universe_limit: Optional[int] = 300,
    progress_callback=None,
) -> Dict:
    """
    다중 년도 × 다중 월 스캔.
    예: years=[2024, 2025], months=[3,4,5] → 2024년 3/4/5월 + 2025년 3/4/5월

    효율성: 가장 이른 날부터 가장 늦은 날까지 한 번에 백테스트하고,
            선택된 월의 데이터만 필터.
    """
    if not years or not months:
        return {"recommendations": pd.DataFrame(), "stats": {}, "period": (None, None)}

    years = sorted(set(years))
    months = sorted(set(months))

    # 전체 기간 시작/종료
    min_year, max_year = years[0], years[-1]
    min_month, max_month = months[0], months[-1]
    start = f"{min_year:04d}-{min_month:02d}-01"
    last_day = calendar.monthrange(max_year, max_month)[1]
    end = f"{max_year:04d}-{max_month:02d}-{last_day:02d}"

    bt_cfg = BacktestConfig(start_date=start, end_date=end, sell_strategy=sell_strategy)

    picks = run_backtest(
        start, end,
        filter_cfg, weights, params, recommend_cfg, bt_cfg,
        universe_limit=universe_limit,
        progress_callback=progress_callback,
    )

    # 선택된 년/월에 해당하는 행만 필터
    if not picks.empty and "Date" in picks.columns:
        picks["_dt"] = pd.to_datetime(picks["Date"])
        picks["_year"] = picks["_dt"].dt.year
        picks["_month"] = picks["_dt"].dt.month
        picks = picks[
            picks["_year"].isin(years) & picks["_month"].isin(months)
        ].drop(columns=["_dt", "_year", "_month"]).reset_index(drop=True)

    return {
        "recommendations": picks,
        "stats": summarize_backtest(picks) if not picks.empty else {},
        "period": (start, end),
        "years": years,
        "months": months,
    }


def summarize_by_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """년/월별 집계."""
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["YM"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m")
    if "return_pct" not in df.columns:
        return df.groupby("YM").size().reset_index(name="건수")
    grouped = df.groupby("YM").agg(
        건수=("Code", "count"),
        평균수익률=("return_pct", "mean"),
        승률=("return_pct", lambda x: (x > 0).mean() * 100),
        평균점수=("Score", "mean"),
    ).round(2).reset_index()
    return grouped.sort_values("YM")


def filter_by_trade_type(df: pd.DataFrame, trade_type: str) -> pd.DataFrame:
    """매매 타입 필터."""
    if df is None or df.empty or trade_type == "전체":
        return df
    return df[df.get("TradeType", "") == trade_type].reset_index(drop=True)


def get_available_years() -> List[int]:
    """사용 가능한 년도 (2020~현재)."""
    current_year = datetime.now().year
    return list(range(2020, current_year + 1))


def get_months() -> List[int]:
    return list(range(1, 13))


def summarize_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """패턴별 집계."""
    if df is None or df.empty or "TradeType" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("TradeType").agg(
        건수=("Code", "count"),
        평균수익률=("return_pct", "mean"),
        승률=("return_pct", lambda x: (x > 0).mean() * 100),
        평균점수=("Score", "mean"),
    ).round(2).reset_index()
    return grouped.sort_values("건수", ascending=False)
