"""
백테스트 모듈 — 2020~현재까지의 일별 추천 + 다음 날 수익률 검증.

주의: FinanceDataReader StockListing은 *현재 시점* 스냅샷만 제공.
백테스트는 universe 종목들의 OHLCV를 미리 캐싱한 뒤 day-by-day 재현.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
import pickle
from datetime import datetime

from config import FilterConfig, ScoreWeights, SignalParams, RecommendConfig, BacktestConfig
from scanner import compute_signals, total_score


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_universe(filter_cfg: FilterConfig) -> pd.DataFrame:
    """백테스트 유니버스 (현재 시점 기준 시총/시장 조건만 적용한 후보군)."""
    frames = []
    if filter_cfg.include_kospi:
        frames.append(fdr.StockListing("KOSPI").assign(Market="KOSPI"))
    if filter_cfg.include_kosdaq:
        frames.append(fdr.StockListing("KOSDAQ").assign(Market="KOSDAQ"))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Marcap": "MarketCap", "ChagesRatio": "ChangeRatio"})
    if filter_cfg.exclude_etf:
        exclude_kw = ["ETF", "ETN", "스팩", "SPAC", "리츠", "REIT"]
        df = df[~df["Name"].str.contains("|".join(exclude_kw), na=False)]
    # 현재 시총 기준으로 1차 거름 (백테스트 시 너무 작던 종목 제외)
    df = df[df["MarketCap"] >= filter_cfg.min_marcap * 0.5]
    return df[["Code", "Name", "Market", "MarketCap"]].reset_index(drop=True)


import signal


class _DLTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _DLTimeout()


def fetch_long_ohlcv(code: str, start: str, end: str, timeout: int = 10) -> Optional[pd.DataFrame]:
    """
    단일 종목 OHLCV. signal.alarm으로 강제 timeout (메인 스레드 전용).
    """
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        df = fdr.DataReader(code, start, end)
        signal.alarm(0)
        if df is None or df.empty:
            return None
        return df
    except (_DLTimeout, Exception):
        signal.alarm(0)
        return None
    finally:
        signal.alarm(0)


def cache_universe_ohlcv(
    universe: pd.DataFrame, start: str, end: str, force: bool = False,
    progress_callback=None, max_workers: int = 1, per_future_timeout: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    유니버스 종목들의 OHLCV를 직렬로 다운로드 (signal.alarm timeout 사용).

    - 메인 스레드 직렬 다운로드 → signal.alarm 강제 timeout 가능
    - 각 종목 10초 timeout (정상 0.1초)
    - 매 50종목마다 부분 캐시 저장
    - 1000종목 추정: 정상 100~200초, 일부 timeout 있어도 5분 내
    """
    cache_file = CACHE_DIR / f"ohlcv_{start}_{end}.pkl"

    # 기존 캐시 로드 (부분 캐시도 활용)
    data: Dict[str, pd.DataFrame] = {}
    if cache_file.exists() and not force:
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            print(f"[OHLCV CACHE] 기존 {len(data)}종목 로드됨", flush=True)
            # universe의 모든 종목이 캐시에 있으면 그대로 반환
            need = set(universe["Code"]) - set(data.keys())
            if not need:
                print(f"[OHLCV CACHE HIT] 전체 종목 캐시 사용", flush=True)
                return data
            print(f"[OHLCV] 추가 다운로드 필요: {len(need)}종목", flush=True)
        except Exception as e:
            print(f"[OHLCV CACHE ERROR] {e}, 재다운로드", flush=True)
            data = {}

    total = len(universe)
    failures = []
    print(f"[OHLCV DOWNLOAD] {total}종목, 직렬 다운로드, timeout={per_future_timeout}초/종목",
          flush=True)

    import time as _t
    t0 = _t.time()

    for i, (_, row) in enumerate(universe.iterrows(), 1):
        code = row["Code"]
        if code in data:
            continue  # 이미 캐시됨

        df = fetch_long_ohlcv(code, start, end, timeout=per_future_timeout)
        if df is not None:
            data[code] = df
        else:
            failures.append(code)

        if progress_callback:
            progress_callback(i, total)

        # 매 50종목마다 진행 로그
        if i % 50 == 0 or i == total:
            elapsed = _t.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"[OHLCV] {i}/{total} 진행 (캐시 {len(data)}, "
                  f"실패 {len(failures)}) · {elapsed:.0f}초 경과 · "
                  f"ETA {eta:.0f}초", flush=True)

        # 매 100종목마다 부분 캐시 저장
        if i % 100 == 0:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(data, f)
            except Exception:
                pass

    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    return data


def compute_amount(df: pd.DataFrame) -> pd.Series:
    """거래대금 = 종가 × 거래량 (근사치, 정확치 아님)."""
    return df["Close"] * df["Volume"]


def replay_day(
    target_date: pd.Timestamp,
    ohlcv_data: Dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    filter_cfg: FilterConfig,
    weights: ScoreWeights,
    params: SignalParams,
    recommend_cfg: RecommendConfig,
) -> pd.DataFrame:
    """특정 날짜 기준으로 필터+점수 계산 → TOP N 추천."""
    name_map = dict(zip(universe["Code"], universe["Name"]))
    market_map = dict(zip(universe["Code"], universe["Market"]))

    rows = []
    for code, df in ohlcv_data.items():
        if target_date not in df.index:
            continue
        # 그날까지의 데이터만 사용
        window = df.loc[:target_date]
        if len(window) < params.ma_long + 5:
            continue
        today = window.iloc[-1]
        prev = window.iloc[-2]

        # 거래대금 (근사: 종가*거래량)
        amount = today["Close"] * today["Volume"]
        change_ratio = (today["Close"] - prev["Close"]) / prev["Close"] * 100 if prev["Close"] > 0 else 0

        # 필터
        if amount < filter_cfg.min_amount:
            continue
        if not (filter_cfg.change_min <= change_ratio <= filter_cfg.change_max):
            continue

        sig = compute_signals(window, params)
        score = total_score(sig, weights)
        if score < recommend_cfg.min_score:
            continue

        rows.append(
            {
                "Date": target_date.strftime("%Y-%m-%d"),
                "Code": code,
                "Name": name_map.get(code, code),
                "Market": market_map.get(code, "?"),
                "Close": today["Close"],
                "ChangeRatio": round(change_ratio, 2),
                "Amount": amount,
                "Score": score,
                **sig,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("Score", ascending=False).head(recommend_cfg.top_n)
    out["Rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def simulate_trade(
    pick_row: pd.Series, df: pd.DataFrame, sell_strategy: str
) -> Dict:
    """매수 후 다음 날 수익률 시뮬레이션."""
    buy_date = pd.to_datetime(pick_row["Date"])
    buy_price = pick_row["Close"]

    # 다음 거래일 찾기
    future = df.loc[df.index > buy_date]
    if future.empty:
        return {"sell_price": None, "return_pct": None, "sell_date": None}

    nxt = future.iloc[0]
    nxt_date = future.index[0]

    if sell_strategy == "next_open":
        sell_price = nxt["Open"]
    elif sell_strategy == "next_high":
        sell_price = nxt["High"]
    elif sell_strategy == "next_close":
        sell_price = nxt["Close"]
    elif sell_strategy == "t2_close":
        if len(future) >= 2:
            sell_price = future.iloc[1]["Close"]
            nxt_date = future.index[1]
        else:
            sell_price = nxt["Close"]
    else:
        sell_price = nxt["Open"]

    return_pct = (sell_price - buy_price) / buy_price * 100
    return {
        "sell_date": nxt_date.strftime("%Y-%m-%d"),
        "sell_price": sell_price,
        "return_pct": round(return_pct, 2),
    }


def run_backtest(
    start: str,
    end: str,
    filter_cfg: FilterConfig,
    weights: ScoreWeights,
    params: SignalParams,
    recommend_cfg: RecommendConfig,
    backtest_cfg: BacktestConfig,
    universe_limit: Optional[int] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """전체 백테스트 실행."""
    universe = get_universe(filter_cfg)
    if universe_limit:
        universe = universe.head(universe_limit)

    ohlcv_data = cache_universe_ohlcv(universe, start, end, progress_callback=progress_callback)

    # 거래일 인덱스 구축 (삼성전자 기준)
    base = ohlcv_data.get("005930")
    if base is None or base.empty:
        # 임의의 종목으로 대체
        base = next(iter(ohlcv_data.values()))
    trade_days = base.loc[start:end].index.tolist()

    all_picks = []
    for i, day in enumerate(trade_days):
        picks = replay_day(day, ohlcv_data, universe, filter_cfg, weights, params, recommend_cfg)
        if not picks.empty:
            # 매도 결과 계산
            for idx, row in picks.iterrows():
                df = ohlcv_data.get(row["Code"])
                if df is not None:
                    res = simulate_trade(row, df, backtest_cfg.sell_strategy)
                    for k, v in res.items():
                        picks.at[idx, k] = v
            all_picks.append(picks)
        if progress_callback:
            progress_callback(i + 1, len(trade_days), phase="replay")

    if not all_picks:
        return pd.DataFrame()
    return pd.concat(all_picks, ignore_index=True)


def summarize_backtest(results: pd.DataFrame) -> Dict:
    """백테스트 결과 통계."""
    if results.empty or "return_pct" not in results.columns:
        return {}
    rets = results["return_pct"].dropna()
    if rets.empty:
        return {}
    return {
        "총 추천 건수": len(rets),
        "평균 수익률(%)": round(rets.mean(), 2),
        "중간값 수익률(%)": round(rets.median(), 2),
        "승률(%)": round((rets > 0).mean() * 100, 1),
        "최대 수익(%)": round(rets.max(), 2),
        "최대 손실(%)": round(rets.min(), 2),
        "표준편차": round(rets.std(), 2),
        "샤프 비율(연환산 근사)": round(rets.mean() / rets.std() * np.sqrt(252), 2) if rets.std() > 0 else 0,
    }
