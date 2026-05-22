"""
종가매수 추천 스캐너 — 코어 로직

10대 시그널 + 워치리스트 보너스:
  S1: 박스권(횡보) 후 상단 돌파 = 하바로셀 "힘의 응축"
  S2: 거래량 폭증 = "수급 선점"
  S3: 장대양봉 (시가→종가 +X% 이상)
  S4: 이평선 정배열 / MA20·60 상회
  S5: 전고점(N일 고가) 근접/돌파
  S6: 과열 회피 (직전 5일 누적 +25% 미만이면 점수 ↑)
  S7: 눌림목 셋업 (1차 슈팅 후 MA20 근처 조정 → 재반등 시작)
  S8: 수급 연속성 (최근 5일 거래량 > 30일 평균 거래량)
  S9: 장기 이평(120/240/480) 돌파 = 하승훈 "대시세 초입"
 S10: 상대강도 (시장 대비 강세) = 하승훈 "주도주 판별"
  BONUS: 하바로셀/사용자 워치리스트 종목 OR 같은 테마 동조 강세
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from config import FilterConfig, ScoreWeights, SignalParams, RecommendConfig
from watchlist import is_in_habarocell, is_in_haseunghoon, is_in_user_list, find_theme
from case_studies import classify_pattern
from pattern_detector import diagnose_all_patterns


def get_market_snapshot(filter_cfg: FilterConfig) -> pd.DataFrame:
    """오늘(가장 최근 거래일) 전체 시장 스냅샷에서 1차 필터 적용.

    1순위: KIS API (한국투자증권 OpenAPI) — 키 설정 시
    2순위: fdr.StockListing (한국 IP만 작동, 실시간)
    3순위: cache/market_snapshot.parquet (매일 업데이트되는 캐시, 어느 IP에서도)
    """
    from pathlib import Path
    cache_path = Path("cache/market_snapshot.parquet")
    all_df = None

    # 1순위: KIS API (한국투자증권) — 등락률+거래량 통합 통해 광범위 후보
    try:
        import kis_api
        if kis_api.is_available():
            cand = kis_api.get_change_rank_combined(top_per_market=50)
            if not cand.empty:
                # 마켓 필터 적용
                markets = []
                if filter_cfg.include_kospi: markets.append("KOSPI")
                if filter_cfg.include_kosdaq: markets.append("KOSDAQ")
                cand = cand[cand["Market"].isin(markets)]
                # MarketCap + Amount 보강 (캐시 마스터에서)
                if cache_path.exists():
                    ms = pd.read_parquet(cache_path)[["Code", "MarketCap", "Amount"]]
                    ms = ms.rename(columns={"Amount": "Amount_cache"})
                    cand = cand.merge(ms, on="Code", how="left")
                    # KIS의 Amount가 0이면 캐시 사용
                    cand["Amount"] = cand["Amount"].where(cand["Amount"] > 0, cand["Amount_cache"])
                    cand = cand.drop(columns=["Amount_cache"], errors="ignore")
                else:
                    cand["MarketCap"] = filter_cfg.min_marcap  # 통과시킴
                if not cand.empty:
                    all_df = cand
    except Exception:
        pass

    # 2순위: fdr (한국 IP) — 타임아웃 3초, 차단 시 즉시 fallback
    if all_df is None:
        import socket
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(3)
            frames = []
            if filter_cfg.include_kospi:
                df = fdr.StockListing("KOSPI").assign(Market="KOSPI")
                frames.append(df)
            if filter_cfg.include_kosdaq:
                df = fdr.StockListing("KOSDAQ").assign(Market="KOSDAQ")
                frames.append(df)
            if frames:
                all_df = pd.concat(frames, ignore_index=True)
                rename_map = {"ChagesRatio": "ChangeRatio", "Marcap": "MarketCap"}
                all_df = all_df.rename(columns=rename_map)
        except Exception:
            pass
        finally:
            socket.setdefaulttimeout(old_timeout)

    # 3순위: 캐시
    if all_df is None or all_df.empty:
        if not cache_path.exists():
            # 빈 DataFrame 반환 — 호출 측에서 에러 메시지 표시
            return pd.DataFrame()
        all_df = pd.read_parquet(cache_path)
        markets = []
        if filter_cfg.include_kospi: markets.append("KOSPI")
        if filter_cfg.include_kosdaq: markets.append("KOSDAQ")
        all_df = all_df[all_df["Market"].isin(markets)]
        if all_df.empty:
            return pd.DataFrame()

    # ETF/ETN/SPAC 등 제외 (이름 기반 휴리스틱)
    if filter_cfg.exclude_etf:
        exclude_keywords = ["ETF", "ETN", "스팩", "SPAC", "리츠", "REIT"]
        mask = ~all_df["Name"].str.contains("|".join(exclude_keywords), na=False)
        all_df = all_df[mask]

    # 하드 필터 적용
    filtered = all_df[
        (all_df["Amount"] >= filter_cfg.min_amount)
        & (all_df["MarketCap"] >= filter_cfg.min_marcap)
        & (all_df["ChangeRatio"].between(filter_cfg.change_min, filter_cfg.change_max))
    ].copy()

    return filtered.reset_index(drop=True)


# OHLCV 캐시 (메모리)
_OHLCV_CACHE_PKL: Optional[Dict[str, pd.DataFrame]] = None


def _load_ohlcv_cache_pkl() -> Dict[str, pd.DataFrame]:
    """ohlcv_*.pkl 캐시를 메모리에 로딩 (한 번만)."""
    global _OHLCV_CACHE_PKL
    if _OHLCV_CACHE_PKL is not None:
        return _OHLCV_CACHE_PKL
    import pickle
    from pathlib import Path as _P
    cache_dir = _P("cache")
    pkls = sorted(cache_dir.glob("ohlcv_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pkls:
        try:
            with open(p, "rb") as f:
                _OHLCV_CACHE_PKL = pickle.load(f)
            return _OHLCV_CACHE_PKL
        except Exception:
            continue
    _OHLCV_CACHE_PKL = {}
    return _OHLCV_CACHE_PKL


def fetch_ohlcv(code: str, days: int = 90, end_date: Optional[str] = None,
                  fast: bool = True) -> Optional[pd.DataFrame]:
    """단일 종목 과거 OHLCV. 실패 시 None.

    1순위: fdr.DataReader (한국 IP만 작동, 실시간)
    2순위: cache/ohlcv_*.pkl (캐시 데이터, 최근 1주일 ~ 며칠 지연)
    """
    end = pd.to_datetime(end_date) if end_date else datetime.now()
    if fast:
        lookback = max(days, 250) + 10
    else:
        lookback = max(days, 500) + 30
    start = end - timedelta(days=lookback)

    # 1순위: KIS API
    try:
        import kis_api
        if kis_api.is_available():
            df = kis_api.get_ohlcv(code, days=int(lookback))
            if df is not None and not df.empty:
                return df
    except Exception:
        pass

    # 2순위: fdr (한국 IP)
    try:
        df = fdr.DataReader(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 3순위: 캐시
    cache = _load_ohlcv_cache_pkl()
    cached = cache.get(code)
    if cached is None or cached.empty:
        return None
    sub = cached[(cached.index >= start) & (cached.index <= end)]
    if sub.empty:
        return cached.tail(min(len(cached), int(lookback)))
    return sub


# 시장 벤치마크 캐시 (상대강도 계산용)
_BENCHMARK_CACHE: Dict[str, pd.DataFrame] = {}


def get_benchmark(symbol: str = "KS11", days: int = 60) -> Optional[pd.DataFrame]:
    """벤치마크 지수(KOSPI=KS11, KOSDAQ=KQ11) OHLCV. 캐싱됨."""
    cache_key = f"{symbol}_{days}"
    if cache_key in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE[cache_key]
    try:
        end = datetime.now()
        start = end - timedelta(days=days + 30)
        df = fdr.DataReader(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return None
        _BENCHMARK_CACHE[cache_key] = df
        return df
    except Exception:
        return None


def compute_signals(ohlcv: pd.DataFrame, params: SignalParams) -> Dict[str, float]:
    """단일 종목의 5대 시그널 점수 (0~100 각각)."""
    if ohlcv is None or len(ohlcv) < params.ma_long + 5:
        return {f"s{i}": 0.0 for i in range(1, 6)} | {"valid": False}

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    open_ = ohlcv["Open"].astype(float)
    vol = ohlcv["Volume"].astype(float)

    today_close = close.iloc[-1]
    today_open = open_.iloc[-1]
    today_high = high.iloc[-1]
    today_vol = vol.iloc[-1]
    yesterday_close = close.iloc[-2]

    # S1: 박스권 돌파
    box_window = close.iloc[-params.box_period - 1 : -1]
    if len(box_window) > 0:
        box_high = box_window.max()
        box_low = box_window.min()
        box_range_pct = (box_high - box_low) / box_low * 100 if box_low > 0 else 999
        is_box = box_range_pct <= params.box_max_range_pct
        breakout_pct = (today_close - box_high) / box_high * 100 if box_high > 0 else 0
        if is_box and breakout_pct > 0:
            s1 = min(100, 50 + breakout_pct * 5)
        elif is_box and breakout_pct > -2:
            s1 = 40  # 박스 상단 근접
        elif breakout_pct > 0:
            s1 = 30  # 박스는 아니지만 돌파는 함
        else:
            s1 = 0
    else:
        s1 = 0

    # S2: 거래량 폭증
    vol_ma = vol.iloc[-params.volume_ma_period - 1 : -1].mean()
    if vol_ma > 0:
        vol_ratio = today_vol / vol_ma
        if vol_ratio >= params.volume_surge_multiplier:
            s2 = min(100, 50 + (vol_ratio - params.volume_surge_multiplier) * 10)
        else:
            s2 = max(0, vol_ratio / params.volume_surge_multiplier * 50)
    else:
        s2 = 0

    # S3: 장대양봉 (시가 대비 종가 상승률)
    if today_open > 0:
        candle_pct = (today_close - today_open) / today_open * 100
        if candle_pct >= params.long_candle_min_pct:
            s3 = min(100, 50 + (candle_pct - params.long_candle_min_pct) * 5)
        else:
            s3 = max(0, candle_pct / params.long_candle_min_pct * 50)
    else:
        s3 = 0
    # 윗꼬리 페널티
    body = abs(today_close - today_open)
    upper_wick = today_high - max(today_close, today_open)
    if body > 0 and upper_wick / body > 1.5:
        s3 *= 0.5

    # S4: 이평선 정배열
    ma_s = close.rolling(params.ma_short).mean().iloc[-1]
    ma_m = close.rolling(params.ma_mid).mean().iloc[-1]
    ma_l = close.rolling(params.ma_long).mean().iloc[-1]
    score = 0
    if today_close > ma_s:
        score += 25
    if today_close > ma_m:
        score += 25
    if today_close > ma_l:
        score += 25
    if ma_s > ma_m > ma_l:
        score += 25
    s4 = score

    # S5: 전고점 근접/돌파
    period_high = high.iloc[-params.box_period - 1 : -1].max()
    if period_high > 0:
        ratio = today_close / period_high
        if ratio >= 1.0:
            s5 = min(100, 70 + (ratio - 1.0) * 100)
        elif ratio >= params.near_high_threshold:
            s5 = (ratio - params.near_high_threshold) / (1.0 - params.near_high_threshold) * 70
        else:
            s5 = 0
    else:
        s5 = 0

    # S6: 과열 회피 (점수 = "안 과열일수록 높음")
    if len(close) >= params.overheat_period + 1:
        cum_gain = (today_close - close.iloc[-params.overheat_period - 1]) / close.iloc[-params.overheat_period - 1] * 100
        if cum_gain <= 0:
            s6 = 100
        elif cum_gain >= params.overheat_threshold:
            s6 = 0
        else:
            s6 = 100 * (1 - cum_gain / params.overheat_threshold)
    else:
        s6 = 50
        cum_gain = 0

    # S7: 눌림목 셋업
    # 최근 20일 내 +10% 이상 슈팅 캔들이 있었고, 그 후 MA20 근처까지 조정,
    # 그리고 오늘 양봉으로 반등 시작했다면 점수 ↑
    s7 = 0
    pull_window = close.iloc[-params.pullback_lookback :]
    if len(pull_window) >= 5:
        # 슈팅 캔들 찾기 (전일 대비 +N% 이상)
        rets = close.pct_change().iloc[-params.pullback_lookback :] * 100
        shoot_days = rets[rets >= params.pullback_first_shoot_pct]
        if not shoot_days.empty:
            shoot_idx = shoot_days.index[-1]  # 가장 최근 슈팅일
            shoot_pos = close.index.get_loc(shoot_idx)
            # 슈팅 후 조정 깊이
            after_shoot = close.iloc[shoot_pos:]
            if len(after_shoot) >= 3:
                min_after = after_shoot.min()
                ma20_at_min = close.rolling(params.ma_mid).mean().iloc[shoot_pos:].min()
                if ma20_at_min and abs(min_after - ma20_at_min) / ma20_at_min < 0.05:
                    if today_close > min_after * 1.02:  # 반등 시작
                        s7 = 80
                    else:
                        s7 = 40

    # S8: 수급 연속성
    if len(vol) >= params.demand_baseline_period:
        recent_vol = vol.iloc[-params.demand_recent_period :].mean()
        baseline_vol = vol.iloc[-params.demand_baseline_period :].mean()
        if baseline_vol > 0:
            demand_ratio = recent_vol / baseline_vol
            if demand_ratio >= 1.5:
                s8 = min(100, 50 + (demand_ratio - 1.5) * 50)
            elif demand_ratio >= 1.0:
                s8 = (demand_ratio - 1.0) / 0.5 * 50
            else:
                s8 = 0
        else:
            s8 = 0
    else:
        s8 = 0

    # S9: 장기 이평선 돌파 (하승훈 — 대시세 초입)
    # 120/240/480일선이 우하향→횡보로 바뀌고 주가가 대량거래로 돌파한 뒤 지지
    s9 = 0
    s9_breakout_list = []
    for ma_period in params.longterm_ma_periods:
        if len(close) < ma_period + params.longterm_ma_flat_window:
            continue
        ma_series = close.rolling(ma_period).mean()
        ma_today = ma_series.iloc[-1]
        if pd.isna(ma_today) or ma_today <= 0:
            continue
        # 횡보 여부: 최근 N일 이평선 변동폭이 작음
        recent_ma = ma_series.iloc[-params.longterm_ma_flat_window :]
        ma_change_pct = (recent_ma.max() - recent_ma.min()) / recent_ma.mean() * 100
        is_flat = ma_change_pct <= params.longterm_ma_flat_pct
        # 돌파 여부: 종가가 이평선 위 + 어제는 이평선 아래/근처였음
        was_below = close.iloc[-2] <= ma_series.iloc[-2] * 1.02
        is_above = today_close > ma_today
        breakout = was_below and is_above
        ratio_above = (today_close - ma_today) / ma_today * 100
        if breakout and is_flat:
            s9 += {120: 30, 240: 35, 480: 35}[ma_period]
            s9_breakout_list.append(f"MA{ma_period}")
        elif is_above and is_flat and ratio_above < 5:
            s9 += {120: 15, 240: 18, 480: 18}[ma_period]
    s9 = min(100, s9)

    # S10: 상대강도 (시장 대비 강세) — 하승훈 "주도주" 판별
    # 종목 N일 수익률 / 벤치마크 N일 수익률
    bench_symbol = params.rs_benchmark
    bench_df = get_benchmark(bench_symbol, days=params.rs_period + 10)
    s10 = 0
    rs_ratio = 1.0
    if bench_df is not None and len(close) >= params.rs_period + 1:
        stock_ret = (today_close - close.iloc[-params.rs_period - 1]) / close.iloc[-params.rs_period - 1] * 100
        if len(bench_df) >= params.rs_period + 1:
            bench_today = bench_df["Close"].iloc[-1]
            bench_past = bench_df["Close"].iloc[-params.rs_period - 1]
            bench_ret = (bench_today - bench_past) / bench_past * 100
            # 상대강도 = 종목수익률 - 벤치수익률
            rs_diff = stock_ret - bench_ret
            if rs_diff >= 30:
                s10 = 100
            elif rs_diff >= 15:
                s10 = 70 + (rs_diff - 15) / 15 * 30
            elif rs_diff >= 5:
                s10 = 40 + (rs_diff - 5) / 10 * 30
            elif rs_diff >= 0:
                s10 = rs_diff / 5 * 40
            else:
                s10 = max(0, 40 + rs_diff)
            rs_ratio = round((1 + stock_ret/100) / (1 + bench_ret/100), 3) if bench_ret > -100 else 1.0

    # 윗꼬리 비율 (진입 룰에서 사용)
    body = abs(today_close - today_open)
    upper_wick = today_high - max(today_close, today_open)
    upper_wick_ratio = upper_wick / body if body > 0 else 0

    # MA3/5/10 (하승훈 단기 손절 기준)
    ma3 = close.rolling(3).mean().iloc[-1]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]

    # S11: 갭 + 이평선 중첩 (마스터 가이드)
    patterns = diagnose_all_patterns(ohlcv)
    s11 = patterns["gap_support"]["score"]

    # S12: 패턴 품질 종합 (첫 눌림 + 컵앤핸들 + 역H&S + 진짜 지지)
    s12_components = []
    if patterns["first_pullback"]["is_first"]:
        s12_components.append(70)
    elif patterns["first_pullback"]["shoot_days"] == 0:
        s12_components.append(0)  # 슈팅 자체가 없음
    else:
        s12_components.append(20)  # n차 눌림 페널티
    s12_components.append(patterns["pullback_quality"]["score"])
    s12_components.append(patterns["cup_and_handle"]["score"])
    s12_components.append(patterns["inverse_hns"]["score"])
    s12 = max(s12_components) if s12_components else 0  # 최강 패턴 점수 사용

    return {
        "s1": round(s1, 1),
        "s2": round(s2, 1),
        "s3": round(s3, 1),
        "s4": round(s4, 1),
        "s5": round(s5, 1),
        "s6": round(s6, 1),
        "s7": round(s7, 1),
        "s8": round(s8, 1),
        "s9": round(s9, 1),
        "s10": round(s10, 1),
        "s11": round(s11, 1),
        "s12": round(s12, 1),
        "valid": True,
        "vol_ratio": round(today_vol / vol_ma, 2) if vol_ma > 0 else 0,
        "candle_pct": round(candle_pct, 2) if today_open > 0 else 0,
        "cum_5d_gain": round(cum_gain, 2),
        "upper_wick_ratio": round(upper_wick_ratio, 2),
        "rs_ratio": rs_ratio,
        "longterm_ma_breakouts": ",".join(s9_breakout_list),
        "ma3": round(ma3, 2) if pd.notna(ma3) else None,
        "ma5": round(ma5, 2) if pd.notna(ma5) else None,
        "ma10": round(ma10, 2) if pd.notna(ma10) else None,
        # 패턴 상세 (UI 표시용)
        "is_first_pullback": patterns["first_pullback"]["is_first"],
        "pullback_quality": patterns["pullback_quality"]["quality"],
        "cup_and_handle_detected": patterns["cup_and_handle"]["detected"],
        "inverse_hns_detected": patterns["inverse_hns"]["detected"],
        "gap_support_detected": patterns["gap_support"]["detected"],
    }


def total_score(sig: Dict[str, float], weights: ScoreWeights, name: str = "") -> float:
    """가중 합계 점수 (0~100) + 워치리스트 보너스"""
    if not sig.get("valid", False):
        return 0.0
    total_w = (
        weights.s1_box_breakout + weights.s2_volume_surge + weights.s3_long_candle
        + weights.s4_ma_alignment + weights.s5_near_high
        + weights.s6_no_overheating + weights.s7_pullback_setup + weights.s8_demand_continuity
        + weights.s9_longterm_ma_breakout + weights.s10_relative_strength
        + weights.s11_gap_ma_confluence + weights.s12_pattern_quality
    )
    if total_w == 0:
        return 0.0
    score = (
        sig["s1"] * weights.s1_box_breakout
        + sig["s2"] * weights.s2_volume_surge
        + sig["s3"] * weights.s3_long_candle
        + sig["s4"] * weights.s4_ma_alignment
        + sig["s5"] * weights.s5_near_high
        + sig.get("s6", 0) * weights.s6_no_overheating
        + sig.get("s7", 0) * weights.s7_pullback_setup
        + sig.get("s8", 0) * weights.s8_demand_continuity
        + sig.get("s9", 0) * weights.s9_longterm_ma_breakout
        + sig.get("s10", 0) * weights.s10_relative_strength
        + sig.get("s11", 0) * weights.s11_gap_ma_confluence
        + sig.get("s12", 0) * weights.s12_pattern_quality
    ) / total_w

    # 워치리스트 보너스 (최대 weights.bonus_watchlist 점)
    bonus = 0.0
    if name:
        if is_in_habarocell(name):
            bonus += weights.bonus_watchlist * 0.5  # 하바로셀 검증
        if is_in_haseunghoon(name):
            bonus += weights.bonus_watchlist * 0.5  # 하승훈 분석
        if is_in_user_list(name):
            bonus += weights.bonus_watchlist * 0.3  # 사용자 본인 매매
        themes = find_theme(name)
        if themes:
            bonus += weights.bonus_watchlist * 0.15 * min(len(themes), 2)
    bonus = min(bonus, weights.bonus_watchlist)

    return round(min(100, score + bonus), 2)


def _process_one(code: str, name: str, market: str, params: SignalParams, weights: ScoreWeights, end_date: Optional[str]):
    ohlcv = fetch_ohlcv(code, days=params.ohlcv_lookback_days, end_date=end_date)
    sig = compute_signals(ohlcv, params)
    score = total_score(sig, weights, name=name)
    themes = find_theme(name)
    # 패턴 자동 분류 (돌파매매/눌림목매매/대시세초입)
    pattern_info = classify_pattern(sig)
    return {
        "Code": code, "Name": name, "Market": market, "Score": score,
        "Themes": ", ".join(themes) if themes else "",
        "InHabarocell": is_in_habarocell(name),
        "InHaseunghoon": is_in_haseunghoon(name),
        "InUserList": is_in_user_list(name),
        "TradeType": pattern_info["trade_type"],
        "Pattern": pattern_info["primary_pattern"].value,
        "PatternConfidence": pattern_info["confidence"],
        "BreakoutScore": pattern_info.get("breakout_score", 0),
        "PullbackScore": pattern_info.get("pullback_score", 0),
        **sig,
    }


def filter_by_trade_type(picks: pd.DataFrame, trade_type: str) -> pd.DataFrame:
    """
    매매 타입별 필터링.
    trade_type: '전체' / '돌파매매' / '눌림목매매' / '대시세 초입'
    """
    if picks is None or picks.empty or trade_type == "전체":
        return picks
    return picks[picks["TradeType"] == trade_type].reset_index(drop=True)


def scan_recommendations(
    filter_cfg: FilterConfig,
    weights: ScoreWeights,
    params: SignalParams,
    recommend_cfg: RecommendConfig,
    end_date: Optional[str] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """오늘의 추천 종목 산출 (필터 통과 종목들 점수 계산 → TOP N)."""
    snapshot = get_market_snapshot(filter_cfg)
    if snapshot.empty:
        return pd.DataFrame()

    results = []
    total = len(snapshot)
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {
            ex.submit(_process_one, r["Code"], r["Name"], r["Market"], params, weights, end_date): r["Code"]
            for _, r in snapshot.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception:
                pass
            if progress_callback:
                progress_callback(i, total)

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # 스냅샷 정보(거래대금/시총/등락률) 머지
    out = out.merge(
        snapshot[["Code", "Close", "ChangeRatio", "Amount", "MarketCap"]],
        on="Code",
        how="left",
    )

    out = out[out["Score"] >= recommend_cfg.min_score]
    out = out.sort_values("Score", ascending=False).head(recommend_cfg.top_n)
    out["Rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


# ============================================================================
# 4 프리셋 앙상블 스캔 (V/S/A/B 등급 시스템용)
# ============================================================================

def _process_one_signals_only(code: str, name: str, market: str,
                                params: SignalParams, end_date: Optional[str]):
    """시그널만 계산 (점수 X). 4 프리셋 점수는 별도로 계산."""
    ohlcv = fetch_ohlcv(code, days=params.ohlcv_lookback_days, end_date=end_date)
    sig = compute_signals(ohlcv, params)
    themes = find_theme(name)
    pattern_info = classify_pattern(sig)
    return {
        "Code": code, "Name": name, "Market": market,
        "Themes": ", ".join(themes) if themes else "",
        "InHabarocell": is_in_habarocell(name),
        "InHaseunghoon": is_in_haseunghoon(name),
        "InUserList": is_in_user_list(name),
        "TradeType": pattern_info["trade_type"],
        "Pattern": pattern_info["primary_pattern"].value,
        "PatternConfidence": pattern_info["confidence"],
        "BreakoutScore": pattern_info.get("breakout_score", 0),
        "PullbackScore": pattern_info.get("pullback_score", 0),
        **sig,
    }


def scan_ensemble(
    filter_cfg: FilterConfig,
    params: SignalParams,
    preset_keys: List[str],
    end_date: Optional[str] = None,
    progress_callback=None,
    min_recommend_score: float = 40.0,
) -> pd.DataFrame:
    """4 프리셋 앙상블 통합 스캔.

    - 시장 1회 + 시그널 1회 + 가중치 N번 (프리셋 수)
    - 각 종목에 대해 N개 프리셋이 추천 임계값을 넘는지 카운트
    - 반환: 종목별 ['Code', 'Name', ..., 'n_presets', 'avg_score', ...]
    """
    from rules import PRESETS

    snapshot = get_market_snapshot(filter_cfg)
    if snapshot.empty:
        return pd.DataFrame()

    # 1) 시그널 1회 계산 (병렬 30 workers — 속도 최적화)
    results = []
    total = len(snapshot)
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {
            ex.submit(_process_one_signals_only, r["Code"], r["Name"], r["Market"],
                      params, end_date): r["Code"]
            for _, r in snapshot.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception:
                pass
            if progress_callback:
                progress_callback(i, total)

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.merge(
        snapshot[["Code", "Close", "ChangeRatio", "Amount", "MarketCap"]],
        on="Code", how="left",
    )

    # 2) 각 프리셋 가중치로 점수 N번 계산 + 추천여부 체크
    preset_scores = {}  # preset_key -> list of scores
    preset_recommended = {}  # preset_key -> list of bool
    for pk in preset_keys:
        if pk not in PRESETS: continue
        w = PRESETS[pk]["weights"]
        weights = ScoreWeights(
            s1_box_breakout=w.get("s1", 18), s2_volume_surge=w.get("s2", 18),
            s3_long_candle=w.get("s3", 8), s4_ma_alignment=w.get("s4", 8),
            s5_near_high=w.get("s5", 10), s6_no_overheating=w.get("s6", 4),
            s7_pullback_setup=w.get("s7", 5), s8_demand_continuity=w.get("s8", 4),
            s9_longterm_ma_breakout=w.get("s9", 7), s10_relative_strength=w.get("s10", 7),
            s11_gap_ma_confluence=w.get("s11", 5), s12_pattern_quality=w.get("s12", 6),
            bonus_watchlist=w.get("bonus", 5),
        )
        scores = []
        recommended = []
        for _, row in out.iterrows():
            sig = {k: row.get(k, 0) for k in
                   ["valid", "s1", "s2", "s3", "s4", "s5", "s6",
                    "s7", "s8", "s9", "s10", "s11", "s12"]}
            sc = total_score(sig, weights, name=row.get("Name", ""))
            scores.append(sc)
            recommended.append(sc >= min_recommend_score)
        preset_scores[pk] = scores
        preset_recommended[pk] = recommended
        out[f"_score_{pk}"] = scores
        out[f"_rec_{pk}"] = recommended

    # 3) n_presets, avg_score 계산
    rec_cols = [f"_rec_{pk}" for pk in preset_keys if pk in PRESETS]
    score_cols = [f"_score_{pk}" for pk in preset_keys if pk in PRESETS]
    out["n_presets"] = out[rec_cols].sum(axis=1)
    # avg_score는 추천한 프리셋들의 평균 점수
    def _avg_rec(row):
        recs = [row[c] for c in rec_cols]
        scs = [row[c] for c in score_cols]
        rec_scs = [s for s, r in zip(scs, recs) if r]
        if rec_scs:
            return sum(rec_scs) / len(rec_scs)
        return sum(scs) / len(scs) if scs else 0
    out["avg_score"] = out.apply(_avg_rec, axis=1)
    # Score 컬럼도 채워둠 (기존 카드 호환)
    out["Score"] = out["avg_score"]

    # 정렬
    out = out.sort_values(["n_presets", "avg_score"], ascending=[False, False])
    out["Rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)
