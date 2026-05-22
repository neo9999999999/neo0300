"""
백테스트 UI 보조 — 일자별 시그널 표시용 데이터 가공.
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional


CACHE_DIR = Path(__file__).parent / "cache"
_OHLCV_CACHE: Dict[str, pd.DataFrame] = {}


def load_ohlcv_dict() -> Dict[str, pd.DataFrame]:
    """전체 OHLCV 캐시 로드 (메모리 캐시)."""
    if _OHLCV_CACHE:
        return _OHLCV_CACHE
    cache_file = CACHE_DIR / "ohlcv_2020-01-01_2026-05-21.pkl"
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        _OHLCV_CACHE.update(data)
        return data
    except Exception:
        return {}


def forward_return(code: str, buy_date: pd.Timestamp, days: int = 10) -> Optional[float]:
    """매수일 N영업일 후 종가 수익률 (%)."""
    return forward_price_return(code, buy_date, days, kind="close")


def forward_price_return(
    code: str, buy_date: pd.Timestamp, days: int, kind: str = "close"
) -> Optional[float]:
    """
    매수일 + N영업일의 OHLC 중 하나 vs 매수일 종가 수익률.
    kind: 'open' | 'high' | 'low' | 'close'
    """
    ohlcv = load_ohlcv_dict().get(code)
    if ohlcv is None or ohlcv.empty:
        return None
    buy_date = pd.to_datetime(buy_date)
    if buy_date not in ohlcv.index:
        return None
    try:
        idx = ohlcv.index.get_loc(buy_date)
        if idx + days >= len(ohlcv):
            return None
        buy_close = ohlcv.iloc[idx]["Close"]
        future = ohlcv.iloc[idx + days]
        col = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}.get(kind, "Close")
        future_px = future[col]
        if buy_close <= 0:
            return None
        return float((future_px - buy_close) / buy_close * 100)
    except Exception:
        return None


def past_20d_return(code: str, ref_date: pd.Timestamp) -> Optional[float]:
    return past_n_return(code, ref_date, 20)


def past_n_return(code: str, ref_date: pd.Timestamp, n: int = 5) -> Optional[float]:
    """직전 N영업일 누적 상승률 (%)."""
    ohlcv = load_ohlcv_dict().get(code)
    if ohlcv is None or ohlcv.empty:
        return None
    ref_date = pd.to_datetime(ref_date)
    if ref_date not in ohlcv.index:
        return None
    try:
        idx = ohlcv.index.get_loc(ref_date)
        if idx < n:
            return None
        cur = ohlcv.iloc[idx]["Close"]
        past = ohlcv.iloc[idx - n]["Close"]
        if past <= 0:
            return None
        return float((cur - past) / past * 100)
    except Exception:
        return None


def short_reason(row) -> str:
    """추천사유 한 줄 요약."""
    parts = []
    # 직전 20일 누적
    if "past_20d" in row.index and pd.notna(row.get("past_20d")):
        p20 = row["past_20d"]
        if p20 >= 10:
            parts.append(f"20일+{p20:.0f}%")
    # 거래대금 100억+
    amount = row.get("Amount", 0)
    if amount >= 50_000_000_000:  # 500억+
        parts.append(f"대금{amount/1e8:.0f}억")
    elif amount >= 10_000_000_000:  # 100억+
        parts.append(f"대금{amount/1e8:.0f}억")
    # 당일 등락
    chg = row.get("ChangeRatio", 0)
    if chg >= 13:
        parts.append(f"당일+{chg:.0f}%")
    # 막판 매수세 (vol_ratio 고)
    vr = row.get("vol_ratio", 0)
    if vr >= 5:
        parts.append("막판매수")
    # 패턴
    if row.get("is_first_pullback"):
        parts.append("첫눌림")
    if row.get("cup_and_handle_detected"):
        parts.append("컵앤핸들")
    if row.get("gap_support_detected"):
        parts.append("갭지지")
    if not parts:
        parts.append(f"점수 {row.get('Score', 0):.0f}")
    return " · ".join(parts[:3])


def similarity_to_top_case(row) -> float:
    """추천 종목의 최고 유사 사례 유사도 (0~1)."""
    try:
        from case_matcher import find_similar_cases
        sims = find_similar_cases(row, top_n=1)
        if sims:
            return sims[0]["similarity"] / 100  # 0~1 스케일
    except Exception:
        pass
    return 0.0


def enrich_trades(df: pd.DataFrame, with_forward: bool = True) -> pd.DataFrame:
    """
    단타용 enrich: D+1 시가/종가, D+2 시가/종가만 계산.
    + 직전20일 + 추천사유(사례 기반) + 유사 사례 종목명
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])
    df["hold_days"] = (df["sell_date"] - df["Date"]).dt.days.fillna(1).astype(int)
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    if with_forward:
        # 단타: D+1 OHLC 4개
        for k in ["open", "high", "low", "close"]:
            df[f"ret_d1_{k}"] = df.apply(
                lambda r, _k=k: forward_price_return(r["Code"], r["Date"], 1, _k), axis=1
            )
        # 중기/장기 스윙: 20~365일 종가
        for d in [20, 30, 60, 90, 120, 180, 240, 365]:
            df[f"ret_{d}d"] = df.apply(
                lambda r, _d=d: forward_price_return(r["Code"], r["Date"], _d, "close"), axis=1
            )
        # 매매타입 분류 (돌파/눌림목/대시세 초입)
        try:
            from case_studies import classify_pattern
            tt_list = []
            for _, r in df.iterrows():
                sig = {f"s{i}": r.get(f"s{i}", 0) for i in range(1, 13)}
                info = classify_pattern(sig)
                tt_list.append(info["trade_type"])
            df["TradeType"] = tt_list
        except Exception:
            df["TradeType"] = ""
        # 직전 5일 누적 (이미 cum_5d_gain 있을 수 있으나 확실하게 재계산)
        if "cum_5d_gain" not in df.columns:
            df["past_5d"] = df.apply(
                lambda r: past_n_return(r["Code"], r["Date"], 5), axis=1
            )
        else:
            df["past_5d"] = df["cum_5d_gain"]
        # 추천사유 (사례 기반) + 유사 사례 종목
        df["reason"] = ""
        df["similar_stock"] = ""
        df["similar_pct"] = 0.0
        if len(df) <= 300:
            try:
                from case_matcher import find_similar_cases
                for idx, r in df.iterrows():
                    sims = find_similar_cases(r, top_n=1)
                    if sims:
                        case = sims[0]["case"]
                        sigs = case.get("key_signals", [])
                        # 추천사유: 사례의 시그널 키워드 2개 + 우리 시그널 수치
                        case_reason = " · ".join(sigs[:2]) if sigs else ""
                        own_metrics = []
                        p20 = r.get("past_20d")
                        if pd.notna(p20) and p20 >= 10:
                            own_metrics.append(f"20일+{p20:.0f}%")
                        amount = r.get("Amount", 0)
                        if amount >= 50_000_000_000:
                            own_metrics.append(f"대금{amount/1e8:.0f}억")
                        combined = " · ".join(own_metrics + ([case_reason] if case_reason else []))
                        df.at[idx, "reason"] = combined[:60] if combined else short_reason(r)
                        df.at[idx, "similar_stock"] = case["stock"]
                        df.at[idx, "similar_pct"] = sims[0]["similarity"]
                    else:
                        df.at[idx, "reason"] = short_reason(r)
            except Exception:
                df["reason"] = df.apply(short_reason, axis=1)
        else:
            df["reason"] = df.apply(short_reason, axis=1)
    return df


def monthly_pnl_table(df: pd.DataFrame, position_size: int = 1_000_000) -> pd.DataFrame:
    """단타용 월별 손익 — D+1 시가/종가, D+2 시가/종가."""
    if df is None or df.empty:
        return pd.DataFrame()

    horizons = [
        ("D+1 종가", "ret_d1_close"),
        ("20일", "ret_20d"),
        ("60일", "ret_60d"),
        ("90일", "ret_90d"),
        ("120일", "ret_120d"),
    ]

    def _agg(sub):
        out = {"N": len(sub)}
        for label, col in horizons:
            if col in sub:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    avg = vals.mean()
                    pnl = avg / 100 * position_size * len(vals)
                    out[f"{label} 평균"] = avg
                    out[f"{label} 손익"] = pnl
                else:
                    out[f"{label} 평균"] = None
                    out[f"{label} 손익"] = None
        return out

    rows = []
    for ym in sorted(df["YearMonth"].unique()):
        m = df[df["YearMonth"] == ym]
        rows.append({"월": ym, **_agg(m)})
    rows.append({"월": "🏆 합계", **_agg(df)})
    return pd.DataFrame(rows)


def pattern_comparison_table(df: pd.DataFrame, position_size: int = 1_000_000) -> pd.DataFrame:
    """
    시점(D+1 OHLC + 20/60/90/120일) × 패턴(돌파/눌림/대시세) 비교.
    행: 시점 8개  ·  열: 패턴3개 × (평균/승률) + N + 전체 평균/승률
    """
    if df is None or df.empty or "TradeType" not in df.columns:
        return pd.DataFrame()

    horizons = [
        ("D+1 시가", "ret_d1_open"),
        ("D+1 고가", "ret_d1_high"),
        ("D+1 저가", "ret_d1_low"),
        ("D+1 종가", "ret_d1_close"),
        ("20일",    "ret_20d"),
        ("60일",    "ret_60d"),
        ("90일",    "ret_90d"),
        ("120일",   "ret_120d"),
    ]
    patterns = ["돌파매매", "눌림목매매", "대시세 초입"]

    def _stats(sub, col):
        v = sub[col].dropna() if col in sub else pd.Series(dtype=float)
        if len(v) == 0:
            return None, None, 0
        return float(v.mean()), float((v > 0).mean() * 100), len(v)

    rows = []
    for label, col in horizons:
        row = {"시점": label}
        # 패턴별
        for pat in patterns:
            sub = df[df["TradeType"] == pat]
            avg, wr, n = _stats(sub, col)
            row[f"{pat} 평균"] = avg
            row[f"{pat} 승률"] = wr
            row[f"{pat} N"] = n
        # 전체
        avg, wr, n = _stats(df, col)
        row["전체 평균"] = avg
        row["전체 승률"] = wr
        row["전체 N"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def year_month_pattern_table(df: pd.DataFrame, position_size: int = 1_000_000,
                              metric: str = "ret_d1_close") -> pd.DataFrame:
    """
    년/월별 × 패턴별 평균 수익률 + 승률.
    metric: 어떤 시점 기준 (ret_d1_open / ret_d1_high / ret_d1_low / ret_d1_close)
    """
    if df is None or df.empty or "TradeType" not in df.columns:
        return pd.DataFrame()
    patterns = ["돌파매매", "눌림목매매", "대시세 초입"]
    months = sorted(df["YearMonth"].unique())

    def _agg(sub, col):
        v = sub[col].dropna() if col in sub else pd.Series(dtype=float)
        if len(v) == 0:
            return None, None, 0
        return float(v.mean()), float((v > 0).mean() * 100), len(v)

    rows = []
    for ym in months:
        m = df[df["YearMonth"] == ym]
        row = {"월": ym, "N": len(m)}
        for pat in patterns:
            sub = m[m["TradeType"] == pat]
            avg, wr, n = _agg(sub, metric)
            row[f"{pat} 평균"] = avg
            row[f"{pat} 승률"] = wr
            row[f"{pat} N"] = n
        rows.append(row)

    # 합계
    row = {"월": "🏆 합계", "N": len(df)}
    for pat in patterns:
        sub = df[df["TradeType"] == pat]
        avg, wr, n = _agg(sub, metric)
        row[f"{pat} 평균"] = avg
        row[f"{pat} 승률"] = wr
        row[f"{pat} N"] = n
    rows.append(row)
    return pd.DataFrame(rows)


def format_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """월별 손익 표를 표시용 포맷팅."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in ["익일 평균", "10일 평균", "30일 평균"]:
        if c in df:
            df[c] = df[c].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    for c in ["익일 손익", "10일 손익", "30일 손익"]:
        if c in df:
            df[c] = df[c].map(lambda x: f"{x/10000:+.0f}만원" if pd.notna(x) else "—")
    return df
