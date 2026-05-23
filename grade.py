"""
V/S/A/B 등급 시스템 — 4 프리셋 앙상블 기반 등급 분류 + 추천사유 생성.

[등급 정의]
V급 (Very Best · 50만/건): 점수≥75 — 6년 32건 (연 5회)
  → 최고급 셋업, 평균 +96.9% (180일 보유)
S급 (Super · 30만/건):    4프리셋 만장일치 + 점수≥65 — 6년 99건 (연 17회)
  → 4개 전략 동시 추천, 큰손실률 4%
A급 (Advanced · 20만/건):  점수≥65 + 등락 10~18% — 6년 43건 (연 7회)
  → 좁은 등락 범위 + 높은 점수
B급 (Basic · 10만/건):     V1 (1개+ 추천) + 등락 7~25% — 매일
  → 베이스 픽, 매일 1종목

[공통 조건]
- 시장: 코스닥(KOSDAQ)
- 매매타입: 돌파매매
- 보유: 180일 (이전 120일 → 180일이 +29.57% vs +19.12%로 압도)
- 손절/익절 없음

[중첩 처리]
V > S > A > B 순 우선 → 상위 등급 채택 시 하위에서 제외
V/S는 동일 등급 내 여러 종목 허용 (2~3개)
A/B는 점수 1위 1종목만
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# 등급별 자본 비중 (원)
GRADE_WEIGHTS = {
    "V": 500_000,
    "S": 300_000,
    "A": 200_000,
    "B": 100_000,
}

# 4 프리셋 (V1 앙상블)
PRESETS_4 = ["default", "box_breakout", "habarocell", "pullback"]

# 등급별 색상/이모지
GRADE_INFO = {
    "V": {
        "emoji": "🏆",
        "name": "V급 (Very Best)",
        "color": "#FFD700",  # Gold
        "bg": "rgba(255, 215, 0, 0.10)",
        "border": "rgba(255, 215, 0, 0.50)",
        "weight_str": "50만원",
        "frequency": "연 5회 (2~3개월에 1번)",
        "avg_return": "+96.9%",
        "big_loss_rate": "7%",
    },
    "S": {
        "emoji": "💎",
        "name": "S급 (Super)",
        "color": "#00BFFF",  # DeepSkyBlue
        "bg": "rgba(0, 191, 255, 0.10)",
        "border": "rgba(0, 191, 255, 0.50)",
        "weight_str": "30만원",
        "frequency": "월 1.4회 (연 17회)",
        "avg_return": "+56.8%",
        "big_loss_rate": "4%",
    },
    "A": {
        "emoji": "⭐",
        "name": "A급 (Advanced)",
        "color": "#FF6B9D",  # Pink
        "bg": "rgba(255, 107, 157, 0.10)",
        "border": "rgba(255, 107, 157, 0.50)",
        "weight_str": "20만원",
        "frequency": "월 2회 (연 22회)",
        "avg_return": "+34.1%",
        "big_loss_rate": "9%",
    },
    "B": {
        "emoji": "🟢",
        "name": "B급 (Basic)",
        "color": "#66BB6A",  # Green
        "bg": "rgba(102, 187, 106, 0.10)",
        "border": "rgba(102, 187, 106, 0.50)",
        "weight_str": "10만원",
        "frequency": "거의 매일 (연 143회)",
        "avg_return": "+34.4%",
        "big_loss_rate": "12%",
    },
}


def classify_one(row: Dict) -> Optional[str]:
    """단일 후보의 등급 분류. None이면 등급 미달.

    row 필수 키: Market, ChangeRatio, n_presets, avg_score (또는 Score)
    """
    market = row.get("Market", "")
    if market != "KOSDAQ":
        return None
    cr = row.get("ChangeRatio", 0)
    n_presets = row.get("n_presets", 0)
    score = row.get("avg_score", row.get("Score", 0))

    # V급: 등락 7~25% + 점수 ≥ 75 (최고급)
    if 7 <= cr <= 25 and score >= 75:
        return "V"

    # S급: 4프리셋 모두 추천 + 점수 ≥ 65
    if 7 <= cr <= 25 and n_presets >= 4 and score >= 65:
        return "S"

    # A급: 등락 10~18% + 점수 ≥ 65
    if 10 <= cr <= 18 and score >= 65:
        return "A"

    # B급: 등락 7~25% + V1 (최소 1개 프리셋)
    if 7 <= cr <= 25 and n_presets >= 1:
        return "B"

    return None


def classify_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """후보 DataFrame에 등급 칼럼 추가."""
    if df.empty:
        df = df.copy()
        df["grade"] = None
        return df
    df = df.copy()
    df["grade"] = df.apply(lambda r: classify_one(r.to_dict()), axis=1)
    return df


def build_grade_buckets(df: pd.DataFrame, vs_max: int = 10,
                         a_max: int = 10, b_max: int = 5,
                         show_all: bool = True) -> Dict[str, pd.DataFrame]:
    """등급별로 종목 묶기.

    show_all=True (기본): 각 등급 안에서 조건 만족한 모든 종목 표시 (vs_max/a_max/b_max 한도 내)
    show_all=False: V/S는 vs_max, A/B는 1개
    """
    df = df.copy()
    if "avg_score" not in df.columns and "Score" in df.columns:
        df["avg_score"] = df["Score"]

    buckets = {}

    # V급
    v = df[df["grade"] == "V"].sort_values("avg_score", ascending=False).head(vs_max)
    buckets["V"] = v
    used_codes = set(v.get("Code", pd.Series()).tolist())

    # S급 (V 제외)
    s = df[(df["grade"] == "S") & (~df["Code"].isin(used_codes))]
    s = s.sort_values("avg_score", ascending=False).head(vs_max)
    buckets["S"] = s
    used_codes.update(s.get("Code", pd.Series()).tolist())

    # A급 (V/S 제외)
    a = df[(df["grade"] == "A") & (~df["Code"].isin(used_codes))]
    if show_all:
        a = a.sort_values("avg_score", ascending=False).head(a_max)
    else:
        a = a.sort_values("avg_score", ascending=False).head(1)
    buckets["A"] = a
    used_codes.update(a.get("Code", pd.Series()).tolist())

    # B급 (V/S/A 제외)
    b = df[(df["grade"] == "B") & (~df["Code"].isin(used_codes))]
    if show_all:
        b = b.sort_values("avg_score", ascending=False).head(b_max)
    else:
        b = b.sort_values("avg_score", ascending=False).head(1)
    buckets["B"] = b

    return buckets


def grade_reason(row: Dict) -> List[str]:
    """등급별 추천사유 텍스트 라인."""
    grade = row.get("grade")
    if grade is None:
        return ["⚠️ 등급 분류 불가 (조건 미달)"]

    info = GRADE_INFO[grade]
    cr = row.get("ChangeRatio", 0)
    n_p = row.get("n_presets", 0)
    score = row.get("avg_score", row.get("Score", 0))
    lines = []
    lines.append(f"{info['emoji']} **{info['name']}** · 비중 {info['weight_str']} · 빈도 {info['frequency']}")

    if grade == "V":
        lines.append(
            f"🌟 **이유:** 앙상블 점수 **{score:.1f}점** (≥75) — 4개 전략 평균 점수가 75점 이상인 "
            f"**최고급 셋업**. 6년간 단 32회만 출현."
        )
        lines.append(
            f"📊 **기대 수익:** 평균 **+96.9%** (180일 보유 기준) · 큰손실률 **7%** · "
            f"+200% 대박 확률 7%"
        )
        lines.append("💰 **비중:** 50만원 — 자본의 가장 큰 몫 배분")

    elif grade == "S":
        lines.append(
            f"💎 **이유:** **4개 전략 모두 추천** ({n_p}/4 만장일치) + 점수 **{score:.1f}점** "
            f"(≥65) — default·박스돌파·하바로셀·풀백 4개 시스템이 동시 동의한 강력한 셋업"
        )
        lines.append(
            f"📊 **기대 수익:** 평균 **+56.8%** · **큰손실률 단 4%** (역대 최저, 최안전 대박)"
        )
        lines.append("💰 **비중:** 30만원")

    elif grade == "A":
        lines.append(
            f"⭐ **이유:** 점수 **{score:.1f}점** (≥65) + 등락률 **{cr:.1f}%** (10~18% 안전 구간) — "
            f"과열 회피 + 점수 안정"
        )
        lines.append(
            f"📊 **기대 수익:** 평균 **+34.1%** · 큰손실률 **9%** · +50% 적중 24%"
        )
        lines.append("💰 **비중:** 20만원")

    else:  # B
        lines.append(
            f"🟢 **이유:** V1 통과 (4개 전략 중 {n_p}개 추천) + 등락 **{cr:.1f}%** (7~25%) — "
            f"코스닥 + 돌파매매 베이스 픽"
        )
        lines.append(
            f"📊 **기대 수익:** 평균 **+34.4%** · 큰손실률 **12%**"
        )
        lines.append("💰 **비중:** 10만원")

    # 공통: 매도/리스크
    lines.append(
        f"📅 **매도:** 매수 후 **180일** 자동 청산 (손절/익절 없음)"
    )
    return lines


def grade_badge_html(grade: str) -> str:
    """등급 배지 HTML."""
    if grade not in GRADE_INFO:
        return ""
    info = GRADE_INFO[grade]
    return (
        f'<span style="display:inline-block;padding:4px 12px;background:{info["bg"]};'
        f'border:1.5px solid {info["border"]};border-radius:16px;'
        f'font-size:13px;font-weight:800;color:{info["color"]};'
        f'letter-spacing:0.5px;">'
        f'{info["emoji"]} {grade}급'
        f'</span>'
    )


def grade_priority(grade: str) -> int:
    return {"V": 4, "S": 3, "A": 2, "B": 1}.get(grade, 0)


# === 4 프리셋 통합 후보 빌드 ===

def build_ensemble_candidates_from_picks(picks_list: List[pd.DataFrame]) -> pd.DataFrame:
    """여러 프리셋 스캔 결과를 통합 → 종목별 n_presets, avg_score 계산.

    picks_list: List[DataFrame], 각 DF는 한 프리셋의 스캔 결과
    각 DF에는 Code, Name, Market, ChangeRatio, Score, Close 등 컬럼 포함.
    """
    if not picks_list:
        return pd.DataFrame()

    # 각 프리셋에 라벨 부여
    frames = []
    for i, df in enumerate(picks_list):
        if df is None or df.empty: continue
        d = df.copy()
        d["__preset_idx"] = i
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Code 기준 그룹 → n_presets, avg_score
    agg_dict = {
        "n_presets": ("__preset_idx", "nunique"),
        "avg_score": ("Score", "mean"),
    }
    # 첫 값 보존
    first_cols = ["Name", "Market", "Close", "ChangeRatio", "Amount", "TradeType",
                  "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
                  "vol_ratio", "candle_pct", "rs_ratio", "past_5d", "ma3", "ma5", "ma10",
                  "is_first_pullback", "cup_and_handle_detected", "gap_support_detected",
                  "pullback_quality", "reason", "Rank"]
    for c in first_cols:
        if c in combined.columns:
            agg_dict[c] = (c, "first")

    grouped = combined.groupby("Code").agg(**agg_dict).reset_index()
    # 점수 정렬
    grouped = grouped.sort_values("avg_score", ascending=False).reset_index(drop=True)
    return grouped


def build_ensemble_from_enriched_for_date(target_date) -> pd.DataFrame:
    """캐시된 enriched_*.parquet에서 특정 날짜 후보 통합 (백테스트용)."""
    from pathlib import Path
    CACHE = Path("cache")
    frames = []
    target_date = pd.to_datetime(target_date)
    for p in PRESETS_4:
        path = CACHE / f"enriched_{p}.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        sub = df[(df["Date"] == target_date) & (df["TradeType"] == "돌파매매")].copy()
        if not sub.empty:
            sub["__preset_idx"] = p
            frames.append(sub)
    if not frames: return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    agg_dict = {
        "n_presets": ("__preset_idx", "nunique"),
        "avg_score": ("Score", "mean"),
        "Name": ("Name", "first"),
        "Market": ("Market", "first"),
        "Close": ("Close", "first"),
        "ChangeRatio": ("ChangeRatio", "first"),
    }
    for c in ["Amount", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
              "vol_ratio", "candle_pct", "rs_ratio", "past_5d",
              "ret_120d", "ret_d1_close", "ret_60d", "ret_90d"]:
        if c in combined.columns:
            agg_dict[c] = (c, "first")
    grouped = combined.groupby("Code").agg(**agg_dict).reset_index()
    return grouped


def build_ensemble_all_enriched() -> pd.DataFrame:
    """전 기간 enriched에서 일자×종목 통합 데이터 (백테스트용)."""
    from pathlib import Path
    CACHE = Path("cache")
    frames = []
    for p in PRESETS_4:
        path = CACHE / f"enriched_{p}.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        sub = df[df["TradeType"] == "돌파매매"].copy()
        sub["__preset_idx"] = p
        frames.append(sub)
    if not frames: return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    agg_dict = {
        "n_presets": ("__preset_idx", "nunique"),
        "avg_score": ("Score", "mean"),
        "Name": ("Name", "first"),
        "Market": ("Market", "first"),
        "Close": ("Close", "first"),
        "ChangeRatio": ("ChangeRatio", "first"),
    }
    for c in ["Amount", "TradeType", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
              "vol_ratio", "candle_pct", "rs_ratio", "past_5d",
              "ret_120d", "ret_180d", "ret_240d", "ret_365d",
              "ret_d1_close", "ret_60d", "ret_90d", "ret_30d", "ret_20d"]:
        if c in combined.columns:
            agg_dict[c] = (c, "first")
    grouped = combined.groupby(["Date", "Code"]).agg(**agg_dict).reset_index()
    return grouped
