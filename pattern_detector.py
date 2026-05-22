"""
고급 패턴 감지 모듈 (마스터 가이드 v4)

검증 기능:
  1. First Pullback Validator — 진짜 "첫 눌림" 인지 검증 (n차 눌림 거름)
  2. Real vs Fake Pullback Classifier — 거래량/캔들/분봉 흐름 종합 판단
  3. Cup-and-Handle Heuristic — 컵앤핸들 패턴 감지
  4. Gap Detector — 과거 갭 + 이평 중첩 확인
  5. Inverse Head & Shoulders — 역헤드앤숄더 휴리스틱

모든 함수는 OHLCV DataFrame을 받아 점수(0~100) + 상세 진단 반환.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# =============================================================================
# 1. First Pullback Validator
# =============================================================================
def is_first_pullback(ohlcv: pd.DataFrame, lookback: int = 30) -> Dict:
    """
    "첫 눌림" 검증:
      - 최근 N일 내 단 1번의 강한 슈팅(상한가 또는 +10% 장대양봉)
      - 그 슈팅 이후의 첫 조정인지 확인
      - 2차/3차 눌림이면 ❌
    """
    if ohlcv is None or len(ohlcv) < lookback + 1:
        return {"is_first": False, "shoot_days": 0, "reason": "데이터 부족"}

    close = ohlcv["Close"]
    rets = close.pct_change() * 100
    recent = rets.iloc[-lookback:]

    # +10% 이상 강한 양봉 카운트
    big_up_days = recent[recent >= 10]
    shoot_count = len(big_up_days)

    if shoot_count == 0:
        return {"is_first": False, "shoot_days": 0,
                "reason": "최근 강한 슈팅 없음 — 눌림목 셋업 자체가 아님"}
    if shoot_count == 1:
        # 단 1번만 슈팅 → 진짜 첫 눌림
        shoot_idx = big_up_days.index[-1]
        days_since = len(close.loc[shoot_idx:]) - 1
        return {
            "is_first": True, "shoot_days": days_since,
            "shoot_date": shoot_idx.strftime("%Y-%m-%d"),
            "shoot_return": round(big_up_days.iloc[-1], 1),
            "reason": f"1차 슈팅 후 {days_since}일째 — 진짜 첫 눌림 ✅",
        }
    # 2번 이상 → n차 눌림
    return {
        "is_first": False, "shoot_days": shoot_count,
        "reason": f"슈팅 {shoot_count}회 발생 — n차 눌림은 신규 진입 ❌",
    }


# =============================================================================
# 2. Real vs Fake Pullback Classifier
# =============================================================================
def classify_pullback_quality(ohlcv: pd.DataFrame, pullback_window: int = 5) -> Dict:
    """
    조정 구간의 품질 판정:
      진짜 지지: 거래량 급감 + 짧은 음봉 + 저점 점진 상승
      가짜 눌림: 거래량 유지/증가 + 장대음봉 + 저점 지속 하락
    """
    if ohlcv is None or len(ohlcv) < pullback_window + 20:
        return {"quality": "unknown", "score": 0, "reasons": ["데이터 부족"]}

    recent = ohlcv.iloc[-pullback_window:]
    baseline = ohlcv.iloc[-pullback_window - 20 : -pullback_window]

    score = 0
    reasons = []

    # 1) 거래량 변화: 조정 기간 평균 vs 직전 20일 평균
    recent_vol = recent["Volume"].mean()
    baseline_vol = baseline["Volume"].mean()
    vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1
    if vol_ratio <= 0.7:
        score += 35
        reasons.append(f"✅ 거래량 급감 (×{vol_ratio:.2f}) = 매도세 소진")
    elif vol_ratio <= 1.0:
        score += 20
        reasons.append(f"⚠️ 거래량 유지 (×{vol_ratio:.2f})")
    else:
        reasons.append(f"❌ 거래량 증가 (×{vol_ratio:.2f}) = 매도 압력 ↑")

    # 2) 음봉 길이: 평균 음봉 몸통 / 평균 일중 변동폭
    bodies = (recent["Close"] - recent["Open"]).abs()
    ranges = recent["High"] - recent["Low"]
    avg_body_ratio = (bodies / ranges).mean() if (ranges > 0).all() else 0.5
    bear_candles = recent[recent["Close"] < recent["Open"]]
    if len(bear_candles) > 0:
        avg_bear_body_pct = ((bear_candles["Open"] - bear_candles["Close"]) / bear_candles["Open"] * 100).mean()
        if avg_bear_body_pct <= 2.0:
            score += 30
            reasons.append(f"✅ 음봉 몸통 평균 {avg_bear_body_pct:.1f}% = 완만한 조정")
        elif avg_bear_body_pct <= 5.0:
            score += 15
            reasons.append(f"⚠️ 음봉 평균 {avg_bear_body_pct:.1f}% = 보통 조정")
        else:
            reasons.append(f"❌ 음봉 평균 {avg_bear_body_pct:.1f}% = 강한 매도")

    # 3) 저점 추이: 저점들이 우상향인지 우하향인지
    lows = recent["Low"].values
    if len(lows) >= 3:
        # 선형 회귀 기울기
        slope = np.polyfit(range(len(lows)), lows, 1)[0]
        slope_pct = slope / lows.mean() * 100
        if slope_pct >= 0:
            score += 35
            reasons.append(f"✅ 저점 점진 상승 (기울기 +{slope_pct:.2f}%)")
        elif slope_pct >= -1.5:
            score += 15
            reasons.append(f"⚠️ 저점 보합 ({slope_pct:+.2f}%)")
        else:
            reasons.append(f"❌ 저점 지속 하락 ({slope_pct:.2f}%)")

    quality = "진짜 지지" if score >= 70 else "보통 조정" if score >= 40 else "가짜 눌림"
    return {"quality": quality, "score": score, "reasons": reasons,
            "vol_ratio": round(vol_ratio, 2)}


# =============================================================================
# 3. Cup-and-Handle Heuristic
# =============================================================================
def detect_cup_and_handle(ohlcv: pd.DataFrame, min_cup_days: int = 30,
                          handle_max_days: int = 15) -> Dict:
    """
    컵앤핸들 패턴 감지 (휴리스틱):
      1) 컵: 좌측 고점 → 완만한 하락 → 바닥 → 완만한 상승 → 우측 고점 (좌≈우 ±10%)
      2) 핸들: 우측 고점 후 짧고 얕은 조정 (-5~-15%)
      3) 핸들 끝에서 거래량 증가하며 돌파 시 매수 시점
    """
    if ohlcv is None or len(ohlcv) < min_cup_days + handle_max_days:
        return {"detected": False, "score": 0, "reason": "데이터 부족"}

    # 최근 60일 정도를 컵 후보 영역으로
    window = ohlcv.iloc[-(min_cup_days + handle_max_days):]
    close = window["Close"].values
    high = window["High"].values

    # 좌측 고점: 앞쪽 1/4에서 최고가
    quarter = len(window) // 4
    left_high_idx = quarter // 2 + np.argmax(high[:quarter])
    left_high = high[left_high_idx]

    # 우측 고점: 뒤쪽 1/3 - handle_max_days 부분
    right_search_start = max(left_high_idx + min_cup_days // 2, len(window) - handle_max_days - 10)
    right_search_end = max(right_search_start + 5, len(window) - handle_max_days)
    if right_search_end <= right_search_start:
        return {"detected": False, "score": 0, "reason": "윈도우 부족"}
    right_high_relative = np.argmax(high[right_search_start:right_search_end])
    right_high_idx = right_search_start + right_high_relative
    right_high = high[right_high_idx]

    # 바닥
    bottom_idx = left_high_idx + np.argmin(close[left_high_idx:right_high_idx]) if right_high_idx > left_high_idx else left_high_idx
    bottom = close[bottom_idx]

    # 검증
    score = 0
    reasons = []
    # 1) 컵 좌우 고점 비슷 (±10%)
    if left_high > 0:
        symmetry = abs(left_high - right_high) / left_high * 100
        if symmetry <= 10:
            score += 30
            reasons.append(f"좌우 고점 대칭 (차이 {symmetry:.1f}%)")
        else:
            reasons.append(f"좌우 고점 비대칭 ({symmetry:.1f}%)")
    # 2) 컵 깊이 (10~30% 적당)
    if right_high > 0:
        cup_depth = (right_high - bottom) / right_high * 100
        if 10 <= cup_depth <= 35:
            score += 30
            reasons.append(f"컵 깊이 {cup_depth:.1f}% (적정)")
        else:
            reasons.append(f"컵 깊이 {cup_depth:.1f}% (부적정)")
    # 3) 핸들 (최근 N일이 짧은 조정)
    handle = window.iloc[right_high_idx:]
    if len(handle) >= 3 and right_high > 0:
        handle_low = handle["Low"].min()
        handle_depth = (right_high - handle_low) / right_high * 100
        if 3 <= handle_depth <= 15:
            score += 30
            reasons.append(f"핸들 깊이 {handle_depth:.1f}% (적정 -5~-15%)")
        else:
            reasons.append(f"핸들 깊이 {handle_depth:.1f}%")
    # 4) 오늘 종가가 우측 고점 근처
    today_close = window["Close"].iloc[-1]
    if right_high > 0 and today_close >= right_high * 0.95:
        score += 10
        reasons.append("종가가 우측 고점 95%+ 근접 (돌파 임박)")

    return {
        "detected": score >= 70,
        "score": min(100, score),
        "reasons": reasons,
        "cup_depth_pct": round((right_high - bottom) / right_high * 100, 1) if right_high > 0 else 0,
    }


# =============================================================================
# 4. Gap Detector — 갭+이평 중첩 자리 확인
# =============================================================================
def detect_gap_support(ohlcv: pd.DataFrame, lookback: int = 90,
                       ma_periods: Tuple[int, ...] = (15, 20, 60, 120)) -> Dict:
    """
    과거 갭 자리 + 이평선 중첩 = 강력 지지 후보:
      1) 직전 N일 내 발생한 상승 갭 위치 식별
      2) 현재 종가가 그 갭 + 이평선 중첩 자리에 도달했는지 확인
    """
    if ohlcv is None or len(ohlcv) < lookback:
        return {"detected": False, "score": 0, "gaps": []}

    window = ohlcv.iloc[-lookback:]
    today_close = window["Close"].iloc[-1]

    # 상승 갭 식별: 오늘 저가 > 어제 고가
    gaps: List[Dict] = []
    for i in range(1, len(window)):
        prev_high = window["High"].iloc[i - 1]
        cur_low = window["Low"].iloc[i]
        if cur_low > prev_high * 1.01:  # 1%+ 갭
            gap_top = cur_low
            gap_bottom = prev_high
            gaps.append({
                "date": window.index[i].strftime("%Y-%m-%d"),
                "top": gap_top, "bottom": gap_bottom,
                "size_pct": round((gap_top - gap_bottom) / gap_bottom * 100, 2),
            })

    # 현재가가 어떤 갭 영역과 겹치는지
    in_gap = None
    for g in gaps:
        if g["bottom"] * 0.98 <= today_close <= g["top"] * 1.02:
            in_gap = g
            break

    # 이평선 중첩 여부
    close = window["Close"]
    ma_at_today = {p: close.rolling(p).mean().iloc[-1] for p in ma_periods if len(close) >= p}
    nearby_mas = [
        p for p, v in ma_at_today.items()
        if pd.notna(v) and abs(today_close - v) / today_close < 0.03  # 3% 이내
    ]

    score = 0
    reasons = []
    if in_gap:
        score += 50
        reasons.append(f"갭 메우기 자리 ({in_gap['date']}, +{in_gap['size_pct']}%)")
    if len(nearby_mas) >= 2:
        score += 50
        reasons.append(f"이평선 {len(nearby_mas)}개 중첩 ({', '.join(f'MA{p}' for p in nearby_mas)})")
    elif len(nearby_mas) == 1:
        score += 25
        reasons.append(f"단일 이평선 근접 (MA{nearby_mas[0]})")

    return {
        "detected": score >= 50,
        "score": min(100, score),
        "gaps_found": len(gaps),
        "in_gap": in_gap,
        "nearby_mas": nearby_mas,
        "reasons": reasons,
    }


# =============================================================================
# 5. Inverse Head & Shoulders Heuristic
# =============================================================================
def detect_inverse_hns(ohlcv: pd.DataFrame, window_days: int = 60) -> Dict:
    """
    역헤드앤숄더 휴리스틱:
      - 좌어깨 저점 ≈ 우어깨 저점 (±10%)
      - 머리 저점 < 어깨 저점 (10% 이상 깊음)
      - 넥라인(좌어깨 고점, 우어깨 고점 비슷) 돌파 시 매수
    """
    if ohlcv is None or len(ohlcv) < window_days:
        return {"detected": False, "score": 0}

    window = ohlcv.iloc[-window_days:]
    lows = window["Low"].values
    highs = window["High"].values

    if len(lows) < 30:
        return {"detected": False, "score": 0}

    # 3등분
    third = len(window) // 3
    left = lows[:third]
    middle = lows[third:2 * third]
    right = lows[2 * third:]

    left_low = left.min()
    middle_low = middle.min()
    right_low = right.min()

    score = 0
    reasons = []
    # 어깨 대칭
    if left_low > 0:
        sym = abs(left_low - right_low) / left_low * 100
        if sym <= 10:
            score += 30
            reasons.append(f"좌우 어깨 대칭 (차이 {sym:.1f}%)")
    # 머리 더 깊음
    if middle_low < min(left_low, right_low) * 0.95:
        depth = (min(left_low, right_low) - middle_low) / min(left_low, right_low) * 100
        score += 35
        reasons.append(f"머리 어깨보다 {depth:.1f}% 깊음")
    # 넥라인 돌파
    neckline = max(highs[:third].max(), highs[2 * third:].max())
    today_close = window["Close"].iloc[-1]
    if today_close > neckline:
        score += 35
        reasons.append(f"넥라인({neckline:.0f}) 돌파 ✅")
    elif today_close > neckline * 0.97:
        score += 15
        reasons.append("넥라인 97%+ 근접")

    return {
        "detected": score >= 70,
        "score": min(100, score),
        "reasons": reasons,
        "left_shoulder": round(left_low, 2),
        "head": round(middle_low, 2),
        "right_shoulder": round(right_low, 2),
        "neckline": round(neckline, 2),
    }


# =============================================================================
# 종합 패턴 진단
# =============================================================================
def diagnose_all_patterns(ohlcv: pd.DataFrame) -> Dict:
    """모든 패턴 검사를 한번에 실행."""
    return {
        "first_pullback": is_first_pullback(ohlcv),
        "pullback_quality": classify_pullback_quality(ohlcv),
        "cup_and_handle": detect_cup_and_handle(ohlcv),
        "gap_support": detect_gap_support(ohlcv),
        "inverse_hns": detect_inverse_hns(ohlcv),
    }
