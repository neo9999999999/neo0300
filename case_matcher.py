"""
유사 사례 매칭 + 추천 사유 생성기
"""
from typing import Dict, List, Tuple
import numpy as np
from case_studies import CASE_STUDIES, Pattern


# 시그널 한글명 매핑
SIGNAL_NAMES = {
    "s1": "박스권 돌파 (힘의 응축)",
    "s2": "거래량 폭증 (수급 선점)",
    "s3": "장대양봉 (시가→종가 강세)",
    "s4": "이평선 정배열 (단기 강세)",
    "s5": "전고점 돌파 (신고가)",
    "s6": "과열 회피 (5일 누적 미과열)",
    "s7": "눌림목 셋업 (1차 슈팅 후 첫 눌림)",
    "s8": "수급 연속성 (거래량 지속)",
    "s9": "장기이평 돌파 (대시세 초입)",
    "s10": "상대강도 (시장 대비 강세)",
    "s11": "갭+이평 중첩 (강력 지지자리)",
    "s12": "패턴 품질 (컵앤핸들/역H&S/첫눌림)",
}

# 사례 → 시그널 패턴 추정 (key_signals 텍스트 → 12차원 벡터)
CASE_SIGNAL_KEYWORDS = {
    "박스권": "s1", "박스": "s1", "응축": "s1",
    "거래량": "s2", "대량거래": "s2", "거래대금": "s2",
    "장대양봉": "s3", "양봉": "s3", "상승률": "s3",
    "정배열": "s4", "이평": "s4", "MA": "s4",
    "신고가": "s5", "최고가": "s5", "전고점": "s5", "돌파": "s5",
    "5일선": "s4", "3일선": "s4", "10일선": "s4",
    "눌림": "s7", "조정": "s7",
    "거래량 급감": "s7", "거래 급감": "s7",
    "장기이평": "s9", "120일선": "s9", "240일선": "s9", "480일선": "s9",
    "주도주": "s10", "대장주": "s10", "테마": "s10",
    "갭": "s11", "저지바": "s11", "저항": "s11",
    "도지": "s12", "쌍도지": "s12", "망치형": "s12", "아래꼬리": "s12",
    "컵앤핸들": "s12", "역헤드앤숄더": "s12", "역H&S": "s12",
    "이중바닥": "s12", "쌍바닥": "s12", "분봉": "s12",
    "첫 눌림": "s7", "첫눌림": "s7",
    "수렴": "s4", "정배": "s4",
}


def estimate_case_signals(case: Dict) -> np.ndarray:
    """
    사례의 key_signals 텍스트 + trigger + theme 으로부터 12차원 시그널 벡터 추정.
    """
    text = (
        " ".join(case.get("key_signals", []))
        + " " + case.get("trigger", "")
        + " " + case.get("theme", "")
    ).lower()

    vec = np.zeros(12)
    for keyword, sig_key in CASE_SIGNAL_KEYWORDS.items():
        if keyword in text or keyword.lower() in text:
            sig_idx = int(sig_key[1:]) - 1
            vec[sig_idx] += 30  # 키워드 매칭마다 30점 가산

    # 패턴 별 기본 시그널 보강
    pat = case.get("pattern")
    if pat == Pattern.A_BREAKOUT:
        vec[0] += 30  # s1 박스권
        vec[1] += 30  # s2 거래량
        vec[2] += 30  # s3 양봉
        vec[4] += 30  # s5 전고점
    elif pat == Pattern.B_PULLBACK:
        vec[3] += 30  # s4 정배열
        vec[5] += 30  # s6 미과열
        vec[6] += 50  # s7 눌림목
        vec[11] += 20  # s12 패턴
    elif pat == Pattern.D_LONGTERM:
        vec[8] += 60  # s9 장기이평
        vec[9] += 30  # s10 상대강도

    return np.clip(vec, 0, 100)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """코사인 유사도."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def find_similar_cases(stock_row, top_n: int = 3) -> List[Dict]:
    """
    추천 종목과 유사한 실전 사례 찾기.

    Returns:
        [{
            "case": case dict,
            "similarity": 0~100,
            "match_reason": str (왜 유사한지),
        }, ...]
    """
    # 종목 시그널 벡터
    stock_vec = np.array([float(stock_row.get(f"s{i}", 0)) for i in range(1, 13)])
    stock_pattern = stock_row.get("TradeType", "")

    pattern_to_trade_type = {
        Pattern.A_BREAKOUT: "돌파매매",
        Pattern.B_PULLBACK: "눌림목매매",
        Pattern.C_DOUBLE_BOTTOM: "눌림목매매",
        Pattern.D_LONGTERM: "대시세 초입",
    }

    results = []
    for case in CASE_STUDIES:
        case_vec = estimate_case_signals(case)
        sim = cosine_similarity(stock_vec, case_vec)
        # 패턴 일치 시 부스트
        case_tt = pattern_to_trade_type.get(case.get("pattern"), "")
        if case_tt == stock_pattern:
            sim = sim * 1.15  # 패턴 일치 가산
        # 유사율 (0~100%)
        sim_pct = min(100, sim * 100)

        # 매칭 이유 생성
        match_keywords = []
        case_text = (
            " ".join(case.get("key_signals", [])) + " " + case.get("trigger", "")
        )
        for sig_key in sorted(SIGNAL_NAMES.keys()):
            sig_val = stock_row.get(sig_key, 0)
            if sig_val >= 50:
                # 사례 텍스트에도 해당 시그널 키워드 있는지
                sig_name_short = SIGNAL_NAMES[sig_key].split(" ")[0]
                if sig_name_short in case_text:
                    match_keywords.append(sig_name_short)
        match_reason = (
            f"공통: {', '.join(match_keywords[:3])}"
            if match_keywords else f"패턴 일치: {case_tt}"
        )

        results.append({
            "case": case,
            "similarity": round(sim_pct, 1),
            "match_reason": match_reason,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]


def generate_recommendation_reasons(row) -> List[str]:
    """
    추천 사유를 자연어 문장으로 생성.
    카드 하단에 표시할 짧은 설명 리스트.
    """
    reasons = []

    # 1) 강한 시그널들 (70점 이상)
    strong = []
    for sig_key, sig_name in SIGNAL_NAMES.items():
        val = row.get(sig_key, 0)
        if val >= 70:
            short = sig_name.split(" (")[0]
            strong.append((short, val))
    strong.sort(key=lambda x: -x[1])
    if strong:
        top_strong = " · ".join(f"{name} {v:.0f}점" for name, v in strong[:4])
        reasons.append(f"💪 **강한 시그널**: {top_strong}")

    # 2) 패턴 진단
    pattern_msgs = []
    if row.get("is_first_pullback"):
        pattern_msgs.append("🎯 **1차 슈팅 후 첫 눌림목** (진입 적기, n차 눌림 ❌)")
    if row.get("cup_and_handle_detected"):
        pattern_msgs.append("☕ **컵앤핸들 패턴** 감지 — 손익비 우수 셋업")
    if row.get("inverse_hns_detected"):
        pattern_msgs.append("🔁 **역헤드앤숄더** 넥라인 돌파")
    if row.get("gap_support_detected"):
        pattern_msgs.append("🧲 **과거 갭 + 이평선 중첩** = 강력 지지자리")
    pq = row.get("pullback_quality")
    if pq == "진짜 지지":
        pattern_msgs.append("✅ **진짜 지지** 판정 (거래량 급감 + 짧은 음봉 + 저점 점진 상승)")
    elif pq == "가짜 눌림":
        pattern_msgs.append("⚠️ **가짜 눌림** 주의 (거래량 유지 + 장대음봉)")
    reasons.extend(pattern_msgs)

    # 3) 수치 근거
    metrics = []
    vol_ratio = row.get("vol_ratio", 0)
    if vol_ratio >= 3:
        metrics.append(f"거래량 평소 대비 **×{vol_ratio:.1f}배 폭증**")
    elif vol_ratio >= 1.5:
        metrics.append(f"거래량 ×{vol_ratio:.1f}배")
    candle_pct = row.get("candle_pct", 0)
    if candle_pct >= 5:
        metrics.append(f"시가→종가 **+{candle_pct:.1f}%** (장대양봉)")
    cum_5d = row.get("cum_5d_gain", 0)
    if 0 <= cum_5d <= 10:
        metrics.append(f"최근 5일 누적 +{cum_5d:.1f}% (미과열)")
    rs = row.get("rs_ratio", 1.0)
    if rs >= 1.1:
        metrics.append(f"시장 대비 +{(rs-1)*100:.0f}% 강세 (주도주)")
    if metrics:
        reasons.append(f"📊 **수치**: " + " / ".join(metrics))

    # 4) 워치리스트 매칭
    wl = []
    if row.get("InHabarocell"):
        wl.append("🎓 하바로셀이 강의/방송에서 다룬 종목")
    if row.get("InHaseunghoon"):
        wl.append("📺 하승훈 채널 분석 종목")
    if row.get("InUserList"):
        wl.append("⭐ 사용자 기록 종목")
    if row.get("Themes"):
        wl.append(f"🏷️ 테마: {row['Themes']}")
    reasons.extend(wl)

    # 5) 하승훈 5조건
    h_data = {
        "themes": row.get("Themes", ""), "amount": row.get("Amount", 0),
        "s5": row.get("s5", 0), "s9": row.get("s9", 0),
        "s3": row.get("s3", 0), "s2": row.get("s2", 0), "s10": row.get("s10", 0),
    }
    from rules import haseunghoon_5_conditions
    passed, msgs = haseunghoon_5_conditions(h_data, {})
    if passed:
        passed_items = [m for m in msgs if m != "—" and not any(
            x in m for x in ["미확인", "약", "부족", "아님"])]
        reasons.append(f"✅ **하승훈 5조건 통과**: {' · '.join(passed_items)}")

    return reasons


# 영문 컬럼 → 한글 매핑 (DataFrame 한글화용)
COLUMN_KOREAN = {
    "Name": "종목명",
    "Code": "종목코드",
    "Market": "시장",
    "Score": "점수",
    "Close": "종가",
    "ChangeRatio": "등락률(%)",
    "Amount": "거래대금",
    "MarketCap": "시가총액",
    "TradeType": "매매타입",
    "Pattern": "패턴",
    "PatternConfidence": "패턴 신뢰도",
    "Themes": "테마",
    "InHabarocell": "하바로셀",
    "InHaseunghoon": "하승훈",
    "InUserList": "사용자",
    "BreakoutScore": "돌파점수",
    "PullbackScore": "눌림목점수",
    "Rank": "순위",
    "Date": "날짜",
    "s1": "S1 박스권",
    "s2": "S2 거래량",
    "s3": "S3 장대양봉",
    "s4": "S4 정배열",
    "s5": "S5 전고점",
    "s6": "S6 미과열",
    "s7": "S7 눌림목",
    "s8": "S8 수급연속",
    "s9": "S9 장기이평",
    "s10": "S10 상대강도",
    "s11": "S11 갭+이평",
    "s12": "S12 패턴품질",
    "vol_ratio": "거래량배수",
    "candle_pct": "양봉률",
    "cum_5d_gain": "5일누적",
    "upper_wick_ratio": "윗꼬리비",
    "rs_ratio": "RS비율",
    "longterm_ma_breakouts": "장기이평돌파",
    "ma3": "MA3",
    "ma5": "MA5",
    "ma10": "MA10",
    "is_first_pullback": "첫눌림",
    "pullback_quality": "눌림품질",
    "cup_and_handle_detected": "컵앤핸들",
    "inverse_hns_detected": "역H&S",
    "gap_support_detected": "갭지지",
    "sell_date": "매도일",
    "sell_price": "매도가",
    "return_pct": "수익률(%)",
    "valid": "유효",
}


def koreanize_dataframe(df):
    """DataFrame 컬럼명을 한글로 변환."""
    import pandas as pd
    if df is None or df.empty:
        return df
    rename_map = {k: v for k, v in COLUMN_KOREAN.items() if k in df.columns}
    return df.rename(columns=rename_map)
