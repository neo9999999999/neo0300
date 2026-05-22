"""
진입/청산 룰 코드화

하바로셀 5대 원칙 + 차트 분석을 기계적 룰로 구현.
모든 룰은 함수 단위로 분리되어 백테스트 / 실전 모두에서 호출 가능.

원칙:
  1) 수급 선점         — 갭 도박 ❌, 종가 매수
  2) 3:20 PM 확인      — 장 마감 직전 세력 흔적
  3) 힘의 응축          — 박스권/조정 후 진입 (과열 종목 ❌)
  4) 눌림목 불타기      — 1차 +10% 익절 → 지지 확인 → 재진입
  5) 분할 청산 + 손절   — 욕심 ❌, 지지선 이탈 시 즉시 손절
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd


# =============================================================================
# 진입 결정 단계 (5단계 — 시간외 단일가 확장)
# =============================================================================
class EntryStage(Enum):
    STAGE1_FILTER = "1차 필터 (3:00 PM~)"
    STAGE2_SETUP = "차트 셋업 확인 (3:10 PM~)"
    STAGE3_DEMAND = "3:20 PM 수급 확인 (동시호가)"
    STAGE4_EXECUTE = "정규장 종가 진입 (3:25~3:30)"
    STAGE5_AFTERHOURS = "시간외 단일가 추가 매수 (4:00~6:00 PM)"


@dataclass
class EntryDecision:
    pass_filter: bool
    pass_setup: bool
    pass_demand: bool          # 호가/외인기관 데이터 필요 (현재는 점수 대체)
    final_decision: bool
    reason: str
    stage_results: Dict[str, bool]


# =============================================================================
# Stage 1 — 1차 필터 (재료 확인)
# =============================================================================
def stage1_basic_filter(
    amount: float,
    market_cap: float,
    change_ratio: float,
    cum_5d_gain: float = 0,
    min_amount: float = 5_000_000_000,
    min_marcap: float = 200_000_000_000,
    change_min: float = 7.0,
    change_max: float = 29.0,
    max_5d_gain: float = 25.0,
) -> Tuple[bool, str]:
    """
    1차 필터:
      ✓ 거래대금 ≥ 50억
      ✓ 시총 ≥ 2000억
      ✓ 등락률 7~29%
      ✗ 직전 5일 누적 +25% 이상 = 과열 제외
    """
    if amount < min_amount:
        return False, f"거래대금 부족 ({amount/1e8:.0f}억 < {min_amount/1e8:.0f}억)"
    if market_cap < min_marcap:
        return False, f"시총 부족 ({market_cap/1e8:.0f}억 < {min_marcap/1e8:.0f}억)"
    if not (change_min <= change_ratio <= change_max):
        return False, f"등락률 범위 외 ({change_ratio:.2f}%)"
    if cum_5d_gain > max_5d_gain:
        return False, f"5일 과열 ({cum_5d_gain:.1f}% > {max_5d_gain:.0f}%)"
    return True, "Stage1 통과"


# =============================================================================
# Stage 2 — 차트 셋업 (시그널 점수 기반)
# =============================================================================
def stage2_chart_setup(
    score: float,
    upper_wick_ratio: float,
    min_score: float = 40.0,
    max_upper_wick: float = 1.5,
) -> Tuple[bool, str]:
    """
    2차 셋업:
      ✓ 종합 점수 ≥ 40 (S1~S8 가중 평균)
      ✗ 윗꼬리 비율(꼬리/몸통) > 1.5 = 매도 압력
    """
    if score < min_score:
        return False, f"점수 미달 ({score:.1f} < {min_score})"
    if upper_wick_ratio > max_upper_wick:
        return False, f"윗꼬리 과대 ({upper_wick_ratio:.2f})"
    return True, "Stage2 통과"


# =============================================================================
# Stage 3 — 3:20 PM 수급 (호가 / 외인기관)
#   * 실시간 호가는 HTS 영역. 여기선 점수 기반 대체.
#   * 실전에서는 HTS API 연동 또는 수동 확인 권장.
# =============================================================================
def stage3_demand_proxy(
    score: float,
    vol_ratio: float,
    s2_score: float,
    high_score_threshold: float = 60.0,
    high_vol_ratio: float = 5.0,
) -> Tuple[bool, str]:
    """
    수급 프록시:
      - 점수 60+ AND 거래량 5배+ = 강한 수급 추정
      - 그 외에는 보수적으로 패스
    """
    if score >= high_score_threshold and vol_ratio >= high_vol_ratio:
        return True, f"강한 수급 (점수 {score:.0f}, 거래량 ×{vol_ratio:.1f})"
    if s2_score >= 70:
        return True, f"S2(거래량) 강세 (S2={s2_score:.0f})"
    return False, "수급 약함 → HTS에서 호가창 직접 확인 권장"


# =============================================================================
# 통합 진입 결정
# =============================================================================
def decide_entry(
    stock_data: Dict,
    config: Dict,
) -> EntryDecision:
    """
    모든 단계 종합 진입 판단.
    stock_data 필수 키: amount, market_cap, change_ratio, cum_5d_gain, score,
                       upper_wick_ratio, vol_ratio, s2_score
    """
    s1_ok, s1_reason = stage1_basic_filter(
        stock_data["amount"], stock_data["market_cap"], stock_data["change_ratio"],
        stock_data.get("cum_5d_gain", 0),
        config.get("min_amount", 5e9), config.get("min_marcap", 2e11),
        config.get("change_min", 7), config.get("change_max", 29),
        config.get("max_5d_gain", 25),
    )
    s2_ok, s2_reason = stage2_chart_setup(
        stock_data["score"], stock_data.get("upper_wick_ratio", 0),
        config.get("min_score", 40), config.get("max_upper_wick", 1.5),
    )
    s3_ok, s3_reason = stage3_demand_proxy(
        stock_data["score"], stock_data.get("vol_ratio", 0),
        stock_data.get("s2_score", 0),
    )
    # 최종: S1 + S2 필수, S3는 가산점 (없어도 진입은 가능)
    final = s1_ok and s2_ok
    return EntryDecision(
        pass_filter=s1_ok,
        pass_setup=s2_ok,
        pass_demand=s3_ok,
        final_decision=final,
        reason=" | ".join([s1_reason, s2_reason, s3_reason]),
        stage_results={"stage1": s1_ok, "stage2": s2_ok, "stage3": s3_ok},
    )


# =============================================================================
# 청산 시나리오 (4가지)
# =============================================================================
class ExitScenario(Enum):
    GAP_UP = "갭상승 +1.5%↑"
    FLAT = "보합 ±1%"
    GAP_DOWN = "갭다운 -1%↓"
    SHOOT_10 = "1차 슈팅 +10% 도달"


@dataclass
class ExitPlan:
    scenario: ExitScenario
    initial_action: str
    follow_up: str
    stop_loss_pct: float


def classify_open_scenario(buy_price: float, next_open: float) -> ExitScenario:
    """다음 날 시초가 기준 시나리오 분류."""
    gap_pct = (next_open - buy_price) / buy_price * 100
    if gap_pct >= 1.5:
        return ExitScenario.GAP_UP
    if gap_pct <= -1.0:
        return ExitScenario.GAP_DOWN
    return ExitScenario.FLAT


def make_exit_plan(scenario: ExitScenario) -> ExitPlan:
    """시나리오별 청산 계획."""
    plans = {
        ExitScenario.GAP_UP: ExitPlan(
            scenario=scenario,
            initial_action="시초 30분 내 1/3 청산 (즉시 익절)",
            follow_up="트레일링 -2% 자동, 추가 슈팅 시 1/3씩 분할",
            stop_loss_pct=-3.0,
        ),
        ExitScenario.FLAT: ExitPlan(
            scenario=scenario,
            initial_action="장중 고가 형성 시 1/2 청산",
            follow_up="눌림목 발생 → 지지(전일 종가/시가) 확인 → 잔여 청산 또는 불타기",
            stop_loss_pct=-3.0,
        ),
        ExitScenario.GAP_DOWN: ExitPlan(
            scenario=scenario,
            initial_action="시초가 즉시 손절 (미체결 매수 금지)",
            follow_up="빠른 반등 시 본전 부근에서 청산 (욕심 ❌)",
            stop_loss_pct=-3.0,
        ),
        ExitScenario.SHOOT_10: ExitPlan(
            scenario=scenario,
            initial_action="1/2 즉시 청산 (수익 확정)",
            follow_up="잔여 50% → 눌림 지지 확인 후 보유 결정, 다음 날 종가까지 청산",
            stop_loss_pct=-3.0,
        ),
    }
    return plans[scenario]


def simulate_exit(
    buy_price: float,
    ohlcv_next: pd.DataFrame,  # 매수 다음 날부터 T+2까지 OHLCV
    strategy: str = "scenario",  # scenario / next_open / next_high / next_close / t2_close
) -> Dict:
    """
    매도 시뮬레이션.
    'scenario' 모드: 시초 갭 기반으로 룰 적용.
    그 외: 단순 매도 가격 사용 (백테스트 비교용).
    """
    if ohlcv_next.empty:
        return {"sell_price": None, "return_pct": None, "scenario": None, "exit_date": None}

    nxt = ohlcv_next.iloc[0]
    nxt_date = ohlcv_next.index[0]

    if strategy == "next_open":
        return {"sell_price": nxt["Open"], "return_pct": _pct(buy_price, nxt["Open"]),
                "scenario": "단순_시초", "exit_date": nxt_date}
    if strategy == "next_high":
        return {"sell_price": nxt["High"], "return_pct": _pct(buy_price, nxt["High"]),
                "scenario": "단순_고가", "exit_date": nxt_date}
    if strategy == "next_close":
        return {"sell_price": nxt["Close"], "return_pct": _pct(buy_price, nxt["Close"]),
                "scenario": "단순_종가", "exit_date": nxt_date}
    if strategy == "t2_close":
        if len(ohlcv_next) >= 2:
            return {"sell_price": ohlcv_next.iloc[1]["Close"],
                    "return_pct": _pct(buy_price, ohlcv_next.iloc[1]["Close"]),
                    "scenario": "T+2_종가", "exit_date": ohlcv_next.index[1]}
        return {"sell_price": nxt["Close"], "return_pct": _pct(buy_price, nxt["Close"]),
                "scenario": "T+2_종가(대체)", "exit_date": nxt_date}

    # scenario 모드
    scenario = classify_open_scenario(buy_price, nxt["Open"])
    if scenario == ExitScenario.GAP_DOWN:
        sell_price = nxt["Open"]
    elif scenario == ExitScenario.GAP_UP:
        sell_price = nxt["High"] * 0.7 + nxt["Open"] * 0.3
    else:
        sell_price = (nxt["High"] + nxt["Close"]) / 2
    # 슈팅 +10% 체크
    if (nxt["High"] - buy_price) / buy_price * 100 >= 10:
        sell_price = max(sell_price, buy_price * 1.10)
        scenario = ExitScenario.SHOOT_10
    # 손절 체크
    if nxt["Low"] <= buy_price * 0.97:
        sell_price = min(sell_price, buy_price * 0.97)
    return {
        "sell_price": round(sell_price, 2),
        "return_pct": _pct(buy_price, sell_price),
        "scenario": scenario.value,
        "exit_date": nxt_date,
    }


def _pct(a: float, b: float) -> float:
    return round((b - a) / a * 100, 2) if a > 0 else 0


# =============================================================================
# 회피 필터 (7가지)
# =============================================================================
AVOID_RULES: List[Dict] = [
    {"id": "OVERHEATED", "desc": "직전 5일 누적 +25% 이상", "stage": 1},
    {"id": "UPPER_WICK", "desc": "윗꼬리/몸통 > 1.5 (매도 압력)", "stage": 2},
    {"id": "VOL_DECLINE", "desc": "거래량 감소하면서 상승 (수급 약함)", "stage": 2},
    {"id": "MA20_BREAK", "desc": "MA20 이탈 후 회복 못 함", "stage": 2},
    {"id": "NEW_LISTING", "desc": "상장 6개월 미만 (변동성 과대)", "stage": 1},
    {"id": "SINGLE_PRICE", "desc": "단일가 매매 종목", "stage": 1},
    {"id": "PENNY", "desc": "주가 500원 미만 (동전주)", "stage": 1},
    {"id": "TRADING_HALT", "desc": "거래정지 이력", "stage": 1},
]


# =============================================================================
# 하승훈 종가베팅 5조건 (하승훈 채널 핵심 룰)
# =============================================================================
def haseunghoon_5_conditions(stock_data: Dict, config: Dict) -> Tuple[bool, List[str]]:
    """
    하승훈식 종가베팅 5조건 점검:
      1) 시장의 핵심 테마에 속함
      2) 강력한 거래대금 (시총 상위 + 거래대금 100억+)
      3) 신고가 또는 의미있는 돌파 흐름 (S5 또는 S9)
      4) 장 막판까지 살아있는 매수세 (장대양봉 + 거래량)
      5) 해당 테마의 대장주 (상대강도 강함)
    """
    passed: List[str] = []
    missing: List[str] = []

    # 1) 테마 보유
    if stock_data.get("themes"):
        passed.append("①테마 매칭")
    else:
        missing.append("①테마 미확인")

    # 2) 거래대금
    if stock_data.get("amount", 0) >= 10_000_000_000:  # 100억+
        passed.append("②거래대금 강")
    else:
        missing.append("②거래대금 약")

    # 3) 돌파 흐름 (S5 전고점 OR S9 장기이평 돌파)
    if stock_data.get("s5", 0) >= 50 or stock_data.get("s9", 0) >= 30:
        passed.append("③돌파 흐름")
    else:
        missing.append("③돌파 흐름 부족")

    # 4) 살아있는 매수세 (S3 장대양봉 + S2 거래량)
    if stock_data.get("s3", 0) >= 50 and stock_data.get("s2", 0) >= 40:
        passed.append("④막판 매수세")
    else:
        missing.append("④매수세 약")

    # 5) 대장주 (상대강도 S10)
    if stock_data.get("s10", 0) >= 60:
        passed.append("⑤대장주")
    else:
        missing.append("⑤대장주 아님")

    # 5조건 중 4개 이상 통과해야 진입
    return len(passed) >= 4, passed + ["—"] + missing


# =============================================================================
# 단기이평 손절 룰 (하승훈: 3일선/5일선 이탈 시 즉시 손절)
# =============================================================================
def haseunghoon_stoploss(
    current_price: float, ma3: Optional[float], ma5: Optional[float],
    base_stoploss_pct: float = -3.0
) -> Tuple[float, str]:
    """
    하승훈식 손절가 계산:
      - 3일선 OR 5일선 이탈하면 즉시 손절
      - 둘 중 더 높은 지지선을 손절가로 (보수적)
    """
    candidates = []
    if ma3 is not None and ma3 > 0:
        candidates.append(("MA3", ma3))
    if ma5 is not None and ma5 > 0:
        candidates.append(("MA5", ma5))
    if not candidates:
        return current_price * (1 + base_stoploss_pct / 100), "기본 -3%"
    # 가장 가까운 지지선 (현재가 바로 아래)
    valid = [(name, p) for name, p in candidates if p < current_price]
    if not valid:
        return current_price * (1 + base_stoploss_pct / 100), "이평 미충족 → 기본 -3%"
    name, support = max(valid, key=lambda x: x[1])
    return support * 0.995, f"{name} 이탈 (-0.5% 버퍼)"


# =============================================================================
# 프리셋 모드 (가중치/필터 묶음)
# =============================================================================
PRESETS: Dict[str, Dict] = {
    "default": {
        "name": "기본 (균형 v4)",
        "desc": "12개 시그널 균형 — 하바로셀 + 하승훈 + 마스터 가이드",
        "weights": {"s1": 18, "s2": 18, "s3": 8, "s4": 8, "s5": 10, "s6": 4, "s7": 5, "s8": 4, "s9": 7, "s10": 7, "s11": 5, "s12": 6},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 2000, "change_min": 7, "change_max": 29},
        "min_score": 40,
    },
    "conservative": {
        "name": "보수형 (대형주 위주)",
        "desc": "안정적 종목만. 적중률 우선.",
        "weights": {"s1": 22, "s2": 10, "s3": 8, "s4": 18, "s5": 10, "s6": 5, "s7": 3, "s8": 2, "s9": 7, "s10": 8, "s11": 3, "s12": 4},
        "filter": {"min_amount_eok": 200, "min_marcap_eok": 10000, "change_min": 5, "change_max": 15},
        "min_score": 55,
    },
    "aggressive": {
        "name": "공격형 (급등 추격)",
        "desc": "큰 등락폭 + 거래량 폭증 종목.",
        "weights": {"s1": 8, "s2": 28, "s3": 22, "s4": 5, "s5": 10, "s6": 0, "s7": 5, "s8": 5, "s9": 5, "s10": 7, "s11": 2, "s12": 3},
        "filter": {"min_amount_eok": 100, "min_marcap_eok": 1000, "change_min": 12, "change_max": 29},
        "min_score": 50,
    },
    "box_breakout": {
        "name": "박스권 돌파형 (하바로셀)",
        "desc": "S1 박스권 돌파 가중. '힘의 응축' 정석.",
        "weights": {"s1": 38, "s2": 18, "s3": 8, "s4": 5, "s5": 5, "s6": 3, "s7": 5, "s8": 2, "s9": 7, "s10": 4, "s11": 2, "s12": 3},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 2000, "change_min": 7, "change_max": 25},
        "min_score": 45,
    },
    "habarocell": {
        "name": "하바로셀식 (수급 선점)",
        "desc": "수급 선점 + 힘의 응축 + 눌림목 + 과열 회피.",
        "weights": {"s1": 20, "s2": 20, "s3": 8, "s4": 8, "s5": 8, "s6": 10, "s7": 8, "s8": 4, "s9": 5, "s10": 4, "s11": 2, "s12": 3},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 1500, "change_min": 7, "change_max": 25},
        "min_score": 50,
    },
    "haseunghoon": {
        "name": "하승훈식 (대시세 + 대장주)",
        "desc": "장기이평 돌파 + 상대강도 + 신고가 + 거래대금.",
        "weights": {"s1": 8, "s2": 18, "s3": 10, "s4": 5, "s5": 13, "s6": 3, "s7": 4, "s8": 4, "s9": 13, "s10": 13, "s11": 4, "s12": 5},
        "filter": {"min_amount_eok": 100, "min_marcap_eok": 2000, "change_min": 5, "change_max": 25},
        "min_score": 55,
    },
    "pullback": {
        "name": "눌림목 매수형 (첫눌림)",
        "desc": "첫눌림(S12) + 단기이평 지지 + 진짜지지(거래량급감).",
        "weights": {"s1": 10, "s2": 10, "s3": 6, "s4": 18, "s5": 5, "s6": 5, "s7": 18, "s8": 5, "s9": 6, "s10": 4, "s11": 6, "s12": 12},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 1500, "change_min": 3, "change_max": 15},
        "min_score": 45,
    },
    "mega_trend": {
        "name": "대시세 초입형 (하승훈)",
        "desc": "장기이평(240/480) 우하향→횡보→돌파.",
        "weights": {"s1": 6, "s2": 16, "s3": 8, "s4": 5, "s5": 10, "s6": 3, "s7": 5, "s8": 5, "s9": 28, "s10": 8, "s11": 3, "s12": 4},
        "filter": {"min_amount_eok": 100, "min_marcap_eok": 2000, "change_min": 5, "change_max": 25},
        "min_score": 50,
    },
    "master_guide": {
        "name": "마스터 가이드 (캔거지파)",
        "desc": "갭+이평 중첩(S11) + 컵앤핸들/역H&S(S12) 정밀 패턴 매매.",
        "weights": {"s1": 12, "s2": 15, "s3": 8, "s4": 8, "s5": 12, "s6": 5, "s7": 8, "s8": 4, "s9": 6, "s10": 6, "s11": 8, "s12": 8},
        "filter": {"min_amount_eok": 100, "min_marcap_eok": 2000, "change_min": 5, "change_max": 25},
        "min_score": 55,
    },
    "ai_optimized_1": {
        "name": "AI 최적화 1위 (샤프 3.93)",
        "desc": "2,460 조합 탐색 결과 1위. 과열회피 + 박스권 돌파 + 수급연속 + 갭+이평.",
        "weights": {"s1": 26, "s2": 0, "s3": 0, "s4": 0, "s5": 1, "s6": 51, "s7": 0, "s8": 10, "s9": 6, "s10": 0, "s11": 6, "s12": 0},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 2000, "change_min": 5, "change_max": 29},
        "min_score": 45,
    },
    "ai_optimized_2": {
        "name": "AI 최적화 2위 (샤프 4.00)",
        "desc": "샤프 비율 최고. 과열회피 + 박스권 돌파 + 패턴품질.",
        "weights": {"s1": 29, "s2": 0, "s3": 1, "s4": 0, "s5": 0, "s6": 49, "s7": 0, "s8": 8, "s9": 0, "s10": 0, "s11": 0, "s12": 13},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 2000, "change_min": 5, "change_max": 29},
        "min_score": 45,
    },
    "ai_optimized_3": {
        "name": "AI 최적화 3위 (수익률 최고)",
        "desc": "평균 수익률 최고 (+0.50%). 갭+이평 중첩 + 첫눌림.",
        "weights": {"s1": 11, "s2": 0, "s3": 2, "s4": 5, "s5": 1, "s6": 0, "s7": 20, "s8": 5, "s9": 3, "s10": 0, "s11": 51, "s12": 2},
        "filter": {"min_amount_eok": 50, "min_marcap_eok": 2000, "change_min": 5, "change_max": 29},
        "min_score": 45,
    },
}


def get_preset(name: str) -> Optional[Dict]:
    return PRESETS.get(name)


def list_presets() -> List[str]:
    return list(PRESETS.keys())
