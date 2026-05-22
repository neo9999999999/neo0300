"""
종가매수 추천 시스템 — 기본 설정
사용자가 Streamlit UI에서 변경 가능. 이 파일은 디폴트값.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class FilterConfig:
    min_amount: float = 5_000_000_000      # 거래대금 최소 (50억)
    min_marcap: float = 200_000_000_000    # 시가총액 최소 (2000억)
    change_min: float = 7.0                # 당일 등락률 최소 (%)
    change_max: float = 29.0               # 당일 등락률 최대 (상한가 30% 제외)
    include_kospi: bool = True
    include_kosdaq: bool = True
    exclude_etf: bool = True               # ETF/ETN/SPAC 제외


@dataclass
class ScoreWeights:
    """
    8대 시그널 가중치 (총합 자동 정규화)
    하바로셀 5대 원칙 + 차트 27장 분석 + 추가 회피/눌림목 시그널
    """
    # 메인 5대 시그널
    s1_box_breakout: float = 25      # 박스권 돌파 = 힘의 응축
    s2_volume_surge: float = 25      # 거래량 폭증 = 수급 선점
    s3_long_candle: float = 12       # 장대양봉
    s4_ma_alignment: float = 12      # 이평선 정배열
    s5_near_high: float = 12         # 전고점 근접/돌파
    # 추가 시그널
    s6_no_overheating: float = 5     # 과열 회피 (5일 누적 상승 페널티 반전)
    s7_pullback_setup: float = 5     # 눌림목 셋업 (1차 슈팅 후 조정 → 재진입)
    s8_demand_continuity: float = 4  # 수급 연속성 (최근 거래량 추세 ↑)
    # 하승훈 추가 시그널
    s9_longterm_ma_breakout: float = 5  # 장기 이평(120/240/480) 돌파 = 대시세 초입
    s10_relative_strength: float = 5    # 상대강도 (시장 대비 강세) = 주도주 판별
    # 마스터 가이드 추가 시그널
    s11_gap_ma_confluence: float = 5    # 과거 갭 + 이평선 중첩 = 강력 지지 자리
    s12_pattern_quality: float = 5      # 컵앤핸들/역H&S/첫눌림 검증 종합 패턴 품질
    # 보너스
    bonus_watchlist: float = 5       # 하바로셀/사용자 워치리스트 + 테마 동조 보너스


@dataclass
class SignalParams:
    """시그널 계산 임계값 — 사용자가 미세조정 가능"""
    box_period: int = 60             # 박스권 기준 일수
    box_max_range_pct: float = 25    # 박스권으로 인정할 최대 변동폭 (%)
    volume_ma_period: int = 20       # 거래량 평균 기간
    volume_surge_multiplier: float = 3.0   # 거래량 폭증 배수
    long_candle_min_pct: float = 5.0  # 장대양봉 최소 등락률 (%)
    ma_short: int = 5
    ma_mid: int = 20
    ma_long: int = 60
    near_high_threshold: float = 0.95  # 전고점 95% 이상이면 근접
    ohlcv_lookback_days: int = 90    # 시그널 계산용 과거 일수
    # S6 과열 회피
    overheat_period: int = 5             # 직전 N일 누적 상승률 측정
    overheat_threshold: float = 25.0     # 누적 +X% 이상이면 과열
    # S7 눌림목 셋업
    pullback_lookback: int = 20          # 최근 N일 안에 1차 슈팅 찾기
    pullback_first_shoot_pct: float = 10 # 1차 슈팅 최소 상승률
    pullback_dip_to_ma: int = 20         # 조정 시 MA20까지 근접 허용
    # S8 수급 연속성
    demand_recent_period: int = 5
    demand_baseline_period: int = 30
    # S9 장기 이평선 돌파 (하승훈: 대시세 초입)
    longterm_ma_periods: tuple = (120, 240, 480)
    longterm_ma_flat_pct: float = 5.0    # 횡보 인정 변동폭 (%)
    longterm_ma_flat_window: int = 20    # 횡보 확인 기간 (일)
    # S10 상대강도 (시장 대비 강세)
    rs_period: int = 20                   # 상대강도 측정 기간 (일)
    rs_benchmark: str = "KS11"            # 코스피 지수 (KS11) / 코스닥 (KQ11)


@dataclass
class RecommendConfig:
    top_n: int = 3                   # 추천 종목 수
    sort_by: str = "total_score"     # 정렬 기준
    min_score: float = 40            # 최소 점수 (이 이하는 추천 안함)


@dataclass
class BacktestConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2026-05-20"
    sell_strategy: str = "next_open"  # next_open / next_high / next_close / t2_close
    initial_capital: float = 10_000_000   # 1천만원
    position_size_pct: float = 33    # 종목당 비중 (3종목 ≈ 100%)


# =============================================================================
# 시스템 고정값 (사용자 수정 불가)
# =============================================================================
SYSTEM_FIXED = {
    "universe_limit": 1000,            # 시총 상위 1000종목 고정
    "backtest_start": "2020-01-01",    # 백테스트 시작 고정
    "top_results": 3,                  # TOP 3만 표시
}


# 기본 통합 설정
DEFAULT_CONFIG = {
    "filter": FilterConfig(),
    "weights": ScoreWeights(),
    "params": SignalParams(),
    "recommend": RecommendConfig(),
    "backtest": BacktestConfig(),
}


def config_to_dict(config: Dict) -> Dict:
    """설정을 일반 dict로 변환 (Streamlit session_state 저장용)"""
    return {k: asdict(v) for k, v in config.items()}
