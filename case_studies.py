"""
종가매매 실전 사례 데이터베이스

자료 출처:
  - 하바로셀 강의/방송
  - 하승훈 채널 (주식투자TV)
  - 사용자 본인 매매 기록

3가지 핵심 패턴 분류:
  A. 매물 소화 후 돌파 '천양봉(첫 장대양봉)' 패턴 — 돌파매매
  B. 대량거래 급등 후 첫 '눌림목 지지' 패턴 — 눌림목매매
  C. 분봉상 이중바닥 + 이평선 정배열 수렴 패턴 — 눌림목매매 (분봉)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Pattern(Enum):
    """매매 패턴 분류"""
    A_BREAKOUT = "A: 매물소화 후 돌파 (천양봉)"
    B_PULLBACK = "B: 급등 후 첫 눌림목 (단기이평 지지)"
    C_DOUBLE_BOTTOM = "C: 분봉 이중바닥 + 정배열 수렴"
    D_LONGTERM = "D: 장기이평 돌파 (대시세 초입)"
    E_RISK = "E: 리스크 관리 사례 (손절)"


# =============================================================================
# 실전 사례 데이터베이스 (자료 직접 추출)
# =============================================================================
CASE_STUDIES: List[Dict] = [
    # =============== 패턴 A: 매물 소화 후 돌파 (천양봉) ===============
    {
        "stock": "SK하이닉스", "code": "000660", "date": "2024-02-22",
        "pattern": Pattern.A_BREAKOUT, "theme": "AI 반도체",
        "trigger": "24년 최고가 15만500원 부근 매물 소화 후 돌파 첫 양봉",
        "key_signals": ["대량거래", "사상최고가 돌파", "AI 주도 테마"],
        "outcome": "다음 날 큰 수익 실현",
        "lesson": "장기 횡보 매물대를 강력하게 뚫는 첫 장대양봉 + 종가 안착 시 진입",
        "source": "하승훈/하바로셀",
    },
    {
        "stock": "삼성전자", "code": "005930", "date": "2024-05-04",
        "pattern": Pattern.A_BREAKOUT, "theme": "반도체",
        "trigger": "30분봉상 박스권 돌파 마감 + 애프터마켓 상승",
        "key_signals": ["30분봉 박스 돌파", "AM 매수세", "거래대금 충분"],
        "outcome": "익일 갭상승",
        "lesson": "일봉뿐만 아니라 분봉 박스권 돌파 + 시간외 강세 확인",
        "source": "하승훈",
    },
    {
        "stock": "현대차", "code": "005380", "date": "2024-05-13",
        "pattern": Pattern.A_BREAKOUT, "theme": "자동차 (현대차그룹 주도)",
        "trigger": "5일선 아래꼬리 이틀 + 사상최고가 돌파 장대양봉, AM 재돌파",
        "key_signals": ["5일선 지지", "사상최고가", "AM 재돌파"],
        "outcome": "익일 상승",
        "lesson": "주도 테마 대장주 + 5일선 지지 + 신고가 돌파 트리플 조건",
        "source": "하승훈",
    },
    {
        "stock": "현대오토에버", "code": "307950", "date": "2024-05-13",
        "pattern": Pattern.A_BREAKOUT, "theme": "자동차 (현대차그룹)",
        "trigger": "5일선 아래꼬리 + 전고점 돌파 신고가 흐름",
        "key_signals": ["5일선 지지", "전고점 돌파", "테마 동조"],
        "outcome": "현대차와 동반 상승",
        "lesson": "대장주(현대차)와 동조 매매. 같은 그룹/테마 후행주 함께 매수",
        "source": "하승훈",
    },
    {
        "stock": "LG CNS", "code": "064400", "date": "2024-05-14",
        "pattern": Pattern.A_BREAKOUT, "theme": "AI/IT 서비스 (LG 대장주)",
        "trigger": "바닥권 박스 돌파 대량거래 장대양봉 + AM 당일 최고가 재돌파",
        "key_signals": ["바닥 박스 돌파", "대량거래", "AM 최고가 재돌파"],
        "outcome": "익일 +15% 상승",
        "lesson": "AM에서 당일 최고가 다시 돌파 = 강력한 익일 시그널",
        "source": "하승훈",
    },
    {
        "stock": "포스코홀딩스", "code": "005490", "date": "2023-04-28",
        "pattern": Pattern.A_BREAKOUT, "theme": "2차전지 (저항 돌파)",
        "trigger": "대량거래로 신고가 돌파 후, 30일 5일선 위 아래꼬리 도지",
        "key_signals": ["대량거래", "52주 신고가", "5일선 도지 지지"],
        "outcome": "추가 상승",
        "lesson": "신고가 돌파 → 단기 조정 → 5일선 도지 지지 = 안정적 진입",
        "source": "하승훈",
    },
    {
        "stock": "포스코퓨처엠", "code": "003670", "date": "2023-07-21",
        "pattern": Pattern.A_BREAKOUT, "theme": "2차전지 (사상최고가)",
        "trigger": "사상최고가 저항 매물 출회 후 장 막판 매수세 + 고가 마감",
        "key_signals": ["사상최고가", "매물 소화", "장 막판 매수세"],
        "outcome": "익일 상승",
        "lesson": "사상최고가 저항대에서 종일 매물 받아내고 종가 고가 마감 = 강세 신호",
        "source": "하승훈",
    },
    {
        "stock": "셀바스AI", "code": "108860", "date": "2023-07-31",
        "pattern": Pattern.A_BREAKOUT, "theme": "AI",
        "trigger": "2,300억 대량거래 장대양봉 박스권 돌파 + 막판 매수세",
        "key_signals": ["거래대금 2300억", "박스권 돌파", "AI 주도 대장"],
        "outcome": "추가 슈팅",
        "lesson": "AI 테마 대장주 + 대량거래 천양봉 = 종배 최우선 후보",
        "source": "하바로셀/하승훈",
    },
    {
        "stock": "폴라리스오피스", "code": "041020", "date": "2023-08-04",
        "pattern": Pattern.A_BREAKOUT, "theme": "AI",
        "trigger": "저항 매물 소화 후 돌파 첫 양봉 + 장 막판 매수세 마감",
        "key_signals": ["매물 소화", "AI 테마", "장 막판 매수세"],
        "outcome": "익일 상승",
        "lesson": "테마 후순위라도 매물 소화 패턴 명확하면 진입 가능",
        "source": "하승훈",
    },
    {
        "stock": "삼성물산", "code": "028260", "date": "2024-02-16",
        "pattern": Pattern.A_BREAKOUT, "theme": "저PBR",
        "trigger": "저PBR 테마 매물 소화 후 돌파 첫 장대양봉",
        "key_signals": ["저PBR 주도", "매물 소화 돌파", "테마 대장"],
        "outcome": "포착 (진입 후보)",
        "lesson": "정책 모멘텀 테마(저PBR) + 매물 소화 셋업이 가장 강력",
        "source": "하승훈",
    },
    {
        "stock": "에코프로비엠", "code": "247540", "date": "2023-07-21",
        "pattern": Pattern.A_BREAKOUT, "theme": "2차전지",
        "trigger": "사상최고가 돌파 강한 상승 후 조정 → 장 막판 재돌파",
        "key_signals": ["사상최고가", "주도 테마", "장 막판 재돌파"],
        "outcome": "이후 추가 상승",
        "lesson": "사상최고가 돌파 후 단기 조정도 종배 기회. 막판 재돌파 확인",
        "source": "하승훈",
    },
    {
        "stock": "STX엔진", "code": "077970", "date": "2024-04-21",
        "pattern": Pattern.A_BREAKOUT, "theme": "방산",
        "trigger": "매물 소화 후 신고가 돌파 첫날 종가 매수 + 다음날 추가 매수",
        "key_signals": ["매물 소화", "신고가", "3일선 눌림 추가"],
        "outcome": "분할 매수 성공",
        "lesson": "신고가 첫날 진입 + 다음날 단기이평 지지에서 불타기",
        "source": "하승훈",
    },
    {
        "stock": "위메이드", "code": "112040", "date": "2024-02-16",
        "pattern": Pattern.A_BREAKOUT, "theme": "게임/메타버스 (바닥권)",
        "trigger": "바닥권 상한가 + 3,770억 거래대금 + 저항 지지 전환",
        "key_signals": ["3770억 대금", "바닥권 첫 상한가", "저항→지지"],
        "outcome": "단타 성공",
        "lesson": "바닥권 첫 상한가 + 거래대금 폭발 = A패턴 정석",
        "source": "하승훈",
    },
    {
        "stock": "에코프로", "code": "086520", "date": "2024-02-07",
        "pattern": Pattern.A_BREAKOUT, "theme": "2차전지",
        "trigger": "컵앤핸들 완성 + 사상최고가 돌파 + 6,300억 거래대금",
        "key_signals": ["컵앤핸들", "사상최고가", "6300억 대금", "분봉 수렴 후 정배열"],
        "outcome": "큰 수익",
        "lesson": "컵앤핸들 패턴 완성 + 신고가 = 최고 손익비",
        "source": "하승훈",
    },

    # =============== 패턴 B: 급등 후 눌림목 (단기이평 지지) ===============
    {
        "stock": "삼성전자", "code": "005930", "date": "2024-05-07",
        "pattern": Pattern.B_PULLBACK, "theme": "반도체",
        "trigger": "신고가 돌파 후 3일선/5일선 영역 쌍도지 + 분봉 이중바닥",
        "key_signals": ["3일선/5일선 지지", "쌍도지", "분봉 이중바닥"],
        "outcome": "재진입 성공",
        "lesson": "신고가 후 급락에도 단기이평 지지 + 쌍도지 확인 시 재진입",
        "source": "하승훈",
    },
    {
        "stock": "기아", "code": "000270", "date": "2024-02-20",
        "pattern": Pattern.B_PULLBACK, "theme": "자동차",
        "trigger": "1월말 급등 → 5일선 지지 + 장 막판 매수세 유입",
        "key_signals": ["5일선 지지", "월봉 역헤드앤숄더", "신고가 후 눌림"],
        "outcome": "익일 상승",
        "lesson": "월봉 역H&S 패턴 + 일봉 5일선 지지 결합 시 강한 매수",
        "source": "하승훈",
    },
    {
        "stock": "넥스턴바이오", "code": "078140", "date": "2024-10-07",
        "pattern": Pattern.B_PULLBACK, "theme": "바이오",
        "trigger": "상한가 급등 후 5일선 분봉 이중바닥 + 시가 회복 매수세",
        "key_signals": ["상한가 후 조정", "5일선", "분봉 W패턴", "시가 회복"],
        "outcome": "익일 상승",
        "lesson": "상한가 다음날 5일선 지지 + 분봉 쌍바닥 = B패턴 정석",
        "source": "하바로셀/하승훈",
    },
    {
        "stock": "성우하이텍", "code": "015750", "date": "2024-10-07",
        "pattern": Pattern.B_PULLBACK, "theme": "자동차 부품",
        "trigger": "1천억 대금 상한가 후, 3일선 + 저지바(저항→지지) 정배열 전환",
        "key_signals": ["1000억 대금", "3일선", "저지바", "이평 정배열"],
        "outcome": "익일 상승",
        "lesson": "상한가 후 3일선 자리에 옛 저항이 지지로 변한 자리 = 강력 진입",
        "source": "하승훈",
    },
    {
        "stock": "지아이텍", "code": "352910", "date": "?",
        "pattern": Pattern.B_PULLBACK, "theme": "2차전지",
        "trigger": "1,136억 장대양봉 후 거래 급감 조정, 5일선 아래꼬리 + 분봉 W",
        "key_signals": ["1136억 대금", "5일선 아래꼬리", "분봉 이중바닥"],
        "outcome": "포착 (지지 확인)",
        "lesson": "거래량 급감 + 이쁜 음봉 + 5일선 닿음 = 매도 압력 약화 사인",
        "source": "하승훈",
    },
    {
        "stock": "원익피앤이", "code": "131390", "date": "?",
        "pattern": Pattern.B_PULLBACK, "theme": "반도체 소부장",
        "trigger": "장대양봉 상단 횡보 후 3일선 도지 + 장 막판 들어올림",
        "key_signals": ["3일선", "도지", "장 막판 매수세"],
        "outcome": "익일 상승",
        "lesson": "장대양봉 후 횡보 자체가 조정. 3일선 닿을 때 도지 = 진입",
        "source": "하승훈",
    },
    {
        "stock": "미래생명자원", "code": "218150", "date": "2024-04-11",
        "pattern": Pattern.B_PULLBACK, "theme": "농생명",
        "trigger": "장기이평(240/480) 돌파 후 10일선+480일선 겹치는 자리 쌍도지",
        "key_signals": ["10일선+480일선 중첩", "이틀 도지", "장기 돌파 후 첫 눌림"],
        "outcome": "장기 상승 초입",
        "lesson": "단기+장기 이평이 겹친 자리 = 가장 강력한 지지",
        "source": "하승훈",
    },
    {
        "stock": "SM C&C", "code": "048550", "date": "2021-12-27",
        "pattern": Pattern.B_PULLBACK, "theme": "엔터",
        "trigger": "900억 상한가 후 3일선 + 저지바 자리 아래꼬리",
        "key_signals": ["900억 상한가", "3일선", "여러번 겹친 저지바"],
        "outcome": "익일 상승",
        "lesson": "과거 지지/저항이 여러 번 겹친 자리 = 강한 지지 신뢰",
        "source": "하바로셀",
    },
    {
        "stock": "NPC", "code": "004250", "date": "2021-12-24",
        "pattern": Pattern.B_PULLBACK, "theme": "산업재 (사상최고가)",
        "trigger": "사상최고가 돌파 후 5일선 지지 + 분봉 이평 수렴→정배열",
        "key_signals": ["사상최고가 후 5일선", "분봉 수렴→정배열"],
        "outcome": "연말 상승",
        "lesson": "분봉상 이평 수렴 후 정배열 = 진짜 반등 신호",
        "source": "하바로셀",
    },
    {
        "stock": "대한전선", "code": "001440", "date": "?",
        "pattern": Pattern.B_PULLBACK, "theme": "전선/케이블",
        "trigger": "대량거래 장대양봉 후 120일선+10일선 겹친 자리 쌍도지",
        "key_signals": ["120일선+10일선 중첩", "이틀 도지"],
        "outcome": "포착 (지지 성공)",
        "lesson": "단기-장기 이평 동시 지지 = 패턴 B 최강",
        "source": "하승훈",
    },
    {
        "stock": "AP위성", "code": "211270", "date": "?-02-02",
        "pattern": Pattern.B_PULLBACK, "theme": "우주/위성",
        "trigger": "상한가 후 3일선 첫 눌림 + 분봉 수렴→정배열",
        "key_signals": ["상한가 첫 눌림", "3일선", "분봉 정배열"],
        "outcome": "익일 상승",
        "lesson": "상한가 → 첫 눌림 → 분봉 정배열 = 종배 핵심 셋업",
        "source": "하승훈",
    },
    {
        "stock": "미래에셋벤처투자", "code": "100790", "date": "2024-04-21",
        "pattern": Pattern.B_PULLBACK, "theme": "VC/금융",
        "trigger": "박스 돌파 상한가 후 조정 → 5일선 지지 양봉",
        "key_signals": ["박스 돌파 상한가 후 눌림", "5일선 양봉 지지"],
        "outcome": "상한가 도달",
        "lesson": "박스 돌파 후 첫 5일선 지지 양봉 = 진입 시그널",
        "source": "하바로셀",
    },
    {
        "stock": "한국전력", "code": "015760", "date": "?-02-22",
        "pattern": Pattern.B_PULLBACK, "theme": "유틸리티",
        "trigger": "대량거래 저항 돌파 후 5일선+옛저항 도지",
        "key_signals": ["저항 돌파 대량거래", "5일선+옛저항 중첩 도지"],
        "outcome": "포착",
        "lesson": "공기업도 거래량 폭발 + 이평 지지 시 진입 후보",
        "source": "하승훈",
    },
    {
        "stock": "가온칩스", "code": "399720", "date": "?",
        "pattern": Pattern.B_PULLBACK, "theme": "반도체 디자인",
        "trigger": "사상최고가 돌파 후 10일선 도지 지지",
        "key_signals": ["사상최고가", "10일선 도지"],
        "outcome": "포착",
        "lesson": "10일선 지지도 패턴 B에 포함. 종목별 강한 지지선 찾기",
        "source": "하승훈",
    },
    {
        "stock": "엔켐", "code": "348370", "date": "?",
        "pattern": Pattern.B_PULLBACK, "theme": "2차전지 전해질",
        "trigger": "사상최고가 돌파 후 5일선 지지",
        "key_signals": ["사상최고가", "5일선"],
        "outcome": "포착",
        "lesson": "고가 종목도 5일선이 가장 안정적 지지선",
        "source": "하승훈",
    },
    {
        "stock": "현대로템", "code": "064350", "date": "2022-07-21",
        "pattern": Pattern.B_PULLBACK, "theme": "방산",
        "trigger": "하락 추세선 돌파 + 5일선 지지 양 단봉",
        "key_signals": ["추세선 돌파", "5일선 지지 양봉"],
        "outcome": "이후 큰 상승",
        "lesson": "장기 하락 추세선 돌파 + 즉시 단기이평 지지 = 추세 전환",
        "source": "하승훈",
    },

    # =============== 패턴 D: 장기이평 돌파 (대시세 초입) ===============
    {
        "stock": "씨젠", "code": "096530", "date": "대시세 초입",
        "pattern": Pattern.D_LONGTERM, "theme": "코로나 진단키트",
        "trigger": "120/240/480일선 우하향→횡보→대량거래 돌파",
        "key_signals": ["장기이평 3개 모두 돌파", "대량거래", "장기 횡보 후"],
        "outcome": "대시세 분출",
        "lesson": "장기 이평 3개를 한꺼번에 돌파 = 수십% ~ 수백% 상승 가능",
        "source": "하승훈",
    },
    {
        "stock": "신풍제약", "code": "019170", "date": "대시세 초입",
        "pattern": Pattern.D_LONGTERM, "theme": "코로나 치료제",
        "trigger": "장기 이평 우하향→횡보→대량거래 돌파",
        "key_signals": ["장기이평 돌파", "테마 모멘텀"],
        "outcome": "대시세",
        "lesson": "씨젠과 동일 패턴. 같은 테마 종목 모두 함께 움직임",
        "source": "하승훈",
    },
    {
        "stock": "금양", "code": "001570", "date": "2023-07-25",
        "pattern": Pattern.D_LONGTERM, "theme": "2차전지 후행",
        "trigger": "7/25 상한가 이후 대량거래 수반 대시세 분출",
        "key_signals": ["상한가 후 대시세", "대량거래 연속"],
        "outcome": "수배 상승",
        "lesson": "상한가 → 대시세 분출 케이스. 후속 거래량 유지 확인",
        "source": "하승훈",
    },

    # =============== 패턴 E: 리스크 관리 (손절 사례) ===============
    {
        "stock": "박셀바이오", "code": "323990", "date": "2022-07-18",
        "pattern": Pattern.E_RISK, "theme": "바이오 (간암 치료제)",
        "trigger": "박스권 돌파 상한가 + 눌림 분봉 이중바닥 매수",
        "key_signals": ["분봉 W 매수", "지지선 이탈 손절"],
        "outcome": "지지 실패 → 손절",
        "lesson": "지지 예상 자리에서 반등 못하고 파동 저점 이탈 시 기계적 손절",
        "source": "하승훈",
    },
    {
        "stock": "현대오토에버", "code": "307950", "date": "?",
        "pattern": Pattern.E_RISK, "theme": "(좋은 종목도 수급 꺾이면)",
        "trigger": "수급 꺾임 시 손절 우선",
        "key_signals": ["수급 약화", "기대감 차단"],
        "outcome": "타이밍 실패",
        "lesson": "좋은 종목도 시점이 안 맞으면 손절. 기대감 ❌, 실시간 대응 ✓",
        "source": "하바로셀",
    },
]


# =============================================================================
# 유틸: 패턴별 사례 추출
# =============================================================================
def get_cases_by_pattern(pattern: Pattern) -> List[Dict]:
    return [c for c in CASE_STUDIES if c["pattern"] == pattern]


def get_cases_summary() -> Dict[str, int]:
    return {p.value: len(get_cases_by_pattern(p)) for p in Pattern}


def get_case_by_stock(stock_name: str) -> List[Dict]:
    return [c for c in CASE_STUDIES if c["stock"] == stock_name]


# =============================================================================
# 패턴 자동 분류 함수 (시그널 점수 기반)
# =============================================================================
def classify_pattern(signals: Dict[str, float]) -> Dict:
    """
    시그널 점수를 보고 종목의 매매 패턴 성격을 자동 분류.

    Returns:
        {
            "primary_pattern": Pattern,
            "confidence": float (0~1),
            "trade_type": "돌파매매" / "눌림목매매" / "대시세초입",
            "reasoning": str,
        }
    """
    s1 = signals.get("s1", 0)   # 박스권 돌파
    s2 = signals.get("s2", 0)   # 거래량 폭증
    s3 = signals.get("s3", 0)   # 장대양봉
    s4 = signals.get("s4", 0)   # 이평 정배열
    s5 = signals.get("s5", 0)   # 전고점 돌파
    s6 = signals.get("s6", 0)   # 과열 회피
    s7 = signals.get("s7", 0)   # 눌림목 셋업
    s9 = signals.get("s9", 0)   # 장기이평 돌파
    s10 = signals.get("s10", 0) # 상대강도

    # 패턴 D: 장기이평 돌파 (최우선)
    if s9 >= 50:
        return {
            "primary_pattern": Pattern.D_LONGTERM,
            "confidence": min(1.0, s9 / 100),
            "trade_type": "대시세 초입",
            "reasoning": f"S9 장기이평 돌파 {s9:.0f}점. 120/240/480일선 우하향→횡보→돌파 패턴.",
        }

    # 패턴 A: 매물 소화 후 돌파 (천양봉)
    # → 박스권 돌파(S1) + 거래량 폭증(S2) + 장대양봉(S3) + 전고점(S5)
    breakout_score = (s1 * 0.3 + s2 * 0.25 + s3 * 0.2 + s5 * 0.25)

    # 패턴 B/C: 눌림목
    # → 눌림목 셋업(S7) + 정배열(S4) + 과열 회피(S6)
    pullback_score = (s7 * 0.5 + s4 * 0.25 + s6 * 0.25)

    if breakout_score >= pullback_score and breakout_score >= 40:
        primary = Pattern.A_BREAKOUT
        trade_type = "돌파매매"
        reasoning = (
            f"돌파 점수 {breakout_score:.0f} > 눌림목 점수 {pullback_score:.0f}. "
            f"S1박스={s1:.0f} S2거래량={s2:.0f} S3양봉={s3:.0f} S5전고점={s5:.0f}."
        )
        confidence = min(1.0, breakout_score / 100)
    elif pullback_score > breakout_score and pullback_score >= 40:
        # B vs C 구분: 정배열 강하면 C, 눌림목 강하면 B
        if s7 >= 60 and s4 >= 75:
            primary = Pattern.C_DOUBLE_BOTTOM
        else:
            primary = Pattern.B_PULLBACK
        trade_type = "눌림목매매"
        reasoning = (
            f"눌림목 점수 {pullback_score:.0f} > 돌파 점수 {breakout_score:.0f}. "
            f"S4정배열={s4:.0f} S6미과열={s6:.0f} S7눌림목={s7:.0f}."
        )
        confidence = min(1.0, pullback_score / 100)
    else:
        primary = Pattern.A_BREAKOUT  # 기본
        trade_type = "셋업 약함"
        reasoning = f"명확한 패턴 미형성 (돌파 {breakout_score:.0f}, 눌림목 {pullback_score:.0f})."
        confidence = 0.3

    return {
        "primary_pattern": primary,
        "confidence": round(confidence, 2),
        "trade_type": trade_type,
        "reasoning": reasoning,
        "breakout_score": round(breakout_score, 1),
        "pullback_score": round(pullback_score, 1),
    }
