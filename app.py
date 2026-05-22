"""
종가매수 추천 시스템 — 사용자 친화 UI v9
- 라이트 모드 디폴트 (핑크/레드)
- 사이드바: 메뉴 3개만
- 인라인 설정 (페이지 상단)
- 전체 한글화
- 추천 사유 + 유사 사례 매칭
- 년/월 버튼 토글
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from config import (
    FilterConfig, ScoreWeights, SignalParams, RecommendConfig,
    BacktestConfig, SYSTEM_FIXED,
)
from scanner import scan_recommendations, filter_by_trade_type, scan_ensemble
from grade import (
    GRADE_INFO, GRADE_WEIGHTS, PRESETS_4,
    classify_one, classify_candidates, build_grade_buckets,
    grade_reason, grade_badge_html, build_ensemble_all_enriched,
)
from backtest import run_backtest, summarize_backtest
from historical import (
    scan_historical_period, get_available_years, get_months,
    summarize_by_pattern, summarize_by_year_month,
)
from walk_forward import walk_forward_validation, get_cached_walk_forward
from rules import (
    PRESETS, list_presets, AVOID_RULES, make_exit_plan, ExitScenario,
    haseunghoon_5_conditions, haseunghoon_stoploss,
)
from watchlist import HABAROCELL_PICKS, HASEUNGHOON_PICKS, USER_PICKS, THEMES, watchlist_summary
from case_studies import CASE_STUDIES, Pattern, get_cases_by_pattern, get_cases_summary
from case_matcher import (
    find_similar_cases, generate_recommendation_reasons,
    koreanize_dataframe, COLUMN_KOREAN, SIGNAL_NAMES,
)
from theme import get_css, get_logo_html, PALETTE


st.set_page_config(
    page_title="종가매수 추천",
    page_icon="🟢",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ===== 비밀번호 인증 =====
def _check_password():
    """간단한 비밀번호 게이트. secrets.toml의 APP_PASSWORD 또는 기본값 사용."""
    try:
        correct = st.secrets.get("APP_PASSWORD", "123456")
    except Exception:
        correct = "123456"

    if st.session_state.get("auth_ok"):
        return True

    # 로그인 화면
    st.markdown(
        '<div style="max-width:420px;margin:80px auto;text-align:center;">'
        '<div style="font-size:48px;margin-bottom:12px;">🟢</div>'
        '<h2 style="margin:0 0 8px 0;">종가매수 추천 시스템</h2>'
        '<p style="color:#888;font-size:14px;margin-bottom:32px;">V/S/A/B 등급제 · 코스닥 돌파매매</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("비밀번호", type="password", key="_pwd_input",
                              placeholder="비밀번호 입력")
        login = st.button("로그인", type="primary", use_container_width=True, key="_login_btn")

        if login or pwd:
            if pwd == str(correct):
                st.session_state.auth_ok = True
                st.rerun()
            elif pwd:
                st.error("비밀번호가 일치하지 않습니다.")
    st.stop()


_check_password()


# ===== 세션 디폴트 =====
DEFAULTS = {
    "page": "today",
    "theme": "light",
    "preset": "default",
    "trade_type": "전체",
    "min_amount_eok": 50,
    "min_marcap_eok": 2000,
    "change_min": 7.0,
    "change_max": 29.0,
    "top_n": 3,
    "min_score": 40,
    "last_picks": None,
    "history_years": [2024, 2025],
    "history_months": list(range(1, 13)),
    "position_size": 1_000_000,  # 종목당 매수금 (만원 단위 입력 → 원으로 저장)
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# 통합 디자인 시스템 CSS 주입 (theme.py에서 일괄 처리)
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)


# =============================================================================
# 사이드바
# =============================================================================
with st.sidebar:
    st.markdown(get_logo_html(st.session_state.theme), unsafe_allow_html=True)

    PAGES = [
        ("오늘의 종가매수 추천", "today"),
        ("백테스트 결과", "results"),
        ("사례 & 가이드", "library"),
    ]
    for label, key in PAGES:
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state.page = key
            st.rerun()

    st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
    theme_label = "라이트 모드" if st.session_state.theme == "dark" else "다크 모드"
    if st.button(theme_label, use_container_width=True,
                  type="secondary", key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()


# =============================================================================
# 설정 객체 빌더
# =============================================================================
def build_configs():
    p = PRESETS[st.session_state.preset]
    w = p["weights"]
    filter_cfg = FilterConfig(
        min_amount=st.session_state.min_amount_eok * 100_000_000,
        min_marcap=st.session_state.min_marcap_eok * 100_000_000,
        change_min=st.session_state.change_min,
        change_max=st.session_state.change_max,
    )
    weights = ScoreWeights(
        s1_box_breakout=w.get("s1", 18), s2_volume_surge=w.get("s2", 18),
        s3_long_candle=w.get("s3", 8), s4_ma_alignment=w.get("s4", 8),
        s5_near_high=w.get("s5", 10), s6_no_overheating=w.get("s6", 4),
        s7_pullback_setup=w.get("s7", 5), s8_demand_continuity=w.get("s8", 4),
        s9_longterm_ma_breakout=w.get("s9", 7), s10_relative_strength=w.get("s10", 7),
        s11_gap_ma_confluence=w.get("s11", 5), s12_pattern_quality=w.get("s12", 6),
        bonus_watchlist=w.get("bonus", 5),
    )
    params = SignalParams()
    rec_cfg = RecommendConfig(top_n=st.session_state.top_n, min_score=st.session_state.min_score)
    return filter_cfg, weights, params, rec_cfg


# =============================================================================
# 인라인 설정 위젯 (페이지 상단)
# =============================================================================
def inline_settings(show_period: bool = False):
    """페이지 상단 설정 위젯. 펼치기 형태로 노출."""
    with st.expander("⚙️  설정 (필요한 경우만 변경)", expanded=False):
        # 전략 + 매매 타입
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📌 매매 전략**")
            preset_keys = list_presets()
            sel_idx = preset_keys.index(st.session_state.preset) if st.session_state.preset in preset_keys else 0
            st.session_state.preset = st.selectbox(
                "전략", preset_keys, index=sel_idx,
                format_func=lambda k: PRESETS[k]["name"],
                label_visibility="collapsed", key="set_preset",
            )
            st.caption(f"💡 {PRESETS[st.session_state.preset]['desc']}")
        with c2:
            st.markdown("**🎯 매매 타입 필터**")
            tt_opts = ["전체", "돌파매매", "눌림목매매", "대시세 초입"]
            st.session_state.trade_type = st.selectbox(
                "타입", tt_opts,
                index=tt_opts.index(st.session_state.trade_type),
                label_visibility="collapsed", key="set_tt",
            )

        # 종목 필터
        st.markdown("**🔍 종목 필터**")
        c3, c4, c5, c6 = st.columns(4)
        st.session_state.min_amount_eok = c3.number_input(
            "거래대금(억)", 10, 10000, st.session_state.min_amount_eok, 10, key="set_amt",
        )
        st.session_state.min_marcap_eok = c4.number_input(
            "시총(억)", 100, 100000, st.session_state.min_marcap_eok, 100, key="set_mc",
        )
        st.session_state.change_min = c5.number_input(
            "등락 최소(%)", 0.0, 30.0, st.session_state.change_min, 0.5, key="set_chmin",
        )
        st.session_state.change_max = c6.number_input(
            "등락 최대(%)", 0.0, 30.0, st.session_state.change_max, 0.5, key="set_chmax",
        )
        c7, c8, c9 = st.columns(3)
        st.session_state.top_n = c7.number_input(
            "추천 종목 수", 1, 10, st.session_state.top_n, key="set_topn",
        )
        st.session_state.min_score = c8.number_input(
            "최소 점수", 0, 100, st.session_state.min_score, key="set_ms",
        )
        ps_man = c9.number_input(
            "종목당 매수금 (만원)", 10, 100000,
            int(st.session_state.get("position_size", 1_000_000) / 10000), 10,
            key="set_psize",
        )
        st.session_state.position_size = ps_man * 10000

        # 기간 (백테스트용)
        if show_period:
            st.markdown("**📅 기간 선택**")
            year_month_picker()


def year_month_picker():
    """년도/월 다중 선택 — 버튼 토글 방식."""
    years = get_available_years()
    months = list(range(1, 13))

    # 빠른 선택 (4개 — 참고 화면)
    qa = st.columns(4)
    if qa[0].button("전체", key="qa_all", use_container_width=True):
        st.session_state.history_years = list(years)
        st.session_state.history_months = list(range(1, 13))
        st.rerun()
    if qa[1].button("1Q", key="qa_q1", use_container_width=True):
        st.session_state.history_months = [1, 2, 3]
        st.rerun()
    if qa[2].button("최근3", key="qa_r3", use_container_width=True):
        st.session_state.history_years = years[-3:]
        st.rerun()
    if qa[3].button("해제", key="qa_clr", use_container_width=True):
        st.session_state.history_years = []
        st.session_state.history_months = []
        st.rerun()

    # 년도 버튼 (7개 — 한 줄 OK)
    st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;margin-top:18px;margin-bottom:8px;">년도 (중복 선택 가능)</div>',
                 unsafe_allow_html=True)
    yr_cols = st.columns(len(years))
    for i, year in enumerate(years):
        is_selected = year in st.session_state.history_years
        btn_type = "primary" if is_selected else "secondary"
        if yr_cols[i].button(f"{year}", key=f"yr_{year}", use_container_width=True,
                              type=btn_type):
            if year in st.session_state.history_years:
                st.session_state.history_years.remove(year)
            else:
                st.session_state.history_years.append(year)
                st.session_state.history_years.sort()
            st.rerun()

    # 월 버튼 (6 × 2 그리드 — 글자 깨짐 방지)
    st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;margin-top:18px;margin-bottom:8px;">월 (중복 선택 가능)</div>',
                 unsafe_allow_html=True)

    # 1~6월
    row1 = st.columns(6)
    for i, month in enumerate(months[:6]):
        is_selected = month in st.session_state.history_months
        btn_type = "primary" if is_selected else "secondary"
        if row1[i].button(f"{month}월", key=f"mo_{month}", use_container_width=True,
                          type=btn_type):
            if month in st.session_state.history_months:
                st.session_state.history_months.remove(month)
            else:
                st.session_state.history_months.append(month)
                st.session_state.history_months.sort()
            st.rerun()

    # 7~12월
    row2 = st.columns(6)
    for i, month in enumerate(months[6:]):
        is_selected = month in st.session_state.history_months
        btn_type = "primary" if is_selected else "secondary"
        if row2[i].button(f"{month}월", key=f"mo_{month}", use_container_width=True,
                          type=btn_type):
            if month in st.session_state.history_months:
                st.session_state.history_months.remove(month)
            else:
                st.session_state.history_months.append(month)
                st.session_state.history_months.sort()
            st.rerun()

    # 선택 요약
    n_y = len(st.session_state.history_years)
    n_m = len(st.session_state.history_months)
    total = n_y * n_m
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    if total > 0:
        st.caption(f"선택됨: 년도 {n_y}개 × 월 {n_m}개 = **{total}개월 기간**")
    else:
        st.caption("년도와 월을 선택해주세요")

    # 재분석 버튼
    if total > 0:
        if st.button("이 기간으로 재분석", type="primary",
                      use_container_width=True, key="period_reanalyze"):
            st.session_state["period_filter_active"] = True
            st.success(f"선택한 {total}개월 기간으로 결과를 필터링합니다.")
            st.rerun()


# =============================================================================
# 종목 카드 (한 줄 HTML)
# =============================================================================
def render_grade_card(row, grade: str):
    """V/S/A/B 등급 카드. row는 dict 또는 Series."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    p = PALETTE[st.session_state.theme]
    info = GRADE_INFO[grade]

    name = row.get("Name", "")
    code = row.get("Code", "")
    close = int(row.get("Close", 0)) if row.get("Close") else 0
    change = row.get("ChangeRatio", 0) or 0
    score = row.get("avg_score", row.get("Score", 0)) or 0
    n_presets = int(row.get("n_presets", 0)) if pd.notna(row.get("n_presets", 0)) else 0
    market = row.get("Market", "")
    trade_type = row.get("TradeType", "돌파매매")
    weight = GRADE_WEIGHTS[grade]
    n_shares = int(weight / close) if close > 0 else 0
    invest = n_shares * close

    ch_class = "up" if change > 0 else "down"

    # 등급 배지 + 가격 헤더
    card_html = (
        f'<div class="tcard" style="border-left:5px solid {info["color"]};">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
        f'<div>'
        f'<div style="margin-bottom:6px;">{grade_badge_html(grade)}'
        f'<span style="font-size:11px;color:{p["text_tertiary"]};margin-left:8px;">'
        f'비중 {info["weight_str"]} · {info["frequency"]}</span></div>'
        f'<div style="font-size:20px;font-weight:700;color:{p["text"]};margin-bottom:2px;">{name}</div>'
        f'<div style="font-size:12px;color:{p["text_tertiary"]};">{code} · {market} · {trade_type}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:22px;font-weight:700;color:{p["text"]};">{close:,}원</div>'
        f'<div class="{ch_class}" style="font-size:14px;">{change:+.2f}%</div>'
        f'</div>'
        f'</div>'
        # 주요 수치 3분할
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding-top:14px;border-top:1px solid {p["border"]};">'
        f'<div>'
        f'<div class="big-number-label">앙상블 점수</div>'
        f'<div style="font-size:18px;font-weight:700;color:{info["color"]};">{score:.1f}</div>'
        f'<div style="font-size:11px;color:{p["text_tertiary"]};">/ 100</div>'
        f'</div>'
        f'<div>'
        f'<div class="big-number-label">추천 프리셋</div>'
        f'<div style="font-size:18px;font-weight:700;color:{p["text"]};">{n_presets}/4</div>'
        f'<div style="font-size:11px;color:{p["text_tertiary"]};">개 동의</div>'
        f'</div>'
        f'<div>'
        f'<div class="big-number-label">매수 금액</div>'
        f'<div style="font-size:18px;font-weight:700;color:{p["accent"]};">{invest:,}원</div>'
        f'<div style="font-size:11px;color:{p["text_tertiary"]};">{n_shares}주</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # 사유 펼치기
    with st.expander(f"💡  {grade}급 추천사유 · {name} 상세"):
        reasons = grade_reason({**row, "grade": grade})
        for r in reasons:
            st.markdown(f"- {r}")

        # 추가 시그널 정보 + 추천 사유
        st.markdown("### 📊 12 시그널 점수")
        sig_data = []
        for sig_key, sig_name in SIGNAL_NAMES.items():
            val = row.get(sig_key, 0) or 0
            sig_data.append({"시그널": sig_name, "점수": val,
                              "막대": "█" * int(val / 5)})
        st.dataframe(pd.DataFrame(sig_data), use_container_width=True,
                      height=460, hide_index=True)

        st.markdown("### 📈 추가 추천 사유")
        recs = generate_recommendation_reasons(row)
        if recs:
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.caption("자동 사유 없음")


def render_no_grade(grade: str):
    """등급에 해당 종목이 없을 때 표시."""
    p = PALETTE[st.session_state.theme]
    info = GRADE_INFO[grade]
    st.markdown(
        f'<div class="tcard" style="border-left:5px solid {info["color"]};opacity:0.5;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div>'
        f'{grade_badge_html(grade)}'
        f'<span style="margin-left:12px;font-size:14px;color:{p["text_secondary"]};">'
        f'오늘은 {grade}급 조건 만족 종목 <b>없음</b></span>'
        f'</div>'
        f'<div style="font-size:11px;color:{p["text_tertiary"]};">{info["frequency"]}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_stock_card(row):
    p = PALETTE[st.session_state.theme]
    name = row.get('Name', '')
    code = row.get('Code', '')
    close = int(row.get('Close', 0))
    change = row.get('ChangeRatio', 0)
    score = row.get('Score', 0)
    trade_type = row.get('TradeType', '')
    pat_emoji = {"돌파매매": "🚀", "눌림목매매": "📉↗", "대시세 초입": "🌊"}.get(trade_type, "•")
    ch_class = "up" if change > 0 else "down"

    sl_price = 0
    if close > 0:
        sl, _ = haseunghoon_stoploss(close, row.get('ma3'), row.get('ma5'))
        sl_price = int(sl)
    sl_pct = (close - sl_price) / close * 100 if close > 0 and sl_price > 0 else 0

    badges = []
    if row.get('InHabarocell'): badges.append("🎓 하바로셀")
    if row.get('InHaseunghoon'): badges.append("📺 하승훈")
    if row.get('InUserList'): badges.append("⭐ 내 종목")
    badges_html = " ".join(f'<span class="badge badge-secondary">{b}</span>' for b in badges)

    insights = []
    if row.get('is_first_pullback'): insights.append("🎯 첫눌림")
    if row.get('cup_and_handle_detected'): insights.append("☕ 컵앤핸들")
    if row.get('gap_support_detected'): insights.append("🧲 갭지지")
    if row.get('pullback_quality') == "진짜 지지": insights.append("✅ 진짜지지")
    insights_html = " ".join(f'<span class="badge">{i}</span>' for i in insights)

    card_html = (
        f'<div class="tcard">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">'
        f'<div>'
        f'<div style="font-size:20px;font-weight:700;color:{p["text"]};margin-bottom:2px;">{name}</div>'
        f'<div style="font-size:12px;color:{p["text_tertiary"]};">{code} · {row.get("Market", "")}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:22px;font-weight:700;color:{p["text"]};">{close:,}원</div>'
        f'<div class="{ch_class}" style="font-size:14px;">{change:+.2f}%</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin:8px 0 14px 0;">'
        f'<span class="badge">{pat_emoji} {trade_type}</span>'
        f'{insights_html} {badges_html}'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding-top:14px;border-top:1px solid {p["border"]};">'
        f'<div>'
        f'<div class="big-number-label">매수가</div>'
        f'<div style="font-size:18px;font-weight:700;color:{p["text"]};">{close:,}</div>'
        f'</div>'
        f'<div>'
        f'<div class="big-number-label">손절가</div>'
        f'<div style="font-size:18px;font-weight:700;color:{p["down"]};">{sl_price:,}</div>'
        f'<div style="font-size:11px;color:{p["text_tertiary"]};">-{sl_pct:.1f}%</div>'
        f'</div>'
        f'<div>'
        f'<div class="big-number-label">점수</div>'
        f'<div style="font-size:18px;font-weight:700;color:{p["accent"]};">{score:.0f}</div>'
        f'<div style="font-size:11px;color:{p["text_tertiary"]};">/ 100</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # 추천 사유 (펼치기)
    with st.expander(f"💡 **{name}** 추천 사유 & 유사 사례"):
        # 추천 사유
        st.markdown("### 📊 추천 사유")
        reasons = generate_recommendation_reasons(row)
        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.caption("강한 시그널이 없거나 데이터 부족")

        # 시그널 분포
        st.markdown("### 📈 12 시그널 점수")
        sig_data = []
        for sig_key, sig_name in SIGNAL_NAMES.items():
            val = row.get(sig_key, 0)
            sig_data.append({"시그널": sig_name, "점수": val})
        sig_df = pd.DataFrame(sig_data)
        sig_df["bar"] = sig_df["점수"].apply(lambda x: "█" * int(x / 5))
        st.dataframe(sig_df, use_container_width=True, height=460, hide_index=True)

        # 유사 사례
        st.markdown("### 🔍 유사 실전 사례 TOP 3")
        st.caption("과거 실전 사례 35건 중 시그널 패턴이 가장 유사한 종목")
        similar = find_similar_cases(row, top_n=3)
        for s in similar:
            case = s["case"]
            sim = s["similarity"]
            pat_emoji2 = {
                Pattern.A_BREAKOUT: "🚀", Pattern.B_PULLBACK: "📉↗",
                Pattern.C_DOUBLE_BOTTOM: "📊", Pattern.D_LONGTERM: "🌊",
                Pattern.E_RISK: "⚠️",
            }.get(case["pattern"], "•")
            st.markdown(
                f'<div class="tcard" style="padding:14px 18px;margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                f'<div>'
                f'<div style="font-size:16px;font-weight:700;color:{p["text"]};">{pat_emoji2} {case["stock"]}</div>'
                f'<div style="font-size:11px;color:{p["text_tertiary"]};margin-top:2px;">{case["date"]} · {case["theme"]}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:22px;font-weight:800;color:{p["accent"]};">{sim:.0f}%</div>'
                f'<div style="font-size:10px;color:{p["text_tertiary"]};">유사도</div>'
                f'</div>'
                f'</div>'
                f'<div style="font-size:12px;color:{p["text_secondary"]};margin-top:8px;">'
                f'<b>📝 트리거:</b> {case["trigger"]}'
                f'</div>'
                f'<div style="font-size:12px;color:{p["text_secondary"]};margin-top:4px;">'
                f'<b>🎯 핵심 시그널:</b> {" · ".join(case["key_signals"])}'
                f'</div>'
                f'<div style="font-size:12px;color:{p["text_secondary"]};margin-top:4px;">'
                f'<b>📈 결과:</b> {case["outcome"]}'
                f'</div>'
                f'<div style="font-size:12px;color:{p["text_secondary"]};margin-top:4px;">'
                f'<b>💡 교훈:</b> {case["lesson"]}'
                f'</div>'
                f'<div style="font-size:11px;color:{p["text_tertiary"]};margin-top:6px;">'
                f'출처: {case["source"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# 페이지: 오늘의 종가매수 추천
# =============================================================================
def page_today():
    p = PALETTE[st.session_state.theme]
    st.markdown(
        f'<h1>오늘의 종가매수 추천 — V/S/A/B 등급제</h1>'
        f'<p style="color:{p["text_secondary"]};font-size:14px;margin-bottom:18px;">'
        f'4개 프리셋 통합 분석 (default · 박스돌파 · 하바로셀 · 풀백). 코스닥 + 돌파매매 + 등락 7~25% 기준.'
        f'</p>',
        unsafe_allow_html=True,
    )

    # 등급 가이드
    with st.expander("📖 등급 시스템 안내 (V > S > A > B)", expanded=False):
        st.markdown("""
**자본을 등급별로 배분합니다. 같은 종목이 여러 등급이면 상위 등급만 적용.**

| 등급 | 조건 | 비중 | 빈도 | 평균수익(180일) | 큰손실률 |
|---|---|---|---|---|---|
| 🏆 **V급** | 코스닥 · 등락 7~25% · 앙상블 점수 ≥ 75 | **50만원** | 연 5회 (2~3개월 1번) | **+96.9%** | 7% |
| 💎 **S급** | 코스닥 · 등락 7~25% · **4개 전략 만장일치** + 점수 ≥ 65 | **30만원** | 연 17회 (월 1.4회) | +56.8% | **4%** |
| ⭐ **A급** | 코스닥 · 등락 **10~18%** · 점수 ≥ 65 | **20만원** | 연 22회 (월 2회) | +34.1% | 9% |
| 🟢 **B급** | 코스닥 · 등락 7~25% · V1 통과 (1개+ 추천) | **10만원** | 매일 (연 143회) | +34.4% | 12% |

- **V/S급은 여러 종목** 추천 (점수 상위 3개까지)
- **A/B급은 점수 1위 1개**만 추천
- 매도: **180일** 자동 청산 (손절/익절 없음)
- 필요 자본: 약 **2,100만원** (피크 기준)
        """)

    # 매수금 설정만 간단히
    with st.expander("⚙️ 등급별 비중 조정 (선택)", expanded=False):
        cc = st.columns(4)
        for i, g in enumerate(["V", "S", "A", "B"]):
            cur = st.session_state.get(f"weight_{g}", GRADE_WEIGHTS[g])
            v = cc[i].number_input(
                f"{GRADE_INFO[g]['emoji']} {g}급 (만원)",
                10, 1000, int(cur/10000), 10, key=f"set_weight_{g}",
            )
            st.session_state[f"weight_{g}"] = v * 10000

    # 메인 액션 — 실시간 스캔만
    do_scan = st.button(
        "🔍 지금 스캔하기 (실시간 · 1~2분)",
        type="primary", use_container_width=True, key="main_scan",
    )

    # 데이터 소스 상태 확인
    try:
        import kis_api
        kis_ok = kis_api.is_available()
    except Exception:
        kis_ok = False
    from pathlib import Path
    cache_ok = Path("cache/market_snapshot.parquet").exists()

    if kis_ok:
        st.markdown(
            f'<div style="background:rgba(33,150,243,0.10);border:1px solid rgba(33,150,243,0.40);'
            f'border-radius:8px;padding:10px 14px;margin:10px 0 18px 0;font-size:12px;color:{p["text_secondary"]};">'
            f'🟦 <b>한국투자 OpenAPI 연결됨</b> — 어디서나 실시간 데이터 사용 가능'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif cache_ok:
        st.markdown(
            f'<div style="background:rgba(102,187,106,0.10);border:1px solid rgba(102,187,106,0.40);'
            f'border-radius:8px;padding:10px 14px;margin:10px 0 18px 0;font-size:12px;color:{p["text_secondary"]};">'
            f'🟢 <b>캐시 데이터 사용 가능</b> (어제 마감 기준, 매일 자동 갱신) · '
            f'한국 IP면 실시간 시도 후 실패 시 캐시 fallback'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:rgba(255,193,7,0.10);border:1px solid rgba(255,193,7,0.40);'
            f'border-radius:8px;padding:10px 14px;margin:10px 0 18px 0;font-size:12px;color:{p["text_secondary"]};">'
            f'⚠️ 캐시 없음 + KIS API 미설정. 한국 IP에서만 작동.'
            f'</div>',
            unsafe_allow_html=True,
        )

    if do_scan:
        filter_cfg, _, params, _ = build_configs()
        # V/S/A/B 등급에 맞춰 사전 필터 강화 (후보 ↓ → 속도 ↑)
        filter_cfg.change_min = 7.0
        filter_cfg.change_max = 25.0
        filter_cfg.include_kospi = False   # KOSDAQ만
        filter_cfg.include_kosdaq = True
        progress = st.progress(0, text="시장 데이터 가져오는 중...")
        start_t = datetime.now()

        def cb(i, t):
            elapsed = (datetime.now() - start_t).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            eta = (t - i) / rate if rate > 0 else 0
            progress.progress(
                min(i / t, 1.0),
                text=f"⚡ {i}/{t} · 경과 {elapsed:.0f}초 · 남은시간 ~{eta:.0f}초",
            )

        try:
            ensemble = scan_ensemble(
                filter_cfg, params, PRESETS_4,
                progress_callback=cb, min_recommend_score=40,
            )
            progress.empty()
            elapsed_total = (datetime.now() - start_t).total_seconds()
            if not ensemble.empty:
                ensemble = ensemble[ensemble["Market"] == "KOSDAQ"]
                if "TradeType" in ensemble.columns:
                    ensemble = ensemble[ensemble["TradeType"] == "돌파매매"]
            st.session_state.last_ensemble = ensemble
            st.session_state.last_scan_time = datetime.now()
            st.session_state.last_scan_elapsed = elapsed_total
            st.success(f"✅ 스캔 완료 — {elapsed_total:.0f}초 소요 · 후보 {len(ensemble)}개")
        except Exception as e:
            progress.empty()
            st.error(
                f"⚠️ **스캔 실패**\n\n"
                f"**해결책 (가장 빠른 길):**\n\n"
                f"1️⃣ **한국투자 OpenAPI 키 설정** (영구 해결)\n"
                f"   - https://apiportal.koreainvestment.com 가입\n"
                f"   - 앱 등록 → APP_KEY, APP_SECRET 발급\n"
                f"   - Streamlit Cloud → Settings → Secrets 에 추가:\n"
                f"     ```\n"
                f"     KIS_APP_KEY = \"...\"\n"
                f"     KIS_APP_SECRET = \"...\"\n"
                f"     ```\n"
                f"2️⃣ 또는 로컬 PC에서 `streamlit run app.py` (한국 IP)\n\n"
                f"에러: `{type(e).__name__}: {str(e)[:200]}`"
            )

    ensemble = st.session_state.get("last_ensemble")
    if ensemble is not None and not ensemble.empty:
        # 등급 분류 + 버킷 빌드
        graded = classify_candidates(ensemble)
        buckets = build_grade_buckets(graded, vs_max=10, a_max=10, b_max=5, show_all=True)

        scan_time = st.session_state.get("last_scan_time")
        elapsed = st.session_state.get("last_scan_elapsed", 0)
        time_str = scan_time.strftime("%Y-%m-%d %H:%M") if hasattr(scan_time, "strftime") else "-"
        st.markdown(
            f'<div style="margin-top:20px;margin-bottom:8px;">'
            f'<h2 style="margin:0;">📋 등급별 추천</h2>'
            f'<p class="subtle">🔴 실시간 · {time_str} 기준 · {elapsed:.0f}초 소요 · 4 프리셋 통합 분석</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 등급별 표시
        for g in ["V", "S", "A", "B"]:
            info = GRADE_INFO[g]
            sub = buckets[g]
            count_str = f"({len(sub)}건)" if len(sub) > 0 else "(없음)"
            st.markdown(
                f'<div style="margin:24px 0 12px 0;">'
                f'<h3 style="margin:0;color:{info["color"]};">{info["emoji"]} {info["name"]} {count_str}</h3>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if sub.empty:
                render_no_grade(g)
            else:
                for _, row in sub.iterrows():
                    render_grade_card(row, g)

        # 자본 요약
        total_invest = 0
        for g in ["V", "S", "A", "B"]:
            sub = buckets[g]
            for _, row in sub.iterrows():
                close = row.get("Close", 0) or 0
                if close > 0:
                    n_sh = int(GRADE_WEIGHTS[g] / close)
                    total_invest += n_sh * close

        st.markdown(
            f'<div class="tcard" style="margin-top:24px;background:{p.get("surface_alt", p.get("bg", "#F5F5F5"))};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<div style="font-size:13px;color:{p["text_tertiary"]};">💰 오늘 총 매수 필요액</div>'
            f'<div style="font-size:24px;font-weight:800;color:{p["accent"]};">{total_invest:,}원</div>'
            f'</div>'
            f'<div style="text-align:right;font-size:11px;color:{p["text_tertiary"]};">'
            f'V {len(buckets["V"])} · S {len(buckets["S"])} · '
            f'A {len(buckets["A"])} · B {len(buckets["B"])}'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # CSV 다운로드
        export_rows = []
        for g in ["V", "S", "A", "B"]:
            for _, row in buckets[g].iterrows():
                export_rows.append({
                    "등급": g,
                    "종목명": row.get("Name", ""),
                    "코드": row.get("Code", ""),
                    "시장": row.get("Market", ""),
                    "종가": int(row.get("Close", 0) or 0),
                    "등락률": row.get("ChangeRatio", 0),
                    "앙상블점수": round(row.get("avg_score", 0), 1),
                    "추천프리셋수": int(row.get("n_presets", 0) or 0),
                    "비중(원)": GRADE_WEIGHTS[g],
                    "매수금액": int(GRADE_WEIGHTS[g] / row["Close"]) * row["Close"] if row.get("Close", 0) > 0 else 0,
                })
        if export_rows:
            csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 추천 종목 CSV 다운로드", csv,
                file_name=f"VSAB추천_{datetime.now():%Y%m%d}.csv",
                use_container_width=True)

    elif ensemble is not None:
        st.markdown(
            f'<div class="empty-state"><div class="emoji">🔍</div>'
            f'<p>조건에 맞는 종목이 없습니다.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        # 첫 방문 안내
        st.markdown(
            f'<div class="empty-state"><div class="emoji">👋</div>'
            f'<p style="font-size:16px;color:{p["text"]};">처음 오셨군요</p>'
            f'<p>위 <b>🔍 실시간 스캔</b> 또는 <b>⚡ 캐시 데이터</b> 버튼을 누르면 V/S/A/B 등급별 추천을 확인할 수 있습니다.</p>'
            f'<p style="font-size:12px;margin-top:18px;">💡 처음엔 <b>⚡ 캐시 데이터</b>로 빠르게 확인하시고, 실시간 시세 반영이 필요하면 <b>실시간 스캔</b>을 누르세요.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


from backtest_helpers import (
    enrich_trades, monthly_pnl_table,
    pattern_comparison_table, year_month_pattern_table,
)


def _color_pct(val):
    """단순 % 셀 HTML (배경 강조 포함)."""
    UP = "#E91E63"; DOWN = "#3B82F6"
    UP_BG = "rgba(233, 30, 99, 0.08)"; DOWN_BG = "rgba(59, 130, 246, 0.08)"
    if val is None or pd.isna(val):
        return '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
    color = UP if val > 0 else (DOWN if val < 0 else "#9CA3AF")
    bg = ""
    if abs(val) >= 10:
        bg_color = UP_BG if val > 0 else DOWN_BG
        bg = f"background-color:{bg_color};"
    return f'<td style="text-align:right;padding:8px;color:{color};font-weight:700;{bg}">{val:+.2f}%</td>'


def _color_pnl(val_pnl, val_ref=None):
    """손익 셀 HTML — val_ref >= 10%면 배경 강조."""
    UP = "#E91E63"; DOWN = "#3B82F6"
    UP_BG = "rgba(233, 30, 99, 0.08)"; DOWN_BG = "rgba(59, 130, 246, 0.08)"
    if val_pnl is None or pd.isna(val_pnl):
        return '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
    color = UP if val_pnl > 0 else (DOWN if val_pnl < 0 else "#9CA3AF")
    bg = ""
    ref = val_ref if val_ref is not None and pd.notna(val_ref) else val_pnl
    if abs(ref) >= 10:
        bg_color = UP_BG if ref > 0 else DOWN_BG
        bg = f"background-color:{bg_color};"
    man = val_pnl / 10000
    text = f"{man:+,.0f}만원" if abs(man) >= 1 else f"{val_pnl:+,.0f}원"
    return f'<td style="text-align:right;padding:8px;color:{color};font-weight:700;{bg}">{text}</td>'


def render_strategy_comparison(position_size: int = 100_000):
    """안정/공격/욕심/중장기 전략 명확 비교 (사용자 매수금 기준)."""
    import numpy as np
    df = _load_all_enriched()
    if df.empty:
        st.warning("enriched 캐시 없음")
        return

    def fmt_won(p):
        if abs(p) >= 1e8: return f"{p/1e8:+,.2f}억"
        if abs(p) >= 1e4: return f"{p/1e4:+,.0f}만원"
        return f"{p:+,.0f}원"

    # 단타 전략들
    strategies = [
        {"label": "🛡️ 안정", "desc": "시가 즉시 매도", "rule": "익절/손절 없음 · 익일 시초가 청산",
         "kind": "open", "target": None, "stop": None, "type": "단타"},
        {"label": "💼 안정-타이트", "desc": "익절 +2% / 손절 -1.5%", "rule": "작은 익절 · 짧은 손절",
         "kind": None, "target": 2.0, "stop": -1.5, "type": "단타"},
        {"label": "⚔️ 공격", "desc": "익절 +5% / 손절 -3%", "rule": "중간 익절 · 중간 손절 (손익비 1.67)",
         "kind": None, "target": 5.0, "stop": -3.0, "type": "단타"},
        {"label": "🔥 욕심", "desc": "익절 +7% / 손절 -5%", "rule": "큰 익절 · 큰 손절 (손익비 1.40)",
         "kind": None, "target": 7.0, "stop": -5.0, "type": "단타"},
        {"label": "🎯 큰욕심", "desc": "익절 +10% / 손절 -5%", "rule": "큰 익절 노림 (손익비 2.0)",
         "kind": None, "target": 10.0, "stop": -5.0, "type": "단타"},
        {"label": "📌 단순", "desc": "종가 매도", "rule": "익일 종가에 무조건 청산",
         "kind": "close", "target": None, "stop": None, "type": "단타"},
    ]

    rows_html = ""
    for s in strategies:
        results = _simulate_exit(df, s["target"] or 999, s["stop"] or -999)
        if s["kind"] == "open" and "ret_d1_open" in df.columns:
            results = df["ret_d1_open"].dropna().values
        elif s["kind"] == "close" and "ret_d1_close" in df.columns:
            results = df["ret_d1_close"].dropna().values
        elif s["target"] is None or s["stop"] is None:
            continue

        if len(results) == 0: continue
        avg = float(np.mean(results))
        wr = float((results > 0).mean() * 100)
        n = len(results)
        pnl = avg / 100 * position_size * n

        color = "#E91E63" if pnl > 0 else "#3B82F6"
        bg = "rgba(233,30,99,0.06)" if pnl > 0 else "rgba(59,130,246,0.04)"
        badge = "✅ 수익" if pnl > 0 else "❌ 손실"
        badge_color = "#E91E63" if pnl > 0 else "#3B82F6"

        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:14px;font-weight:800;font-size:14px;">{s["label"]}</td>'
            f'<td style="padding:14px;font-weight:700;font-size:13px;color:var(--text);">{s["desc"]}<div style="font-size:10px;color:var(--text-3);font-weight:500;margin-top:2px;">{s["rule"]}</div></td>'
            f'<td style="text-align:right;padding:14px;font-weight:700;color:#E91E63;">{wr:.1f}%</td>'
            f'<td style="text-align:right;padding:14px;font-weight:700;color:{color};">{avg:+.2f}%</td>'
            f'<td style="text-align:right;padding:14px;font-weight:800;color:{color};font-size:15px;">{fmt_won(pnl)}</td>'
            f'<td style="text-align:center;padding:14px;color:{badge_color};font-weight:700;">{badge}</td>'
            '</tr>'
        )

    # 중장기 보유 (참고)
    horizons_swing = [("💎 20일 보유", "ret_20d"), ("💎 30일 보유", "ret_30d"),
                       ("💎 60일 보유", "ret_60d"), ("💎 90일 보유", "ret_90d"),
                       ("🥇 120일 보유", "ret_120d")]
    rows_html += (
        '<tr><td colspan="6" style="padding:8px 14px;background:var(--surface-alt);'
        'color:var(--text-2);font-weight:700;font-size:12px;">─ 참고: 중장기 보유 ─</td></tr>'
    )
    for label, col in horizons_swing:
        if col not in df.columns: continue
        v = df[col].dropna()
        if len(v) == 0: continue
        avg = float(v.mean())
        wr = float((v > 0).mean() * 100)
        n = len(v)
        pnl = avg / 100 * position_size * n
        color = "#E91E63" if pnl > 0 else "#3B82F6"
        bg = "rgba(233,30,99,0.08)" if "120" in label else "rgba(233,30,99,0.03)"
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:14px;font-weight:800;font-size:14px;">{label}</td>'
            f'<td style="padding:14px;font-weight:700;font-size:13px;color:var(--text);">매수 후 {label.split()[1]} 보유</td>'
            f'<td style="text-align:right;padding:14px;font-weight:700;color:#E91E63;">{wr:.1f}%</td>'
            f'<td style="text-align:right;padding:14px;font-weight:700;color:{color};">{avg:+.2f}%</td>'
            f'<td style="text-align:right;padding:14px;font-weight:800;color:{color};font-size:15px;">{fmt_won(pnl)}</td>'
            f'<td style="text-align:center;padding:14px;color:#E91E63;font-weight:700;">✅ 수익</td>'
            '</tr>'
        )

    table_html = (
        '<div style="border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:12px 14px;text-align:left;font-size:12px;color:var(--text-2);">전략</th>'
        '<th style="padding:12px 14px;text-align:left;font-size:12px;color:var(--text-2);">설명</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">승률</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">평균수익률</th>'
        f'<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">최종 누적<br><span style="font-size:9px;font-weight:500;">({position_size/10000:,.0f}만원×31,154건)</span></th>'
        '<th style="padding:12px;text-align:center;font-size:12px;color:var(--text-2);">결과</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # 핵심 메시지
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.success(
        f"### 💡 명확한 결론\n\n"
        f"**단타로 6년 돌리면 최대 +798만원** (10만원 매수 기준, 시가 즉시 매도)\n\n"
        f"**120일 보유하면 +3.45억** ← 43배 차이\n\n"
        f"⚠️ 종가매수 시그널은 사실 '단타용'이 아니라 **'중기 매집 진입점'**으로 해석해야 함.\n"
        f"단타 노린다면 **시가 즉시 매도** 또는 **+7% 익절 / -5% 손절** 두 개만 통계적으로 양수."
    )


def _rr_stats(v):
    """평균수익/손실/승률/손익비/기대값."""
    v = v.dropna()
    if len(v) == 0:
        return None
    wins = v[v > 0]
    losses = v[v < 0]
    wr = len(wins) / len(v) * 100
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
    expectancy = (avg_win * wr / 100) - (avg_loss * (100 - wr) / 100)
    return {
        "n": len(v), "wr": wr, "avg_win": avg_win, "avg_loss": -avg_loss,
        "rr": rr, "exp": expectancy,
    }


def _simulate_exit(df, target_pct, stop_pct):
    """동적 익절/손절. 익일 OHLC로 실현 수익률 계산."""
    import numpy as np
    cols = ["ret_d1_open", "ret_d1_high", "ret_d1_low", "ret_d1_close"]
    if not all(c in df.columns for c in cols):
        return np.array([])
    sub = df[cols].dropna()
    if sub.empty:
        return np.array([])
    o = sub["ret_d1_open"].values
    h = sub["ret_d1_high"].values
    l = sub["ret_d1_low"].values
    c = sub["ret_d1_close"].values
    results = np.zeros(len(sub))
    for i in range(len(sub)):
        if o[i] >= target_pct:
            results[i] = o[i]
        elif o[i] <= stop_pct:
            results[i] = o[i]
        else:
            hit_t = h[i] >= target_pct
            hit_s = l[i] <= stop_pct
            if hit_t and hit_s:
                results[i] = stop_pct  # 보수적
            elif hit_t:
                results[i] = target_pct
            elif hit_s:
                results[i] = stop_pct
            else:
                results[i] = c[i]
    return results


def render_risk_reward_section(position_size: int = 1_000_000):
    """손익비 표 + 동적 익절/손절 시뮬레이터."""
    df_all = _load_all_enriched()
    if df_all.empty:
        st.warning("enriched 캐시 없음")
        return

    def fmt_won(p):
        if p is None or pd.isna(p): return "—"
        man = p / 10000
        if abs(man) >= 10000: return f"{man/10000:+,.2f}억"
        return f"{man:+,.0f}만"

    # === [A] 매도 시점별 손익비 표 ===
    st.markdown("### 📊 D+1 매도 시점별 손익비")
    st.caption(
        "손익비 = 평균수익 ÷ 평균손실 · 기대값 = 평균수익×승률 - 평균손실×손실률 "
        "(양수면 통계적으로 유리)"
    )

    horizons = [("익일 시가", "ret_d1_open"),
                ("익일 고가 (이상치)", "ret_d1_high"),
                ("익일 저가 (최악)", "ret_d1_low"),
                ("익일 종가", "ret_d1_close")]
    rows_html = ""
    best_rr = -1; best_label = None
    for label, col in horizons:
        s = _rr_stats(df_all[col])
        if not s: continue
        if s["rr"] > best_rr and s["rr"] != float('inf'):
            best_rr = s["rr"]; best_label = label
        exp_pnl = s["exp"] / 100 * position_size * s["n"]
        exp_color = "#E91E63" if s["exp"] > 0 else "#3B82F6"
        rr_color = "#E91E63" if s["rr"] >= 1.5 else ("#22C55E" if s["rr"] >= 1.0 else "#3B82F6")
        rows_html += (
            '<tr>'
            f'<td style="padding:12px 14px;font-weight:700;background:rgba(233,30,99,0.04);">{label}</td>'
            f'<td style="text-align:right;padding:10px;color:var(--text-2);">{s["n"]:,}</td>'
            f'<td style="text-align:right;padding:10px;color:#E91E63;font-weight:700;">{s["wr"]:.1f}%</td>'
            f'<td style="text-align:right;padding:10px;color:#E91E63;font-weight:700;">{s["avg_win"]:+.2f}%</td>'
            f'<td style="text-align:right;padding:10px;color:#3B82F6;font-weight:700;">{s["avg_loss"]:+.2f}%</td>'
            f'<td style="text-align:right;padding:10px;color:{rr_color};font-weight:800;font-size:16px;">{s["rr"]:.2f}</td>'
            f'<td style="text-align:right;padding:10px;color:{exp_color};font-weight:800;">{s["exp"]:+.2f}%</td>'
            f'<td style="text-align:right;padding:10px;color:{exp_color};font-weight:800;font-size:14px;">{fmt_won(exp_pnl)}</td>'
            '</tr>'
        )
    rr_table = (
        '<div style="border:1px solid var(--border);border-radius:10px;margin-bottom:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:12px 14px;text-align:left;font-size:12px;color:var(--text-2);">매도 시점</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">N</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">승률</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">평균수익</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">평균손실</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">손익비</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">기대값/거래</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">전체 기대손익</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )
    st.markdown(rr_table, unsafe_allow_html=True)
    st.caption(
        "🥇 손익비 1.5+ (분홍) / 1.0+ (그린) / 1.0↓ (파랑) · "
        "고가는 이상치 (실제로 잡기 어려움)"
    )

    # === [B] 동적 익절/손절 시뮬레이터 ===
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🎯 동적 익절/손절 시뮬레이터")
    st.caption("익일 고가가 익절가 도달 → 익절 / 저가가 손절가 도달 → 손절 / 둘다 미달 → 종가 청산")

    col_t, col_s, col_p = st.columns(3)
    target = col_t.slider("익절가 (%)", 0.5, 15.0, 5.0, 0.5, key="rr_target")
    stop = col_s.slider("손절가 (%)", -10.0, -0.5, -3.0, 0.5, key="rr_stop")
    pat_filter = col_p.selectbox(
        "매매타입 필터",
        ["전체", "돌파매매", "눌림목매매", "대시세 초입"],
        key="rr_pattern",
    )

    sub = df_all if pat_filter == "전체" else df_all[df_all["TradeType"] == pat_filter]
    results = _simulate_exit(sub, target, stop)
    if len(results) == 0:
        st.warning("결과 없음")
        return

    import numpy as np
    avg = float(np.mean(results))
    wr = float((results > 0).mean() * 100)
    n = len(results)
    pnl = avg / 100 * position_size * n
    avg_color = "#E91E63" if avg > 0 else "#3B82F6"

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="tcard" style="text-align:center;padding:18px;">'
            f'<div class="big-number-label">평균 수익률</div>'
            f'<div style="font-size:24px;font-weight:800;color:{avg_color};">{avg:+.2f}%</div>'
            f'</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            f'<div class="tcard" style="text-align:center;padding:18px;">'
            f'<div class="big-number-label">승률</div>'
            f'<div style="font-size:24px;font-weight:800;color:#E91E63;">{wr:.1f}%</div>'
            f'</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f'<div class="tcard" style="text-align:center;padding:18px;">'
            f'<div class="big-number-label">거래 수</div>'
            f'<div style="font-size:24px;font-weight:800;color:var(--text);">{n:,}</div>'
            f'</div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(
            f'<div class="tcard" style="text-align:center;padding:18px;">'
            f'<div class="big-number-label">누적 손익</div>'
            f'<div style="font-size:22px;font-weight:800;color:{avg_color};">{fmt_won(pnl)}</div>'
            f'</div>', unsafe_allow_html=True)

    # === [C] 익절/손절 매트릭스 (전체 조합) ===
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 익절/손절 전체 조합 매트릭스 (최적 조합 자동 탐색)"):
        target_list = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
        stop_list = [-1.0, -1.5, -2.0, -3.0, -5.0]
        best = {"pnl": -1e18, "tp": None, "sp": None, "stats": None}

        matrix_html = '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        matrix_html += '<thead><tr style="background:var(--surface-alt);">'
        matrix_html += '<th style="padding:10px;text-align:left;">익절↓ 손절→</th>'
        for sp in stop_list:
            matrix_html += f'<th style="padding:10px;text-align:right;">{sp:+.1f}%</th>'
        matrix_html += '</tr></thead><tbody>'

        for tp in target_list:
            matrix_html += '<tr>'
            matrix_html += f'<td style="padding:10px;font-weight:700;background:rgba(233,30,99,0.04);">+{tp:.1f}%</td>'
            for sp in stop_list:
                results = _simulate_exit(sub, tp, sp)
                if len(results) == 0:
                    matrix_html += '<td style="padding:10px;text-align:right;color:#9CA3AF;">—</td>'
                    continue
                avg_m = float(np.mean(results))
                pnl_m = avg_m / 100 * position_size * len(results)
                if pnl_m > best["pnl"]:
                    best["pnl"] = pnl_m
                    best["tp"] = tp; best["sp"] = sp
                    best["stats"] = (avg_m, float((results > 0).mean() * 100), len(results))
                color = "#E91E63" if pnl_m > 0 else "#3B82F6"
                bg = "background:rgba(233,30,99,0.10);" if pnl_m > 0 else ""
                man = pnl_m / 10000
                if abs(man) >= 10000:
                    text = f"{man/10000:+,.1f}억"
                else:
                    text = f"{man:+,.0f}만"
                matrix_html += (
                    f'<td style="padding:10px;text-align:right;color:{color};'
                    f'font-weight:700;{bg}">{text}</td>'
                )
            matrix_html += '</tr>'
        matrix_html += '</tbody></table>'
        st.markdown(matrix_html, unsafe_allow_html=True)

        if best["tp"]:
            stats = best["stats"]
            st.success(
                f"🥇 **최적 조합**: 익절 **+{best['tp']:.1f}%** / 손절 **{best['sp']:+.1f}%** → "
                f"평균 **{stats[0]:+.2f}%** · 승률 **{stats[1]:.1f}%** · "
                f"누적 **{fmt_won(best['pnl'])}** ({stats[2]:,}건)"
            )


def _load_all_enriched():
    """9 프리셋 합본 로드."""
    from pathlib import Path
    presets = ["default", "conservative", "aggressive", "box_breakout", "habarocell",
               "haseunghoon", "pullback", "mega_trend", "master_guide"]
    frames = []
    for p in presets:
        path = Path(f"cache/enriched_{p}.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            df["preset"] = p
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def render_comprehensive_guide(position_size: int = 1_000_000):
    """년도별 가이드 + TOP3 전략 + 매매타입별 (9 프리셋 합본)."""
    import numpy as np

    df_all = _load_all_enriched()
    if df_all.empty:
        st.warning("enriched 캐시 없음. `python3 precompute_enriched.py` 실행 필요.")
        return

    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all["Year"] = df_all["Date"].dt.year

    horizons = [("익일", "ret_d1_close"), ("20일", "ret_20d"), ("30일", "ret_30d"),
                ("60일", "ret_60d"), ("90일", "ret_90d"), ("120일", "ret_120d")]

    def fmt_won(pnl):
        if pnl is None or pd.isna(pnl): return "—"
        man = pnl / 10000
        if abs(man) >= 10000: return f"{man/10000:+,.2f}억"
        return f"{man:+,.0f}만원"

    def stats_of(v, n):
        if len(v) == 0: return None, None, None
        avg = float(v.mean()); wr = float((v > 0).mean() * 100)
        pnl = avg / 100 * position_size * n
        return avg, wr, pnl

    UP_BG = "rgba(233,30,99,0.06)"

    # === [1] 년도별 최고 보유기간 가이드 ===
    st.markdown("### 📅 년도별 최고 보유기간 가이드")
    rows_html = ""
    for y in sorted(df_all["Year"].unique()):
        ydf = df_all[df_all["Year"] == y]
        best_avg = -999; best_h = None; best_pnl = 0
        for label, col in horizons:
            v = ydf[col].dropna() if col in ydf else pd.Series(dtype=float)
            avg, wr, pnl = stats_of(v, len(v))
            if avg is not None and avg > best_avg:
                best_avg = avg; best_h = label; best_pnl = pnl
        market_emoji = "🔥" if best_avg > 10 else "📈" if best_avg > 0 else "📉"
        avg_color = "#E91E63" if best_avg > 0 else "#3B82F6"
        rows_html += (
            '<tr>'
            f'<td style="padding:10px 14px;font-weight:800;font-size:15px;background:{UP_BG};">{y}</td>'
            f'<td style="text-align:center;padding:10px;color:var(--text-2);">{len(ydf):,}건</td>'
            f'<td style="text-align:center;padding:10px;font-size:18px;font-weight:800;color:var(--accent);">{best_h}</td>'
            f'<td style="text-align:right;padding:10px;font-weight:700;color:{avg_color};">{best_avg:+.2f}%</td>'
            f'<td style="text-align:right;padding:10px;font-weight:800;font-size:15px;color:{avg_color};">{fmt_won(best_pnl)} {market_emoji}</td>'
            '</tr>'
        )

    # 전체
    best_avg = -999; best_h = None; best_pnl = 0
    for label, col in horizons:
        v = df_all[col].dropna() if col in df_all else pd.Series(dtype=float)
        avg, wr, pnl = stats_of(v, len(v))
        if avg is not None and avg > best_avg:
            best_avg = avg; best_h = label; best_pnl = pnl
    rows_html += (
        '<tr style="border-top:3px solid var(--accent);background:linear-gradient(0deg, rgba(233,30,99,0.08), rgba(233,30,99,0.02));">'
        f'<td style="padding:14px;font-weight:800;font-size:16px;">🏆 전체</td>'
        f'<td style="text-align:center;padding:14px;font-weight:700;">{len(df_all):,}건</td>'
        f'<td style="text-align:center;padding:14px;font-size:20px;font-weight:800;color:var(--accent);">{best_h}</td>'
        f'<td style="text-align:right;padding:14px;font-weight:800;color:#E91E63;font-size:15px;">{best_avg:+.2f}%</td>'
        f'<td style="text-align:right;padding:14px;font-weight:800;font-size:18px;color:#E91E63;">{fmt_won(best_pnl)}</td>'
        '</tr>'
    )

    table_html = (
        '<div style="border:1px solid var(--border);border-radius:10px;margin-bottom:8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:12px 14px;text-align:left;font-size:12px;color:var(--text-2);">년도</th>'
        '<th style="padding:12px;text-align:center;font-size:12px;color:var(--text-2);">시그널수</th>'
        '<th style="padding:12px;text-align:center;font-size:12px;color:var(--text-2);">🥇 최고 보유기간</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">평균 수익률</th>'
        '<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">누적 손익</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("🔥 +10%↑ · 📈 양수 · 📉 음수 (베어마켓)")

    # === [2] TOP 3 매매전략 ===
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🏆 최고 누적 수익금 TOP 3 매매전략")
    st.caption("9 프리셋 × 6 보유기간 = 54개 조합 중 누적 손익 상위 3")

    combos = []
    for p in df_all["preset"].unique():
        sub = df_all[df_all["preset"] == p]
        for label, col in horizons:
            v = sub[col].dropna() if col in sub else pd.Series(dtype=float)
            if len(v) == 0: continue
            avg, wr, pnl = stats_of(v, len(v))
            combos.append({
                "preset": p, "horizon": label, "n": len(v),
                "avg": avg, "wr": wr, "pnl": pnl,
            })
    combos.sort(key=lambda x: -x["pnl"])
    top3 = combos[:3]

    from rules import PRESETS as PRESET_DEFS
    medals = ["🥇", "🥈", "🥉"]
    cards_html = '<div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;margin-bottom:8px;">'
    for i, c in enumerate(top3):
        pname = PRESET_DEFS.get(c["preset"], {}).get("name", c["preset"])
        cards_html += (
            '<div class="tcard" style="text-align:center;padding:20px 16px;'
            f'border:2px solid var(--accent);">'
            f'<div style="font-size:32px;margin-bottom:6px;">{medals[i]}</div>'
            f'<div style="font-size:13px;font-weight:700;color:var(--text-2);margin-bottom:2px;">{c["horizon"]} 보유</div>'
            f'<div style="font-size:14px;font-weight:800;color:var(--text);margin-bottom:10px;">{pname}</div>'
            f'<div style="font-size:24px;font-weight:800;color:#E91E63;letter-spacing:-0.5px;">{fmt_won(c["pnl"])}</div>'
            f'<div style="font-size:11px;color:var(--text-2);margin-top:6px;">평균 <b style="color:#E91E63;">{c["avg"]:+.2f}%</b> · 승률 <b>{c["wr"]:.1f}%</b></div>'
            f'<div style="font-size:11px;color:var(--text-3);margin-top:2px;">{c["n"]:,}건</div>'
            '</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # === [3] 매매타입별 × 보유기간 ===
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 매매타입별 × 보유기간 수익률")
    st.caption("돌파매매 / 눌림목매매 / 대시세 초입 — 어느 패턴이 어느 시점에 가장 좋은가")

    if "TradeType" not in df_all.columns:
        st.warning("매매타입 컬럼 없음. `python3 precompute_enriched.py` 다시 실행 필요.")
        return

    patterns = ["돌파매매", "눌림목매매", "대시세 초입"]
    pat_emoji = {"돌파매매": "🚀", "눌림목매매": "📉↗", "대시세 초입": "🌊"}

    rows_html = ""
    for pat in patterns:
        sub = df_all[df_all["TradeType"] == pat]
        n = len(sub)
        if n == 0: continue
        best_pnl_val = -1e18; best_h = None
        cells_html = ""
        cells_data = []
        for label, col in horizons:
            v = sub[col].dropna() if col in sub else pd.Series(dtype=float)
            avg, wr, pnl = stats_of(v, len(v))
            cells_data.append((label, avg, wr, pnl))
            if pnl is not None and pnl > best_pnl_val:
                best_pnl_val = pnl; best_h = label
        for label, avg, wr, pnl in cells_data:
            is_best = label == best_h
            color = "#E91E63" if (pnl or 0) > 0 else "#3B82F6"
            bg = "background-color:rgba(233,30,99,0.10);" if is_best else ""
            weight = 800 if is_best else 700
            badge = " 🥇" if is_best else ""
            cells_html += (
                f'<td style="text-align:right;padding:10px;color:{color};font-weight:{weight};{bg}">'
                f'<div style="font-size:13px;">{avg:+.2f}%{badge}</div>'
                f'<div style="font-size:10px;color:var(--text-3);font-weight:500;">{fmt_won(pnl)}</div>'
                f'</td>'
            )
        rows_html += (
            '<tr>'
            f'<td style="padding:14px;font-weight:800;background:{UP_BG};">'
            f'<div style="font-size:14px;">{pat_emoji.get(pat, "")} {pat}</div>'
            f'<div style="font-size:10px;color:var(--text-3);font-weight:500;">{n:,}건</div>'
            f'</td>'
            f'{cells_html}'
            '</tr>'
        )

    header_html = (
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:12px;text-align:left;font-size:12px;color:var(--text-2);">매매타입</th>'
    )
    for label, _ in horizons:
        header_html += f'<th style="padding:12px;text-align:right;font-size:12px;color:var(--text-2);">{label}</th>'
    header_html += '</tr></thead>'

    table_html = (
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'{header_html}<tbody>{rows_html}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_year_horizon_matrix(trades_df, position_size: int = 1_000_000):
    """년도 × 매도 시점 매트릭스 — 평균/승률/누적손익 + 각 년도 최고 시점 하이라이트."""
    if trades_df is None or trades_df.empty:
        return

    horizons = [
        ("익일 종가", "ret_d1_close"),
        ("20일", "ret_20d"),
        ("30일", "ret_30d"),
        ("60일", "ret_60d"),
        ("90일", "ret_90d"),
        ("120일", "ret_120d"),
    ]

    df = trades_df.copy()
    df["Year"] = df["Date"].dt.year
    years = sorted(df["Year"].unique())

    def _stats(sub, col):
        if col not in sub.columns:
            return None, None, None, 0
        v = sub[col].dropna()
        if len(v) == 0:
            return None, None, None, 0
        avg = float(v.mean())
        wr = float((v > 0).mean() * 100)
        pnl = avg / 100 * position_size * len(v)
        return avg, wr, pnl, len(v)

    # 각 년도별 최고 시점 (avg 기준)
    best_per_year = {}
    for y in years:
        ydf = df[df["Year"] == y]
        best_avg = -999
        best_h = None
        for label, col in horizons:
            avg, *_ = _stats(ydf, col)
            if avg is not None and avg > best_avg:
                best_avg = avg
                best_h = label
        best_per_year[y] = best_h

    # 표 모드 선택: 평균 / 승률 / 손익
    mode = st.radio(
        "표시", ["평균 수익률(%)", "승률(%)", "누적 손익(만원)"],
        horizontal=True, label_visibility="collapsed", key="matrix_mode",
    )

    rows_html = ""
    for y in years:
        ydf = df[df["Year"] == y]
        n_year = len(ydf)
        rows_html += "<tr>"
        rows_html += (
            f'<td style="padding:10px 12px;font-weight:800;color:var(--text);'
            f'background:var(--surface-alt);">{y}</td>'
        )
        rows_html += (
            f'<td style="text-align:right;padding:8px;color:var(--text-2);">{n_year}</td>'
        )
        for label, col in horizons:
            avg, wr, pnl, n = _stats(ydf, col)
            is_best = best_per_year.get(y) == label
            badge = ' 🥇' if is_best else ''
            if mode == "평균 수익률(%)":
                if avg is None:
                    cell = '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
                else:
                    color = "#E91E63" if avg > 0 else "#3B82F6"
                    bg = ""
                    if abs(avg) >= 10:
                        bg_color = "rgba(233, 30, 99, 0.08)" if avg > 0 else "rgba(59, 130, 246, 0.08)"
                        bg = f"background-color:{bg_color};"
                    elif is_best:
                        bg = "background-color:rgba(233, 30, 99, 0.04);"
                    weight = 800 if is_best else 700
                    cell = (
                        f'<td style="text-align:right;padding:8px;color:{color};'
                        f'font-weight:{weight};{bg}">{avg:+.2f}%{badge}</td>'
                    )
            elif mode == "승률(%)":
                if wr is None:
                    cell = '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
                else:
                    color = "#E91E63" if wr >= 50 else "#3B82F6"
                    bg = "background-color:rgba(233, 30, 99, 0.04);" if is_best else ""
                    weight = 800 if is_best else 700
                    cell = (
                        f'<td style="text-align:right;padding:8px;color:{color};'
                        f'font-weight:{weight};{bg}">{wr:.1f}%{badge}</td>'
                    )
            else:  # 누적 손익
                if pnl is None:
                    cell = '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
                else:
                    color = "#E91E63" if pnl > 0 else "#3B82F6"
                    man = pnl / 10000
                    # 만원 단위로 명확히 표시
                    if abs(man) >= 10000:  # 1억 이상
                        text = f"{man/10000:+,.1f}억"
                    else:
                        text = f"{man:+,.0f}만원"
                    bg = "background-color:rgba(233, 30, 99, 0.04);" if is_best else ""
                    weight = 800 if is_best else 700
                    cell = (
                        f'<td style="text-align:right;padding:8px;color:{color};'
                        f'font-weight:{weight};font-size:14px;{bg}">{text}{badge}</td>'
                    )
            rows_html += cell
        rows_html += "</tr>"

    # 전체 합계 — 강조
    rows_html += (
        '<tr style="border-top:3px solid var(--accent);'
        'background:linear-gradient(0deg, rgba(233,30,99,0.06), rgba(233,30,99,0.02));">'
    )
    rows_html += (
        '<td style="padding:14px 12px;font-weight:800;color:var(--text);font-size:15px;">🏆 전체</td>'
    )
    rows_html += (
        f'<td style="text-align:right;padding:8px;color:var(--text);font-weight:800;font-size:14px;">{len(df):,}</td>'
    )
    for label, col in horizons:
        avg, wr, pnl, n = _stats(df, col)
        if mode == "평균 수익률(%)" and avg is not None:
            color = "#E91E63" if avg > 0 else "#3B82F6"
            cell = f'<td style="text-align:right;padding:14px 8px;color:{color};font-weight:800;font-size:16px;">{avg:+.2f}%</td>'
        elif mode == "승률(%)" and wr is not None:
            color = "#E91E63" if wr >= 50 else "#3B82F6"
            cell = f'<td style="text-align:right;padding:14px 8px;color:{color};font-weight:800;font-size:16px;">{wr:.1f}%</td>'
        elif mode == "누적 손익(만원)" and pnl is not None:
            color = "#E91E63" if pnl > 0 else "#3B82F6"
            man = pnl / 10000
            if abs(man) >= 10000:
                text = f"{man/10000:+,.2f}억"
            else:
                text = f"{man:+,.0f}만원"
            cell = (
                f'<td style="text-align:right;padding:14px 8px;color:{color};'
                f'font-weight:800;font-size:17px;">{text}</td>'
            )
        else:
            cell = '<td style="text-align:right;padding:14px 8px;color:#9CA3AF;">—</td>'
        rows_html += cell
    rows_html += "</tr>"

    header_cells = (
        '<th style="padding:10px 12px;text-align:left;font-size:12px;font-weight:700;color:var(--text-2);">년도</th>'
        '<th style="padding:10px;text-align:right;font-size:12px;font-weight:700;color:var(--text-2);">N</th>'
    )
    for label, _ in horizons:
        header_cells += (
            f'<th style="padding:10px 6px;text-align:right;font-size:12px;font-weight:700;color:var(--text-2);">{label}</th>'
        )

    table_html = (
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;margin-top:8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:var(--surface-alt);">{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(
        f"🥇 각 년도에서 가장 수익률 좋았던 시점 · 빨강=양수, 파랑=음수, 진하면 10%↑ · "
        f"종목당 매수금 {position_size/10000:,.0f}만원 기준"
    )

    # ───────── 누적 손익 모드 — 큰 금액 카드 ─────────
    if mode == "누적 손익(만원)":
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<h3 style="margin-bottom:8px;">💰 매도 시점별 최종 누적 손익</h3>'
            f'<p class="muted" style="margin-bottom:14px;">'
            f'2020~2026 전체 {len(df):,}건 × 종목당 {position_size/10000:,.0f}만원 매수 기준</p>',
            unsafe_allow_html=True,
        )

        # 각 horizon의 최종 손익을 큰 카드로
        cards_html = '<div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;">'
        for label, col in horizons:
            avg, wr, pnl, n = _stats(df, col)
            if pnl is None:
                continue
            color = "#E91E63" if pnl > 0 else "#3B82F6"
            man = pnl / 10000
            if abs(man) >= 10000:
                main_text = f"{man/10000:+,.2f}억"
                sub_text = f"{man:+,.0f}만원"
            else:
                main_text = f"{man:+,.0f}만원"
                sub_text = ""
            cards_html += (
                '<div class="tcard" style="text-align:center;padding:18px 16px;">'
                f'<div class="big-number-label" style="font-size:13px;font-weight:700;">{label}</div>'
                f'<div style="font-size:26px;font-weight:800;color:{color};margin:8px 0 4px 0;letter-spacing:-0.5px;">'
                f'{main_text}</div>'
                + (f'<div style="font-size:11px;color:var(--text-3);">{sub_text}</div>' if sub_text else '')
                + f'<div style="font-size:11px;color:var(--text-2);margin-top:6px;">'
                f'평균 <b style="color:{color};">{avg:+.2f}%</b> · '
                f'승률 <b>{wr:.1f}%</b> · '
                f'{n:,}건</div>'
                '</div>'
            )
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # 최고 horizon 강조
        best_pnl_h = max(horizons, key=lambda h: _stats(df, h[1])[2] or -1e18)
        best_avg, best_wr, best_pnl, best_n = _stats(df, best_pnl_h[1])
        if best_pnl is not None:
            best_man = best_pnl / 10000
            best_text = (
                f"{best_man/10000:+,.2f}억" if abs(best_man) >= 10000
                else f"{best_man:+,.0f}만원"
            )
            st.success(
                f"🏆 **{best_pnl_h[0]}** 매도가 최고 — 누적 **{best_text}** · "
                f"평균 {best_avg:+.2f}% · 승률 {best_wr:.1f}% · {best_n:,}건"
            )


def render_pattern_comparison_html(pat_df):
    """
    시점 × 패턴 비교 표 (transpose 구조).
    행: 시점 8개 (D+1 OHLC 4 + 20/60/90/120일)
    열: 돌파/눌림/대시세/전체 × (평균 + 승률)
    """
    if pat_df is None or pat_df.empty:
        return
    patterns = ["돌파매매", "눌림목매매", "대시세 초입", "전체"]
    pat_emoji = {"돌파매매": "🚀", "눌림목매매": "📉↗", "대시세 초입": "🌊", "전체": "🏆"}

    rows_html = ""
    for _, row in pat_df.iterrows():
        sp = str(row.get("시점", ""))
        # 단타↔중기 구분선
        is_swing_start = sp == "20일"
        border_top = (
            'border-top:2px solid var(--accent);'
            if is_swing_start else ''
        )
        rows_html += f'<tr style="{border_top}">'
        rows_html += (
            f'<td style="padding:10px 12px;font-weight:700;color:var(--text);white-space:nowrap;">{sp}</td>'
        )
        for pat in patterns:
            avg = row.get(f"{pat} 평균")
            wr = row.get(f"{pat} 승률")
            n = int(row.get(f"{pat} N", 0))
            rows_html += _color_pct(avg)
            # 승률 색상
            if wr is None or pd.isna(wr):
                rows_html += '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
            else:
                wr_color = "#E91E63" if wr >= 50 else "#3B82F6"
                rows_html += f'<td style="text-align:right;padding:8px;color:{wr_color};font-weight:700;">{wr:.1f}%</td>'
            rows_html += f'<td style="text-align:right;padding:8px;color:var(--text-3);font-size:11px;">{n}</td>'
        rows_html += '</tr>'

    header_cells = (
        '<th style="padding:10px 12px;text-align:left;font-size:12px;font-weight:700;color:var(--text-2);">시점</th>'
    )
    for pat in patterns:
        emoji = pat_emoji.get(pat, "")
        header_cells += (
            f'<th colspan="3" style="padding:10px 6px;text-align:center;font-size:12px;font-weight:700;color:var(--text-2);border-left:1px solid var(--border);">{emoji} {pat}</th>'
        )

    sub_header = '<th style="padding:6px 12px;background:var(--surface);"></th>'
    for pat in patterns:
        sub_header += (
            '<th style="padding:6px;text-align:right;font-size:10px;font-weight:500;color:var(--text-3);border-left:1px solid var(--border);">평균</th>'
            '<th style="padding:6px;text-align:right;font-size:10px;font-weight:500;color:var(--text-3);">승률</th>'
            '<th style="padding:6px;text-align:right;font-size:10px;font-weight:500;color:var(--text-3);">N</th>'
        )

    table_html = (
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:var(--surface-alt);">{header_cells}</tr>'
        f'<tr style="background:var(--surface-alt);">{sub_header}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_ym_pattern_html(df):
    """년/월 × 패턴별 평균+승률 표."""
    if df is None or df.empty:
        return
    patterns = ["돌파매매", "눌림목매매", "대시세 초입"]

    rows_html = ""
    for _, row in df.iterrows():
        is_total = "합계" in str(row.get("월", ""))
        row_style = (
            'border-top:2px solid var(--accent);background:var(--surface-alt);'
            if is_total else ''
        )
        rows_html += f'<tr style="{row_style}">'
        rows_html += (
            f'<td style="padding:10px 12px;font-weight:'
            f'{800 if is_total else 700};color:var(--text);">{row["월"]}</td>'
        )
        rows_html += (
            f'<td style="text-align:right;padding:8px;color:var(--text-2);">{int(row["N"])}</td>'
        )
        for pat in patterns:
            avg = row.get(f"{pat} 평균")
            wr = row.get(f"{pat} 승률")
            n = int(row.get(f"{pat} N", 0))
            rows_html += _color_pct(avg)
            if wr is None or pd.isna(wr):
                rows_html += '<td style="text-align:right;padding:8px;color:#9CA3AF;">—</td>'
            else:
                wr_color = "#E91E63" if wr >= 50 else "#3B82F6"
                rows_html += f'<td style="text-align:right;padding:8px;color:{wr_color};font-weight:700;">{wr:.0f}%</td>'
            rows_html += f'<td style="text-align:right;padding:8px;color:var(--text-3);font-size:11px;">{n}</td>'
        rows_html += '</tr>'

    header_cells = (
        '<th style="padding:10px 12px;text-align:left;font-size:12px;font-weight:700;color:var(--text-2);">월</th>'
        '<th style="padding:10px;text-align:right;font-size:12px;font-weight:700;color:var(--text-2);">N</th>'
    )
    for pat in patterns:
        emoji = {"돌파매매": "🚀", "눌림목매매": "📉↗", "대시세 초입": "🌊"}.get(pat, "")
        header_cells += (
            f'<th style="padding:10px 6px;text-align:right;font-size:11px;font-weight:700;color:var(--text-2);">{emoji}<br>{pat}<br><span style="font-size:9px;font-weight:500;">평균</span></th>'
            f'<th style="padding:10px 6px;text-align:right;font-size:11px;font-weight:700;color:var(--text-2);"><br><br><span style="font-size:9px;font-weight:500;">승률</span></th>'
            f'<th style="padding:10px 6px;text-align:right;font-size:11px;font-weight:700;color:var(--text-2);"><br><br><span style="font-size:9px;font-weight:500;">N</span></th>'
        )
    table_html = (
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:var(--surface-alt);">{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_monthly_pnl_html(pnl_df, position_size: int = 1_000_000):
    """월별 손익 표 HTML — 일별 시그널 표와 동일한 색상 시스템."""
    if pnl_df is None or pnl_df.empty:
        return

    UP = "#E91E63"
    DOWN = "#3B82F6"
    UP_BG = "rgba(233, 30, 99, 0.08)"
    DOWN_BG = "rgba(59, 130, 246, 0.08)"
    NEUTRAL = "#9CA3AF"

    horizons = ["D+1 종가", "20일", "60일", "90일", "120일"]

    def fmt_avg_cell(val):
        if val is None or pd.isna(val):
            return f'<td style="text-align:right;padding:8px;color:{NEUTRAL};">—</td>'
        color = UP if val > 0 else (DOWN if val < 0 else NEUTRAL)
        bg = ""
        if abs(val) >= 10:
            bg_color = UP_BG if val > 0 else DOWN_BG
            bg = f"background-color:{bg_color};"
        return f'<td style="text-align:right;padding:8px;color:{color};font-weight:700;{bg}">{val:+.2f}%</td>'

    def fmt_pnl_cell(val_pnl, val_avg):
        """손익은 평균 부호와 동일하므로 같은 강조 적용."""
        if val_pnl is None or pd.isna(val_pnl):
            return f'<td style="text-align:right;padding:8px;color:{NEUTRAL};">—</td>'
        color = UP if val_pnl > 0 else (DOWN if val_pnl < 0 else NEUTRAL)
        bg = ""
        if val_avg is not None and pd.notna(val_avg) and abs(val_avg) >= 10:
            bg_color = UP_BG if val_avg > 0 else DOWN_BG
            bg = f"background-color:{bg_color};"
        man = val_pnl / 10000
        if abs(man) >= 1:
            text = f"{man:+,.0f}만원"
        else:
            text = f"{val_pnl:+,.0f}원"
        return f'<td style="text-align:right;padding:8px;color:{color};font-weight:700;{bg}">{text}</td>'

    rows_html = ""
    for _, row in pnl_df.iterrows():
        is_total = "합계" in str(row.get("월", ""))
        row_style = (
            'border-top:2px solid var(--accent);background:var(--surface-alt);'
            if is_total else ''
        )
        rows_html += f'<tr style="{row_style}">'
        rows_html += (
            f'<td style="padding:10px 12px;font-weight:'
            f'{800 if is_total else 700};color:var(--text);">{row["월"]}</td>'
        )
        rows_html += (
            f'<td style="text-align:right;padding:8px;color:var(--text-2);">'
            f'{int(row["N"]) if pd.notna(row.get("N")) else "—"}</td>'
        )
        for h in horizons:
            avg = row.get(f"{h} 평균")
            pnl = row.get(f"{h} 손익")
            rows_html += fmt_avg_cell(avg)
            rows_html += fmt_pnl_cell(pnl, avg)
        rows_html += '</tr>'

    header_cells = (
        '<th style="padding:10px 12px;text-align:left;font-size:12px;font-weight:700;color:var(--text-2);">월</th>'
        '<th style="padding:10px;text-align:right;font-size:12px;font-weight:700;color:var(--text-2);">N</th>'
    )
    for h in horizons:
        header_cells += (
            f'<th style="padding:10px;text-align:right;font-size:11px;font-weight:700;color:var(--text-2);">{h} 평균</th>'
            f'<th style="padding:10px;text-align:right;font-size:11px;font-weight:700;color:var(--text-2);">{h} 손익</th>'
        )

    table_html = (
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:var(--surface-alt);">{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# 헬퍼: 캐시된 백테스트 trade 로드
# =============================================================================
def load_backtest_trades(preset_key: str):
    """선택된 preset의 모든 trade DataFrame 반환."""
    import pickle
    from pathlib import Path
    cache_file = Path("cache/wf_full_2020-01-01_2026-05-21_u1000.pkl")
    if not cache_file.exists():
        return pd.DataFrame()
    try:
        with open(cache_file, "rb") as f:
            all_trades = pickle.load(f)
        df = all_trades.get(preset_key, pd.DataFrame())
        if df.empty:
            return df
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["sell_date"] = pd.to_datetime(df["sell_date"])
        df["hold_days"] = (df["sell_date"] - df["Date"]).dt.days
        df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
        return df.sort_values("Date", ascending=False)
    except Exception:
        return pd.DataFrame()


def filter_trades_by_period(df, years, months):
    """선택된 년/월에 해당하는 trade만 추출."""
    if df is None or df.empty or not years or not months:
        return df if df is not None else pd.DataFrame()
    return df[
        (df["Date"].dt.year.isin(years)) & (df["Date"].dt.month.isin(months))
    ].reset_index(drop=True)


def render_daily_signals(df, position_size: int = 1_000_000):
    """
    단타용 일자별 시그널 HTML 테이블.
    - D+1 시가/종가, D+2 시가/종가만
    - 빨강 = 양수, 파랑 = 음수
    - 10%+ 손익은 배경 5% opacity 강조
    - 시장 컬럼 제거
    - 추천사유/유사사례는 오른쪽 끝
    - 가로 스크롤 없이 화면 안에 들어가게 컬럼 압축
    """
    if df is None or df.empty:
        return

    df = df.copy()
    dates = sorted(df["Date"].dt.normalize().unique(), reverse=True)
    KOREAN_DOW = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    rank_badges = {1: "🏆 대장주", 2: "🥈 2순위", 3: "🥉 3순위"}

    UP = "#E91E63"      # 빨강 (한국식 상승)
    DOWN = "#3B82F6"    # 파랑 (한국식 하락)
    UP_BG = "rgba(233, 30, 99, 0.08)"
    DOWN_BG = "rgba(59, 130, 246, 0.08)"
    NEUTRAL = "#9CA3AF"

    def fmt_ret_cell(val):
        """수익률 셀 HTML — 색상 + 10%+ 배경 강조 + 손익(만원)."""
        if val is None or pd.isna(val):
            return f'<td style="text-align:right;color:{NEUTRAL};">—</td>'
        color = UP if val > 0 else (DOWN if val < 0 else NEUTRAL)
        bg = ""
        if abs(val) >= 10:
            bg_color = UP_BG if val > 0 else DOWN_BG
            bg = f"background-color:{bg_color};"
        won = val / 100 * position_size
        if abs(won) >= 10000:
            won_str = f"{won/10000:+.0f}만"
        else:
            won_str = f"{won:+,.0f}원"
        return (
            f'<td style="text-align:right;color:{color};font-weight:700;{bg}">'
            f'{val:+.1f}%'
            f'<div style="font-size:10px;font-weight:500;opacity:0.7;">({won_str})</div>'
            f'</td>'
        )

    def fmt_pct_cell(val):
        """% 셀 — 배경 강조 포함 (직전20일 등)."""
        if val is None or pd.isna(val):
            return f'<td style="text-align:right;color:{NEUTRAL};">—</td>'
        color = UP if val > 0 else (DOWN if val < 0 else NEUTRAL)
        bg = ""
        if abs(val) >= 10:
            bg_color = UP_BG if val > 0 else DOWN_BG
            bg = f"background-color:{bg_color};"
        return f'<td style="text-align:right;padding:8px;color:{color};font-weight:700;{bg}">{val:+.2f}%</td>'

    def fmt_pct_cell_plain(val):
        """% 셀 — 색상만, 배경 강조 없음 (당일등락용)."""
        if val is None or pd.isna(val):
            return f'<td style="text-align:right;color:{NEUTRAL};">—</td>'
        color = UP if val > 0 else (DOWN if val < 0 else NEUTRAL)
        return f'<td style="text-align:right;padding:8px;color:{color};font-weight:700;">{val:+.2f}%</td>'

    for d in dates[:50]:
        day_df = df[df["Date"].dt.normalize() == d].sort_values("Rank")
        if day_df.empty:
            continue

        dow = KOREAN_DOW[d.weekday()]
        st.markdown(
            f'<div style="background:var(--accent-soft);border-radius:10px;'
            f'padding:10px 16px;margin:18px 0 8px 0;border:1px solid var(--border);">'
            f'<span style="font-weight:800;color:var(--text);">'
            f'📅 {d.strftime("%Y-%m-%d")} ({dow})</span> '
            f'<span class="muted">· {len(day_df)}종목</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # HTML 테이블 구축
        rows_html = ""
        for _, r in day_df.iterrows():
            rank = int(r.get("Rank", 0))
            rank_badge = rank_badges.get(rank, str(rank))
            similar_stock = r.get("similar_stock", "")
            similar_pct = r.get("similar_pct", 0)
            similar_html = ""
            if similar_stock and similar_pct > 0:
                similar_html = (
                    f'<div style="font-weight:700;color:var(--text);">{similar_stock}</div>'
                    f'<div style="font-size:10px;color:var(--text-3);">{similar_pct:.0f}%</div>'
                )
            else:
                similar_html = '<span style="color:var(--text-3);">—</span>'

            rows_html += "<tr>"
            rows_html += f'<td style="padding:8px;white-space:nowrap;">{rank_badge}</td>'
            rows_html += (
                f'<td style="padding:8px;">'
                f'<div style="font-weight:700;color:var(--text);">{r["Name"]}</div>'
                f'<div style="font-size:10px;color:var(--text-3);">{r["Code"]}</div>'
                f'</td>'
            )
            rows_html += f'<td style="padding:8px;text-align:right;font-weight:600;">{int(r["Close"]):,}</td>'
            # 당일은 색상만 (배경 강조 없음)
            rows_html += fmt_pct_cell_plain(r.get("ChangeRatio"))
            # D+1 OHLC 4개
            rows_html += fmt_ret_cell(r.get("ret_d1_open"))
            rows_html += fmt_ret_cell(r.get("ret_d1_high"))
            rows_html += fmt_ret_cell(r.get("ret_d1_low"))
            rows_html += fmt_ret_cell(r.get("ret_d1_close"))
            # 추천사유 + 유사사례
            rows_html += (
                f'<td style="padding:8px;font-size:11px;color:var(--text-2);max-width:160px;">'
                f'{r.get("reason", "")}'
                f'</td>'
            )
            rows_html += f'<td style="padding:8px;text-align:center;font-size:11px;">{similar_html}</td>'
            rows_html += "</tr>"

        table_html = (
            '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
            '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            '<thead><tr style="background:var(--surface-alt);">'
            '<th style="padding:10px 6px;text-align:left;font-weight:700;font-size:11px;color:var(--text-2);">순위</th>'
            '<th style="padding:10px 6px;text-align:left;font-weight:700;font-size:11px;color:var(--text-2);">종목</th>'
            '<th style="padding:10px 6px;text-align:right;font-weight:700;font-size:11px;color:var(--text-2);">매수가</th>'
            '<th style="padding:10px 6px;text-align:right;font-weight:700;font-size:11px;color:var(--text-2);">당일</th>'
            '<th style="padding:10px 6px;text-align:right;font-weight:700;font-size:11px;color:var(--text-2);">D+1 시</th>'
            '<th style="padding:10px 6px;text-align:right;font-weight:700;font-size:11px;color:var(--text-2);">D+1 고</th>'
            '<th style="padding:10px 6px;text-align:right;font-weight:700;font-size:11px;color:var(--text-2);">D+1 저</th>'
            '<th style="padding:10px 6px;text-align:right;font-weight:700;font-size:11px;color:var(--text-2);">D+1 종</th>'
            '<th style="padding:10px 6px;text-align:left;font-weight:700;font-size:11px;color:var(--text-2);">추천사유</th>'
            '<th style="padding:10px 6px;text-align:center;font-weight:700;font-size:11px;color:var(--text-2);">유사사례</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# V/S/A/B 일자별 + 월별 손익
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_vsab_history():
    """전체 enriched 통합 → 일자별 V/S/A/B 분류 결과."""
    df = build_ensemble_all_enriched()
    if df.empty:
        return pd.DataFrame()
    df["grade"] = df.apply(lambda r: classify_one(r.to_dict()), axis=1)
    return df


def _vsab_pick_daily(df_graded: pd.DataFrame, vs_max: int = 10,
                       a_max: int = 10, b_max: int = 5) -> pd.DataFrame:
    """일자별 V/S/A/B 픽 (모든 종목 표시 모드)."""
    df = df_graded[df_graded["grade"].notna()].copy()
    grade_pri = {"V": 4, "S": 3, "A": 2, "B": 1}
    daily_picks = []
    df_sorted = df.sort_values(["Date", "Code"]).copy()
    for date, day_df in df_sorted.groupby("Date"):
        day_df = day_df.copy()
        day_df["_pri"] = day_df["grade"].map(grade_pri).fillna(0)
        day_df = day_df.sort_values("_pri", ascending=False).drop_duplicates("Code", keep="first")
        v = day_df[day_df["grade"] == "V"].nlargest(vs_max, "avg_score")
        s = day_df[day_df["grade"] == "S"].nlargest(vs_max, "avg_score")
        used = set(v["Code"]).union(set(s["Code"]))
        a = day_df[(day_df["grade"] == "A") & (~day_df["Code"].isin(used))].nlargest(a_max, "avg_score")
        used.update(a["Code"])
        b = day_df[(day_df["grade"] == "B") & (~day_df["Code"].isin(used))].nlargest(b_max, "avg_score")
        daily_picks.append(pd.concat([v, s, a, b]))
    if not daily_picks:
        return pd.DataFrame()
    return pd.concat(daily_picks, ignore_index=True)


def render_vsab_monthly_pnl(df_graded: pd.DataFrame, ret_col: str = "ret_180d"):
    """월별 손익 표 — V/S/A/B 등급별 손익 + 합계."""
    if df_graded.empty:
        st.caption("데이터 없음")
        return

    df = df_graded[df_graded["grade"].notna()].copy()
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # 일자별 → 등급별 픽 (모두 표시 모드)
    picks_df = _vsab_pick_daily(df)
    if picks_df.empty:
        st.caption("선택된 픽 없음")
        return
    picks_df["YearMonth"] = picks_df["Date"].dt.to_period("M").astype(str)

    # 비중 부여
    picks_df["weight"] = picks_df["grade"].map(GRADE_WEIGHTS)
    # 손익 = ret / 100 * weight
    if ret_col not in picks_df.columns:
        # ret_180d 없으면 ret_120d로 fallback
        if "ret_120d" in picks_df.columns:
            ret_col = "ret_120d"
        else:
            st.caption(f"{ret_col} 없음")
            return
    picks_df["pnl"] = picks_df[ret_col].fillna(0) / 100 * picks_df["weight"]

    # 월별 집계
    monthly = picks_df.groupby(["YearMonth", "grade"]).agg(
        n=("Code", "count"),
        total_pnl=("pnl", "sum"),
        avg_ret=(ret_col, "mean"),
    ).reset_index()

    months = sorted(picks_df["YearMonth"].unique())
    # HTML 테이블
    UP = "#FF3B30"; DOWN = "#0066FF"; NEU = "var(--text)"

    def fmt_pnl(v):
        if v is None or pd.isna(v) or v == 0: return f'<td style="text-align:right;color:var(--text-3);">—</td>'
        color = UP if v > 0 else DOWN
        if abs(v) >= 1e8: s = f"{v/1e8:+,.2f}억"
        elif abs(v) >= 1e4: s = f"{v/1e4:+,.0f}만"
        else: s = f"{v:+,.0f}원"
        return f'<td style="text-align:right;color:{color};font-weight:700;">{s}</td>'

    rows = ""
    cumul = 0
    for m in months:
        sub = monthly[monthly["YearMonth"] == m]
        v_pnl = sub[sub["grade"] == "V"]["total_pnl"].sum()
        s_pnl = sub[sub["grade"] == "S"]["total_pnl"].sum()
        a_pnl = sub[sub["grade"] == "A"]["total_pnl"].sum()
        b_pnl = sub[sub["grade"] == "B"]["total_pnl"].sum()
        tot = v_pnl + s_pnl + a_pnl + b_pnl
        cumul += tot
        v_n = int(sub[sub["grade"] == "V"]["n"].sum())
        s_n = int(sub[sub["grade"] == "S"]["n"].sum())
        a_n = int(sub[sub["grade"] == "A"]["n"].sum())
        b_n = int(sub[sub["grade"] == "B"]["n"].sum())
        rows += f"<tr><td style='padding:8px;font-weight:700;'>{m}</td>"
        rows += f"<td style='text-align:center;'>{v_n}</td>"
        rows += fmt_pnl(v_pnl)
        rows += f"<td style='text-align:center;'>{s_n}</td>"
        rows += fmt_pnl(s_pnl)
        rows += f"<td style='text-align:center;'>{a_n}</td>"
        rows += fmt_pnl(a_pnl)
        rows += f"<td style='text-align:center;'>{b_n}</td>"
        rows += fmt_pnl(b_pnl)
        rows += fmt_pnl(tot)
        rows += fmt_pnl(cumul)
        rows += "</tr>"

    table_html = (
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:10px;text-align:left;font-weight:700;">월</th>'
        '<th style="padding:10px;text-align:center;">🏆 V건</th>'
        '<th style="padding:10px;text-align:right;">🏆 V 손익</th>'
        '<th style="padding:10px;text-align:center;">💎 S건</th>'
        '<th style="padding:10px;text-align:right;">💎 S 손익</th>'
        '<th style="padding:10px;text-align:center;">⭐ A건</th>'
        '<th style="padding:10px;text-align:right;">⭐ A 손익</th>'
        '<th style="padding:10px;text-align:center;">🟢 B건</th>'
        '<th style="padding:10px;text-align:right;">🟢 B 손익</th>'
        '<th style="padding:10px;text-align:right;font-weight:800;">월 합계</th>'
        '<th style="padding:10px;text-align:right;font-weight:800;">누적</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_vsab_daily_signals(df_graded: pd.DataFrame, max_days: int = 50,
                                ret_col: str = "ret_180d",
                                sort_mode: str = "newest"):
    """일자별 V/S/A/B 등급별 종목 리스트. 모든 등급의 모든 종목 표시.

    sort_mode: 'newest' / 'oldest' / 'return_desc' / 'return_asc' / 'score_desc'
    """
    if df_graded.empty:
        st.caption("데이터 없음")
        return

    df = df_graded[df_graded["grade"].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if ret_col not in df.columns:
        # fallback
        if "ret_120d" in df.columns:
            ret_col = "ret_120d"

    # 정렬
    if sort_mode == "newest":
        dates = sorted(df["Date"].unique(), reverse=True)
    elif sort_mode == "oldest":
        dates = sorted(df["Date"].unique())
    elif sort_mode in ("return_desc", "return_asc", "score_desc"):
        # 거래별 정렬: 모든 픽을 한 표로
        return _render_vsab_flat_table(df, ret_col, sort_mode, max_days)
    else:
        dates = sorted(df["Date"].unique(), reverse=True)

    KOREAN_DOW = ["월", "화", "수", "목", "금", "토", "일"]
    grade_pri = {"V": 4, "S": 3, "A": 2, "B": 1}

    shown = 0
    for d in dates:
        if shown >= max_days: break
        day_df = df[df["Date"] == d].copy()
        day_df["_pri"] = day_df["grade"].map(grade_pri).fillna(0)
        day_df = day_df.sort_values("_pri", ascending=False).drop_duplicates("Code", keep="first")
        v = day_df[day_df["grade"] == "V"].nlargest(10, "avg_score")
        s = day_df[day_df["grade"] == "S"].nlargest(10, "avg_score")
        used = set(v["Code"]).union(set(s["Code"]))
        a = day_df[(day_df["grade"] == "A") & (~day_df["Code"].isin(used))].nlargest(10, "avg_score")
        used.update(a["Code"])
        b = day_df[(day_df["grade"] == "B") & (~day_df["Code"].isin(used))].nlargest(5, "avg_score")
        picks = pd.concat([v, s, a, b])
        if picks.empty: continue

        dow = KOREAN_DOW[pd.Timestamp(d).weekday()]
        st.markdown(
            f'<div style="background:var(--accent-soft);border-radius:10px;'
            f'padding:10px 16px;margin:18px 0 8px 0;border:1px solid var(--border);">'
            f'<span style="font-weight:800;color:var(--text);">'
            f'📅 {pd.Timestamp(d).strftime("%Y-%m-%d")} ({dow})</span> '
            f'<span class="muted">· V {len(v)} · S {len(s)} · A {len(a)} · B {len(b)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        rows_html = ""
        UP = "#FF3B30"; DOWN = "#0066FF"
        for grade in ["V", "S", "A", "B"]:
            sub = picks[picks["grade"] == grade]
            info = GRADE_INFO[grade]
            if sub.empty:
                rows_html += (
                    f'<tr style="opacity:0.4;">'
                    f'<td style="padding:8px;font-weight:700;color:{info["color"]};">{info["emoji"]} {grade}</td>'
                    f'<td colspan="6" style="padding:8px;color:var(--text-3);">없음</td>'
                    f'</tr>'
                )
                continue
            for _, r in sub.iterrows():
                ret = r.get(ret_col, 0) or 0
                ret_color = UP if ret > 0 else (DOWN if ret < 0 else "var(--text)")
                rows_html += "<tr>"
                rows_html += (
                    f'<td style="padding:8px;font-weight:800;color:{info["color"]};">'
                    f'{info["emoji"]} {grade}</td>'
                )
                rows_html += (
                    f'<td style="padding:8px;">'
                    f'<div style="font-weight:700;">{r["Name"]}</div>'
                    f'<div style="font-size:10px;color:var(--text-3);">{r["Code"]}</div>'
                    f'</td>'
                )
                rows_html += f'<td style="padding:8px;text-align:right;">{int(r["Close"]):,}원</td>'
                rows_html += f'<td style="padding:8px;text-align:right;color:{UP if r["ChangeRatio"] > 0 else DOWN};font-weight:700;">{r["ChangeRatio"]:+.2f}%</td>'
                rows_html += f'<td style="padding:8px;text-align:center;">{r["avg_score"]:.1f}</td>'
                rows_html += f'<td style="padding:8px;text-align:center;">{int(r["n_presets"])}/4</td>'
                if pd.notna(ret) and ret != 0:
                    rows_html += f'<td style="padding:8px;text-align:right;color:{ret_color};font-weight:700;">{ret:+.1f}%</td>'
                else:
                    rows_html += f'<td style="padding:8px;text-align:right;color:var(--text-3);">—</td>'
                rows_html += "</tr>"

        ret_label = "180일" if ret_col == "ret_180d" else ret_col.replace("ret_", "").replace("d", "일")
        st.markdown(
            '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
            '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            '<thead><tr style="background:var(--surface-alt);">'
            '<th style="padding:10px;text-align:left;">등급</th>'
            '<th style="padding:10px;text-align:left;">종목</th>'
            '<th style="padding:10px;text-align:right;">매수가</th>'
            '<th style="padding:10px;text-align:right;">당일</th>'
            '<th style="padding:10px;text-align:center;">점수</th>'
            '<th style="padding:10px;text-align:center;">프리셋</th>'
            f'<th style="padding:10px;text-align:right;">{ret_label}수익</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )
        shown += 1


def _render_vsab_flat_table(df: pd.DataFrame, ret_col: str, sort_mode: str, max_rows: int):
    """수익률/점수순으로 평면 정렬한 표."""
    picks = _vsab_pick_daily(df)
    if picks.empty:
        st.caption("선택된 픽 없음")
        return
    if ret_col not in picks.columns:
        ret_col = "ret_120d"

    if sort_mode == "return_desc":
        picks = picks.sort_values(ret_col, ascending=False, na_position="last")
    elif sort_mode == "return_asc":
        picks = picks.sort_values(ret_col, ascending=True, na_position="last")
    elif sort_mode == "score_desc":
        picks = picks.sort_values("avg_score", ascending=False)

    picks = picks.head(max_rows * 10)  # 일자별 평균 4건 가정
    rows_html = ""
    UP = "#FF3B30"; DOWN = "#0066FF"
    for _, r in picks.iterrows():
        grade = r["grade"]
        info = GRADE_INFO[grade]
        ret = r.get(ret_col, 0) or 0
        ret_color = UP if ret > 0 else (DOWN if ret < 0 else "var(--text)")
        date_str = pd.Timestamp(r["Date"]).strftime("%Y-%m-%d")
        rows_html += "<tr>"
        rows_html += f'<td style="padding:8px;font-weight:700;">{date_str}</td>'
        rows_html += f'<td style="padding:8px;font-weight:800;color:{info["color"]};">{info["emoji"]} {grade}</td>'
        rows_html += (
            f'<td style="padding:8px;">'
            f'<div style="font-weight:700;">{r["Name"]}</div>'
            f'<div style="font-size:10px;color:var(--text-3);">{r["Code"]}</div>'
            f'</td>'
        )
        rows_html += f'<td style="padding:8px;text-align:right;">{int(r["Close"]):,}원</td>'
        rows_html += f'<td style="padding:8px;text-align:right;color:{UP if r["ChangeRatio"] > 0 else DOWN};font-weight:700;">{r["ChangeRatio"]:+.2f}%</td>'
        rows_html += f'<td style="padding:8px;text-align:center;">{r["avg_score"]:.1f}</td>'
        if pd.notna(ret) and ret != 0:
            rows_html += f'<td style="padding:8px;text-align:right;color:{ret_color};font-weight:700;">{ret:+.1f}%</td>'
        else:
            rows_html += f'<td style="padding:8px;text-align:right;color:var(--text-3);">—</td>'
        rows_html += "</tr>"

    ret_label = "180일" if ret_col == "ret_180d" else ret_col.replace("ret_", "").replace("d", "일")
    st.markdown(
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:10px;text-align:left;">날짜</th>'
        '<th style="padding:10px;text-align:left;">등급</th>'
        '<th style="padding:10px;text-align:left;">종목</th>'
        '<th style="padding:10px;text-align:right;">매수가</th>'
        '<th style="padding:10px;text-align:right;">당일</th>'
        '<th style="padding:10px;text-align:center;">점수</th>'
        f'<th style="padding:10px;text-align:right;">{ret_label}수익</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _render_vsab_summary(df_graded: pd.DataFrame, ret_col: str = "ret_180d"):
    """선택 기간 통계 요약 — 등급별 / 전체. 모든 종목 표시 모드."""
    if df_graded.empty:
        return
    df = df_graded[df_graded["grade"].notna()].copy()

    # 일자별 버킷 빌드 (모든 등급 모두 표시)
    pdf = _vsab_pick_daily(df)
    if pdf.empty:
        return
    if ret_col not in pdf.columns:
        ret_col = "ret_120d" if "ret_120d" in pdf.columns else None
        if ret_col is None: return
    pdf["weight"] = pdf["grade"].map(GRADE_WEIGHTS)
    pdf["pnl"] = pdf[ret_col].fillna(0) / 100 * pdf["weight"]

    # 등급별 통계 카드
    UP = "#FF3B30"; DOWN = "#0066FF"
    rows = ""
    total_n = 0; total_pnl = 0
    for g in ["V", "S", "A", "B"]:
        info = GRADE_INFO[g]
        sub = pdf[pdf["grade"] == g]
        n = len(sub)
        total_n += n
        if n == 0:
            rows += (
                f'<tr style="opacity:0.4;">'
                f'<td style="padding:10px;font-weight:800;color:{info["color"]};">{info["emoji"]} {g}급</td>'
                f'<td colspan="7" style="padding:10px;color:var(--text-3);">없음</td>'
                f'</tr>'
            )
            continue
        rets = sub[ret_col].dropna()
        avg = rets.mean() if len(rets) > 0 else 0
        wr = (rets > 0).mean() * 100 if len(rets) > 0 else 0
        big_win = (rets >= 50).mean() * 100 if len(rets) > 0 else 0
        big_loss = (rets <= -30).mean() * 100 if len(rets) > 0 else 0
        pnl = sub["pnl"].sum()
        total_pnl += pnl
        invest = n * GRADE_WEIGHTS[g]
        roi = pnl / invest * 100 if invest > 0 else 0
        avg_color = UP if avg > 0 else DOWN
        pnl_color = UP if pnl > 0 else DOWN

        def fmt_pnl(v):
            if abs(v) >= 1e8: return f"{v/1e8:+,.2f}억"
            if abs(v) >= 1e4: return f"{v/1e4:+,.0f}만"
            return f"{v:+,.0f}원"

        rows += "<tr>"
        rows += f'<td style="padding:10px;font-weight:800;color:{info["color"]};">{info["emoji"]} {g}급</td>'
        rows += f'<td style="padding:10px;text-align:center;">{n}</td>'
        rows += f'<td style="padding:10px;text-align:right;color:{avg_color};font-weight:700;">{avg:+.2f}%</td>'
        rows += f'<td style="padding:10px;text-align:center;">{wr:.1f}%</td>'
        rows += f'<td style="padding:10px;text-align:right;color:{UP};font-weight:700;">{big_win:.1f}%</td>'
        rows += f'<td style="padding:10px;text-align:right;color:{DOWN};font-weight:700;">{big_loss:.1f}%</td>'
        rows += f'<td style="padding:10px;text-align:right;color:{pnl_color};font-weight:700;">{fmt_pnl(pnl)}</td>'
        rows += f'<td style="padding:10px;text-align:right;font-weight:700;">{roi:+.1f}%</td>'
        rows += "</tr>"

    # 합계
    avg_overall = pdf[ret_col].dropna().mean() if not pdf[ret_col].dropna().empty else 0
    rows += "<tr style='border-top:2px solid var(--border);background:var(--surface-alt);'>"
    rows += f'<td style="padding:12px;font-weight:800;">전체</td>'
    rows += f'<td style="padding:12px;text-align:center;font-weight:800;">{total_n}</td>'
    rows += f'<td style="padding:12px;text-align:right;font-weight:800;color:{UP if avg_overall > 0 else DOWN};">{avg_overall:+.2f}%</td>'
    rows += f'<td style="padding:12px;"></td><td style="padding:12px;"></td><td style="padding:12px;"></td>'
    def fmt_pnl(v):
        if abs(v) >= 1e8: return f"{v/1e8:+,.2f}억"
        if abs(v) >= 1e4: return f"{v/1e4:+,.0f}만"
        return f"{v:+,.0f}원"
    rows += f'<td style="padding:12px;text-align:right;font-weight:800;color:{UP if total_pnl > 0 else DOWN};">{fmt_pnl(total_pnl)}</td>'
    rows += "<td style='padding:12px;'></td>"
    rows += "</tr>"

    st.markdown(
        '<div style="overflow-x:auto;border:1px solid var(--border);border-radius:10px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:var(--surface-alt);">'
        '<th style="padding:10px;text-align:left;">등급</th>'
        '<th style="padding:10px;text-align:center;">건수</th>'
        '<th style="padding:10px;text-align:right;">평균수익</th>'
        '<th style="padding:10px;text-align:center;">승률</th>'
        '<th style="padding:10px;text-align:right;">+50%↑</th>'
        '<th style="padding:10px;text-align:right;">-30%↓</th>'
        '<th style="padding:10px;text-align:right;">누적 손익</th>'
        '<th style="padding:10px;text-align:right;">ROI</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # CSV 다운로드 — 전체 종목 리스트
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    pdf_export = pdf.copy()
    pdf_export["매수일"] = pdf_export["Date"].dt.strftime("%Y-%m-%d")
    cols_export = ["매수일", "grade", "Name", "Code", "Market",
                    "Close", "ChangeRatio", "avg_score", "n_presets",
                    "weight", ret_col, "pnl"]
    cols_export = [c for c in cols_export if c in pdf_export.columns]
    ret_label_kr = "180일수익률" if ret_col == "ret_180d" else f"{ret_col.replace('ret_','').replace('d','일')}수익률"
    rename = {"grade": "등급", "Name": "종목명", "Code": "코드", "Market": "시장",
                "Close": "매수가", "ChangeRatio": "당일등락",
                "avg_score": "앙상블점수", "n_presets": "추천프리셋수",
                "weight": "매수금액", ret_col: ret_label_kr, "pnl": "손익"}
    pdf_export = pdf_export[cols_export].rename(columns=rename)
    csv = pdf_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 선택 기간 종목 전체 CSV", csv,
        file_name=f"VSAB백테스트_{datetime.now():%Y%m%d}.csv",
        use_container_width=True, key="vsab_csv")


# =============================================================================
# 페이지: 백테스트 결과
# =============================================================================
def page_results():
    p = PALETTE[st.session_state.theme]
    st.markdown(
        '<h1>백테스트 결과 — V/S/A/B 등급제 (OOS 검증)</h1>'
        f'<p style="color:{p["text_secondary"]};">'
        '4 프리셋 앙상블 등급제 — 코스닥 + 돌파매매 + 등락 7~25% · 120일 보유 손익 · '
        '미래 데이터 미사용 (Out-of-Sample).</p>',
        unsafe_allow_html=True,
    )

    # V/S/A/B 일자별 추천 + 월별 손익
    with st.spinner("V/S/A/B 등급 일별 데이터 로딩 중..."):
        df_graded = _load_vsab_history()

    if df_graded.empty:
        st.warning("enriched 캐시 데이터가 없습니다. `python3 precompute_enriched.py` 실행 필요.")
        return

    # ============================================================
    # 년/월 선택기 — V/S/A/B 데이터에 직접 적용
    # ============================================================
    df_graded["Date"] = pd.to_datetime(df_graded["Date"])
    available_years = sorted(df_graded["Date"].dt.year.unique())
    available_months = list(range(1, 13))

    # 세션 디폴트 초기화
    if "vsab_years" not in st.session_state:
        st.session_state.vsab_years = list(available_years)
    if "vsab_months" not in st.session_state:
        st.session_state.vsab_months = list(range(1, 13))
    if "vsab_applied_years" not in st.session_state:
        st.session_state.vsab_applied_years = list(available_years)
    if "vsab_applied_months" not in st.session_state:
        st.session_state.vsab_applied_months = list(range(1, 13))
    if "vsab_sort" not in st.session_state:
        st.session_state.vsab_sort = "newest"
    if "vsab_hold" not in st.session_state:
        # 180일이 있으면 기본 180, 없으면 120
        st.session_state.vsab_hold = "180" if "ret_180d" in df_graded.columns else "120"

    st.markdown('<h3 style="margin-bottom:8px;">📅 기간 선택</h3>', unsafe_allow_html=True)
    st.caption("선택한 년/월의 V/S/A/B 추천만 표시 — 다중 선택 가능")

    # 빠른 선택
    qa = st.columns(5)
    if qa[0].button("전체", key="vsab_qa_all", use_container_width=True):
        st.session_state.vsab_years = list(available_years)
        st.session_state.vsab_months = list(range(1, 13))
        st.rerun()
    if qa[1].button("최근 3년", key="vsab_qa_r3", use_container_width=True):
        st.session_state.vsab_years = available_years[-3:]
        st.session_state.vsab_months = list(range(1, 13))
        st.rerun()
    if qa[2].button("최근 1년", key="vsab_qa_r1", use_container_width=True):
        st.session_state.vsab_years = available_years[-1:]
        st.session_state.vsab_months = list(range(1, 13))
        st.rerun()
    if qa[3].button("강세장만", key="vsab_qa_bull", use_container_width=True,
                     help="2020 + 2025 (강세장)"):
        st.session_state.vsab_years = [y for y in [2020, 2025] if y in available_years]
        st.session_state.vsab_months = list(range(1, 13))
        st.rerun()
    if qa[4].button("해제", key="vsab_qa_clr", use_container_width=True):
        st.session_state.vsab_years = []
        st.session_state.vsab_months = []
        st.rerun()

    # 년도 버튼
    st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;'
                 'margin-top:14px;margin-bottom:6px;">년도</div>',
                 unsafe_allow_html=True)
    yr_cols = st.columns(len(available_years))
    for i, year in enumerate(available_years):
        selected = year in st.session_state.vsab_years
        btn_type = "primary" if selected else "secondary"
        if yr_cols[i].button(f"{year}", key=f"vsab_yr_{year}",
                              use_container_width=True, type=btn_type):
            if year in st.session_state.vsab_years:
                st.session_state.vsab_years.remove(year)
            else:
                st.session_state.vsab_years.append(year)
                st.session_state.vsab_years.sort()
            st.rerun()

    # 월 버튼 (6x2)
    st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;'
                 'margin-top:14px;margin-bottom:6px;">월</div>',
                 unsafe_allow_html=True)
    for row in range(2):
        m_cols = st.columns(6)
        for i in range(6):
            m = row * 6 + i + 1
            selected = m in st.session_state.vsab_months
            btn_type = "primary" if selected else "secondary"
            if m_cols[i].button(f"{m}월", key=f"vsab_m_{m}",
                                  use_container_width=True, type=btn_type):
                if m in st.session_state.vsab_months:
                    st.session_state.vsab_months.remove(m)
                else:
                    st.session_state.vsab_months.append(m)
                    st.session_state.vsab_months.sort()
                st.rerun()

    # ============================================================
    # 보유기간 + 정렬 + 필터 적용 버튼
    # ============================================================
    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    hold_opts = ["180", "120", "90", "60", "30", "20"]
    if "ret_180d" not in df_graded.columns:
        hold_opts = ["120", "90", "60", "30", "20"]
    sort_opts = {
        "newest": "📅 최신순",
        "oldest": "📅 오래된순",
        "return_desc": "🔴 수익률 높은순",
        "return_asc": "🔵 수익률 낮은순",
        "score_desc": "⭐ 점수 높은순",
    }
    cc1, cc2, cc3 = st.columns([1, 2, 2])
    with cc1:
        st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;'
                     'margin-bottom:6px;">보유기간</div>',
                     unsafe_allow_html=True)
        hold = st.selectbox("보유", hold_opts,
                              index=hold_opts.index(st.session_state.vsab_hold) if st.session_state.vsab_hold in hold_opts else 0,
                              label_visibility="collapsed", key="vsab_hold_select")
        st.session_state.vsab_hold = hold
    with cc2:
        st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;'
                     'margin-bottom:6px;">정렬</div>',
                     unsafe_allow_html=True)
        sort_key = st.selectbox("정렬", list(sort_opts.keys()),
                                  format_func=lambda k: sort_opts[k],
                                  index=list(sort_opts.keys()).index(st.session_state.vsab_sort),
                                  label_visibility="collapsed", key="vsab_sort_select")
        st.session_state.vsab_sort = sort_key
    with cc3:
        st.markdown('<div style="font-size:13px;color:var(--text-2);font-weight:700;'
                     'margin-bottom:6px;">&nbsp;</div>',
                     unsafe_allow_html=True)
        if st.button("✅ 필터 적용하기", type="primary", use_container_width=True,
                       key="vsab_apply_filter"):
            st.session_state.vsab_applied_years = list(st.session_state.vsab_years)
            st.session_state.vsab_applied_months = list(st.session_state.vsab_months)
            st.rerun()

    # 실제 적용된 값 사용
    sel_years = st.session_state.vsab_applied_years
    sel_months = st.session_state.vsab_applied_months
    if not sel_years or not sel_months:
        st.warning("⚠️ 년도와 월을 최소 1개씩 선택 후 [✅ 필터 적용하기] 버튼을 눌러주세요.")
        return

    df_filtered = df_graded[
        df_graded["Date"].dt.year.isin(sel_years) &
        df_graded["Date"].dt.month.isin(sel_months)
    ].copy()

    yrs_str = ", ".join(str(y) for y in sel_years)
    mons_str = ", ".join(str(m) for m in sel_months)
    st.markdown(
        f'<div style="background:var(--accent-soft);border-radius:10px;'
        f'padding:10px 16px;margin:14px 0 24px 0;border:1px solid var(--border);">'
        f'<span style="font-weight:700;">📊 선택: {yrs_str} 년 / {mons_str} 월</span> '
        f'<span class="muted">· {len(df_filtered):,}건 후보</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if df_filtered.empty:
        st.info("선택된 기간에 V/S/A/B 후보가 없습니다.")
    else:
        ret_col = f"ret_{st.session_state.vsab_hold}d"
        hold_label = st.session_state.vsab_hold

        # 월별 손익
        st.markdown(f"<h2>💰 V/S/A/B 등급별 월별 손익 ({hold_label}일 보유)</h2>",
                     unsafe_allow_html=True)
        st.caption("등급별 비중 적용 (V:50만/S:30만/A:20만/B:10만) · 이상치 미제거 · "
                    "모든 등급 매일 다 추천 종목 합산")
        render_vsab_monthly_pnl(df_filtered, ret_col=ret_col)

        # 등급별 종목 리스트 — 전체 보이기
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        sort_kr = {"newest":"최신순","oldest":"오래된순","return_desc":"수익률↓",
                    "return_asc":"수익률↑","score_desc":"점수↓"}[st.session_state.vsab_sort]
        st.markdown(f"<h2>📋 일자별 등급별 추천 종목 — {sort_kr}</h2>",
                     unsafe_allow_html=True)
        st.caption(f"V/S/A/B 모두 매일 조건 만족 종목 다 표시 (없는 등급은 '없음' 표시) · "
                    f"{hold_label}일 수익률 기준")
        render_vsab_daily_signals(df_filtered, max_days=10000,
                                    ret_col=ret_col,
                                    sort_mode=st.session_state.vsab_sort)

        # 통계 요약
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<h2>📊 선택 기간 통계 요약 ({hold_label}일 기준)</h2>",
                     unsafe_allow_html=True)
        _render_vsab_summary(df_filtered, ret_col=ret_col)

    st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    show_detail = st.toggle("🔧 [참고] 전체 OOS 검증 + 9 프리셋 백테스트 상세 보기",
                              value=False, key="show_detail_backtest")
    if not show_detail:
        return
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)


    # 인라인 설정 (기간 포함)
    inline_settings(show_period=True)

    cached = get_cached_walk_forward()
    if not cached or "consensus_top3" not in cached:
        st.markdown(
            f'<div class="empty-state"><div class="emoji">⏳</div>'
            f'<p>백테스트 결과가 아직 없습니다.</p>'
            f'<p>백그라운드에서 자동 실행 중입니다 (30~60분 소요).</p>'
            f'<p>완료되면 여기에 자동으로 표시됩니다.</p></div>',
            unsafe_allow_html=True,
        )
        return

    # 기간 필터링 적용
    period_active = st.session_state.get("period_filter_active", False)
    sel_years = st.session_state.history_years
    sel_months = st.session_state.history_months

    if period_active and sel_years and sel_months:
        # 결정 시점 중 선택된 년/월에 속하는 것만 필터링
        filtered_dps = []
        for dp in cached.get("decision_points", []):
            test_period = dp.get("test_period", "")
            try:
                test_start = pd.to_datetime(test_period.split("~")[0].strip())
                if test_start.year in sel_years and test_start.month in sel_months:
                    filtered_dps.append(dp)
            except Exception:
                pass

        if filtered_dps:
            # 필터링된 결정 시점들로 재집계
            n_dp = len(filtered_dps)
            from collections import defaultdict
            vote_count = defaultdict(int)
            preset_returns = defaultdict(list)
            for dp in filtered_dps:
                for k in dp.get("top3", []):
                    vote_count[k] += 1
                for k, perf in dp.get("oos_chunk_perf", {}).items():
                    avg = perf.get("avg", 0)
                    n_in_chunk = perf.get("n", 0)
                    if n_in_chunk > 0:
                        preset_returns[k].extend([avg] * n_in_chunk)

            # 재계산된 consensus
            recomputed = []
            for k, votes in vote_count.items():
                rets = preset_returns.get(k, [])
                if not rets:
                    continue
                import numpy as np
                arr = np.array(rets)
                recomputed.append({
                    "preset_key": k,
                    "preset_name": PRESETS.get(k, {}).get("name", k),
                    "preset_desc": PRESETS.get(k, {}).get("desc", ""),
                    "votes_in_top3": votes,
                    "n_trades": len(rets),
                    "avg_return": float(arr.mean()),
                    "win_rate": float((arr > 0).mean() * 100),
                    "sharpe_annual": float(arr.mean() / arr.std() * (252 ** 0.5)) if arr.std() > 0 else 0,
                })
            recomputed.sort(key=lambda x: (x["votes_in_top3"], x["sharpe_annual"]), reverse=True)
            cached = dict(cached)  # copy
            cached["consensus_top3"] = recomputed[:3]
            cached["n_decision_points"] = n_dp
            cached["per_preset_oos"] = {r["preset_key"]: r for r in recomputed}

            period_label = f"필터: {n_dp}회 결정시점 (선택 {len(sel_years)}년 × {len(sel_months)}월)"
        else:
            st.warning("선택된 기간에 결정 시점이 없습니다. 다른 년/월을 선택해보세요.")
            period_label = "필터: 결정시점 없음"
    else:
        period_label = "전체 기간"
        n_dp = cached["n_decision_points"]

    stab = cached.get("stability_score", 0)
    c_start = cached["period"][0]
    c_end = cached["period"][1]

    if period_active:
        if st.button("필터 해제 (전체 기간 보기)", type="secondary",
                      use_container_width=True, key="clear_period"):
            st.session_state["period_filter_active"] = False
            st.rerun()

    # ============================================================
    # 메인 — 신규 디자인 (참고 화면 기반)
    # ============================================================
    cur_preset = st.session_state.preset
    cur_preset_name = PRESETS.get(cur_preset, {}).get("name", cur_preset)
    position_size = st.session_state.get("position_size", 1_000_000)

    # 최종 설정 요약 박스
    sel_years_txt = ", ".join(str(y) for y in sel_years) if sel_years else "—"
    sel_months_txt = ", ".join(str(m) for m in sel_months) if sel_months else "—"
    st.markdown(
        f'<div style="background:var(--surface);border:1px solid var(--border);'
        f'border-radius:12px;padding:18px 22px;margin-top:20px;">'
        f'<div style="font-weight:800;font-size:15px;margin-bottom:10px;">📊 최종 설정 요약</div>'
        f'<div style="font-size:13px;line-height:1.9;">'
        f'• 전략: <b>{cur_preset_name}</b><br>'
        f'• 연도: [{sel_years_txt}] · 월: [{sel_months_txt}]<br>'
        f'• 일별 추천: <b>{st.session_state.top_n}개</b> · '
        f'매수금: <b>{position_size/10000:,.0f}만원</b><br>'
        f'• 필터: 거래대금≥{st.session_state.min_amount_eok}억 · '
        f'시총≥{st.session_state.min_marcap_eok}억 · '
        f'당일등락 {st.session_state.change_min:.0f}~{st.session_state.change_max:.0f}%'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ───────── 큰 보라색 액션 버튼 ─────────
    st.markdown(
        '<style>'
        '.signal-action-btn button { '
        'background-color: #4F46E5 !important; '
        'color: #FFFFFF !important; '
        'padding: 22px 24px !important; '
        'font-size: 17px !important; '
        '} '
        '.signal-action-btn button:hover { '
        'background-color: #4338CA !important; '
        '}'
        '</style>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="signal-action-btn">', unsafe_allow_html=True)
    do_query = st.button(
        "🔍 시그널 조회 — 위 설정으로 가져오기",
        type="primary", use_container_width=True, key="signal_query",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 데이터 로드 (버튼 누른 후 or 캐시된 결과 사용)
    if cur_preset.startswith("ai_optimized"):
        st.warning(
            "AI 최적화 프리셋은 가중치 탐색 결과이므로 일별 trade 데이터가 없습니다. "
            "[오늘의 종가매수 추천] 페이지에서 [지금 스캔하기]로 직접 적용해보세요."
        )
        trades_df = pd.DataFrame()
        n_total_raw = 0
        n_filtered = 0
    else:
        # 1) parquet 사전 캐시 우선 로드 (precompute_enriched.py 결과)
        from pathlib import Path
        parquet_cache = Path(f"cache/enriched_{cur_preset}.parquet")
        ss_key = f"enriched_full_{cur_preset}"

        if ss_key not in st.session_state or do_query:
            if parquet_cache.exists():
                with st.spinner("캐시 로드 중..."):
                    df_full = pd.read_parquet(parquet_cache)
                    df_full["Date"] = pd.to_datetime(df_full["Date"])
                    if "sell_date" in df_full.columns:
                        df_full["sell_date"] = pd.to_datetime(df_full["sell_date"])
                    st.session_state[ss_key] = df_full
            else:
                # fallback — 즉석 enrich (느릴 수 있음)
                st.warning(
                    "⏳ 사전 캐시 없음. 터미널에서 `python3 precompute_enriched.py` 실행 권장.\n"
                    "즉석 계산 시도 중..."
                )
                trades_df_raw = load_backtest_trades(cur_preset)
                with st.spinner(f"forward return 계산 중 ({len(trades_df_raw)}건, 1~3분)..."):
                    st.session_state[ss_key] = enrich_trades(trades_df_raw, with_forward=True)

        full_enriched = st.session_state.get(ss_key, pd.DataFrame())
        n_total_raw = len(full_enriched)

        # 기간 필터 (parquet 캐시에서 빠르게)
        if sel_years and sel_months and not full_enriched.empty:
            trades_df = full_enriched[
                full_enriched["Date"].dt.year.isin(sel_years) &
                full_enriched["Date"].dt.month.isin(sel_months)
            ].reset_index(drop=True)
        else:
            trades_df = full_enriched
        n_filtered = len(trades_df)

    # 정렬 옵션
    if not trades_df.empty:
        sc1, sc2 = st.columns([3, 1])
        sort_opt = sc2.selectbox(
            "정렬",
            ["최신순", "오래된순", "수익률 높은순", "수익률 낮은순",
             "점수 높은순", "거래대금 높은순"],
            label_visibility="collapsed",
            key="results_sort",
        )
        if sort_opt == "최신순":
            trades_df = trades_df.sort_values(["Date", "Rank"], ascending=[False, True])
        elif sort_opt == "오래된순":
            trades_df = trades_df.sort_values(["Date", "Rank"], ascending=[True, True])
        elif sort_opt == "수익률 높은순":
            trades_df = trades_df.sort_values("ret_d1_close", ascending=False, na_position="last")
        elif sort_opt == "수익률 낮은순":
            trades_df = trades_df.sort_values("ret_d1_close", ascending=True, na_position="last")
        elif sort_opt == "점수 높은순":
            trades_df = trades_df.sort_values("Score", ascending=False)
        elif sort_opt == "거래대금 높은순":
            trades_df = trades_df.sort_values("Amount", ascending=False)
        trades_df = trades_df.reset_index(drop=True)

    # 카운트 안내
    if n_total_raw > 0:
        excluded = n_total_raw - n_filtered
        st.markdown(
            f'<div style="font-size:13px;color:var(--text-2);margin:10px 0 18px 0;">'
            f'🔧 추가 필터로 <b>{n_total_raw:,}건</b> → '
            f'<b style="color:var(--accent);">{n_filtered:,}건</b> '
            f'<span class="subtle">({excluded:,}건 제외)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not trades_df.empty:
        # ───────── 요약 통계 (큰 숫자 카드 4개) ─────────
        n_signals = len(trades_df)
        n_stocks = trades_df["Code"].nunique()
        n_days = trades_df["Date"].dt.normalize().nunique()
        avg_per_day = n_signals / max(n_days, 1)

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        sc = st.columns(4)
        with sc[0]:
            st.markdown(
                f'<div class="tcard" style="text-align:center;">'
                f'<div class="big-number-label">총 시그널</div>'
                f'<div class="big-number" style="color:var(--text);">{n_signals:,}</div>'
                f'<div class="subtle" style="font-size:11px;">선택 기간</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with sc[1]:
            st.markdown(
                f'<div class="tcard" style="text-align:center;">'
                f'<div class="big-number-label">종목 수</div>'
                f'<div class="big-number" style="color:var(--accent);">{n_stocks:,}</div>'
                f'<div class="subtle" style="font-size:11px;">중복 제외</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with sc[2]:
            st.markdown(
                f'<div class="tcard" style="text-align:center;">'
                f'<div class="big-number-label">거래일 수</div>'
                f'<div class="big-number" style="color:var(--text);">{n_days:,}</div>'
                f'<div class="subtle" style="font-size:11px;">시그널 발생일</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with sc[3]:
            st.markdown(
                f'<div class="tcard" style="text-align:center;">'
                f'<div class="big-number-label">일평균 시그널</div>'
                f'<div class="big-number" style="color:var(--accent);">{avg_per_day:.1f}</div>'
                f'<div class="subtle" style="font-size:11px;">건/일</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ───────── 패턴별 비교 (돌파/눌림/대시세) ─────────
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2>🏆 패턴별 승률 · 수익률 비교</h2>", unsafe_allow_html=True)
        st.caption("돌파매매 vs 눌림목매매 vs 대시세 초입 — D+1 시·고·저·종 시점별")
        pat_df = pattern_comparison_table(trades_df, position_size=position_size)
        if not pat_df.empty:
            render_pattern_comparison_html(pat_df)

        # ───────── 월별 손익 표 ─────────
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2>💰 월별 손익 + 전체 손익</h2>", unsafe_allow_html=True)
        st.caption(
            f"종목당 매수금 {position_size/10000:,.0f}만원 기준. "
            "D+1 시가/고가/저가/종가 평균 × 시그널수 × 매수금."
        )
        pnl = monthly_pnl_table(trades_df, position_size=position_size)
        if not pnl.empty:
            render_monthly_pnl_html(pnl, position_size=position_size)

        # ───────── 년/월별 × 패턴별 표 ─────────
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3>📅 월별 × 패턴별 (D+1 종가)</h3>", unsafe_allow_html=True)
        st.caption("매월 어떤 패턴이 잘 되었는지 비교")
        ym_pat_df = year_month_pattern_table(
            trades_df, position_size=position_size, metric="ret_d1_close"
        )
        if not ym_pat_df.empty:
            render_ym_pattern_html(ym_pat_df)

        # ───────── 년도별 × 매도 시점 매트릭스 (단타 vs 중장기) ─────────
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2>📊 년도별 × 매도 시점 매트릭스</h2>", unsafe_allow_html=True)
        st.caption(
            "단타(익일) vs 중기(20/30/60일) vs 장기(90/120일) — 어느 시점에 매도해야 가장 좋았나"
        )
        render_year_horizon_matrix(trades_df, position_size=position_size)

        # ───────── 종합 가이드 (9 프리셋 합본) ─────────
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2>🎯 종합 분석 가이드 (9 프리셋 합본)</h2>", unsafe_allow_html=True)
        st.caption("전체 31,154건 합본 — 년도별 가이드 + TOP 3 전략 + 매매타입별")
        render_comprehensive_guide(position_size=position_size)

        # ───────── 손익비 분석 + 동적 익절/손절 시뮬레이터 ─────────
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2>💰 단타 손익비 & 익절/손절 시뮬레이터</h2>", unsafe_allow_html=True)
        st.caption("어느 시점 매도가 손익비 좋은지 + 동적 익절/손절 조합 실험")
        render_risk_reward_section(position_size=position_size)

        # ───────── 안정/공격/욕심 단타 전략 비교 (사용자 매수금 기준) ─────────
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2>🎯 단타 전략 4종 비교</h2>", unsafe_allow_html=True)
        st.caption(
            f"종목당 {position_size/10000:,.0f}만원 매수 기준 · 2020~2026 31,154건 합본 "
            "· 안정/공격/욕심 + 중장기 보유 비교"
        )
        render_strategy_comparison(position_size=position_size)

        # ───────── 일자별 시그널 표 ─────────
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        ym_label = f"{sel_years[0] if sel_years else '?'}년 " + \
                    (f"{','.join(str(m) for m in sel_months)}월 " if sel_months else "")
        st.markdown(
            f"<h2>📋 {ym_label}일자별 시그널 (일별 상위 {st.session_state.top_n}개)</h2>",
            unsafe_allow_html=True,
        )
        render_daily_signals(trades_df, position_size=position_size)

        # ───────── CSV 다운로드 ─────────
        csv_view = pd.DataFrame({
            "매수일": trades_df["Date"].dt.strftime("%Y-%m-%d"),
            "종목명": trades_df["Name"],
            "종목코드": trades_df["Code"],
            "시장": trades_df.get("Market", ""),
            "매수가": trades_df["Close"],
            "당일등락(%)": trades_df["ChangeRatio"],
            "직전20일(%)": trades_df.get("past_20d", 0),
            "거래대금(억)": (trades_df["Amount"] / 1e8).round(1),
            "익일(%)": trades_df["return_pct"],
            "10일(%)": trades_df.get("ret_10d", pd.NA),
            "30일(%)": trades_df.get("ret_30d", pd.NA),
            "추천사유": trades_df.get("reason", ""),
            "점수": trades_df["Score"],
        })
        csv_bytes = csv_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "전체 시그널 CSV 다운로드", csv_bytes,
            file_name=f"백테스트_{cur_preset}_{datetime.now():%Y%m%d}.csv",
            use_container_width=True,
        )

    # ============================================================
    # 참고 (펼침): OOS 검증 통계
    # ============================================================
    with st.expander("OOS 검증 통계 (참고)"):
        ref_cols = st.columns(3)
        ref_cols[0].metric("결정 시점", f"{n_dp}회")
        ref_cols[1].metric("기간", f"{c_start[:7]}~{c_end[:7]}")
        ref_cols[2].metric("상태", period_label.split(":")[0])
        st.caption(
            "Walk-Forward OOS: 각 결정 시점에서 그 이전 데이터로 TOP 3 선정 → 다음 분기에 적용. "
            "미래 데이터 미사용."
        )

    # TOP 3 전략 카드 — 펼침 처리
    with st.expander("OOS 검증 TOP 3 전략 (요약)"):
        top3 = cached["consensus_top3"][:3]
        medals = ["1위", "2위", "3위"]
        _render_top3_cards(cached, top3, medals)


def _render_top3_cards(cached, top3, medals):
    p = PALETTE[st.session_state.theme]
    n_dp = cached["n_decision_points"]
    for i, item in enumerate(top3):
        avg = item.get("avg_return", 0)
        wr = item.get("win_rate", 0)
        sharpe = item.get("sharpe_annual", 0)
        n_trades = item.get("n_trades", 0)
        votes = item.get("votes_in_top3", 0)
        avg_class = "up" if avg > 0 else "down"

        card = (
            f'<div class="tcard">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
            f'<div>'
            f'<div style="font-size:13px;color:{p["text_tertiary"]};">{medals[i]} {votes}/{n_dp}회 TOP3 진입</div>'
            f'<div style="font-size:20px;font-weight:700;color:{p["text"]};margin-top:2px;">{item["preset_name"]}</div>'
            f'</div>'
            f'<div class="big-number {avg_class}">{avg:+.2f}%</div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding-top:14px;border-top:1px solid {p["border"]};">'
            f'<div><div class="big-number-label">OOS 승률</div>'
            f'<div style="font-size:18px;font-weight:700;color:{p["text"]};">{wr:.1f}%</div></div>'
            f'<div><div class="big-number-label">OOS 샤프</div>'
            f'<div style="font-size:18px;font-weight:700;color:{p["text"]};">{sharpe:.2f}</div></div>'
            f'<div><div class="big-number-label">OOS 거래수</div>'
            f'<div style="font-size:18px;font-weight:700;color:{p["text"]};">{n_trades:,}</div></div>'
            f'</div></div>'
        )
        col_card, col_btn = st.columns([5, 1])
        col_card.markdown(card, unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='height:55px;'></div>", unsafe_allow_html=True)
            if st.button("적용", key=f"apply_{item['preset_key']}",
                          use_container_width=True, type="secondary"):
                st.session_state.preset = item["preset_key"]
                st.success(f"✅ {item['preset_name']} 적용됨")

    with st.expander("전체 9개 전략 비교"):
        per_preset = list(cached.get("per_preset_oos", {}).values())
        if per_preset:
            df = pd.DataFrame([
                {
                    "전략": p["preset_name"],
                    "TOP3 빈도": p["votes_in_top3"],
                    "OOS 평균(%)": round(p.get("avg_return", 0), 2),
                    "OOS 승률(%)": round(p.get("win_rate", 0), 1),
                    "OOS 샤프": round(p.get("sharpe_annual", 0), 2),
                    "거래수": p.get("n_trades", 0),
                }
                for p in per_preset
            ]).sort_values(by=["TOP3 빈도", "OOS 샤프"], ascending=[False, False])
            st.dataframe(df, use_container_width=True, height=300)

    # ============================================================
    # AI 최적화 결과 (Overnight Optimizer 2,460 조합 탐색)
    # ============================================================
    import json
    from pathlib import Path
    overnight_path = Path("cache/overnight_final.json")
    if overnight_path.exists():
        try:
            with open(overnight_path) as f:
                ov_data = json.load(f)
        except Exception:
            ov_data = None
        if ov_data:
            st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
            st.markdown("<h2>AI 최적화 가중치 (2,460개 조합 탐색)</h2>", unsafe_allow_html=True)
            st.markdown(
                f'<p class="muted">기존 9개 프리셋이 아닌, 가중치를 자유롭게 변화시켜 '
                f'OOS 성과가 가장 좋은 조합을 찾은 결과 (Random + Refinement search).</p>',
                unsafe_allow_html=True,
            )

            top3_ai = ov_data.get("top3_combinations", [])[:3]
            medals_ai = ["🥇", "🥈", "🥉"]
            ai_preset_keys = ["ai_optimized_1", "ai_optimized_2", "ai_optimized_3"]

            for i, item in enumerate(top3_ai):
                avg = item.get("oos_avg", 0)
                wr = item.get("oos_win_rate", 0)
                sharpe = item.get("oos_sharpe", 0)
                n_tr = item.get("n_trades", 0)
                w = item.get("weights", [0] * 12)
                avg_class = "up" if avg > 0 else "down"

                # 상위 가중치 시그널
                w_with_idx = [(j + 1, val) for j, val in enumerate(w)]
                w_with_idx.sort(key=lambda x: -x[1])
                top_w = w_with_idx[:5]
                w_str = " · ".join(
                    f"S{idx}={val:.0f}" for idx, val in top_w if val > 0.5
                )

                card = (
                    f'<div class="tcard">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
                    f'<div>'
                    f'<div style="font-size:13px;color:var(--text-3);">{medals_ai[i]} AI 최적화 {i+1}위</div>'
                    f'<div style="font-size:18px;font-weight:700;color:var(--text);margin-top:2px;">'
                    f'샤프 {sharpe:.2f} · 승률 {wr:.1f}%</div>'
                    f'<div style="font-size:12px;color:var(--text-2);margin-top:4px;">{w_str}</div>'
                    f'</div>'
                    f'<div class="big-number {avg_class}">{avg:+.2f}%</div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding-top:14px;border-top:1px solid var(--border);">'
                    f'<div><div class="big-number-label">OOS 점수</div>'
                    f'<div style="font-size:18px;font-weight:700;">{item.get("oos_score", 0):.2f}</div></div>'
                    f'<div><div class="big-number-label">OOS 거래수</div>'
                    f'<div style="font-size:18px;font-weight:700;">{n_tr:,}</div></div>'
                    f'<div><div class="big-number-label">탐색 방식</div>'
                    f'<div style="font-size:14px;font-weight:700;">{item.get("phase", "?")}</div></div>'
                    f'</div></div>'
                )
                col_c, col_b = st.columns([5, 1])
                col_c.markdown(card, unsafe_allow_html=True)
                with col_b:
                    st.markdown("<div style='height:55px;'></div>", unsafe_allow_html=True)
                    if i < len(ai_preset_keys):
                        if st.button("적용", key=f"apply_ai_{i}",
                                      use_container_width=True, type="secondary"):
                            st.session_state.preset = ai_preset_keys[i]
                            st.success(f"AI 최적화 {i+1}위 적용됨")

            # 시그널 중요도
            sig_imp = ov_data.get("signal_importance_top30", {})
            if sig_imp:
                st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
                st.markdown("<h3>12 시그널 중요도 (상위 30 조합 평균)</h3>", unsafe_allow_html=True)
                st.markdown(
                    '<p class="muted">어떤 시그널이 일관되게 중요한지. 가중치 평균이 높을수록 중요.</p>',
                    unsafe_allow_html=True,
                )
                ranked = sig_imp.get("ranked_signals", [])
                from case_matcher import SIGNAL_NAMES
                rows_imp = []
                for label, val in ranked:
                    sig_key = label.lower()
                    long_name = SIGNAL_NAMES.get(sig_key, label)
                    rows_imp.append({"시그널": long_name, "평균 가중치": round(val, 2)})
                df_imp = pd.DataFrame(rows_imp)
                st.dataframe(df_imp, use_container_width=True, height=460, hide_index=True)


# =============================================================================
# 페이지: 사례 & 가이드
# =============================================================================
def page_library():
    p = PALETTE[st.session_state.theme]
    st.markdown("<h1>사례 & 가이드</h1>", unsafe_allow_html=True)

    sub_tabs = st.tabs(["🏆 V/S/A/B 등급 가이드", "📚 실전 사례 35건", "📖 매매 가이드", "📋 워치리스트"])

    with sub_tabs[0]:
        st.markdown("<h2>V/S/A/B 등급 시스템</h2>", unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{p["text_secondary"]};">'
            '4 프리셋 (default · 박스돌파 · 하바로셀 · 풀백) 통합 분석 후 '
            '4단계 등급으로 자동 분류. 자본을 등급별 비중으로 배분.</p>',
            unsafe_allow_html=True,
        )

        # 등급 카드 4개 (V/S/A/B)
        for g in ["V", "S", "A", "B"]:
            info = GRADE_INFO[g]
            conditions = {
                "V": [
                    "✓ 시장: 코스닥",
                    "✓ 매매타입: 돌파매매",
                    "✓ 오늘 등락률: 7~25%",
                    "✓ 앙상블 점수 ≥ 75 (절대 최고급)",
                ],
                "S": [
                    "✓ 시장: 코스닥",
                    "✓ 매매타입: 돌파매매",
                    "✓ 오늘 등락률: 7~25%",
                    "✓ 4개 프리셋 모두 추천 (만장일치)",
                    "✓ 앙상블 점수 ≥ 65",
                ],
                "A": [
                    "✓ 시장: 코스닥",
                    "✓ 매매타입: 돌파매매",
                    "✓ 오늘 등락률: 10~18% (좁은 안전구간)",
                    "✓ 앙상블 점수 ≥ 65",
                ],
                "B": [
                    "✓ 시장: 코스닥",
                    "✓ 매매타입: 돌파매매",
                    "✓ 오늘 등락률: 7~25%",
                    "✓ V1 통과 (4개 프리셋 중 1개 이상 추천)",
                ],
            }[g]

            reasons_text = {
                "V": "4개 전략 평균 점수가 75점 이상인 절대 최고급 셋업. 6년간 단 32회만 출현. 모든 시그널이 완벽하게 정렬된 종목으로, 평균 +96.9% 수익 (180일 보유 기준). +200% 대박 확률 7%로 자본을 가장 크게 배분.",
                "S": "default·박스돌파·하바로셀·풀백 4개 시스템이 동시에 동의한 강력한 셋업. 만장일치라는 강력한 합의 + 점수 65 이상으로 큰손실률 단 4% (역대 최저). 안전한 대박 종목.",
                "A": "점수 65 이상 + 등락률 10~18%의 좁은 안전구간 종목. 과열 회피 + 점수 안정의 균형. +50% 적중률 24%로 평균 +34.1% 수익.",
                "B": "4개 전략 중 1개 이상이 추천한 베이스 픽. 코스닥 + 돌파매매 + 등락 7~25%의 표준 셋업. 매일 1종목 픽으로 안정 운영. 평균 +34.4% 수익.",
            }[g]

            cond_html = "<br>".join(conditions)
            st.markdown(
                f'<div class="tcard" style="border-left:5px solid {info["color"]};margin-bottom:20px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
                f'<div>'
                f'<div style="font-size:24px;font-weight:800;color:{info["color"]};">{info["emoji"]} {info["name"]}</div>'
                f'<div style="font-size:12px;color:{p["text_secondary"]};margin-top:4px;">'
                f'비중 <b>{info["weight_str"]}</b> · 빈도 <b>{info["frequency"]}</b>'
                f'</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:11px;color:{p["text_tertiary"]};">평균수익 (180일)</div>'
                f'<div style="font-size:22px;font-weight:800;color:{info["color"]};">{info["avg_return"]}</div>'
                f'<div style="font-size:11px;color:{p["text_tertiary"]};">큰손실률 {info["big_loss_rate"]}</div>'
                f'</div>'
                f'</div>'
                f'<div style="background:{info["bg"]};border-radius:8px;padding:12px;margin-bottom:12px;">'
                f'<div style="font-size:13px;font-weight:700;color:{info["color"]};margin-bottom:6px;">진입 조건</div>'
                f'<div style="font-size:13px;color:{p["text"]};line-height:1.8;">{cond_html}</div>'
                f'</div>'
                f'<div style="font-size:13px;color:{p["text_secondary"]};line-height:1.7;">'
                f'<b>💡 추천 이유:</b> {reasons_text}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 운용 가이드
        st.markdown("---")
        st.markdown("<h2>📋 등급 시스템 운용 가이드</h2>", unsafe_allow_html=True)
        st.markdown("""
**중첩 처리 규칙**
- 같은 종목이 여러 등급 조건을 만족하면 → **상위 등급만 적용** (V > S > A > B)
- V/S급은 점수 상위 3개까지 모두 매수 가능
- A/B급은 점수 1위 1개만 매수

**자본 배분 예시 (피크 기준 약 2,100만원 필요)**

| 등급 | 비중 | 평균 동시 보유 | 자본 |
|---|---|---|---|
| 🏆 V급 | 50만원 | ~5종목 | 250만원 |
| 💎 S급 | 30만원 | ~16종목 | 480만원 |
| ⭐ A급 | 20만원 | ~7종목 | 140만원 |
| 🟢 B급 | 10만원 | ~110종목 | 1,100만원 |
| **합계** | | | **~2,000만원** (피크 2,100만) |

**매도 규칙**
- 보유 기간: **180일** (5개월)
- 손절 / 익절 **없음**
- 단순 시간 매도가 가장 좋음 (180일 평균 +29.57%, 누적 +6,318만 / 6년)

**기대 성과 (2020~2025 백테스트)**

| 시장 환경 | 년 평균 수익 |
|---|---|
| 🟢 강세 (2020) | +48.0% |
| 🟡 횡보 (2021) | +1.1% |
| 🔴 하락 (2022) | +16.0% ✅ (B급이 받쳐줌) |
| 🟢 회복 (2023) | +19.7% |
| 🔴 하락 (2024) | +15.5% ✅ |
| 🚀 폭등 (2025) | +122.2% 🔥 |

**6년 종합: 자본 2,100만원 → +6,318만원 (연 ROI +50.1%)**
        """)

    with sub_tabs[1]:
        st.markdown("<h2>📚 실전 사례 35건</h2>", unsafe_allow_html=True)
        st.caption("하바로셀/하승훈 검증된 실전 사례 — V/S/A/B 등급과 함께 참고")
        filter_pat = st.selectbox(
            "패턴 필터",
            ["전체", "돌파매매", "눌림목매매", "대시세 초입"],
            label_visibility="collapsed",
        )
        type_to_pattern = {
            "전체": None, "돌파매매": Pattern.A_BREAKOUT,
            "눌림목매매": Pattern.B_PULLBACK, "대시세 초입": Pattern.D_LONGTERM,
        }
        target = type_to_pattern.get(filter_pat)
        cases = CASE_STUDIES if target is None else get_cases_by_pattern(target)
        st.markdown(f'<p class="muted" style="margin-top:12px;">총 {len(cases)}건</p>',
                     unsafe_allow_html=True)
        for case in cases:
            with st.expander(f"**{case['stock']}** · {case['date']} · {case['theme']}"):
                pat_emoji = {
                    Pattern.A_BREAKOUT: "🚀 돌파매매 (V/S/A/B 후보)",
                    Pattern.B_PULLBACK: "📉↗️ 눌림목매매",
                    Pattern.C_DOUBLE_BOTTOM: "📊 분봉눌림목",
                    Pattern.D_LONGTERM: "🌊 대시세 초입",
                    Pattern.E_RISK: "⚠️ 리스크 관리",
                }.get(case['pattern'], "")
                st.markdown(f"**패턴**: {pat_emoji}")
                st.markdown(f"**📝 트리거**: {case['trigger']}")
                st.markdown(f"**🎯 핵심 시그널**: {' · '.join(case['key_signals'])}")
                st.markdown(f"**📈 결과**: {case['outcome']}")
                st.markdown(f"**💡 교훈**: {case['lesson']}")
                st.caption(f"출처: {case['source']}")

    with sub_tabs[2]:
        try:
            with open("docs/master_guide.md", "r") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.error("가이드 파일 없음")

    with sub_tabs[3]:
        sub_inner = st.tabs(["🎓 하바로셀", "📺 하승훈", "⭐ 사용자", "🏷️ 테마"])
        with sub_inner[0]:
            df = koreanize_dataframe(pd.DataFrame(HABAROCELL_PICKS))
            st.dataframe(df, use_container_width=True, height=400)
        with sub_inner[1]:
            df = koreanize_dataframe(pd.DataFrame(HASEUNGHOON_PICKS))
            st.dataframe(df, use_container_width=True, height=400)
        with sub_inner[2]:
            df = koreanize_dataframe(pd.DataFrame(USER_PICKS))
            st.dataframe(df, use_container_width=True, height=400)
        with sub_inner[3]:
            for theme, stocks in THEMES.items():
                with st.expander(f"**{theme}** ({len(stocks)}종목)"):
                    st.write(" · ".join(stocks))


# =============================================================================
# 라우팅
# =============================================================================
ROUTES = {
    "today": page_today,
    "results": page_results,
    "library": page_library,
}
ROUTES.get(st.session_state.page, page_today)()
