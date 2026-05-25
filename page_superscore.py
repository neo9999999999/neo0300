"""
SuperScore 추천 페이지 (Streamlit) — 한글 정리 + 버튼 멀티선택
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")


def _grade_color(grade: str) -> str:
    if "강력매수" in grade: return "#10B981"
    if "추천" in grade: return "#3B82F6"
    if "관망" in grade: return "#9CA3AF"
    if "손절위험" in grade: return "#EF4444"
    return "#6B7280"


def _render_pick_card(row: pd.Series):
    grade = row.get("등급", "")
    code = row.get("Code", "")
    name = row.get("Name", "")
    close = row.get("Close", 0)
    ss = row.get("SuperScore", 0)
    peak_pred = row.get("예상peak%", 0)
    p_sw = row.get("슈퍼위너확률%", 0)
    p100 = row.get("100%+확률", 0)
    p50 = row.get("50%+확률", 0)
    ploss = row.get("손절확률%", 0)
    tags = row.get("가능성태그", "")
    market = row.get("Market", "")
    date = row.get("Date", "")
    if isinstance(date, str): date = date[:10]
    else:
        try: date = pd.to_datetime(date).strftime("%Y-%m-%d")
        except: date = ""

    color = _grade_color(grade)

    st.markdown(f"""
<div style="border-left:4px solid {color};padding:14px 18px;background:rgba(0,0,0,0.02);
            border-radius:6px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="font-size:14px;color:{color};font-weight:700;">{grade}</span>
      <span style="font-size:18px;font-weight:800;margin-left:12px;">{name}</span>
      <span style="font-size:12px;color:#9CA3AF;margin-left:8px;">{code} · {market} · {date}</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;color:#9CA3AF;">매수가</div>
      <div style="font-size:18px;font-weight:700;">{close:,.0f}원</div>
    </div>
  </div>
  <div style="margin-top:12px;display:grid;grid-template-columns:repeat(5,1fr);gap:8px;font-size:12px;">
    <div><b>슈퍼점수</b><br>{ss:.2f}</div>
    <div><b>예상 최고가</b><br><span style="color:{color}">+{peak_pred:.0f}%</span></div>
    <div><b>슈퍼위너 확률</b><br>{p_sw:.0f}%</div>
    <div><b>100%+ 확률</b><br>{p100:.0f}%</div>
    <div><b>50%+ / 손절</b><br>{p50:.0f}% / <span style="color:#EF4444">{ploss:.0f}%</span></div>
  </div>
  {f'<div style="margin-top:8px;font-size:12px;color:#6B7280;">{tags}</div>' if tags else ''}
</div>
""", unsafe_allow_html=True)


def _button_multiselect(label: str, options: list, default: list, key_prefix: str):
    """버튼 형식 멀티선택 (on/off 토글)"""
    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = list(default)

    selected = st.session_state[f"{key_prefix}_selected"]

    st.markdown(f"**{label}**")
    cols = st.columns(min(len(options), 8))
    for i, opt in enumerate(options):
        col = cols[i % len(cols)]
        is_on = opt in selected
        btn_type = "primary" if is_on else "secondary"
        if col.button(str(opt), key=f"{key_prefix}_btn_{opt}", type=btn_type,
                       use_container_width=True):
            if is_on:
                selected.remove(opt)
            else:
                selected.append(opt)
            st.session_state[f"{key_prefix}_selected"] = selected
            st.rerun()
    return selected


def page_superscore():
    st.markdown('<h1 style="margin-bottom:8px;">💎 슈퍼스코어 추천</h1>', unsafe_allow_html=True)
    st.caption("시총 300 풀 + 4 RF 분류기 + 5년 walk-forward OOS")

    tabs = st.tabs(["🎯 오늘 추천", "📅 이번 주", "🗓️ 지난 주", "📊 백테스트 (년월 선택)", "📋 매수 룰"])

    # 공통: json 로드
    json_path = CACHE / "today_picks.json"
    data = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

    # ========== 탭 1: 오늘 추천 ==========
    with tabs[0]:
        if not data:
            st.warning("아직 추천 데이터가 없습니다.")
        else:
            base_date = data.get("base_date", "")
            updated = data.get("updated_at", "")[:16]
            st.markdown(f"**기준일**: {base_date}  ·  **갱신**: {updated}")

            today = data.get("today", {})
            n = today.get("n", 0)

            if n == 0:
                st.info("📭 기준일 발생 추천 없음 — 현금 보유")
            else:
                # 등급 필터 버튼
                grades_avail = ["★ 강력매수", "○ 추천", "- 관망", "⚠️ 손절위험"]
                sel_grades = _button_multiselect(
                    "등급 필터", grades_avail,
                    default=["★ 강력매수", "○ 추천"], key_prefix="today_grade")

                picks = today["picks"]
                picks_filtered = [p for p in picks if p.get("등급") in sel_grades]
                st.markdown(f"### ⭐ 오늘 추천 ({len(picks_filtered)}/{n}건)")
                for p in picks_filtered:
                    _render_pick_card(pd.Series(p))

        st.markdown("---")
        with st.expander("📌 매수 가이드", expanded=False):
            st.markdown(
                "1. **★ 강력매수**만 매수 (○ 추천은 자본 여유 시 선택)\n"
                "2. **⚠️ 손절위험**은 매수 절대 X\n"
                "3. **매수 시점**: 당일 NXT 19:50 시장가 (1순위) / D+1 시초가 (2순위)\n"
                "4. **종목당 10만원** (자본 1억 → 0.1%)\n"
                "5. **매도**: 매수일 + 180거래일 후 정규장 종가 시장가\n"
                "6. **익절/손절 룰 X** (그냥 묻기)"
            )

    # ========== 탭 2: 이번 주 ==========
    with tabs[1]:
        week = data.get("week", {})
        st.markdown(f"### 📅 이번 주 누적 추천 ({week.get('n', 0)}건)")
        st.caption(f"주 시작일: {data.get('week_start', '')}")

        if week.get("n", 0) > 0:
            grades_avail = ["★ 강력매수", "○ 추천", "- 관망", "⚠️ 손절위험"]
            sel_grades = _button_multiselect(
                "등급 필터", grades_avail,
                default=["★ 강력매수", "○ 추천"], key_prefix="week_grade")

            picks = week["picks"]
            picks_filtered = [p for p in picks if p.get("등급") in sel_grades]
            for p in picks_filtered:
                _render_pick_card(pd.Series(p))
        else:
            st.info("이번 주 추천 없음")

    # ========== 탭 3: 지난 주 ==========
    with tabs[2]:
        last_week = data.get("last_week", {})
        st.markdown(f"### 🗓️ 지난 주 추천 ({last_week.get('n', 0)}건)")

        if last_week.get("n", 0) > 0:
            grades_avail = ["★ 강력매수", "○ 추천", "- 관망", "⚠️ 손절위험"]
            sel_grades = _button_multiselect(
                "등급 필터", grades_avail,
                default=["★ 강력매수", "○ 추천"], key_prefix="lw_grade")

            picks = last_week["picks"]
            picks_filtered = [p for p in picks if p.get("등급") in sel_grades]
            for p in picks_filtered:
                _render_pick_card(pd.Series(p))
        else:
            st.info("지난 주 추천 데이터 없음")

    # ========== 탭 4: 백테스트 ==========
    with tabs[3]:
        st.markdown("### 📊 5년 백테스트 (2022-2026 walk-forward OOS)")

        # 년도별 요약
        yr_path = CACHE / "MASTER_best_yearly.csv"
        if yr_path.exists():
            yr = pd.read_csv(yr_path)
            # 한글 컬럼명
            yr = yr.rename(columns={
                "year": "년도", "매수": "매수", "SW": "슈퍼위너",
                "100+": "100%+", "50+": "50%+", "10+": "10%+",
                "손절": "손절", "투자만": "투자(만원)",
                "수익만": "수익(만원)", "수익률%": "수익률(%)"
            })
            st.markdown("#### 📅 년도별 요약")
            st.dataframe(yr, hide_index=True, use_container_width=True)

            cols = st.columns(4)
            cols[0].metric("총 매수", f"{int(yr['매수'].sum()):,}건")
            cols[1].metric("총 투자", f"{int(yr['투자(만원)'].sum()):,}만")
            cols[2].metric("총 수익", f"{int(yr['수익(만원)'].sum()):+,}만")
            tot_inv = yr['투자(만원)'].sum()
            tot_prof = yr['수익(만원)'].sum()
            cols[3].metric("5년 수익률", f"{tot_prof/tot_inv*100:+.1f}%")

        st.markdown("---")
        st.markdown("#### 📋 매수 종목 전체 (1,155건)")

        picks_path = CACHE / "MASTER_best_picks_2020-2026.csv"
        if picks_path.exists():
            picks = pd.read_csv(picks_path)
            picks["Date"] = pd.to_datetime(picks["Date"])
            picks["년도"] = picks["Date"].dt.year
            picks["월"] = picks["Date"].dt.month

            # 결과 분류
            def cls(row):
                p = row.get("peak_180d", 0)
                if pd.isna(p): return "미정"
                if p >= 200: return "🏆 슈퍼위너"
                if p >= 100: return "💯 100%+"
                if p >= 50: return "📈 50%+"
                if p >= 10: return "✅ 10%+"
                if row.get("ret_180d", 0) <= -20: return "❌ 손절"
                return "💤 보합"
            picks["결과"] = picks.apply(cls, axis=1)

            # 버튼 멀티선택 - 년도
            years_avail = sorted(picks["년도"].dropna().unique().astype(int).tolist())
            sel_years = _button_multiselect(
                "년도 (다중 선택)", years_avail, default=years_avail, key_prefix="bt_year")

            # 버튼 멀티선택 - 월
            months_avail = list(range(1, 13))
            sel_months = _button_multiselect(
                "월 (다중 선택)", months_avail, default=months_avail, key_prefix="bt_month")

            # 결과 버튼 멀티선택
            results_avail = ["🏆 슈퍼위너", "💯 100%+", "📈 50%+", "✅ 10%+", "💤 보합", "❌ 손절", "미정"]
            sel_results = _button_multiselect(
                "결과 (다중 선택)", results_avail,
                default=["🏆 슈퍼위너", "💯 100%+", "📈 50%+"], key_prefix="bt_result")

            # 정렬 (버튼 형)
            sort_options = {
                "최신 일자순": ("Date", False),
                "오래된 일자순": ("Date", True),
                "최고가 높은순": ("peak_180d", False),
                "수익률 높은순": ("ret_180d", False),
                "수익률 낮은순": ("ret_180d", True),
                "슈퍼점수 높은순": ("SuperScore_v2", False),
            }
            sort_keys = list(sort_options.keys())

            if "bt_sort_selected" not in st.session_state:
                st.session_state.bt_sort_selected = "최신 일자순"

            st.markdown("**정렬 기준**")
            sort_cols = st.columns(len(sort_keys))
            for i, sk in enumerate(sort_keys):
                is_sel = st.session_state.bt_sort_selected == sk
                btn_type = "primary" if is_sel else "secondary"
                if sort_cols[i].button(sk, key=f"bt_sort_{sk}",
                                         type=btn_type, use_container_width=True):
                    st.session_state.bt_sort_selected = sk
                    st.rerun()

            sort_col_key, sort_asc = sort_options[st.session_state.bt_sort_selected]
            if sort_col_key not in picks.columns:
                sort_col_key = "Date"

            filtered = picks[
                picks["년도"].isin(sel_years) &
                picks["월"].isin(sel_months) &
                picks["결과"].isin(sel_results)
            ]
            filtered = filtered.sort_values(sort_col_key, ascending=sort_asc)

            # 한글 컬럼명
            show_map = {
                "Date": "일자", "년도": "년도", "월": "월",
                "Code": "종목코드", "Name": "종목명", "Market": "시장",
                "Close": "매수가", "결과": "결과",
                "ret_180d": "180일수익률(%)", "peak_180d": "최고가도달(%)",
                "sell_close": "매도가", "sell_date": "매도일",
                "SuperScore_v2": "슈퍼점수",
                "p_sw": "슈퍼위너확률", "p_100plus": "100%+확률",
                "p_50plus": "50%+확률", "p_loss": "손절확률",
            }
            show_cols = [c for c in show_map if c in filtered.columns]
            display = filtered[show_cols].rename(columns=show_map).head(500)

            # 일자 포맷
            if "일자" in display.columns:
                display["일자"] = pd.to_datetime(display["일자"]).dt.strftime("%Y-%m-%d")
            # 확률 % 변환
            for c in ["슈퍼위너확률", "100%+확률", "50%+확률", "손절확률"]:
                if c in display.columns:
                    display[c] = (display[c]*100).round(1).astype(str) + "%"

            st.dataframe(display, hide_index=True, use_container_width=True, height=600)
            st.caption(f"검색 결과 {len(filtered):,}건 중 최대 500건 표시")

    # ========== 탭 5: 매수 룰 ==========
    with tabs[4]:
        st.markdown("""
### 🎯 최종 매수 룰

```
[풀]   시총 상위 300종목 (KRX)
[시그널] 4 프리셋 ensemble + Score ≥ 40
[모델]  RF 4분류기 + peak 회귀

[종합 점수]
  슈퍼점수 = p_sw × 5 + p_100+ × 2 + p_50+ × 1 - p_loss × 3

[등급 자동 부여]
  ★ 강력매수: 점수 상위 20% + 손절확률 < 55%
  ○ 추천:    점수 상위 20-40%
  - 관망:    중간
  ⚠️ 손절위험: 손절확률 ≥ 55%

[매수]
  - ★ 강력매수만 매수 (○ 추천은 옵션)
  - 시점: 당일 NXT 19:50 시장가 (1순위) / D+1 시초가 (2순위)
  - 종목당 10만원 (자본 1억 기준 0.1%)

[매도]
  - 매수일 + 180거래일 후 정규장 종가
  - 익절/손절 룰 X
```

### 5년 OOS (2022-2026)
- 매수 1,155건 / 투자 1억 1,550만 → 수익 +1억 2,975만
- 자본 1억 → **2.3억** (+112.3%)
- 슈퍼위너 327건 (28.3%) / 손절 182건 (15.8%)
""")

        st.markdown("---")
        st.markdown("### 🔍 키움 HTS 검색식")
        st.code("""
[영웅문 0150 조건검색]
A: 시가총액 ≥ 14,000억
B: 전일 거래대금 100억 ~ 3,000억
C: 종가 > 60일 이평선
D: 60일 이평선 > 120일 이평선
E: 종가 > 200일 이평선
F: 252일 신고가의 70% 이상
G: RSI(14) 30 ~ 75
H: 5일 평균 거래량 > 20일 평균 × 1.2
I: 60일 등락률 -10% ~ +60%
J: 외국인 5일 누적 순매수 > 0 (선택)

조건: A AND B AND C AND D AND E AND F AND G AND H AND I
""", language="text")
