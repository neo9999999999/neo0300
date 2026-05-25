"""
SuperScore 추천 페이지 (Streamlit)
==============================
- 오늘의 추천 (SuperScore TOP 5 + 등급/태그/예상수익률)
- 이번 주 누적
- 5년 walk-forward 백테스트 결과
- 년도별 매수 종목 리스트
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")


def _grade_color(grade: str) -> str:
    if "강력매수" in grade: return "#10B981"  # green
    if "추천" in grade: return "#3B82F6"  # blue
    if "관망" in grade: return "#9CA3AF"  # gray
    if "손절위험" in grade: return "#EF4444"  # red
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

    color = _grade_color(grade)

    st.markdown(f"""
<div style="border-left:4px solid {color};padding:14px 18px;background:rgba(0,0,0,0.02);
            border-radius:6px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="font-size:14px;color:{color};font-weight:700;">{grade}</span>
      <span style="font-size:18px;font-weight:800;margin-left:12px;">{name}</span>
      <span style="font-size:12px;color:#9CA3AF;margin-left:8px;">{code} · {market}</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;color:#9CA3AF;">매수가</div>
      <div style="font-size:18px;font-weight:700;">{close:,.0f}원</div>
    </div>
  </div>
  <div style="margin-top:12px;display:grid;grid-template-columns:repeat(5,1fr);gap:8px;font-size:12px;">
    <div><b>SuperScore</b><br>{ss:.2f}</div>
    <div><b>예상 peak</b><br><span style="color:{color}">+{peak_pred:.0f}%</span></div>
    <div><b>슈퍼위너</b><br>{p_sw:.0f}%</div>
    <div><b>100%+</b><br>{p100:.0f}%</div>
    <div><b>50%+ / 손절</b><br>{p50:.0f}% / <span style="color:#EF4444">{ploss:.0f}%</span></div>
  </div>
  {f'<div style="margin-top:8px;font-size:12px;color:#6B7280;">{tags}</div>' if tags else ''}
</div>
""", unsafe_allow_html=True)


def page_superscore():
    """SuperScore 추천 페이지 메인"""
    st.markdown('<h1 style="margin-bottom:8px;">💎 SuperScore 추천</h1>', unsafe_allow_html=True)
    st.caption("시총 300 풀 + 4 RF 분류기 + 5년 walk-forward OOS 검증")

    tabs = st.tabs(["🎯 오늘 추천", "📅 이번 주", "📊 5년 백테스트", "📋 매수 룰"])

    # ========== 탭 1: 오늘 추천 ==========
    with tabs[0]:
        path = CACHE / "today_picks.json"
        if not path.exists():
            st.warning("오늘의 추천 데이터가 아직 없습니다.")
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        st.markdown(f"**기준일**: {data.get('base_date', '')}  ·  **갱신**: {data.get('updated_at', '')[:16]}")

        today = data.get("today", {})
        n = today.get("n", 0)

        if n == 0:
            st.info("📭 오늘 발생한 추천 종목이 없습니다. 현금 보유 권장.")
        else:
            st.markdown(f"### ⭐ 오늘의 추천 ({n}건)")
            for p in today["picks"]:
                _render_pick_card(pd.Series(p))

        st.markdown("---")
        st.markdown("**📌 매수 가이드**")
        st.markdown(
            "1. ★ 강력매수 등급만 매수 (○ 추천은 자본 여유 시 선택)\n"
            "2. ⚠️ 손절위험 등급은 매수 절대 X\n"
            "3. 매수 시점: **당일 NXT 19:50 시장가** 또는 **D+1 시초가**\n"
            "4. 종목당 10만원 (자본 1억 기준 0.1%)\n"
            "5. 매도: 매수일 + 180거래일 후 정규장 종가\n"
            "6. 익절/손절 룰 X (그냥 묻기)"
        )

    # ========== 탭 2: 이번 주 ==========
    with tabs[1]:
        wp_path = CACHE / "week_picks.csv"
        if not wp_path.exists() or wp_path.stat().st_size < 100:
            st.warning("이번 주 추천 데이터가 없습니다.")
        else:
            week = pd.read_csv(wp_path)
            st.markdown(f"### 📅 이번 주 누적 추천 (TOP {len(week)})")
            if len(week) > 0:
                week = week.sort_values("SuperScore", ascending=False)
                show_cols = [c for c in ["Date","등급","Code","Name","Market","Close",
                                         "SuperScore","예상peak%",
                                         "슈퍼위너확률%","100%+확률","50%+확률","손절확률%",
                                         "가능성태그"] if c in week.columns]
                st.dataframe(week[show_cols], hide_index=True, use_container_width=True)
            else:
                st.info("이번 주 발생 종목 없음")

        # 이번 달
        mp_path = CACHE / "month_picks.csv"
        if mp_path.exists() and mp_path.stat().st_size > 100:
            month = pd.read_csv(mp_path)
            if len(month) > 0:
                st.markdown(f"### 📆 이번 달 누적 추천 (TOP {len(month)})")
                month = month.sort_values("SuperScore", ascending=False)
                show_cols = [c for c in ["Date","등급","Code","Name","Market","Close",
                                          "SuperScore","예상peak%"] if c in month.columns]
                st.dataframe(month[show_cols], hide_index=True, use_container_width=True)

    # ========== 탭 3: 5년 백테스트 ==========
    with tabs[2]:
        st.markdown("### 📊 5년 OOS walk-forward 결과")

        # 년도별
        yr_path = CACHE / "MASTER_best_yearly.csv"
        if yr_path.exists():
            yr = pd.read_csv(yr_path)
            st.markdown("#### 📅 년도별")
            st.dataframe(yr, hide_index=True, use_container_width=True)

            total_n = yr["매수"].sum()
            total_inv = yr["투자만"].sum()
            total_prof = yr["수익만"].sum()
            ret = total_prof / total_inv * 100 if total_inv else 0
            cols = st.columns(4)
            cols[0].metric("총 매수", f"{total_n:,}건")
            cols[1].metric("총 투자", f"{total_inv:,.0f}만")
            cols[2].metric("총 수익", f"{total_prof:+,.0f}만")
            cols[3].metric("5년 수익률", f"{ret:+.1f}%")

        # 전체 매수 종목
        st.markdown("#### 📋 매수 종목 전체 (1,155건)")
        picks_path = CACHE / "MASTER_best_picks_2020-2026.csv"
        if picks_path.exists():
            picks = pd.read_csv(picks_path)
            picks["Date"] = pd.to_datetime(picks["Date"])
            picks["Year"] = picks["Date"].dt.year

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

            # 필터
            c1, c2, c3 = st.columns(3)
            years_avail = sorted(picks["Year"].dropna().unique().astype(int).tolist())
            sel_year = c1.multiselect("년도", years_avail, default=years_avail)
            sel_result = c2.multiselect(
                "결과",
                ["🏆 슈퍼위너","💯 100%+","📈 50%+","✅ 10%+","💤 보합","❌ 손절","미정"],
                default=["🏆 슈퍼위너","💯 100%+","📈 50%+"],
            )
            sort_col = c3.selectbox("정렬", ["Date","peak_180d","ret_180d","SuperScore_v2"], index=0)

            filtered = picks[picks["Year"].isin(sel_year) & picks["결과"].isin(sel_result)]
            filtered = filtered.sort_values(sort_col, ascending=False)

            show_cols = [c for c in ["Date","Year","Code","Name","Market","Close",
                                      "결과","ret_180d","peak_180d","sell_close","sell_date",
                                      "SuperScore_v2","p_sw","p_100plus","p_50plus","p_loss"]
                          if c in filtered.columns]
            st.dataframe(filtered[show_cols].head(500), hide_index=True, use_container_width=True)
            st.caption(f"검색 결과 {len(filtered):,}건 중 최대 500건 표시")

    # ========== 탭 4: 매수 룰 ==========
    with tabs[3]:
        st.markdown("""
### 🎯 최종 매수 룰

```
[풀]
  시총 상위 300종목 (KRX)

[시그널]
  4 프리셋 ensemble + Score ≥ 40
  매일 평균 1.5건 발생 (시그널 있는 영업일 56%)

[모델]
  RF 4분류기 (손절/슈퍼위너/100+/50+) + peak 회귀
  StrongScore = p_sw*3 + p_100*1.5 + p_50*1 - p_loss*2
  SuperScore = p_sw*5 + p_100*2 + p_50*1 - p_loss*3

[등급 자동 부여]
  ★ 강력매수: 점수 상위 20% + p_loss < 55%
  ○ 추천:    점수 상위 20-40%
  - 관망:    중간
  ⚠️ 손절위험: p_loss ≥ 55%

[매수]
  - ★ 강력매수만 매수 (○ 추천은 옵션)
  - 시점: 당일 NXT 19:50 시장가 (또는 D+1 시초가)
  - 종목당 10만원 (자본 1억 기준 0.1%)
  - 주 한도 5건

[매도]
  - 매수일 + 180거래일 후 정규장 종가
  - 익절/손절 X (그냥 묻기)

[5년 OOS 결과 (2022-2026)]
  - 매수 1,155건
  - 투자 11,550만 → 수익 +12,975만
  - 자본 1억 → 2.3억 (+112.3%)
  - 슈퍼위너 327건 (28.3%)
  - 손절 182건 (15.8%)
```
""")

        st.markdown("---")
        st.markdown("### 🔍 키움 HTS 검색식 (RF 없이도 시그널 추출)")
        st.code("""
[영웅문 0150 조건검색]
A: 시가총액 ≥ 14,000억
B: 전일 거래대금 100억 ~ 3,000억
C: 종가 > 60일 이평선
D: 60일 이평선 > 120일 이평선 (정배열)
E: 종가 > 200일 이평선
F: 252일 신고가의 70% 이상
G: RSI(14) BETWEEN 30 AND 75
H: 5일 평균 거래량 > 20일 평균 × 1.2
I: 60일 등락률 BETWEEN -10% AND 60%
J: 외국인 5일 누적 순매수 > 0 (선택)

조건: A AND B AND C AND D AND E AND F AND G AND H AND I
""", language="text")
