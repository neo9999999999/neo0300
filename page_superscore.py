"""
SuperScore 추천 페이지들 (좌측 메뉴별 분리)
=======================================
- page_today_pick: 오늘의 추천
- page_this_week: 이번 주
- page_last_week: 지난 주
- page_backtest: 백테스트
- page_case_validation: 추천 사례 검증
- page_buy_rule: 매수 룰
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

CACHE = Path("cache")


def _grade_color(grade: str) -> str:
    if "강력매수" in grade: return "#DC2626"  # 강렬한 빨강 (Red 600)
    if "추천" in grade: return "#F97316"  # 주황 (Orange 500)
    if "관망" in grade: return "#9CA3AF"  # 회색
    if "손절위험" in grade: return "#7F1D1D"  # 진한 다크 레드 (Red 900)
    return "#6B7280"


def _find_similar_cases(code: str, n: int = 5) -> pd.DataFrame:
    path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not path.exists(): return pd.DataFrame()
    hist = pd.read_csv(path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    same = hist[hist["Code"].astype(str) == str(code)].copy()
    return same.sort_values("Date", ascending=False).head(n)


def _reason_text(row: pd.Series) -> list:
    reasons = []
    p_sw = row.get("슈퍼위너확률%", 0)
    p100 = row.get("100%+확률", 0)
    p50 = row.get("50%+확률", 0)
    ploss = row.get("손절확률%", 0)

    if p_sw >= 50: reasons.append(f"🏆 **슈퍼위너 확률 {p_sw:.0f}%** — peak ≥ 200% 도달 가능성 매우 높음")
    elif p_sw >= 30: reasons.append(f"⭐ **슈퍼위너 확률 {p_sw:.0f}%** — peak ≥ 200% 가능")
    elif p_sw >= 15: reasons.append(f"🌟 슈퍼위너 확률 {p_sw:.0f}% — 중상위 가능성")

    if p100 >= 50: reasons.append(f"💯 **100%+ 확률 {p100:.0f}%** — 2배 도달 매우 가능")
    elif p100 >= 30: reasons.append(f"💯 100%+ 확률 {p100:.0f}%")

    if p50 >= 60: reasons.append(f"📈 **50%+ 확률 {p50:.0f}%** — 절반 이상 상승 매우 가능")
    elif p50 >= 40: reasons.append(f"📈 50%+ 확률 {p50:.0f}%")

    slope = row.get("slope60", 0)
    past60 = row.get("past_60", 0)
    pos252 = row.get("pos_252_high", 0)

    if slope and slope >= 1: reasons.append(f"📊 60일 상승추세 강함 (slope60={slope:.1f})")
    elif slope and slope >= 0.3: reasons.append("📊 60일 완만한 상승추세")
    if past60 and 5 < past60 < 30: reasons.append(f"⚖️ 60일 +{past60:.0f}% 적정 상승")
    if pos252 and -30 < pos252 < -5: reasons.append(f"📍 52주 고점 -{abs(pos252):.0f}% — 눌림목 회복")

    if ploss < 30: reasons.append(f"✅ 손절 확률 {ploss:.0f}% — 안전한 자리")
    elif ploss < 50: reasons.append(f"⚖️ 손절 확률 {ploss:.0f}% — 보통")
    else: reasons.append(f"⚠️ 손절 확률 {ploss:.0f}% — 변동성 큰 종목")
    return reasons


def _render_pick_card(row: pd.Series, show_similar: bool = True):
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
    market = row.get("Market", "")
    date = row.get("Date", "")
    if isinstance(date, str): date = date[:10]
    else:
        try: date = pd.to_datetime(date).strftime("%Y-%m-%d")
        except: date = ""

    color = _grade_color(grade)

    st.markdown(f"""
<div style="border-left:4px solid {color};padding:14px 18px;background:rgba(0,0,0,0.02);
            border-radius:6px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="font-size:14px;color:{color};font-weight:700;">{grade}</span>
      <span style="font-size:20px;font-weight:800;margin-left:12px;">{name}</span>
      <span style="font-size:12px;color:#9CA3AF;margin-left:8px;">{code} · {market} · {date}</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;color:#9CA3AF;">매수가</div>
      <div style="font-size:18px;font-weight:700;">{close:,.0f}원</div>
    </div>
  </div>
  <div style="margin-top:12px;display:grid;grid-template-columns:repeat(5,1fr);gap:8px;font-size:12px;">
    <div><b>슈퍼점수</b><br><span style="font-size:14px;color:{color};font-weight:700;">{ss:.2f}</span></div>
    <div><b>예상 최고가</b><br><span style="font-size:14px;color:{color};font-weight:700;">+{peak_pred:.0f}%</span></div>
    <div><b>슈퍼위너 확률</b><br><span style="font-size:14px;font-weight:700;">{p_sw:.0f}%</span></div>
    <div><b>100%+ 확률</b><br><span style="font-size:14px;font-weight:700;">{p100:.0f}%</span></div>
    <div><b>50%+ / 손절</b><br>{p50:.0f}% / <span style="color:#EF4444">{ploss:.0f}%</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    reasons = _reason_text(row)
    if reasons:
        with st.expander(f"📌 {name} 강력추천 사유", expanded=False):
            for r in reasons:
                st.markdown(f"- {r}")

    if show_similar:
        similar = _find_similar_cases(code, n=5)
        if len(similar) > 0:
            with st.expander(f"🔍 {name} 과거 매수 사례 ({len(similar)}건)", expanded=False):
                hist_show = similar[[c for c in ["Date","Close","sell_close","ret_180d","peak_180d"] if c in similar.columns]].copy()
                hist_show = hist_show.rename(columns={
                    "Date":"발생일","Close":"매수가","sell_close":"매도가",
                    "ret_180d":"180일 수익률(%)","peak_180d":"최고가 도달(%)"
                })
                if "발생일" in hist_show.columns:
                    hist_show["발생일"] = pd.to_datetime(hist_show["발생일"]).dt.strftime("%Y-%m-%d")
                for c in ["180일 수익률(%)","최고가 도달(%)"]:
                    if c in hist_show.columns:
                        hist_show[c] = hist_show[c].round(1)
                st.dataframe(hist_show, hide_index=True, use_container_width=True)

                if "peak_180d" in similar.columns:
                    avg_peak = similar["peak_180d"].mean()
                    sw_count = (similar["peak_180d"]>=200).sum()
                    w100_count = (similar["peak_180d"]>=100).sum()
                    st.caption(f"📊 평균 최고가: +{avg_peak:.0f}% · 슈퍼위너 {sw_count}건 · 100%+ {w100_count}건")


def _button_multiselect(label: str, options: list, default: list, key_prefix: str):
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
            if is_on: selected.remove(opt)
            else: selected.append(opt)
            st.session_state[f"{key_prefix}_selected"] = selected
            st.rerun()
    return selected


def _load_json():
    p = CACHE / "today_picks.json"
    if not p.exists(): return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sort_strong(picks_list):
    strong = [p for p in picks_list if p.get("등급") == "★ 강력매수"]
    def priority(p):
        ss = p.get("SuperScore", 0) or 0
        psw = p.get("슈퍼위너확률%", 0) or 0
        return ss * 0.5 + psw * 0.01
    strong.sort(key=priority, reverse=True)
    return strong


# ============ 좌측 메뉴 페이지들 ============

def page_today_pick():
    """🎯 오늘의 추천"""
    st.markdown('<h1>🎯 오늘의 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    if not data:
        st.warning("추천 데이터가 없습니다."); return

    base_date = data.get("base_date", "")
    updated = data.get("updated_at", "")[:16]
    st.markdown(f"**기준일**: {base_date}  ·  **갱신**: {updated}")

    today = data.get("today", {})
    strong = _sort_strong(today.get("picks", []))

    if len(strong) == 0:
        st.info("📭 오늘 ★ 강력매수 추천 없음 — 현금 보유 권장")
        st.caption(f"(전체 시그널 {today.get('n', 0)}건 중 ★ 강력매수 0건)")
    else:
        st.markdown(f"### ⭐ ★ 강력매수 ({len(strong)}건)")
        st.caption("정렬: 슈퍼점수 + 슈퍼위너 확률 종합 우선순위")
        for p in strong:
            _render_pick_card(pd.Series(p), show_similar=True)


def _render_weekly_by_day(picks_list, week_limit=5):
    """일자별 그룹화 + 선착순 매수 (실전 룰: 매일 발견 즉시 매수, 주 누적 한도)"""
    strong = [p for p in picks_list if p.get("등급") == "★ 강력매수"]
    if len(strong) == 0:
        st.info("★ 강력매수 종목 없음")
        return

    # 일자별 그룹화
    by_day = {}
    for p in strong:
        d = p.get("Date", "")
        if isinstance(d, str): d = d[:10]
        else:
            try: d = pd.to_datetime(d).strftime("%Y-%m-%d")
            except: d = ""
        by_day.setdefault(d, []).append(p)
    dates_sorted = sorted(by_day.keys())  # 월→금 순

    # 선착순 매수 처리: 매일 발견 시 즉시 매수, 같은 일 내에서는 슈퍼점수 순
    cumulative = 0
    buy_n = 0
    for d in dates_sorted:
        day_picks = by_day[d]
        # 같은 일 내 슈퍼점수 정렬 (그날 발견된 종목 중에서)
        day_picks.sort(key=lambda p: (p.get("SuperScore", 0) or 0) * 0.5 + (p.get("슈퍼위너확률%", 0) or 0) * 0.01, reverse=True)
        for p in day_picks:
            cumulative += 1
            p["_seq"] = cumulative   # 순서 번호 (전체)
            if buy_n < week_limit:
                p["_will_buy"] = True
                buy_n += 1
                p["_buy_no"] = buy_n
            else:
                p["_will_buy"] = False
                p["_buy_no"] = None

    # 요약
    st.markdown(f"### ★ 강력매수 {len(strong)}건 발생 — 주 한도 {week_limit}건 (선착순 매수)")
    st.caption(f"📌 실전 룰: 매일 발견 즉시 매수 (NXT 19:50) · 같은 일 내 여러 건이면 슈퍼점수 높은 순 · 주 누적 {week_limit}건 채우면 그 주 stop")

    # 일자별 표시
    weekday_kr = ["월","화","수","목","금","토","일"]
    for d in dates_sorted:
        try:
            dt = pd.to_datetime(d)
            wd = weekday_kr[dt.weekday()]
        except:
            wd = ""
        day_picks = by_day[d]
        n_buy_today = sum(1 for p in day_picks if p["_will_buy"])

        st.markdown(f"#### 📆 {d} ({wd}요일) — {len(day_picks)}건 발생 / 매수 {n_buy_today}건")
        for p in day_picks:
            will = p["_will_buy"]
            buy_no = p.get("_buy_no")
            if will:
                badge = (
                    f'<span style="background:#DC2626;color:white;padding:2px 8px;border-radius:4px;'
                    f'font-size:11px;font-weight:700;margin-right:8px;">🔥 그 주 {buy_no}번째 매수</span>'
                )
            else:
                badge = (
                    f'<span style="background:#9CA3AF;color:white;padding:2px 8px;border-radius:4px;'
                    f'font-size:11px;font-weight:700;margin-right:8px;">⏭️ 주 한도 초과</span>'
                )
            ss = p.get("SuperScore", 0) or 0
            psw = p.get("슈퍼위너확률%", 0) or 0
            st.markdown(badge + f"<b>{p['Name']}</b> ({p['Code']})  ·  슈퍼점수 {ss:.2f}  ·  슈퍼위너 {psw:.0f}%", unsafe_allow_html=True)
            with st.expander(f"📌 {p['Name']} 상세", expanded=False):
                _render_pick_card(pd.Series(p), show_similar=True)
        st.markdown("---")


def page_this_week():
    """📅 이번 주 추천"""
    st.markdown('<h1>📅 이번 주 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    week = data.get("week", {})

    st.markdown(f"#### 주 시작일: {data.get('week_start', '')}")
    st.caption("일자별 정리 · 슈퍼점수 TOP 5만 실제 매수 (주 한도)")
    _render_weekly_by_day(week.get("picks", []), week_limit=5)


def page_last_week():
    """🗓️ 지난 주 추천"""
    st.markdown('<h1>🗓️ 지난 주 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    last_week = data.get("last_week", {})
    st.caption("일자별 정리 · 슈퍼점수 TOP 5만 실제 매수")
    _render_weekly_by_day(last_week.get("picks", []), week_limit=5)


def page_backtest():
    """📊 백테스트 (년월 선택)"""
    st.markdown('<h1>📊 백테스트 (2022-2026 walk-forward OOS)</h1>', unsafe_allow_html=True)

    # 년도별 요약
    yr_path = CACHE / "MASTER_best_yearly.csv"
    if yr_path.exists():
        yr = pd.read_csv(yr_path)
        yr = yr.rename(columns={
            "year":"년도","매수":"매수","SW":"슈퍼위너",
            "100+":"100%+","50+":"50%+","10+":"10%+",
            "손절":"손절","투자만":"투자(만원)",
            "수익만":"수익(만원)","수익률%":"수익률(%)"
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
    st.markdown("#### 📋 매수 종목 전체")

    picks_path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not picks_path.exists():
        st.warning("백테스트 데이터 없음"); return

    picks = pd.read_csv(picks_path)
    picks["Date"] = pd.to_datetime(picks["Date"])
    picks["년도"] = picks["Date"].dt.year
    picks["월"] = picks["Date"].dt.month

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

    years_avail = sorted(picks["년도"].dropna().unique().astype(int).tolist())
    sel_years = _button_multiselect(
        "년도 (다중 선택)", years_avail, default=years_avail, key_prefix="bt_year")

    months_avail = list(range(1, 13))
    sel_months = _button_multiselect(
        "월 (다중 선택)", months_avail, default=months_avail, key_prefix="bt_month")

    results_avail = ["🏆 슈퍼위너","💯 100%+","📈 50%+","✅ 10%+","💤 보합","❌ 손절","미정"]
    sel_results = _button_multiselect(
        "결과 (다중 선택)", results_avail,
        default=["🏆 슈퍼위너","💯 100%+","📈 50%+"], key_prefix="bt_result")

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

    show_map = {
        "Date":"일자","년도":"년도","월":"월",
        "Code":"종목코드","Name":"종목명","Market":"시장",
        "Close":"매수가","결과":"결과",
        "ret_180d":"180일수익률(%)","peak_180d":"최고가도달(%)",
        "sell_close":"매도가","sell_date":"매도일",
        "SuperScore_v2":"슈퍼점수",
        "p_sw":"슈퍼위너확률","p_100plus":"100%+확률",
        "p_50plus":"50%+확률","p_loss":"손절확률",
    }
    show_cols = [c for c in show_map if c in filtered.columns]
    display = filtered[show_cols].rename(columns=show_map).head(500)

    if "일자" in display.columns:
        display["일자"] = pd.to_datetime(display["일자"]).dt.strftime("%Y-%m-%d")
    for c in ["슈퍼위너확률","100%+확률","50%+확률","손절확률"]:
        if c in display.columns:
            display[c] = (display[c]*100).round(1).astype(str) + "%"

    st.dataframe(display, hide_index=True, use_container_width=True, height=600)
    st.caption(f"검색 결과 {len(filtered):,}건 중 최대 500건 표시")


def page_case_validation():
    """🔍 추천 사례 검증 (현재 추천 종목들의 과거 성과)"""
    st.markdown('<h1>🔍 추천 사례 검증</h1>', unsafe_allow_html=True)
    st.caption("현재 추천 종목들의 과거 5년 매수 사례 검증")

    data = _load_json()
    picks_all = []
    for key in ["today","week","last_week"]:
        for p in data.get(key, {}).get("picks", []):
            if p.get("등급") == "★ 강력매수":
                p_ = dict(p); p_["기간"] = {"today":"오늘","week":"이번주","last_week":"지난주"}[key]
                picks_all.append(p_)

    # 중복 제거 (같은 종목 한 번만)
    seen = set()
    uniq = []
    for p in picks_all:
        if p["Code"] not in seen:
            seen.add(p["Code"]); uniq.append(p)

    if len(uniq) == 0:
        st.info("현재 ★ 강력매수 종목 없음"); return

    st.markdown(f"### ★ 강력매수 종목 {len(uniq)}개의 과거 매수 사례")

    for p in uniq:
        code = p["Code"]; name = p["Name"]
        st.markdown(f"#### 🔎 {name} ({code}) — {p['기간']} 추천")

        similar = _find_similar_cases(code, n=20)
        if len(similar) == 0:
            st.caption("과거 매수 사례 없음")
            continue

        # 통계
        avg_peak = similar["peak_180d"].mean()
        avg_ret = similar["ret_180d"].mean()
        sw_count = (similar["peak_180d"]>=200).sum()
        w100_count = (similar["peak_180d"]>=100).sum()
        w50_count = (similar["peak_180d"]>=50).sum()
        loss_count = (similar["ret_180d"]<=-20).sum()
        n = len(similar)

        cols = st.columns(6)
        cols[0].metric("매수 사례", f"{n}건")
        cols[1].metric("평균 최고가", f"+{avg_peak:.0f}%")
        cols[2].metric("평균 수익률", f"{avg_ret:+.0f}%")
        cols[3].metric("슈퍼위너", f"{sw_count}건")
        cols[4].metric("100%+", f"{w100_count}건")
        cols[5].metric("손절", f"{loss_count}건")

        # 표
        show = similar[[c for c in ["Date","Close","sell_close","ret_180d","peak_180d"] if c in similar.columns]].copy()
        show = show.rename(columns={
            "Date":"발생일","Close":"매수가","sell_close":"매도가",
            "ret_180d":"180일수익률(%)","peak_180d":"최고가도달(%)"
        })
        if "발생일" in show.columns:
            show["발생일"] = pd.to_datetime(show["발생일"]).dt.strftime("%Y-%m-%d")
        for c in ["180일수익률(%)","최고가도달(%)"]:
            if c in show.columns:
                show[c] = show[c].round(1)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.markdown("---")


def page_buy_rule():
    """📋 매수 룰"""
    st.markdown('<h1>📋 매수 룰</h1>', unsafe_allow_html=True)

    st.markdown("""
### 🎯 최종 매수 룰 (단순)

```
[풀]   시총 상위 300종목 (KRX)
[시그널] 4 프리셋 ensemble + Score ≥ 40
[모델]  RF 4분류기 + peak 회귀

[슈퍼점수]
  슈퍼점수 = p_sw × 5 + p_100+ × 2 + p_50+ × 1 - p_loss × 3

[등급 자동 부여]
  ★ 강력매수: 슈퍼점수 상위 20%
  ○ 추천:    상위 20-40%
  - 관망:    중간
  ⚠️ 손절위험: 점수 낮음 + 손절확률 ≥ 70%

[매수]
  - ★ 강력매수만 매수
  - 우선순위: 슈퍼점수 + 슈퍼위너 확률 종합
  - 시점: 당일 NXT 19:50 시장가 (1순위) / D+1 시초가 (2순위)
  - 종목당 10만원 (자본 1억 → 0.1%)
  - 주 한도 5건

[매도]
  - 매수일 + 180거래일 후 정규장 종가
  - 익절/손절 룰 X
```

### 5년 OOS 결과 (2022-2026)

| 년도 | 매수 | 수익률 | 슈퍼위너 | 손절 |
|---|---|---|---|---|
| 2022 (약세) | 260 | +33.9% | 35 | 78 |
| 2023 (박스) | 260 | +57.9% | 49 | 47 |
| 2024 (혼조) | 265 | +68.5% | 56 | 48 |
| **2025 (강세)** ⭐ | 265 | **+292.5%** | **153** | **4** |
| 2026 (5월) | 105 | +97.0% | 34 | 5 |
| **5년 누적** | **1,155** | **+112.3%** | **327** | **182** |

→ 자본 1억 → **2.3억** (자본 2.3배)
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

    st.markdown("---")
    st.markdown("### 📊 등급 시스템")
    st.markdown("""
| 등급 | 기준 | 매수 여부 |
|---|---|---|
| **★ 강력매수** | 슈퍼점수 상위 20% | ✅ **매수** |
| ○ 추천 | 슈퍼점수 상위 20-40% | △ (자본 여유 시) |
| - 관망 | 중간 | ❌ 매수 X |
| ⚠️ 손절위험 | 점수 낮음 + 손절 ≥ 70% | ❌ 매수 절대 X |

### 가능성 태그 (자동 부여)

| 태그 | 기준 | 의미 |
|---|---|---|
| 🏆 슈퍼위너 강력후보 | 슈퍼위너 확률 ≥ 20% | peak 200%+ 매우 가능 |
| ⭐ 슈퍼위너후보 | ≥ 10% | 슈퍼위너 가능 |
| 💯 100%+ 가능 | 100%+ 확률 ≥ 30% | 2배 가능 |
| 📈 50%+ 가능 | 50%+ 확률 ≥ 50% | 절반 이상 상승 가능 |
| 🔻 손절 주의 | 손절 확률 40-70% | 변동성 주의 |
""")


# 기존 통합 페이지 (호환성)
def page_superscore():
    """기존 통합 페이지 — 좌측 메뉴로 이전됨"""
    st.info("💎 슈퍼스코어 추천 메뉴가 좌측 메뉴로 이전되었습니다. 좌측 메뉴에서 선택하세요.")
    cols = st.columns(3)
    if cols[0].button("🎯 오늘의 추천", use_container_width=True):
        st.session_state.page = "ss_today"; st.rerun()
    if cols[1].button("📅 이번 주", use_container_width=True):
        st.session_state.page = "ss_week"; st.rerun()
    if cols[2].button("📊 백테스트", use_container_width=True):
        st.session_state.page = "ss_backtest"; st.rerun()
