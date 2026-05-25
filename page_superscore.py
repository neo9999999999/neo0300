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
import numpy as np
import json
from pathlib import Path

CACHE = Path("cache")


def _grade_color(grade: str) -> str:
    if "슈퍼 강력매수" in grade: return "#B91C1C"  # 진한 빨강 (Red 700)
    if "강력매수" in grade: return "#F97316"  # 주황 (Orange 500)
    if "추천" in grade: return "#F59E0B"  # 호박색
    if "관망" in grade: return "#9CA3AF"
    if "손절위험" in grade: return "#7F1D1D"
    return "#6B7280"


def _find_similar_cases(code: str, n: int = 5) -> pd.DataFrame:
    """같은 종목 과거 사례"""
    path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not path.exists(): return pd.DataFrame()
    hist = pd.read_csv(path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    same = hist[hist["Code"].astype(str) == str(code)].copy()
    return same.sort_values("Date", ascending=False).head(n)


def _find_similar_stocks(target_row, n: int = 5, exclude_code: str = None) -> pd.DataFrame:
    """다른 종목 중 변수 비슷한 과거 매수 사례 (유클리드 거리)"""
    path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not path.exists(): return pd.DataFrame()
    hist = pd.read_csv(path)
    hist["Date"] = pd.to_datetime(hist["Date"])

    # 자기 자신 제외
    if exclude_code:
        hist = hist[hist["Code"].astype(str) != str(exclude_code)].copy()

    # 비교 변수
    var_cols = ["p_sw", "p_100plus", "p_50plus", "p_loss"]
    var_cols = [c for c in var_cols if c in hist.columns]
    if not var_cols:
        return pd.DataFrame()

    # 결측 제거
    hist = hist.dropna(subset=var_cols).copy()
    if len(hist) == 0: return pd.DataFrame()

    # 타깃 값
    target_vals = []
    for c in var_cols:
        # 0~1 (p_*) or %, 페이지에서 받은 row는 % (슈퍼위너확률% 등)
        v_raw = target_row.get(c, None)
        if v_raw is None:
            # 확률% 컬럼에서 변환
            pct_col = {"p_sw":"슈퍼위너확률%","p_100plus":"100%+확률","p_50plus":"50%+확률","p_loss":"손절확률%"}.get(c)
            if pct_col and pct_col in target_row.index:
                try:
                    v_raw = float(target_row[pct_col]) / 100
                except Exception:
                    v_raw = 0
            else: v_raw = 0
        target_vals.append(float(v_raw) if v_raw is not None else 0)

    # 유클리드 거리
    diff_sq = np.zeros(len(hist))
    for i, c in enumerate(var_cols):
        diff_sq += (hist[c].values - target_vals[i])**2
    hist["_distance"] = np.sqrt(diff_sq)

    # 가장 가까운 N건 (단 같은 일자 중복 제거)
    similar = hist.sort_values("_distance").drop_duplicates(["Date","Code"]).head(n*3)
    # 다른 종목 위주 (동일 종목 너무 중복 안 되게)
    seen_codes = set()
    pick = []
    for _, r in similar.iterrows():
        c = str(r["Code"])
        if c in seen_codes: continue
        seen_codes.add(c)
        pick.append(r)
        if len(pick) >= n: break
    if not pick: return pd.DataFrame()
    return pd.DataFrame(pick)


def _reason_text(row: pd.Series) -> list:
    reasons = []
    p_sw = row.get("슈퍼위너확률%", 0) or 0
    p100 = row.get("100%+확률", 0) or 0
    p50  = row.get("50%+확률", 0) or 0
    p30  = row.get("30%+확률", 0) or 0
    p10  = row.get("10%+확률", 0) or 0
    ploss = row.get("손절확률%", 0) or 0

    # OOS 보정 확률 기반 (실제 적중률)
    if p_sw >= 40: reasons.append(f"🏆 **슈퍼위너 확률 {p_sw:.0f}%** (OOS 실측) — 200% 도달 매우 유력")
    elif p_sw >= 25: reasons.append(f"⭐ **슈퍼위너 확률 {p_sw:.0f}%** (OOS 실측) — 200% 도달 가능")
    elif p_sw >= 15: reasons.append(f"🌟 슈퍼위너 확률 {p_sw:.0f}%")

    if p100 >= 50: reasons.append(f"💯 **100%+ 확률 {p100:.0f}%** — 2배 도달 매우 유력")
    elif p100 >= 30: reasons.append(f"💯 100%+ 확률 {p100:.0f}%")

    if p50 >= 60: reasons.append(f"📈 **50%+ 확률 {p50:.0f}%** — 절반 상승 매우 유력")
    elif p50 >= 40: reasons.append(f"📈 50%+ 확률 {p50:.0f}%")

    if p30 >= 70: reasons.append(f"📊 **30%+ 확률 {p30:.0f}%** — 상승 거의 확실")
    if p10 >= 85: reasons.append(f"✅ **10%+ 확률 {p10:.0f}%** — 최소 10% 상승 거의 보장")

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


def _render_fundamentals_block(row: pd.Series):
    """🏢 기업 분석 — 매출/영업이익 추이 + 성장 평가 + PER/PBR/외인지분율
    OOS 인사이트: 매출↑+영업이익↑가 슈퍼위너에 더 많이 나오지 않음.
    오히려 역성장/이익률 하락 종목이 슈퍼위너 80%+ 적중 (V자 반등).
    """
    name = row.get("Name", "")
    rev_24 = row.get("매출_2024")
    rev_25 = row.get("매출_2025")
    op_24  = row.get("영업이익_2024")
    op_25  = row.get("영업이익_2025")
    om_24  = row.get("영업이익률_2024")
    om_25  = row.get("영업이익률_2025")
    rev_yoy = row.get("매출_YoY")
    op_yoy  = row.get("영업이익_YoY")
    om_diff = row.get("영업이익률_변화")
    per    = row.get("PER_최신") or row.get("PER_num")
    pbr    = row.get("PBR_최신") or row.get("PBR_num")
    roe    = row.get("ROE_최신")
    fgnr   = row.get("외인소진율_num")
    grade  = row.get("성장등급", "데이터X")

    # 데이터 없으면 expander 자체를 생략
    if grade == "데이터X" and (rev_25 is None or (isinstance(rev_25,float) and pd.isna(rev_25))):
        return

    # 성장등급 색상
    grade_color = {
        "🚀 폭발적 성장": "#10B981",
        "📈 성장중":     "#22C55E",
        "⚖️ 보합":      "#9CA3AF",
        "📉 둔화":      "#F59E0B",
        "❌ 역성장":     "#DC2626",
    }.get(grade, "#6B7280")

    # OOS 인사이트 (역성장이 슈퍼위너 더 많이 나옴)
    insight = {
        "🚀 폭발적 성장": ("⚠️ 의외: 폭발 성장 종목은 SW 적중 42% — 의외로 낮음", "#F59E0B"),
        "📈 성장중":     ("📊 안정 성장은 SW 적중 31% — 보수적", "#9CA3AF"),
        "⚖️ 보합":      ("📊 보합 종목은 SW 적중 17% — 낮음", "#6B7280"),
        "📉 둔화":      ("✅ 둔화는 SW 적중 ~75% — 회복 베팅 유효", "#10B981"),
        "❌ 역성장":     ("🔥 V자 반등! 역성장 종목 SW 적중 81% — 최강", "#DC2626"),
        "데이터X":      ("", ""),
    }.get(grade, ("", ""))

    def _fmt_num(v, unit=""):
        if v is None or (isinstance(v,float) and pd.isna(v)): return "—"
        if unit == "억":
            return f"{v/1:,.0f}억"  # 이미 억원 단위로 저장됨 (확인 필요)
        if unit == "%":
            return f"{v:+.1f}%" if isinstance(v,(int,float)) else "—"
        return f"{v:,.2f}" if isinstance(v,(int,float)) else "—"

    # 매출/영업이익 단위 표시 (DART 단위: 억원)
    def _fmt_won(v):
        if v is None or pd.isna(v): return "—"
        try:
            if abs(v) >= 1e4: return f"{v/1e4:,.1f}조"
            else:             return f"{v:,.0f}억"
        except Exception: return "—"

    with st.expander(f"🏢 {name} 기업 분석 (매출·영업이익 추이 + 성장 평가)", expanded=False):
        # 성장 등급 배지 (큰 컬러 박스)
        st.markdown(f"""
<div style="background:{grade_color};color:white;padding:12px 16px;border-radius:8px;margin-bottom:10px;text-align:center;">
  <div style="font-size:11px;opacity:0.85;letter-spacing:1px;">최근 2년 성장 평가</div>
  <div style="font-size:22px;font-weight:900;margin-top:4px;">{grade}</div>
  {f'<div style="font-size:12px;margin-top:6px;opacity:0.95;">{insight[0]}</div>' if insight[0] else ''}
</div>
""", unsafe_allow_html=True)

        # 매출/영업이익 표
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📊 매출액**")
            st.markdown(f"- 2024년: **{_fmt_won(rev_24)}**")
            st.markdown(f"- 2025년: **{_fmt_won(rev_25)}**")
            if pd.notna(rev_yoy):
                color = "#10B981" if rev_yoy > 0 else "#DC2626"
                st.markdown(f"- YoY: <span style='color:{color};font-weight:700;'>{rev_yoy:+.1f}%</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("**💰 영업이익**")
            st.markdown(f"- 2024년: **{_fmt_won(op_24)}**")
            st.markdown(f"- 2025년: **{_fmt_won(op_25)}**")
            if pd.notna(op_yoy):
                color = "#10B981" if op_yoy > 0 else "#DC2626"
                st.markdown(f"- YoY: <span style='color:{color};font-weight:700;'>{op_yoy:+.1f}%</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("**📈 영업이익률**")
            if pd.notna(om_24): st.markdown(f"- 2024년: **{om_24:.2f}%**")
            else: st.markdown("- 2024년: —")
            if pd.notna(om_25): st.markdown(f"- 2025년: **{om_25:.2f}%**")
            else: st.markdown("- 2025년: —")
            if pd.notna(om_diff):
                color = "#10B981" if om_diff > 0 else "#DC2626"
                st.markdown(f"- 변화: <span style='color:{color};font-weight:700;'>{om_diff:+.2f}%p</span>", unsafe_allow_html=True)

        st.markdown("---")
        # 밸류에이션 + 지분
        col4, col5, col6, col7 = st.columns(4)
        def _per_eval(v):
            if v is None or pd.isna(v): return "—", "#6B7280"
            try: v = float(v)
            except: return "—", "#6B7280"
            if v < 0: return f"{v:.1f} (적자)", "#DC2626"
            if v < 10: return f"{v:.1f} (저평가)", "#10B981"
            if v < 20: return f"{v:.1f} (보통)", "#9CA3AF"
            return f"{v:.1f} (고평가)", "#F59E0B"
        def _pbr_eval(v):
            if v is None or pd.isna(v): return "—", "#6B7280"
            try: v = float(v)
            except: return "—", "#6B7280"
            if v < 1: return f"{v:.2f} (자산↓)", "#10B981"
            if v < 2: return f"{v:.2f} (보통)", "#9CA3AF"
            return f"{v:.2f} (자산↑)", "#F59E0B"
        per_txt, per_col = _per_eval(per)
        pbr_txt, pbr_col = _pbr_eval(pbr)
        with col4:
            st.markdown(f"**PER** <span style='color:{per_col};font-weight:700;'>{per_txt}</span>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"**PBR** <span style='color:{pbr_col};font-weight:700;'>{pbr_txt}</span>", unsafe_allow_html=True)
        with col6:
            if pd.notna(roe):
                c = "#10B981" if roe > 10 else ("#F59E0B" if roe > 5 else "#DC2626")
                st.markdown(f"**ROE** <span style='color:{c};font-weight:700;'>{roe:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown("**ROE** —")
        with col7:
            if pd.notna(fgnr):
                st.markdown(f"**외인지분** <span style='font-weight:700;'>{fgnr:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown("**외인지분** —")

        st.caption(
            "💡 **5년 OOS 분석 결과**: 매출+영업이익이 둘 다 큰 폭으로 성장한 종목보다, "
            "**역성장/이익률 하락한 종목**이 슈퍼위너(200%+)에 더 많이 나옴 (V자 반등). "
            "성장세는 안정성 지표이지 슈퍼위너 예측 지표는 아님."
        )


def _prob_bar(label: str, prob_pct: float, color: str, is_main: bool = False):
    """미니 확률 바 (HTML)"""
    width = max(0, min(100, prob_pct))
    if is_main:
        return (
            f'<div style="margin:6px 0;">'
            f'  <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;">'
            f'    <span style="color:{color};">{label}</span>'
            f'    <span style="color:{color};">{prob_pct:.0f}%</span>'
            f'  </div>'
            f'  <div style="background:#F3F4F6;border-radius:6px;height:10px;overflow:hidden;margin-top:2px;">'
            f'    <div style="background:{color};width:{width}%;height:100%;"></div>'
            f'  </div>'
            f'</div>'
        )
    return (
        f'<div style="margin:3px 0;">'
        f'  <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7280;">'
        f'    <span>{label}</span><span>{prob_pct:.0f}%</span>'
        f'  </div>'
        f'  <div style="background:#F3F4F6;border-radius:4px;height:5px;overflow:hidden;">'
        f'    <div style="background:{color};width:{width}%;height:100%;opacity:0.6;"></div>'
        f'  </div>'
        f'</div>'
    )


def _render_pick_card(row: pd.Series, show_similar: bool = True):
    grade = row.get("등급", "")
    code = row.get("Code", "")
    name = row.get("Name", "")
    close = row.get("Close", 0)
    ss = row.get("SuperScore", 0)
    peak_pred = row.get("예상peak%", 0)
    p_sw  = row.get("슈퍼위너확률%", 0) or 0
    p100  = row.get("100%+확률", 0) or 0
    p50   = row.get("50%+확률", 0) or 0
    p30   = row.get("30%+확률", 0) or 0
    p10   = row.get("10%+확률", 0) or 0
    ploss = row.get("손절확률%", 0) or 0
    market = row.get("Market", "")
    date = row.get("Date", "")
    if isinstance(date, str): date = date[:10]
    else:
        try: date = pd.to_datetime(date).strftime("%Y-%m-%d")
        except: date = ""

    color = _grade_color(grade)

    # ===== 메인 배지 (가장 가능성 높은 도달 구간) =====
    main_label    = row.get("메인도달", "")
    main_prob     = row.get("메인확률%", 0) or 0
    main_color    = row.get("메인컬러", color) or color
    main_strength = row.get("메인강도", "")

    # 메인이 없으면 자동 산정 (확률 큰 순)
    if not main_label:
        candidates = [
            ("🏆 슈퍼위너 200%+", p_sw,  "#B91C1C"),
            ("💯 100%+ (2배)",   p100,  "#DC2626"),
            ("📈 50%+",          p50,   "#F97316"),
            ("📊 30%+",          p30,   "#F59E0B"),
            ("✅ 10%+",          p10,   "#10B981"),
        ]
        for lbl, pr, cc in candidates:
            if pr >= 50:
                main_label, main_prob, main_color = lbl, pr, cc
                main_strength = "★★★ 도달 매우 유력"; break
        if not main_label:
            for lbl, pr, cc in candidates:
                if pr >= 30:
                    main_label, main_prob, main_color = lbl, pr, cc
                    main_strength = "★★ 도달 가능"; break
        if not main_label:
            for lbl, pr, cc in candidates[:2]:
                if pr >= 15:
                    main_label, main_prob, main_color = lbl, pr, cc
                    main_strength = "★ 도달 후보"; break
        if not main_label:
            main_label, main_prob, main_color = "✅ 10%+ 권", p10, "#10B981"
            main_strength = ""

    st.markdown(f"""
<div style="border:2px solid {main_color};padding:0;background:white;
            border-radius:10px;margin-bottom:10px;overflow:hidden;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);">

  <!-- 상단 헤더 -->
  <div style="padding:14px 18px;display:flex;justify-content:space-between;align-items:center;
              background:linear-gradient(90deg,{main_color}10,{main_color}05);">
    <div>
      <span style="background:{color};color:white;padding:3px 10px;border-radius:4px;
                   font-size:12px;font-weight:700;">{grade}</span>
      <span style="font-size:22px;font-weight:800;margin-left:12px;color:#111;">{name}</span>
      <div style="font-size:11px;color:#9CA3AF;margin-top:4px;">{code} · {market} · {date}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#9CA3AF;">매수가</div>
      <div style="font-size:20px;font-weight:800;color:#111;">{close:,.0f}원</div>
    </div>
  </div>

  <!-- 메인 배지: 가장 가능성 높은 도달 구간 (강한 표시) -->
  <div style="padding:16px 18px;background:{main_color};color:white;text-align:center;">
    <div style="font-size:11px;opacity:0.85;letter-spacing:1px;">가장 가능성 높은 도달</div>
    <div style="font-size:28px;font-weight:900;margin:4px 0;">
      {main_label}
    </div>
    <div style="font-size:16px;font-weight:700;">
      OOS 적중률 <span style="font-size:22px;">{main_prob:.0f}%</span>
      <span style="margin-left:8px;opacity:0.9;font-size:13px;">· {main_strength}</span>
    </div>
  </div>

  <!-- 보조 정보: 슈퍼점수 + 예상최고가 -->
  <div style="padding:10px 18px;display:flex;gap:18px;background:#FAFAFA;
              border-top:1px solid #F3F4F6;font-size:12px;">
    <div><span style="color:#9CA3AF;">슈퍼점수</span>
         <b style="color:{main_color};margin-left:6px;font-size:14px;">{ss:.2f}</b></div>
    <div><span style="color:#9CA3AF;">예상 최고가</span>
         <b style="color:{main_color};margin-left:6px;font-size:14px;">+{peak_pred:.0f}%</b></div>
    <div><span style="color:#9CA3AF;">손절 확률 (-20%↓)</span>
         <b style="color:{'#DC2626' if ploss >= 25 else '#10B981'};margin-left:6px;font-size:14px;">{ploss:.0f}%</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # 도달 구간별 상세 확률 (미니 바)
    with st.expander(f"📊 {name} 도달 구간별 OOS 적중 확률", expanded=False):
        st.caption("실제 5년 백테스트(825건) 기반 보정 확률")
        bars_html = ""
        rows_data = [
            ("🏆 슈퍼위너 200%+", p_sw,  "#B91C1C"),
            ("💯 100%+ (2배)",   p100,  "#DC2626"),
            ("📈 50%+",          p50,   "#F97316"),
            ("📊 30%+",          p30,   "#F59E0B"),
            ("✅ 10%+",          p10,   "#10B981"),
            ("❌ 손절 (-20%↓)",  ploss, "#7F1D1D"),
        ]
        # 가장 큰 확률(=메인) 강조
        max_prob = max([r[1] for r in rows_data[:-1]])
        for lbl, pr, cc in rows_data:
            is_main = (pr == max_prob and lbl != "❌ 손절 (-20%↓)")
            bars_html += _prob_bar(lbl, pr, cc, is_main=is_main)
        st.markdown(bars_html, unsafe_allow_html=True)

    # 🏢 기업 분석 (펀더멘털)
    _render_fundamentals_block(row)

    reasons = _reason_text(row)
    if reasons:
        with st.expander(f"📌 {name} 강력추천 사유", expanded=False):
            for r in reasons:
                st.markdown(f"- {r}")

    if show_similar:
        # 1) 같은 종목 과거 사례
        similar_same = _find_similar_cases(code, n=5)
        if len(similar_same) > 0:
            with st.expander(f"🔍 {name} 과거 매수 사례 ({len(similar_same)}건)", expanded=False):
                hist_show = similar_same[[c for c in ["Date","Close","sell_close","ret_180d","peak_180d"] if c in similar_same.columns]].copy()
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

                if "peak_180d" in similar_same.columns:
                    avg_peak = similar_same["peak_180d"].mean()
                    sw_count = (similar_same["peak_180d"]>=200).sum()
                    w100_count = (similar_same["peak_180d"]>=100).sum()
                    st.caption(f"📊 평균 최고가: +{avg_peak:.0f}% · 슈퍼위너 {sw_count}건 · 100%+ {w100_count}건")

        # 2) 비슷한 패턴의 다른 종목들 (유사도 기반)
        similar_other = _find_similar_stocks(row, n=5, exclude_code=code)
        if len(similar_other) > 0:
            with st.expander(f"🎭 {name} 와 비슷한 패턴 종목 ({len(similar_other)}건)", expanded=False):
                show = similar_other[[c for c in ["Date","Code","Name","Market","Close","sell_close","ret_180d","peak_180d","p_sw","p_loss"] if c in similar_other.columns]].copy()
                show = show.rename(columns={
                    "Date":"발생일","Code":"종목코드","Name":"종목명","Market":"시장",
                    "Close":"매수가","sell_close":"매도가",
                    "ret_180d":"180일수익률(%)","peak_180d":"최고가도달(%)",
                    "p_sw":"슈퍼위너확률","p_loss":"손절확률"
                })
                if "발생일" in show.columns:
                    show["발생일"] = pd.to_datetime(show["발생일"]).dt.strftime("%Y-%m-%d")
                for c in ["180일수익률(%)","최고가도달(%)"]:
                    if c in show.columns:
                        show[c] = show[c].round(1)
                for c in ["슈퍼위너확률","손절확률"]:
                    if c in show.columns:
                        show[c] = (show[c]*100).round(0).astype(str) + "%"
                st.dataframe(show, hide_index=True, use_container_width=True)

                avg_peak_o = similar_other["peak_180d"].mean()
                sw_count_o = (similar_other["peak_180d"]>=200).sum()
                w100_count_o = (similar_other["peak_180d"]>=100).sum()
                loss_count_o = (similar_other["ret_180d"]<=-20).sum()
                st.caption(f"📊 유사 패턴 평균 최고가: +{avg_peak_o:.0f}% · 슈퍼위너 {sw_count_o}건 · 100%+ {w100_count_o}건 · 손절 {loss_count_o}건")
                st.caption("💡 슈퍼위너/100%+/50%+/손절 확률이 비슷한 다른 종목의 과거 매수 결과")


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
    """슈퍼강력 + 강력매수만 (그 외 제외) - 슈퍼강력 먼저, 그 안에서 점수순"""
    buyable = [p for p in picks_list if p.get("등급") in ("🔥 슈퍼 강력매수", "★ 강력매수")]
    def priority(p):
        grade_bonus = 100 if "슈퍼" in p.get("등급", "") else 0
        ss = p.get("SuperScore", 0) or 0
        psw = p.get("슈퍼위너확률%", 0) or 0
        return grade_bonus + ss * 0.5 + psw * 0.01
    buyable.sort(key=priority, reverse=True)
    return buyable


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


def _render_weekly_by_day(picks_list, week_limit=999):
    """일자별 그룹화 + 선착순 매수 (슈퍼강력 + 강력매수만)"""
    strong = [p for p in picks_list if p.get("등급") in ("🔥 슈퍼 강력매수", "★ 강력매수")]
    if len(strong) == 0:
        st.info("🔥 슈퍼 강력매수 / ★ 강력매수 종목 없음")
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
        # 같은 일 내: 슈퍼강력 먼저, 그 안에서 점수 순
        def pri(p):
            bonus = 100 if "슈퍼" in p.get("등급","") else 0
            return bonus + (p.get("SuperScore",0) or 0)*0.5 + (p.get("슈퍼위너확률%",0) or 0)*0.01
        day_picks.sort(key=pri, reverse=True)
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


def _weekly_limit_selector(key_prefix: str) -> int:
    """주 매수 한도 선택 버튼"""
    options = {
        "무제한 (★ 모두 매수)": 999,
        "10건/주": 10,
        "7건/주": 7,
        "5건/주": 5,
        "3건/주": 3,
    }
    if f"{key_prefix}_limit" not in st.session_state:
        st.session_state[f"{key_prefix}_limit"] = "무제한 (★ 모두 매수)"

    st.markdown("**주 매수 한도** (자본에 맞춰 선택)")
    cols = st.columns(len(options))
    for i, label in enumerate(options.keys()):
        is_sel = st.session_state[f"{key_prefix}_limit"] == label
        btn_type = "primary" if is_sel else "secondary"
        if cols[i].button(label, key=f"{key_prefix}_lim_{label}",
                           type=btn_type, use_container_width=True):
            st.session_state[f"{key_prefix}_limit"] = label
            st.rerun()
    return options[st.session_state[f"{key_prefix}_limit"]]


def page_this_week():
    """📅 이번 주 추천"""
    st.markdown('<h1>📅 이번 주 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    week = data.get("week", {})

    st.markdown(f"#### 주 시작일: {data.get('week_start', '')}")
    st.caption("⭐ 자본 충분하면 무제한(★ 모두 매수) 추천 · 자본 적으면 한도 설정")
    limit = _weekly_limit_selector("week")
    _render_weekly_by_day(week.get("picks", []), week_limit=limit)


def page_last_week():
    """🗓️ 지난 주 추천"""
    st.markdown('<h1>🗓️ 지난 주 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    last_week = data.get("last_week", {})
    st.caption("⭐ 자본 충분하면 무제한(★ 모두 매수) · 한도 자유 선택")
    limit = _weekly_limit_selector("lw")
    _render_weekly_by_day(last_week.get("picks", []), week_limit=limit)


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
    st.caption("현재 🔥 슈퍼 강력매수 / ★ 강력매수 종목들의 과거 5년 매수 사례 검증")

    data = _load_json()
    picks_all = []
    for key in ["today","week","last_week"]:
        for p in data.get(key, {}).get("picks", []):
            if p.get("등급") in ("🔥 슈퍼 강력매수", "★ 강력매수"):
                p_ = dict(p); p_["기간"] = {"today":"오늘","week":"이번주","last_week":"지난주"}[key]
                picks_all.append(p_)

    # 중복 제거 (같은 종목 한 번만)
    seen = set()
    uniq = []
    for p in picks_all:
        if p["Code"] not in seen:
            seen.add(p["Code"]); uniq.append(p)

    if len(uniq) == 0:
        st.info("현재 슈퍼강력/강력매수 종목 없음"); return

    st.markdown(f"### 매수 후보 {len(uniq)}개의 과거 매수 사례")

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

[등급 자동 부여 - 2단계만]
  🔥 슈퍼 강력매수: 슈퍼점수 상위 5%  (일 2-3건)
  ★ 강력매수:    슈퍼점수 상위 5-20% (일 3-5건)
  (그 외 등급은 매수 X, 표시 안 함)

[매수 우선순위]
  1순위: 🔥 슈퍼 강력매수 (점수 높은 순)
  2순위: ★ 강력매수 (점수 + 슈퍼위너 확률)
  시점: 당일 NXT 19:50 시장가 (1순위) / D+1 시초가 (2순위)
  종목당 10만원

[매도]
  - 매수일 + 180거래일 후 정규장 종가
  - 익절/손절 룰 X
```

### 등급별 OOS 성과 (5년)

| 등급 | 일평균 | 매수 (5년) | SW% | 손절% | **수익률** |
|---|---|---|---|---|---|
| **🔥 슈퍼 강력매수** (상위 5%) | 2.3건 | 1,865 | **28.6%** | **11.4%** | **+109.0%** |
| **★ 강력매수** (상위 5-20%) | ~3건 | ~2,437 | ~15% | ~13% | ~+75% |
| 합계 (모두 매수) | ~5건 | 4,302 | 19.9% | 12.5% | +88.0% |

→ **🔥 슈퍼 강력매수**가 농도/효율 최강 (자본 부족하면 이것만)
→ **★ 강력매수**는 보완 (자본 충분할 때 둘 다)

### 5년 OOS 결과 (모두 매수, 자본 1억 가정)

| 년도 | 시장 | 매수 | 수익률 |
|---|---|---|---|
| 2022 (약세) | -25% | 260 | +33.9% |
| 2023 (박스) | +18% | 260 | +57.9% |
| 2024 (혼조) | +9% | 265 | +68.5% |
| **2025 (강세)** ⭐ | +35% | 265 | **+292.5%** |
| 2026 (5월) | +12% | 105 | +97.0% |
| **5년 누적** | | **1,155** | **+112.3%** |

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
