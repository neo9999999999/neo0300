"""
SuperScore 추천 페이지들 — 미니멀 클린 디자인 (이모지 제거 + 면처리 강조)
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

CACHE = Path("cache")

# ============== 유틸: 이모지 / 라벨 정리 ==============

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF☀-➿⬀-⯿←-⇿"
    "✀-➿〰️‍]+",
    flags=re.UNICODE,
)


def _clean(s: str) -> str:
    if s is None: return ""
    s = _EMOJI_RE.sub("", str(s))
    # 별표/특수문자 제거
    s = s.replace("★", "").replace("☆", "").replace("·", "·")
    return s.strip()


def _grade_short(grade: str) -> str:
    """등급명을 깔끔한 텍스트로 — 2단계만 (슈퍼강력매수 / 추천매수)"""
    g = _clean(grade)
    if "슈퍼" in g and "강력" in g: return "슈퍼강력매수"
    if "강력매수" in g: return "추천매수"   # 기존 '강력매수' → '추천매수'
    if "추천매수" in g: return "추천매수"
    if "추천" in g: return "추천매수"
    if "관망" in g: return "관망"
    if "손절" in g: return "위험"
    return g or "일반"


def _grade_color(grade: str) -> str:
    g = _clean(grade)
    if "슈퍼" in g and "강력" in g: return "#B91C1C"   # 슈퍼강력매수 진빨강
    if "강력매수" in g or "추천매수" in g: return "#F97316"   # 추천매수 주황
    if "추천" in g: return "#F59E0B"
    if "위험" in g or "손절" in g: return "#7F1D1D"
    return "#6B7280"


def _bucket_label_plain(label: str) -> str:
    """메인 도달 라벨에서 이모지 제거 → '100%+ 도달', '50%+ 도달' 식"""
    s = _clean(label)
    # 흔한 형태: "💯 100%+ (2배)" → "100%+"
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = s.replace("슈퍼위너", "").strip()
    if not s or s == "+":
        s = "10%+"
    return f"{s} 도달"


# ============== 유사 종목 조회 ==============

def _find_similar_cases(code: str, n: int = 5) -> pd.DataFrame:
    path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not path.exists(): return pd.DataFrame()
    hist = pd.read_csv(path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    same = hist[hist["Code"].astype(str) == str(code)].copy()
    return same.sort_values("Date", ascending=False).head(n)


# === 유사도 차트 특성 (캐시) ===
_ENRICHED_CACHE = {}

def _get_enriched_signals():
    """signals_2000_enriched.parquet 캐시 로드"""
    if "df" not in _ENRICHED_CACHE:
        p = CACHE / "signals_2000_enriched.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Code"] = df["Code"].astype(str).str.zfill(6)
            _ENRICHED_CACHE["df"] = df
        else:
            _ENRICHED_CACHE["df"] = None
    return _ENRICHED_CACHE["df"]


# 유사도 벡터 차원
# - 가격 흐름: past_60, past_120, past_240, slope60, runup60
# - 신고가/위치: pos_252_high, pos_120_high, pos_60_high, days_since_52w_high
# - 변동성/조정: drawdown60, range60_pct, range120_pct
# - 거래/수급: vol_ratio, For_5d, For_20d, Inst_5d, Inst_20d
# - 모델 확률: p_sw, p_100plus, p_50plus
SIMILARITY_FEATURES = {
    "past_60":       {"weight": 1.5, "scale": 30},   # 60일 등락률
    "past_120":      {"weight": 1.2, "scale": 50},
    "past_240":      {"weight": 1.0, "scale": 80},
    "slope60":       {"weight": 1.2, "scale": 2.0},
    "runup60":       {"weight": 1.0, "scale": 40},
    "pos_252_high":  {"weight": 1.5, "scale": 30},   # 신고가 위치
    "pos_120_high":  {"weight": 1.0, "scale": 25},
    "pos_60_high":   {"weight": 1.0, "scale": 20},
    "drawdown60":    {"weight": 1.0, "scale": 20},
    "range60_pct":   {"weight": 0.8, "scale": 30},
    "vol_ratio":     {"weight": 0.8, "scale": 2.0},  # 거래량 비율
    "For_5d":        {"weight": 0.8, "scale": 200},  # 외인 5일 (억)
    "For_20d":       {"weight": 0.6, "scale": 800},
    "Inst_5d":       {"weight": 0.8, "scale": 200},
    "Inst_20d":      {"weight": 0.6, "scale": 800},
    "p_sw":          {"weight": 0.7, "scale": 0.2},  # 모델 확률 (보조)
    "p_100plus":     {"weight": 0.5, "scale": 0.2},
}


def _find_similar_stocks(target_row, n: int = 5, exclude_code: str = None) -> pd.DataFrame:
    """차트/신고가/수급/모델확률 다차원 유사도 (시그널 풀 + 백테스트 결과 머지)"""
    sigs = _get_enriched_signals()
    if sigs is None: return pd.DataFrame()
    master_path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not master_path.exists(): return pd.DataFrame()
    master = pd.read_csv(master_path)
    master["Date"] = pd.to_datetime(master["Date"])
    master["Code"] = master["Code"].astype(str).str.zfill(6)
    master_keys = master.set_index(["Date","Code"])[["peak_180d","ret_180d","sell_close","p_sw","p_100plus","p_50plus","p_loss"]]

    # 시그널 풀에서 결과 있는 것만 (백테스트 결과 머지)
    pool = sigs[sigs["Code"].isin(master["Code"].unique())].copy()
    # 컬럼 충돌 방지 — sigs에 이미 peak_180d/ret_180d/sell_close 있으면 drop
    pool = pool.drop(columns=["peak_180d","ret_180d","sell_close"], errors="ignore")
    pool = pool.merge(master_keys.reset_index()[["Date","Code","peak_180d","ret_180d","sell_close"]],
                       on=["Date","Code"], how="inner")
    if exclude_code:
        pool = pool[pool["Code"] != str(exclude_code).zfill(6)].copy()

    # 사용 가능한 피처만
    feat_cols = [c for c in SIMILARITY_FEATURES if c in pool.columns]
    if not feat_cols: return pd.DataFrame()
    pool = pool.dropna(subset=feat_cols).copy()
    if len(pool) == 0: return pd.DataFrame()

    # 타깃 벡터
    target_vals = []
    for c in feat_cols:
        v = target_row.get(c)
        if v is None and c in ("p_sw","p_100plus","p_50plus","p_loss"):
            pct_col = {"p_sw":"슈퍼위너확률%","p_100plus":"100%+확률","p_50plus":"50%+확률","p_loss":"손절확률%"}.get(c)
            if pct_col and pct_col in target_row.index:
                try: v = float(target_row[pct_col]) / 100
                except Exception: v = None
        try: target_vals.append(float(v) if v is not None and not (isinstance(v,float) and pd.isna(v)) else None)
        except Exception: target_vals.append(None)

    # 결측은 가중 0으로 (해당 차원 제외)
    diff_sq = np.zeros(len(pool))
    for i, c in enumerate(feat_cols):
        cfg = SIMILARITY_FEATURES[c]
        w = cfg["weight"]; s = cfg["scale"]
        tv = target_vals[i]
        if tv is None or s == 0: continue
        vals = pool[c].values
        d = (vals - tv) / s
        diff_sq += w * (d ** 2)
    pool["_distance"] = np.sqrt(diff_sq)

    similar = pool.sort_values("_distance").drop_duplicates(["Date","Code"])

    # 다양성: 동일 종목 중복 회피, 추가로 가장 가까운 5건
    seen = set(); pick = []
    for _, r in similar.iterrows():
        c = str(r["Code"])
        if c in seen: continue
        seen.add(c); pick.append(r)
        if len(pick) >= n: break
    if not pick: return pd.DataFrame()
    return pd.DataFrame(pick)


# ============== 색상 유틸 ==============

def _peak_color(v):
    if pd.isna(v): return "#9CA3AF"
    if v >= 200: return "#B91C1C"
    if v >= 100: return "#DC2626"
    if v >= 50:  return "#F97316"
    if v >= 30:  return "#F59E0B"
    if v >= 10:  return "#10B981"
    return "#9CA3AF"


def _ret_color(v):
    """한국식: + 상승 빨강, - 하락 파랑"""
    if pd.isna(v): return "#9CA3AF"
    if v >= 100:  return "#7F1D1D"   # 진빨강 (대박)
    if v >= 30:   return "#B91C1C"   # 진빨강
    if v >= 5:    return "#DC2626"   # 빨강
    if v > 0:     return "#EF4444"   # 연빨강
    if v == 0:    return "#6B7280"   # 보합 (드묾)
    if v > -5:    return "#3B82F6"   # 연파랑
    if v > -20:   return "#2563EB"   # 파랑
    return "#1D4ED8"                  # 진파랑 (손절)


def _peak_label_plain(v):
    if pd.isna(v): return "—"
    if v >= 200: return "SW"
    if v >= 100: return "100+"
    if v >= 50:  return "50+"
    if v >= 30:  return "30+"
    if v >= 10:  return "10+"
    return "보합"


# ============== 사유 텍스트 (이모지 없음) ==============

def _reason_text(row: pd.Series) -> list:
    reasons = []
    p_sw = row.get("슈퍼위너확률%", 0) or 0
    p100 = row.get("100%+확률", 0) or 0
    p50  = row.get("50%+확률", 0) or 0
    p30  = row.get("30%+확률", 0) or 0
    p10  = row.get("10%+확률", 0) or 0
    ploss = row.get("손절확률%", 0) or 0

    if p_sw >= 40: reasons.append(f"**슈퍼위너 확률 {p_sw:.0f}%** — 200% 도달 매우 유력")
    elif p_sw >= 25: reasons.append(f"**슈퍼위너 확률 {p_sw:.0f}%** — 200% 도달 가능")
    elif p_sw >= 15: reasons.append(f"슈퍼위너 확률 {p_sw:.0f}%")

    if p100 >= 50: reasons.append(f"**100%+ 확률 {p100:.0f}%** — 2배 도달 매우 유력")
    elif p100 >= 30: reasons.append(f"100%+ 확률 {p100:.0f}%")

    if p50 >= 60: reasons.append(f"**50%+ 확률 {p50:.0f}%** — 절반 상승 매우 유력")
    elif p50 >= 40: reasons.append(f"50%+ 확률 {p50:.0f}%")

    if p30 >= 70: reasons.append(f"**30%+ 확률 {p30:.0f}%** — 상승 거의 확실")
    if p10 >= 85: reasons.append(f"**10%+ 확률 {p10:.0f}%** — 최소 10% 상승 거의 보장")

    slope = row.get("slope60", 0)
    past60 = row.get("past_60", 0)
    pos252 = row.get("pos_252_high", 0)
    if slope and slope >= 1: reasons.append(f"60일 상승추세 강함 (slope60={slope:.1f})")
    elif slope and slope >= 0.3: reasons.append("60일 완만한 상승추세")
    if past60 and 5 < past60 < 30: reasons.append(f"60일 +{past60:.0f}% 적정 상승")
    if pos252 and -30 < pos252 < -5: reasons.append(f"52주 고점 -{abs(pos252):.0f}% — 눌림목 회복")

    if ploss < 30: reasons.append(f"손절 확률 {ploss:.0f}% — 안전한 자리")
    elif ploss < 50: reasons.append(f"손절 확률 {ploss:.0f}% — 보통")
    else: reasons.append(f"손절 확률 {ploss:.0f}% — 변동성 큰 종목")
    return reasons


# ============== 기업 분석 블록 (이모지 없음) ==============

def _render_fundamentals_block(row: pd.Series):
    name = row.get("Name", "")
    rev_24 = row.get("매출_2024"); rev_25 = row.get("매출_2025")
    op_24  = row.get("영업이익_2024"); op_25 = row.get("영업이익_2025")
    om_24  = row.get("영업이익률_2024"); om_25 = row.get("영업이익률_2025")
    rev_yoy = row.get("매출_YoY"); op_yoy = row.get("영업이익_YoY")
    om_diff = row.get("영업이익률_변화")
    per = row.get("PER_최신") or row.get("PER_num")
    pbr = row.get("PBR_최신") or row.get("PBR_num")
    roe = row.get("ROE_최신")
    fgnr = row.get("외인소진율_num")
    grade_raw = row.get("성장등급", "데이터X")

    if "데이터X" in str(grade_raw) and (rev_25 is None or (isinstance(rev_25,float) and pd.isna(rev_25))):
        return

    grade = _clean(grade_raw)
    grade_color = {
        "폭발적 성장": "#10B981",
        "성장중":     "#22C55E",
        "보합":      "#9CA3AF",
        "둔화":      "#F59E0B",
        "역성장":     "#DC2626",
    }.get(grade, "#6B7280")
    insight = {
        "폭발적 성장": "주의: 폭발 성장 종목은 SW 적중 42% — 의외로 낮음",
        "성장중":     "안정 성장은 SW 적중 31% — 보수적",
        "보합":      "보합 종목은 SW 적중 17% — 낮음",
        "둔화":      "둔화는 SW 적중 ~75% — 회복 베팅 유효",
        "역성장":     "V자 반등! 역성장 종목 SW 적중 81% — 최강",
    }.get(grade, "")

    def _fmt_won(v):
        if v is None or pd.isna(v): return "—"
        try:
            if abs(v) >= 1e4: return f"{v/1e4:,.1f}조"
            return f"{v:,.0f}억"
        except Exception: return "—"

    with st.expander(f"{name} 기업 분석 (매출 / 영업이익 / PER / PBR)", expanded=False):
        st.markdown(f"""
<div style="background:{grade_color};color:white;padding:14px 18px;border-radius:8px;margin-bottom:12px;text-align:center;">
  <div style="font-size:11px;opacity:0.85;letter-spacing:2px;">최근 2년 성장 평가</div>
  <div style="font-size:22px;font-weight:900;margin-top:4px;">{grade or '데이터X'}</div>
  {f'<div style="font-size:12px;margin-top:6px;opacity:0.95;">{insight}</div>' if insight else ''}
</div>
""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**매출액**")
            st.markdown(f"- 2024년: **{_fmt_won(rev_24)}**")
            st.markdown(f"- 2025년: **{_fmt_won(rev_25)}**")
            if pd.notna(rev_yoy):
                c = "#DC2626" if rev_yoy > 0 else "#2563EB"
                st.markdown(f"- YoY: <span style='color:{c};font-weight:700;'>{rev_yoy:+.1f}%</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("**영업이익**")
            st.markdown(f"- 2024년: **{_fmt_won(op_24)}**")
            st.markdown(f"- 2025년: **{_fmt_won(op_25)}**")
            if pd.notna(op_yoy):
                c = "#DC2626" if op_yoy > 0 else "#2563EB"
                st.markdown(f"- YoY: <span style='color:{c};font-weight:700;'>{op_yoy:+.1f}%</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("**영업이익률**")
            st.markdown(f"- 2024년: **{om_24:.2f}%**" if pd.notna(om_24) else "- 2024년: —")
            st.markdown(f"- 2025년: **{om_25:.2f}%**" if pd.notna(om_25) else "- 2025년: —")
            if pd.notna(om_diff):
                c = "#DC2626" if om_diff > 0 else "#2563EB"
                st.markdown(f"- 변화: <span style='color:{c};font-weight:700;'>{om_diff:+.2f}%p</span>", unsafe_allow_html=True)

        st.markdown("---")
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
        with col4: st.markdown(f"**PER** <span style='color:{per_col};font-weight:700;'>{per_txt}</span>", unsafe_allow_html=True)
        with col5: st.markdown(f"**PBR** <span style='color:{pbr_col};font-weight:700;'>{pbr_txt}</span>", unsafe_allow_html=True)
        with col6:
            if pd.notna(roe):
                c = "#10B981" if roe > 10 else ("#F59E0B" if roe > 5 else "#DC2626")
                st.markdown(f"**ROE** <span style='color:{c};font-weight:700;'>{roe:.1f}%</span>", unsafe_allow_html=True)
            else: st.markdown("**ROE** —")
        with col7:
            if pd.notna(fgnr):
                st.markdown(f"**외인지분** <span style='font-weight:700;'>{fgnr:.1f}%</span>", unsafe_allow_html=True)
            else: st.markdown("**외인지분** —")

        st.caption(
            "5년 OOS 결과: 매출+영업이익이 둘 다 큰 폭 성장한 종목보다, "
            "역성장/이익률 하락한 종목이 슈퍼위너(200%+) 더 많이 적중 (V자 반등). "
            "성장세는 안정성 지표이지 슈퍼위너 예측 지표는 아님."
        )


# ============== 미니 확률 바 (이모지 없음) ==============

def _prob_bar(label: str, prob_pct: float, color: str, is_main: bool = False):
    width = max(0, min(100, prob_pct))
    if is_main:
        return (
            f'<div style="margin:8px 0;">'
            f'  <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;">'
            f'    <span style="color:{color};">{label}</span>'
            f'    <span style="color:{color};">{prob_pct:.0f}%</span>'
            f'  </div>'
            f'  <div style="background:#F3F4F6;border-radius:6px;height:12px;overflow:hidden;margin-top:3px;">'
            f'    <div style="background:{color};width:{width}%;height:100%;"></div>'
            f'  </div>'
            f'</div>'
        )
    return (
        f'<div style="margin:4px 0;">'
        f'  <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7280;">'
        f'    <span>{label}</span><span>{prob_pct:.0f}%</span>'
        f'  </div>'
        f'  <div style="background:#F3F4F6;border-radius:4px;height:6px;overflow:hidden;">'
        f'    <div style="background:{color};width:{width}%;height:100%;opacity:0.7;"></div>'
        f'  </div>'
        f'</div>'
    )


# ============== 유사 종목 카드 (이모지 없음) ==============

def _render_similar_cards(similar_df: pd.DataFrame, show_stock_name: bool = False):
    """단순 텍스트 행 — 일자 / 종목 / 고점% / 180일% (색상 텍스트만)"""
    # 헤더
    html = (
        '<div style="border:1px solid #F3F4F6;border-radius:6px;background:white;overflow:hidden;">'
        '<div style="display:grid;grid-template-columns:90px 1fr 80px 80px;gap:8px;'
        'padding:6px 12px;background:#FAFAFA;border-bottom:1px solid #F3F4F6;'
        'font-size:10px;color:#9CA3AF;font-weight:600;letter-spacing:1px;">'
        '<div>일자</div>'
        f'<div>{"종목" if show_stock_name else "거래"}</div>'
        '<div style="text-align:right;">고점</div>'
        '<div style="text-align:right;">180일</div>'
        '</div>'
    )
    for i, (_, r) in enumerate(similar_df.iterrows()):
        try: d = pd.to_datetime(r.get("Date")).strftime("%Y-%m-%d")
        except: d = ""
        nm = r.get("Name", ""); cd = r.get("Code", "")
        peak = r.get("peak_180d", float("nan"))
        ret_ = r.get("ret_180d", float("nan"))
        close_buy = r.get("Close", 0) or 0
        close_sell = r.get("sell_close", 0) or 0
        peak_col = _peak_color(peak); ret_col = _ret_color(ret_)
        peak_txt = f"{peak:+.1f}%" if pd.notna(peak) else "—"
        ret_txt  = f"{ret_:+.1f}%" if pd.notna(ret_) else "—"
        # 좌측 중앙: 종목명(또는 매수→매도)
        if show_stock_name:
            mid = f'<b style="color:#111;">{nm}</b> <span style="color:#9CA3AF;font-size:10px;">{cd}</span>'
        else:
            mid = f'<span style="color:#6B7280;">매수 {close_buy:,.0f} → 매도 {close_sell:,.0f}</span>'
        bg = "#FAFAFA" if i % 2 else "white"
        html += (
            f'<div style="display:grid;grid-template-columns:90px 1fr 80px 80px;gap:8px;'
            f'padding:8px 12px;background:{bg};font-size:12px;align-items:center;">'
            f'<div style="color:#6B7280;">{d}</div>'
            f'<div>{mid}</div>'
            f'<div style="text-align:right;color:{peak_col};font-weight:700;">{peak_txt}</div>'
            f'<div style="text-align:right;color:{ret_col};font-weight:700;">{ret_txt}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_similar_summary(similar_df: pd.DataFrame, label: str = "평균"):
    """한 줄 요약 — 평균 고점/180일 + 카운트"""
    n = len(similar_df)
    avg_peak = similar_df["peak_180d"].mean()
    avg_ret  = similar_df["ret_180d"].mean()
    sw_n   = int((similar_df["peak_180d"]>=200).sum())
    p100_n = int((similar_df["peak_180d"]>=100).sum())
    p50_n  = int((similar_df["peak_180d"]>=50).sum())
    loss_n = int((similar_df["ret_180d"]<=-20).sum())
    win_n  = int((similar_df["ret_180d"]>0).sum())
    peak_col = _peak_color(avg_peak); ret_col = _ret_color(avg_ret)
    st.markdown(
        f'<div style="margin-top:8px;padding:8px 12px;background:#FAFAFA;border-radius:6px;'
        f'font-size:12px;color:#374151;">'
        f'<b>{label}</b> · '
        f'고점 <span style="color:{peak_col};font-weight:700;">{avg_peak:+.0f}%</span> · '
        f'180일 <span style="color:{ret_col};font-weight:700;">{avg_ret:+.0f}%</span> · '
        f'<span style="color:#9CA3AF;">SW {sw_n} / 100+ {p100_n} / 50+ {p50_n} / 손절 {loss_n} · 승률 {win_n/max(n,1)*100:.0f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============== 메인 카드 (이모지 0, 면처리 강조 %) ==============

def _render_fundamentals_inline(row: pd.Series, accent_color: str) -> str:
    """카드 안에 인라인으로 들어갈 기업분석 (펼치기 없이) — 한 줄 HTML 반환"""
    rev_24 = row.get("매출_2024"); rev_25 = row.get("매출_2025")
    op_24 = row.get("영업이익_2024"); op_25 = row.get("영업이익_2025")
    om_24 = row.get("영업이익률_2024"); om_25 = row.get("영업이익률_2025")
    rev_yoy = row.get("매출_YoY"); op_yoy = row.get("영업이익_YoY")
    om_diff = row.get("영업이익률_변화")
    per = row.get("PER_최신") or row.get("PER_num")
    pbr = row.get("PBR_최신") or row.get("PBR_num")
    roe = row.get("ROE_최신")
    fgnr = row.get("외인소진율_num")
    grade_raw = row.get("성장등급", "")
    grade_g = _clean(grade_raw)

    # 데이터 부족이면 비반환
    if (rev_25 is None or (isinstance(rev_25,float) and pd.isna(rev_25))) and \
       (op_25 is None or (isinstance(op_25,float) and pd.isna(op_25))):
        return ""

    grade_colors = {
        "폭발적 성장":"#10B981","성장중":"#22C55E","보합":"#9CA3AF",
        "둔화":"#F59E0B","역성장":"#DC2626",
    }
    g_col = grade_colors.get(grade_g, "#6B7280")

    def _wonfmt(v):
        if v is None or pd.isna(v): return "—"
        try:
            if abs(v) >= 1e4: return f"{v/1e4:,.1f}조"
            return f"{v:,.0f}억"
        except Exception: return "—"

    def _yoy_html(v, suffix="%"):
        if v is None or pd.isna(v): return ""
        c = "#DC2626" if v > 0 else "#2563EB"  # 한국식: + 빨강 / - 파랑
        return f' <span style="color:{c};font-weight:700;font-size:10px;">{v:+.1f}{suffix}</span>'

    def _per_eval(v):
        if v is None or pd.isna(v): return "—", "#9CA3AF"
        try: v = float(v)
        except: return "—", "#9CA3AF"
        if v < 0: return f"{v:.1f}(적자)", "#DC2626"
        if v < 10: return f"{v:.1f}(저평가)", "#10B981"
        if v < 20: return f"{v:.1f}(보통)", "#6B7280"
        return f"{v:.1f}(고평가)", "#F59E0B"
    def _pbr_eval(v):
        if v is None or pd.isna(v): return "—", "#9CA3AF"
        try: v = float(v)
        except: return "—", "#9CA3AF"
        if v < 1: return f"{v:.2f}(저평가)", "#10B981"
        if v < 2: return f"{v:.2f}(보통)", "#6B7280"
        return f"{v:.2f}(고평가)", "#F59E0B"

    per_txt, per_c = _per_eval(per)
    pbr_txt, pbr_c = _pbr_eval(pbr)
    roe_txt = "—"
    roe_c = "#9CA3AF"
    if pd.notna(roe):
        try:
            rv = float(roe)
            roe_txt = f"{rv:.1f}%"
            roe_c = "#10B981" if rv > 10 else ("#F59E0B" if rv > 5 else "#DC2626")
        except Exception: pass
    fgnr_txt = f"{fgnr:.1f}%" if pd.notna(fgnr) else "—"

    # 영업이익률 표시 텍스트
    if pd.notna(om_24) and pd.notna(om_25):
        om_inner = f'<b style="color:#111;">{om_24:.2f}→{om_25:.2f}%</b>{_yoy_html(om_diff, "%p")}'
    else:
        om_inner = '<b style="color:#111;">—</b>'

    html = (
        '<div style="background:#FAFAFA;border-top:1px solid #F3F4F6;padding:8px 14px;">'
        # 성장등급 배지
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">'
        f'<span style="background:{g_col};color:white;padding:1px 8px;border-radius:3px;'
        f'font-size:10px;font-weight:700;letter-spacing:1px;">{grade_g or "데이터X"}</span>'
        '<span style="font-size:10px;color:#9CA3AF;letter-spacing:1px;">2024 → 2025</span>'
        '</div>'
        # 매출/영업이익/영업이익률 3컬럼 (모바일에서 자동 wrap)
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));'
        'gap:6px;font-size:11px;">'
        '<div><span style="color:#9CA3AF;">매출</span> '
        f'<b style="color:#111;">{_wonfmt(rev_24)}→{_wonfmt(rev_25)}</b>{_yoy_html(rev_yoy)}</div>'
        '<div><span style="color:#9CA3AF;">영업이익</span> '
        f'<b style="color:#111;">{_wonfmt(op_24)}→{_wonfmt(op_25)}</b>{_yoy_html(op_yoy)}</div>'
        f'<div><span style="color:#9CA3AF;">영업이익률</span> {om_inner}</div>'
        '</div>'
        # PER/PBR/ROE/외인 — 한 줄 inline
        '<div style="margin-top:6px;padding-top:6px;border-top:1px dashed #E5E7EB;'
        'display:flex;flex-wrap:wrap;gap:10px;font-size:11px;">'
        f'<span><span style="color:#9CA3AF;">PER</span> <b style="color:{per_c};">{per_txt}</b></span>'
        f'<span><span style="color:#9CA3AF;">PBR</span> <b style="color:{pbr_c};">{pbr_txt}</b></span>'
        f'<span><span style="color:#9CA3AF;">ROE</span> <b style="color:{roe_c};">{roe_txt}</b></span>'
        f'<span><span style="color:#9CA3AF;">외인</span> <b style="color:#111;">{fgnr_txt}</b></span>'
        '</div>'
        '</div>'
    )
    return html


def _simple_summary(row: pd.Series) -> str:
    """카드 헤더에 들어갈 한 줄 평어 요약 — 매출/영업이익/밸류에이션의 핵심만"""
    parts = []
    op_yoy  = row.get("영업이익_YoY")
    rev_yoy = row.get("매출_YoY")
    om_diff = row.get("영업이익률_변화")
    per = row.get("PER_최신") or row.get("PER_num")
    pbr = row.get("PBR_최신") or row.get("PBR_num")
    fgnr = row.get("외인소진율_num")

    # 영업이익 변화 — 가장 중요
    try:
        if pd.notna(op_yoy):
            v = float(op_yoy)
            if v >= 100:   parts.append(f"영업이익 작년比 +{v:.0f}% 폭발")
            elif v >= 30:  parts.append(f"영업이익 +{v:.0f}% 성장")
            elif v >= 0:   parts.append(f"영업이익 +{v:.0f}%")
            elif v >= -30: parts.append(f"영업이익 {v:.0f}% 둔화")
            else:          parts.append(f"영업이익 {v:.0f}% 부진")
    except Exception: pass

    # 매출
    try:
        if pd.notna(rev_yoy):
            v = float(rev_yoy)
            if v >= 20:    parts.append(f"매출 +{v:.0f}%")
            elif v >= 5:   parts.append(f"매출 +{v:.0f}%")
            elif v < -5:   parts.append(f"매출 {v:.0f}%")
    except Exception: pass

    # PER 평가 (간단)
    try:
        if pd.notna(per):
            v = float(per)
            if v < 0:     parts.append("적자(턴어라운드)")
            elif v < 10:  parts.append(f"PER {v:.0f} 저평가")
            elif v > 30:  parts.append(f"PER {v:.0f} 고평가")
    except Exception: pass

    return " · ".join(parts) if parts else "재무 데이터 부족"


def _render_pick_card(row: pd.Series, show_similar: bool = True):
    grade_raw = row.get("등급", "")
    grade = _grade_short(grade_raw)
    code = row.get("Code", ""); name = row.get("Name", "")
    close = row.get("Close", 0) or 0
    ss = row.get("SuperScore", 0) or 0
    peak_pred = row.get("예상peak%", 0) or 0
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

    color = _grade_color(grade_raw)
    summary = _simple_summary(row)

    # 현재 수익률 (지난주/이번주 페이지에서 inject한 값)
    cur_price = row.get("_현재가")
    cur_ret = row.get("_현재수익률")
    cur_date_str = row.get("_현재기준일", "")

    # 메인 도달 구간 산정 (이모지 없는 라벨로 변환)
    main_label_raw = row.get("메인도달", "")
    main_prob = row.get("메인확률%", 0) or 0
    main_color = row.get("메인컬러", color) or color
    main_strength = row.get("메인강도", "")

    if not main_label_raw:
        candidates = [
            ("200%+", p_sw,  "#B91C1C"),
            ("100%+", p100,  "#DC2626"),
            ("50%+",  p50,   "#F97316"),
            ("30%+",  p30,   "#F59E0B"),
            ("10%+",  p10,   "#10B981"),
        ]
        chosen = None
        for lbl, pr, cc in candidates:
            if pr >= 50: chosen=(lbl,pr,cc,"매우 유력"); break
        if not chosen:
            for lbl, pr, cc in candidates:
                if pr >= 30: chosen=(lbl,pr,cc,"가능"); break
        if not chosen:
            for lbl, pr, cc in candidates[:2]:
                if pr >= 15: chosen=(lbl,pr,cc,"후보"); break
        if not chosen:
            chosen = ("10%+", p10, "#10B981", "")
        main_label = f"{chosen[0]} 도달"; main_prob = chosen[1]; main_color = chosen[2]; main_strength = chosen[3]
    else:
        main_label = _bucket_label_plain(main_label_raw)
        # main_strength 도 이모지 제거
        main_strength = _clean(main_strength).replace("도달", "").strip()

    # ===== 카드 HTML (차분) =====
    # 손절 확률은 '하락' 의미 → 한국식 파랑 / 낮으면 회색
    ploss_col = "#1D4ED8" if ploss >= 25 else "#6B7280"
    bar_width = max(2, min(100, main_prob))
    # 현재 수익률 배지 (매수일 이후 경과 시에만)
    cur_html = ""
    if cur_ret is not None and pd.notna(cur_ret):
        cret_col = _ret_color(cur_ret)
        cur_html = (
            f'<div style="display:inline-block;color:{cret_col};font-size:12px;font-weight:700;margin-top:4px;">'
            f'현재 {cur_ret:+.1f}% ({cur_price:,.0f}원)</div>'
        )
    strength_html = (
        f'<span style="color:{main_color};font-size:11px;font-weight:700;margin-left:6px;">{main_strength}</span>'
        if main_strength else ""
    )
    # 현재 수익률 — 컴팩트 흰 박스 (가로형, 모바일에서도 컴팩트)
    cur_html_header = ""
    if cur_ret is not None and pd.notna(cur_ret):
        cret_col = _ret_color(cur_ret)
        cur_html_header = (
            f'<div style="background:white;padding:4px 8px;border-radius:5px;'
            f'margin-top:4px;display:inline-block;box-shadow:0 1px 2px rgba(0,0,0,0.08);">'
            f'<span style="font-size:9px;color:#9CA3AF;letter-spacing:1px;font-weight:700;">현재</span> '
            f'<span style="font-size:14px;color:{cret_col};font-weight:900;">{cur_ret:+.1f}%</span> '
            f'<span style="font-size:10px;color:#6B7280;">({cur_price:,.0f})</span>'
            f'</div>'
        )

    # 기업분석 인라인 HTML (디폴트 노출)
    fundamentals_html = _render_fundamentals_inline(row, main_color)

    card_html = (
        f'<div style="border:1px solid {color}33;background:white;border-radius:10px;'
        f'margin-bottom:10px;overflow:hidden;">'
        # 1) 헤더 — 컴팩트 (align center로 좌우 여백 균등 흡수)
        f'<div style="background:{color};color:white;padding:8px 14px;'
        f'display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:160px;">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span style="background:rgba(255,255,255,0.22);color:white;padding:1px 8px;border-radius:3px;'
        f'font-size:10px;font-weight:700;letter-spacing:1px;">{grade}</span>'
        f'<span style="font-size:17px;font-weight:800;color:white;">{name}</span>'
        f'<span style="font-size:10px;color:rgba(255,255,255,0.75);">{code} · {market} · {date}</span>'
        f'</div>'
        f'<div style="font-size:11px;color:rgba(255,255,255,0.92);margin-top:2px;line-height:1.3;">{summary}</div>'
        f'</div>'
        f'<div style="text-align:right;flex-shrink:0;">'
        f'<div style="font-size:9px;color:rgba(255,255,255,0.7);letter-spacing:1px;">매수가</div>'
        f'<div style="font-size:15px;font-weight:700;color:white;line-height:1.2;">{close:,.0f}원</div>'
        f'{cur_html_header}'
        f'</div></div>'
        # 2) 기업분석 인라인
        f'{fundamentals_html}'
        # 3) 메인 % — 컬러 진행바 + 우측 숫자 (모바일에서도 한 줄 유지)
        f'<div style="padding:10px 14px;background:white;border-top:1px solid #F3F4F6;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:11px;color:#6B7280;margin-bottom:3px;font-weight:600;">'
        f'가장 도달 확률 — <b style="color:{main_color};">{main_label}</b>{strength_html}</div>'
        f'<div style="background:#F3F4F6;border-radius:6px;height:12px;overflow:hidden;">'
        f'<div style="background:{main_color};width:{bar_width}%;height:100%;border-radius:6px;"></div>'
        f'</div></div>'
        f'<div style="font-size:28px;font-weight:900;color:{main_color};line-height:1;min-width:64px;text-align:right;">'
        f'{main_prob:.0f}<span style="font-size:15px;font-weight:700;">%</span>'
        f'</div></div>'
        # 4) 푸터 — 모바일에서 flex-wrap
        f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #F3F4F6;'
        f'display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:11px;color:#6B7280;">'
        f'<span>최고가 <b style="color:{main_color};margin-left:3px;">+{peak_pred:.0f}%</b></span>'
        f'<span>슈퍼점수 <b style="color:#111;margin-left:3px;">{ss:.2f}</b></span>'
        f'<span>손절 <b style="color:{ploss_col};margin-left:3px;">{ploss:.0f}%</b></span>'
        f'</div>'
        f'</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"{name} — 도달 구간별 OOS 적중 확률", expanded=False):
        st.caption("실제 5년 백테스트(825건) 기반 보정 확률")
        rows_data = [
            ("200%+ (슈퍼위너)", p_sw,  "#B91C1C"),
            ("100%+ (2배)",   p100,  "#DC2626"),
            ("50%+",          p50,   "#F97316"),
            ("30%+",          p30,   "#F59E0B"),
            ("10%+",          p10,   "#10B981"),
            ("손절 (-20%↓)",  ploss, "#7F1D1D"),
        ]
        max_prob = max([r[1] for r in rows_data[:-1]])
        bars_html = ""
        for lbl, pr, cc in rows_data:
            is_main = (pr == max_prob and "손절" not in lbl)
            bars_html += _prob_bar(lbl, pr, cc, is_main=is_main)
        st.markdown(bars_html, unsafe_allow_html=True)

    # 기업분석은 카드 안에 이미 인라인으로 표시됨 — expander 제거
    reasons = _reason_text(row)
    if reasons:
        with st.expander(f"{name} — 강력추천 사유", expanded=False):
            for r in reasons: st.markdown(f"- {r}")

    if show_similar:
        similar_same = _find_similar_cases(code, n=5)
        if len(similar_same) > 0:
            with st.expander(f"{name} 과거 매수 사례 ({len(similar_same)}건)", expanded=False):
                _render_similar_cards(similar_same, show_stock_name=False)
                _render_similar_summary(similar_same, label="같은 종목 과거")

        similar_other = _find_similar_stocks(row, n=5, exclude_code=code)
        if len(similar_other) > 0:
            with st.expander(f"{name} 와 비슷한 패턴 종목 ({len(similar_other)}건)", expanded=False):
                _render_similar_cards(similar_other, show_stock_name=True)
                _render_similar_summary(similar_other, label="유사 패턴")
                st.caption("슈퍼위너/100%+/50%+/손절 확률이 비슷한 다른 종목의 과거 매수 결과")


# ============== 버튼 멀티셀렉트 ==============

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
    with open(p, encoding="utf-8") as f: return json.load(f)


def _sort_strong(picks_list):
    """슈퍼강력 + 강력매수만 — 슈퍼강력 먼저, 그 안에서 점수순"""
    buyable = [p for p in picks_list if "강력매수" in str(p.get("등급",""))]
    def priority(p):
        grade_bonus = 100 if "슈퍼" in str(p.get("등급", "")) else 0
        ss = p.get("SuperScore", 0) or 0
        psw = p.get("슈퍼위너확률%", 0) or 0
        return grade_bonus + ss * 0.5 + psw * 0.01
    buyable.sort(key=priority, reverse=True)
    return buyable


# ============== 페이지 ==============

def page_today_pick():
    st.markdown('<h1 style="font-weight:800;">오늘의 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    if not data:
        st.warning("추천 데이터가 없습니다."); return
    base_date = data.get("base_date", "")
    updated = data.get("updated_at", "")[:16]
    st.markdown(f"**기준일**: {base_date}  ·  **갱신**: {updated}")

    today = data.get("today", {})
    strong = _sort_strong(today.get("picks", []))

    if len(strong) == 0:
        st.info("오늘 강력매수 추천 없음 — 현금 보유 권장")
        st.caption(f"(전체 시그널 {today.get('n', 0)}건 중 강력매수 0건)")
    else:
        st.markdown(f"### 강력매수 {len(strong)}건")
        st.caption("정렬: 슈퍼점수 + 슈퍼위너 확률 종합 우선순위")
        for p in strong:
            # 매수일 이후 1일이라도 지났으면 현재 수익률 inject
            cur, ret_now, cur_date = _current_return(p.get("Close"), p.get("Code",""))
            if ret_now is not None:
                p["_현재가"] = cur; p["_현재수익률"] = ret_now
                p["_현재기준일"] = cur_date.strftime("%Y-%m-%d") if cur_date is not None else ""
            _render_pick_card(pd.Series(p), show_similar=True)


_OHLCV_CACHE = {}

def _get_latest_ohlcv():
    """가장 큰 OHLCV pkl 캐시 로드"""
    if "ohlcv" not in _OHLCV_CACHE:
        import pickle
        files = sorted(CACHE.glob("ohlcv_*.pkl"), key=lambda p: p.stat().st_size, reverse=True)
        if files:
            with open(files[0], "rb") as f:
                _OHLCV_CACHE["ohlcv"] = pickle.load(f)
        else:
            _OHLCV_CACHE["ohlcv"] = {}
    return _OHLCV_CACHE["ohlcv"]


def _current_return(buy_close: float, code: str) -> tuple:
    """매수가 대비 현재가 수익률 — (현재가, 수익률%, 경과일)"""
    if not buy_close or buy_close <= 0: return None, None, None
    ohlcv = _get_latest_ohlcv()
    code_s = str(code).zfill(6)
    if code_s not in ohlcv: return None, None, None
    df = ohlcv[code_s]
    if len(df) == 0: return None, None, None
    cur = float(df["Close"].iloc[-1])
    cur_date = df.index[-1]
    ret = (cur - buy_close) / buy_close * 100
    return cur, ret, cur_date


def _render_weekly_by_day(picks_list):
    """일자별로 강력매수 종목 표시 — 카드는 인라인 렌더 (중첩 expander 금지)"""
    strong = [p for p in picks_list if "강력매수" in str(p.get("등급",""))]
    if len(strong) == 0:
        st.info("강력매수 종목 없음"); return

    by_day = {}
    for p in strong:
        d = p.get("Date", "")
        if isinstance(d, str): d = d[:10]
        else:
            try: d = pd.to_datetime(d).strftime("%Y-%m-%d")
            except: d = ""
        by_day.setdefault(d, []).append(p)
    dates_sorted = sorted(by_day.keys())

    st.markdown(f"### 강력매수 {len(strong)}건")
    st.caption("실전 룰: 매일 발견 즉시 매수 (NXT 19:50) · 같은 일 내 여러 건이면 슈퍼점수 높은 순")

    weekday_kr = ["월","화","수","목","금","토","일"]
    for d in dates_sorted:
        try: wd = weekday_kr[pd.to_datetime(d).weekday()]
        except: wd = ""
        day_picks = by_day[d]
        # 슈퍼강력 먼저, 그 안에서 점수 순
        def pri(p):
            bonus = 100 if "슈퍼" in str(p.get("등급","")) else 0
            return bonus + (p.get("SuperScore",0) or 0)*0.5 + (p.get("슈퍼위너확률%",0) or 0)*0.01
        day_picks.sort(key=pri, reverse=True)
        st.markdown(f"#### {d} ({wd}요일) — {len(day_picks)}건")
        for p in day_picks:
            # 현재 수익률 계산 (매수일 이후 경과한 경우)
            buy_close = p.get("Close")
            cur, ret_now, cur_date = _current_return(buy_close, p.get("Code",""))
            if ret_now is not None:
                p["_현재가"] = cur
                p["_현재수익률"] = ret_now
                p["_현재기준일"] = cur_date.strftime("%Y-%m-%d") if cur_date is not None else ""
            # 카드 인라인 렌더
            _render_pick_card(pd.Series(p), show_similar=True)
        st.markdown("---")


def page_this_week():
    st.markdown('<h1 style="font-weight:800;">이번 주 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    week = data.get("week", {})
    st.markdown(f"**주 시작일**: {data.get('week_start', '')}")
    _render_weekly_by_day(week.get("picks", []))


def page_last_week():
    st.markdown('<h1 style="font-weight:800;">지난 주 추천</h1>', unsafe_allow_html=True)
    data = _load_json()
    last_week = data.get("last_week", {})
    _render_weekly_by_day(last_week.get("picks", []))


def page_backtest():
    st.markdown('<h1 style="font-weight:800;">백테스트 (2022-2026 walk-forward OOS)</h1>', unsafe_allow_html=True)

    yr_path = CACHE / "MASTER_best_yearly.csv"
    if yr_path.exists():
        yr = pd.read_csv(yr_path)
        yr = yr.rename(columns={
            "year":"년도","매수":"매수","SW":"슈퍼위너",
            "100+":"100%+","50+":"50%+","10+":"10%+",
            "손절":"손절","투자만":"투자(만원)",
            "수익만":"수익(만원)","수익률%":"수익률(%)"
        })
        st.markdown("#### 년도별 요약")
        st.dataframe(yr, hide_index=True, use_container_width=True)
        cols = st.columns(4)
        cols[0].metric("총 매수", f"{int(yr['매수'].sum()):,}건")
        cols[1].metric("총 투자", f"{int(yr['투자(만원)'].sum()):,}만")
        cols[2].metric("총 수익", f"{int(yr['수익(만원)'].sum()):+,}만")
        tot_inv = yr['투자(만원)'].sum()
        tot_prof = yr['수익(만원)'].sum()
        cols[3].metric("5년 수익률", f"{tot_prof/tot_inv*100:+.1f}%")

    st.markdown("---")
    st.markdown("#### 매수 종목 전체")
    picks_path = CACHE / "MASTER_best_picks_2020-2026.csv"
    if not picks_path.exists():
        st.warning("백테스트 데이터 없음"); return

    picks = pd.read_csv(picks_path)
    picks["Date"] = pd.to_datetime(picks["Date"])
    picks["년도"] = picks["Date"].dt.year
    picks["월"] = picks["Date"].dt.month

    # === 180일 완료 여부 판정 ===
    today = pd.Timestamp.now().normalize()
    HOLD_CAL_DAYS = 260   # 180거래일 ≈ 260달력일
    picks["경과일"] = (today - picks["Date"]).dt.days
    picks["진행상태"] = np.where(
        picks["경과일"] >= HOLD_CAL_DAYS, "완료", "진행중"
    )
    picks["D-남은일"] = (HOLD_CAL_DAYS - picks["경과일"]).clip(lower=0)

    n_done = (picks["진행상태"]=="완료").sum()
    n_ongoing = (picks["진행상태"]=="진행중").sum()
    info_cols = st.columns(3)
    info_cols[0].metric("180일 완료", f"{n_done:,}건")
    info_cols[1].metric("진행중", f"{n_ongoing:,}건")
    info_cols[2].metric("매수일 기준 평균 경과", f"{picks['경과일'].mean():.0f}일")
    st.caption(
        "**진행중**: 매수일+260일이 아직 안 지난 케이스 — 표의 수익률/최고가는 **현재까지의 진행값** "
        "(180일 후 더 오르거나 떨어질 수 있음). 결과 분류는 현재까지 도달한 최고가 기준."
    )
    st.info(
        "**OOS 검증 데이터**: 모델은 walk-forward 학습 — 2026년 picks는 2025년까지 데이터로만 훈련된 모델이 "
        "선정한 종목입니다. 수익률은 실제 KRX OHLCV 데이터 기반 (예: 대한광통신 3,205→24,300원). "
        "단기 폭등 종목이 추출된 건 모델이 의도한 슈퍼위너 발굴 효과."
    )

    # 결과 분류 = 실제 매도가 수익률(ret_180d) 기준 (현실적인 실현 수익)
    # 최고가도달(peak)은 별도 컬럼으로 참고만
    def cls(row):
        r = row.get("ret_180d")
        if pd.isna(r): return "미정"
        if r >= 200: return "슈퍼위너"   # 실제 +200% 이상 익절
        if r >= 100: return "100%+"     # 2배
        if r >= 50:  return "50%+"
        if r >= 10:  return "10%+"
        if r <= -20: return "손절"
        if r > 0:    return "소폭익절"   # 0~10% 미세 익절
        return "보합"                    # 0%~-20% 미세 손실
    picks["결과"] = picks.apply(cls, axis=1)

    # === 종목당 매수 금액 입력 (만원 단위) ===
    if "bt_buy_amount" not in st.session_state:
        st.session_state.bt_buy_amount = 10  # 디폴트 10만원

    st.markdown("**종목당 매수 금액** (만원 단위)")
    bcols = st.columns([1,1,1,1,1,2])
    presets = [
        ("10만", 10), ("50만", 50), ("100만", 100), ("300만", 300), ("500만", 500),
    ]
    for i, (label, amt) in enumerate(presets):
        is_sel = st.session_state.bt_buy_amount == amt
        if bcols[i].button(label, key=f"bt_amt_{amt}",
                           type=("primary" if is_sel else "secondary"),
                           use_container_width=True):
            st.session_state.bt_buy_amount = amt
            st.rerun()
    custom_amt = bcols[5].number_input(
        "직접 입력 (만원)", min_value=1, max_value=100000,
        value=st.session_state.bt_buy_amount, step=10,
        label_visibility="collapsed", key="bt_amt_custom_input",
    )
    if custom_amt != st.session_state.bt_buy_amount:
        st.session_state.bt_buy_amount = int(custom_amt)
        st.rerun()
    buy_amount_man = st.session_state.bt_buy_amount  # 만원
    st.caption(f"현재 설정: 종목당 **{buy_amount_man:,}만원** 매수 가정 (모든 수익금/투자금 자동 재계산)")

    # 수익금(만원 단위) = ret% × 매수금만원 / 100
    picks["수익금_만원"] = (picks["ret_180d"].fillna(0) * buy_amount_man / 100).round(1)

    # === 등급 자동 부여 ===
    # MASTER_best_picks는 이미 weekly_5 top 픽들. 그 안에서 SuperScore 분위로
    # 슈퍼강력매수(상위 30%) vs 추천매수(나머지)로 나눔.
    score_col = "SuperScore_v2" if "SuperScore_v2" in picks.columns else "SuperScore"
    if score_col in picks.columns:
        # 일자별 percentile rank (같은 날 매수 종목들끼리 비교) — 일일 분포 안 좋으면 전체 분위 fallback
        picks["_score_pct_global"] = picks[score_col].rank(pct=True)
        picks["등급"] = np.where(
            picks["_score_pct_global"] >= 0.70, "슈퍼강력매수", "추천매수"
        )
    else:
        picks["등급"] = "추천매수"

    years_avail = sorted(picks["년도"].dropna().unique().astype(int).tolist())
    sel_years = _button_multiselect("년도 (다중 선택)", years_avail, default=years_avail, key_prefix="bt_year")
    months_avail = list(range(1, 13))
    sel_months = _button_multiselect("월 (다중 선택)", months_avail, default=months_avail, key_prefix="bt_month")

    # 등급 필터 (슈퍼강력매수 / 추천매수)
    grade_avail = ["슈퍼강력매수", "추천매수"]
    sel_grades = _button_multiselect("등급 (다중 선택)", grade_avail, default=grade_avail, key_prefix="bt_grade")

    sort_options = {
        "최신 일자순": ("Date", False),
        "오래된 일자순": ("Date", True),
        "최고가 높은순": ("peak_180d", False),
        "수익률 높은순": ("ret_180d", False),
        "수익률 낮은순": ("ret_180d", True),
        "슈퍼점수 높은순": ("SuperScore_v2", False),
    }
    sort_label = st.selectbox(
        "정렬 기준", list(sort_options.keys()),
        index=0, key="bt_sort_dropdown",
    )
    sort_col_key, sort_asc = sort_options[sort_label]
    if sort_col_key not in picks.columns:
        sort_col_key = "Date"

    filtered = picks[
        picks["년도"].isin(sel_years) &
        picks["월"].isin(sel_months) &
        picks["등급"].isin(sel_grades)
    ].copy()
    filtered = filtered.sort_values(sort_col_key, ascending=sort_asc)

    # 진행상태 한 문장으로
    filtered["진행"] = filtered.apply(
        lambda r: f"진행중 D-{int(r['D-남은일'])}" if r["진행상태"]=="진행중" else "완료",
        axis=1,
    )

    # === 요약 카드 (필터 결과 기반) — 표 위에 배치 ===
    if len(filtered) > 0:
        n = len(filtered)
        avg_ret = filtered["ret_180d"].mean()
        med_ret = filtered["ret_180d"].median()
        avg_peak = filtered["peak_180d"].mean()
        total_profit = filtered["수익금_만원"].sum()
        invest_amount = n * buy_amount_man  # 종목당 매수금 × 건수
        sw_n   = int((filtered["ret_180d"]>=200).sum())
        p100_n = int((filtered["ret_180d"]>=100).sum())
        p50_n  = int((filtered["ret_180d"]>=50).sum())
        p10_n  = int((filtered["ret_180d"]>=10).sum())
        win_n  = int((filtered["ret_180d"]>0).sum())
        loss_n = int((filtered["ret_180d"]<=-20).sum())
        sw_peak = int((filtered["peak_180d"]>=200).sum())
        p100_peak = int((filtered["peak_180d"]>=100).sum())
        p50_peak = int((filtered["peak_180d"]>=50).sum())

        def pct(x): return f"{x/max(n,1)*100:.1f}%"
        ret_col = "#DC2626" if avg_ret > 0 else "#2563EB"
        prof_col = "#DC2626" if total_profit > 0 else "#2563EB"

        # Row 1: 핵심 KPI 4개
        st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px;">
  <div style="background:white;border:1px solid #E5E7EB;border-radius:8px;padding:12px 16px;">
    <div style="font-size:10px;color:#6B7280;letter-spacing:1px;">선택 건수</div>
    <div style="font-size:22px;font-weight:800;color:#111;margin-top:2px;">{n:,}<span style="font-size:13px;font-weight:600;color:#6B7280;"> 건</span></div>
    <div style="font-size:10px;color:#9CA3AF;margin-top:2px;">투자 {invest_amount:,}만원</div>
  </div>
  <div style="background:white;border:1px solid #E5E7EB;border-radius:8px;padding:12px 16px;">
    <div style="font-size:10px;color:#6B7280;letter-spacing:1px;">평균 수익률 (매도)</div>
    <div style="font-size:22px;font-weight:800;color:{ret_col};margin-top:2px;">{avg_ret:+.1f}%</div>
    <div style="font-size:10px;color:#9CA3AF;margin-top:2px;">중앙 {med_ret:+.1f}% · 승률 {win_n/max(n,1)*100:.0f}%</div>
  </div>
  <div style="background:white;border:1px solid #E5E7EB;border-radius:8px;padding:12px 16px;">
    <div style="font-size:10px;color:#6B7280;letter-spacing:1px;">평균 최고가 도달</div>
    <div style="font-size:22px;font-weight:800;color:#B91C1C;margin-top:2px;">+{avg_peak:.0f}%</div>
    <div style="font-size:10px;color:#9CA3AF;margin-top:2px;">매수 후 고점 평균</div>
  </div>
  <div style="background:{prof_col};color:white;border-radius:8px;padding:12px 16px;">
    <div style="font-size:10px;opacity:0.85;letter-spacing:1px;">총 수익금 ({buy_amount_man}만/종목)</div>
    <div style="font-size:22px;font-weight:800;margin-top:2px;">{total_profit:+,.0f}<span style="font-size:13px;font-weight:600;opacity:0.95;"> 만</span></div>
    <div style="font-size:10px;opacity:0.9;margin-top:2px;">투자 대비 {total_profit/max(invest_amount,1)*100:+.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Row 2: 매도 실현 단계별 도달
        st.markdown(f"""
<div style="margin-top:8px;background:white;border:1px solid #E5E7EB;border-radius:8px;padding:10px 14px;">
  <div style="font-size:11px;color:#6B7280;font-weight:700;letter-spacing:1px;margin-bottom:6px;">
    매도(180일 종가) 실현 단계별 도달 — 실현 수익 기준
  </div>
  <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px;text-align:center;">
    <div style="padding:6px;background:#FEE2E2;border-radius:6px;">
      <div style="font-size:10px;color:#7F1D1D;font-weight:700;">슈퍼위너 200%+</div>
      <div style="font-size:18px;font-weight:900;color:#7F1D1D;">{sw_n}<span style="font-size:11px;color:#6B7280;font-weight:600;">건 · {pct(sw_n)}</span></div>
      <div style="font-size:9px;color:#9CA3AF;">고점 {sw_peak}건 도달</div>
    </div>
    <div style="padding:6px;background:#FECACA;border-radius:6px;">
      <div style="font-size:10px;color:#B91C1C;font-weight:700;">100%+ (2배)</div>
      <div style="font-size:18px;font-weight:900;color:#B91C1C;">{p100_n}<span style="font-size:11px;color:#6B7280;font-weight:600;">건 · {pct(p100_n)}</span></div>
      <div style="font-size:9px;color:#9CA3AF;">고점 {p100_peak}건 도달</div>
    </div>
    <div style="padding:6px;background:#FECACA;border-radius:6px;">
      <div style="font-size:10px;color:#DC2626;font-weight:700;">50%+</div>
      <div style="font-size:18px;font-weight:900;color:#DC2626;">{p50_n}<span style="font-size:11px;color:#6B7280;font-weight:600;">건 · {pct(p50_n)}</span></div>
      <div style="font-size:9px;color:#9CA3AF;">고점 {p50_peak}건 도달</div>
    </div>
    <div style="padding:6px;background:#FEE2E2;border-radius:6px;">
      <div style="font-size:10px;color:#EF4444;font-weight:700;">10%+</div>
      <div style="font-size:18px;font-weight:900;color:#EF4444;">{p10_n}<span style="font-size:11px;color:#6B7280;font-weight:600;">건 · {pct(p10_n)}</span></div>
      <div style="font-size:9px;color:#9CA3AF;">의미있는 익절</div>
    </div>
    <div style="padding:6px;background:#FEF2F2;border-radius:6px;">
      <div style="font-size:10px;color:#F87171;font-weight:700;">소폭+ (0~10%)</div>
      <div style="font-size:18px;font-weight:900;color:#F87171;">{win_n - p10_n}<span style="font-size:11px;color:#6B7280;font-weight:600;">건 · {pct(win_n - p10_n)}</span></div>
      <div style="font-size:9px;color:#9CA3AF;">소익절</div>
    </div>
    <div style="padding:6px;background:#DBEAFE;border-radius:6px;">
      <div style="font-size:10px;color:#1D4ED8;font-weight:700;">손절 -20%↓</div>
      <div style="font-size:18px;font-weight:900;color:#1D4ED8;">{loss_n}<span style="font-size:11px;color:#6B7280;font-weight:600;">건 · {pct(loss_n)}</span></div>
      <div style="font-size:9px;color:#9CA3AF;">실패율</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        # OOS 검증 안내
        st.markdown(
            '<div style="background:#ECFDF5;border:1px solid #10B981;border-radius:6px;'
            'padding:10px 14px;margin-top:8px;font-size:12px;color:#065F46;line-height:1.5;">'
            '<b>OOS 검증</b> · 모든 picks은 walk-forward 학습 결과. '
            '2025 picks는 2024년까지 데이터로만 훈련된 모델이 선정, 2026 picks는 2025년까지. '
            '수익률은 실제 KRX OHLCV 데이터 검증 (대한광통신 3,205→24,300 등). 미래 정보 누출 없음.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    profit_col_label = f"수익금({buy_amount_man}만/종목,만원)"
    show_map = {
        "Date":"일자","년도":"년도","월":"월",
        "Code":"종목코드","Name":"종목명","Market":"시장",
        "Close":"매수가","등급":"등급","진행":"진행상태",
        "결과":"결과",
        "ret_180d":"수익률(%)","peak_180d":"최고가(%)",
        "sell_close":"매도가/현재가",
        "수익금_만원":profit_col_label,
        "SuperScore_v2":"슈퍼점수",
    }
    show_cols = [c for c in show_map if c in filtered.columns]
    display = filtered[show_cols].rename(columns=show_map).head(500)
    if "일자" in display.columns:
        display["일자"] = pd.to_datetime(display["일자"]).dt.strftime("%Y-%m-%d")
    for c in ["수익률(%)","최고가(%)"]:
        if c in display.columns: display[c] = display[c].round(1)

    # 결과 컬럼 색상 (한국식: 양수 빨강 / 음수 파랑) — ret_180d 기준
    def _result_style(val):
        cmap = {
            "슈퍼위너":  "background-color:#7F1D1D;color:white;font-weight:700;",
            "100%+":    "background-color:#B91C1C;color:white;font-weight:700;",
            "50%+":     "background-color:#DC2626;color:white;font-weight:700;",
            "10%+":     "background-color:#EF4444;color:white;font-weight:700;",
            "소폭익절":  "background-color:#F87171;color:white;",
            "보합":     "background-color:#9CA3AF;color:white;",
            "손절":     "background-color:#1D4ED8;color:white;font-weight:700;",
            "미정":     "background-color:#E5E7EB;color:#6B7280;",
        }
        return cmap.get(str(val), "")

    def _grade_style(val):
        if val == "슈퍼강력매수":
            return "background-color:#B91C1C;color:white;font-weight:700;"
        if val == "추천매수":
            return "background-color:#F97316;color:white;font-weight:700;"
        return ""

    def _return_style(val):
        try:
            v = float(val)
            if v >= 100:  return "color:#7F1D1D;font-weight:800;"
            if v >= 30:   return "color:#B91C1C;font-weight:700;"
            if v >= 10:   return "color:#DC2626;font-weight:700;"
            if v > 0:     return "color:#EF4444;"
            if v == 0:    return "color:#6B7280;"
            if v > -20:   return "color:#2563EB;font-weight:600;"
            return "color:#1D4ED8;font-weight:800;"
        except Exception: return ""

    # pandas >=2.1.0 권장: Styler.map (applymap 은 deprecated)
    styler = display.style
    style_apply = getattr(styler, "map", None) or styler.applymap
    if "결과" in display.columns:
        styler = style_apply(_result_style, subset=["결과"])
        style_apply = getattr(styler, "map", None) or styler.applymap
    if "등급" in display.columns:
        styler = style_apply(_grade_style, subset=["등급"])
        style_apply = getattr(styler, "map", None) or styler.applymap
    for c in ["수익률(%)","최고가(%)", profit_col_label]:
        if c in display.columns:
            styler = style_apply(_return_style, subset=[c])
            style_apply = getattr(styler, "map", None) or styler.applymap

    # 컬럼별 숫자 포맷 (소수점 제거 / 콤마 천 단위)
    fmt_map = {}
    for c in ["매수가","매도가/현재가"]:
        if c in display.columns: fmt_map[c] = "{:,.0f}"   # 원 단위, 콤마, 소수 없음
    for c in ["수익률(%)","최고가(%)"]:
        if c in display.columns: fmt_map[c] = "{:+.1f}"   # 부호 포함, 소수 1자리
    if profit_col_label in display.columns:
        fmt_map[profit_col_label] = "{:+,.1f}"        # 만원 단위 부호 + 콤마
    if "슈퍼점수" in display.columns:
        fmt_map["슈퍼점수"] = "{:.2f}"
    if fmt_map:
        styler = styler.format(fmt_map, na_rep="—")

    st.dataframe(styler, hide_index=True, use_container_width=True, height=600)

    st.caption(
        f"검색 결과 {len(filtered):,}건 중 최대 500건 표시. "
        f"**진행중 D-XX** = 매수일+260일까지 남은 일수. 수익률/최고가/매도가는 진행중이면 **현재까지 진행값**, "
        f"수익금은 **{buy_amount_man}만원 매수 시** 만원 단위 손익."
    )


def page_case_validation():
    st.markdown('<h1 style="font-weight:800;">추천 사례 검증</h1>', unsafe_allow_html=True)
    st.caption("현재 강력매수 종목들의 과거 5년 매수 사례 검증")
    data = _load_json()
    picks_all = []
    for key in ["today","week","last_week"]:
        for p in data.get(key, {}).get("picks", []):
            if "강력매수" in str(p.get("등급","")):
                p_ = dict(p); p_["기간"] = {"today":"오늘","week":"이번주","last_week":"지난주"}[key]
                picks_all.append(p_)
    seen = set(); uniq = []
    for p in picks_all:
        if p["Code"] not in seen:
            seen.add(p["Code"]); uniq.append(p)
    if len(uniq) == 0:
        st.info("현재 강력매수 종목 없음"); return
    st.markdown(f"### 매수 후보 {len(uniq)}개의 과거 매수 사례")
    for p in uniq:
        code = p["Code"]; name = p["Name"]
        st.markdown(f"#### {name} ({code}) — {p['기간']} 추천")
        similar = _find_similar_cases(code, n=20)
        if len(similar) == 0:
            st.caption("과거 매수 사례 없음"); continue
        avg_peak = similar["peak_180d"].mean()
        avg_ret = similar["ret_180d"].mean()
        sw_count = (similar["peak_180d"]>=200).sum()
        w100_count = (similar["peak_180d"]>=100).sum()
        loss_count = (similar["ret_180d"]<=-20).sum()
        n = len(similar)
        cols = st.columns(6)
        cols[0].metric("매수 사례", f"{n}건")
        cols[1].metric("평균 최고가", f"+{avg_peak:.0f}%")
        cols[2].metric("평균 수익률", f"{avg_ret:+.0f}%")
        cols[3].metric("슈퍼위너", f"{sw_count}건")
        cols[4].metric("100%+", f"{w100_count}건")
        cols[5].metric("손절", f"{loss_count}건")
        show = similar[[c for c in ["Date","Close","sell_close","ret_180d","peak_180d"] if c in similar.columns]].copy()
        show = show.rename(columns={
            "Date":"발생일","Close":"매수가","sell_close":"매도가",
            "ret_180d":"180일수익률(%)","peak_180d":"최고가도달(%)"
        })
        if "발생일" in show.columns:
            show["발생일"] = pd.to_datetime(show["발생일"]).dt.strftime("%Y-%m-%d")
        for c in ["180일수익률(%)","최고가도달(%)"]:
            if c in show.columns: show[c] = show[c].round(1)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.markdown("---")


def page_buy_rule():
    st.markdown('<h1 style="font-weight:800;">매수 룰</h1>', unsafe_allow_html=True)

    st.markdown("### 슈퍼강력매수 vs 추천매수 — 차이 한눈에")
    st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:8px 0 16px;">

  <div style="border:2px solid #B91C1C;border-radius:10px;padding:14px 18px;background:#FEF2F2;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <span style="background:#B91C1C;color:white;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;">슈퍼강력매수</span>
      <span style="font-size:11px;color:#6B7280;">1순위 매수</span>
    </div>
    <div style="font-size:13px;color:#374151;line-height:1.6;">
      · <b>슈퍼점수 상위 30%</b> (per pool)<br>
      · 일평균 <b>2~3건</b> 발생<br>
      · <b>200%+ 매도 27.7%</b> · 100%+ 매도 47.0%<br>
      · 손절률 <b>8.9%</b> · 승률 <b>81.6%</b><br>
      · 평균 수익률 <b style="color:#B91C1C;">+151%</b><br>
      · 자본 부족해도 <b>필수 매수</b>
    </div>
  </div>

  <div style="border:2px solid #F97316;border-radius:10px;padding:14px 18px;background:#FFF7ED;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <span style="background:#F97316;color:white;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;">추천매수</span>
      <span style="font-size:11px;color:#6B7280;">2순위 매수</span>
    </div>
    <div style="font-size:13px;color:#374151;line-height:1.6;">
      · <b>슈퍼점수 하위 70%</b> (per pool)<br>
      · 일평균 <b>3~5건</b> 발생<br>
      · <b>200%+ 매도 14.2%</b> · 100%+ 매도 27.6%<br>
      · 손절률 <b>18.7%</b> · 승률 <b>63.9%</b><br>
      · 평균 수익률 <b style="color:#F97316;">+96%</b><br>
      · 자본 여유 시 <b>보완 매수</b>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### 등급 산정 기준 (상세)

**공통 입력**:
- 시총 상위 300 종목 KRX 종목 풀
- 4 프리셋 시그널 ensemble + Score ≥ 40 (4 시그널 패턴 중 1개 이상 발생)
- 회피 6 자동 적용 (chart_pattern, KOSPI 약세, 과열, 거래대금 ≥3,000억 등)
- 6 RF 분류기 (loss / sw / 100+ / 50+ / 30+ / 10+) + peak 회귀

**슈퍼점수 공식**:
```
슈퍼점수 = p_sw × 5 + p_100+ × 2 + p_50+ × 1 − p_loss × 3
            └ OOS 보정 확률 (실제 적중률) 기반 ┘
```

**등급 분기**:

| 항목 | 슈퍼강력매수 | 추천매수 |
|---|---|---|
| 일자별 슈퍼점수 분위 | 상위 **5%** (top 5%) | 상위 **5~20%** |
| 슈퍼위너 확률 (p_sw) | 대체로 ≥ 0.20 | 대체로 0.10~0.20 |
| 100%+ 확률 (p_100+) | 대체로 ≥ 0.40 | 대체로 0.25~0.40 |
| 손절 확률 (p_loss) | 대체로 < 0.20 | 0.20~0.35 허용 |
| 일평균 발생 | 2~3건 | 3~5건 |

### 실제 5년 OOS 성과 — 매도 실현(180일 종가) 기준
""")

    # === 슈퍼강력매수 vs 추천매수 비교 표 ===
    cmp_df = pd.DataFrame([
        {"항목": "매수 건수",        "슈퍼강력매수": "347건",     "추천매수": "808건"},
        {"항목": "200%+ 매도 (대박)", "슈퍼강력매수": "27.7%",     "추천매수": "14.2%"},
        {"항목": "100%+ 매도 (2배)",  "슈퍼강력매수": "47.0%",     "추천매수": "27.6%"},
        {"항목": "50%+ 매도",        "슈퍼강력매수": "66.6%",     "추천매수": "42.5%"},
        {"항목": "10%+ 매도",        "슈퍼강력매수": "76.1%",     "추천매수": "59.2%"},
        {"항목": "승률 (익절)",       "슈퍼강력매수": "81.6%",     "추천매수": "63.9%"},
        {"항목": "손절률 -20%↓",      "슈퍼강력매수": "8.9%",      "추천매수": "18.7%"},
        {"항목": "평균 수익률",       "슈퍼강력매수": "+151.1%",   "추천매수": "+95.7%"},
        {"항목": "총 실현 수익 (10만/종목)", "슈퍼강력매수": "+5,242만", "추천매수": "+7,733만"},
        {"항목": "5년 수익률 (등급 한정)",   "슈퍼강력매수": "+151.1%", "추천매수": "+95.7%"},
    ])
    # 슈퍼강력매수 = 빨강 강조, 추천매수 = 주황 강조, 승자만 굵게/진하게
    def _sw_style(val):
        return "background-color:#FEF2F2;color:#B91C1C;font-weight:700;text-align:center;"
    def _rc_style(val):
        return "background-color:#FFF7ED;color:#F97316;font-weight:700;text-align:center;"
    def _label_style(val):
        return "font-weight:600;color:#374151;"
    cmp_styler = cmp_df.style
    sm = getattr(cmp_styler, "map", None) or cmp_styler.applymap
    cmp_styler = sm(_label_style, subset=["항목"])
    sm = getattr(cmp_styler, "map", None) or cmp_styler.applymap
    cmp_styler = sm(_sw_style, subset=["슈퍼강력매수"])
    sm = getattr(cmp_styler, "map", None) or cmp_styler.applymap
    cmp_styler = sm(_rc_style, subset=["추천매수"])
    st.dataframe(cmp_styler, hide_index=True, use_container_width=True)

    # 합계 박스
    st.markdown(
        '<div style="margin-top:8px;padding:12px 16px;background:linear-gradient(90deg,#DC2626,#F97316);'
        'color:white;border-radius:8px;text-align:center;">'
        '<div style="font-size:11px;opacity:0.9;letter-spacing:2px;">5년 합계 (2022~2026)</div>'
        '<div style="font-size:22px;font-weight:900;margin-top:2px;">'
        '총 1,155건 매수 · +12,975만 실현 수익 · 수익률 +112.3%</div>'
        '<div style="font-size:12px;opacity:0.95;margin-top:2px;">'
        '투자 1.16억 → 실현 1.30억 (10만원/종목 기준)</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "**적중률·승률·평균 수익률**은 슈퍼강력매수가 압도, "
        "**총 수익금**은 추천매수가 건수 2배 많아서 더 큼. "
        "→ 자본 적으면 슈퍼강력매수만, 자본 충분하면 둘 다 매수."
    )

    # === 년도별 표 ===
    st.markdown("#### 년도별 매도 실현 (10만원/종목 기준)")
    yr_data = pd.DataFrame([
        {"년도":2022, "매수":260, "200%+":"8.8%", "100%+":"13.5%", "50%+":"23.5%", "10%+":"40.4%", "손절":"30.0%", "승률":"46.5%", "평균수익률":"+33.9%", "수익금":"+882만"},
        {"년도":2023, "매수":260, "200%+":"10.0%","100%+":"23.5%", "50%+":"36.9%", "10%+":"55.0%", "손절":"18.1%", "승률":"60.4%", "평균수익률":"+57.9%", "수익금":"+1,506만"},
        {"년도":2024, "매수":265, "200%+":"9.1%", "100%+":"24.5%", "50%+":"49.8%", "10%+":"66.0%", "손절":"18.1%", "승률":"68.7%", "평균수익률":"+68.5%", "수익금":"+1,816만"},
        {"년도":2025, "매수":265, "200%+":"46.0%","100%+":"71.7%", "50%+":"85.7%", "10%+":"93.2%", "손절":"1.5%",  "승률":"95.8%", "평균수익률":"+292.5%","수익금":"+7,752만"},
        {"년도":"2026 (진행중)", "매수":105, "200%+":"15.2%","100%+":"33.3%", "50%+":"55.2%", "10%+":"68.6%", "손절":"4.8%",  "승률":"81.0%", "평균수익률":"+97.0%", "수익금":"+1,019만"},
    ])
    st.dataframe(yr_data, hide_index=True, use_container_width=True)
    st.caption(
        "**2025년 폭발적 강세장 (+292% 평균)** 영향으로 5년 평균이 +112%로 끌어올려짐. "
        "약세장(2022) -1.5% 수익은 가능하나 손절 30% 발생. **2025 같은 강세 사이클에서 슈퍼위너 집중 발굴**되는 게 핵심."
    )

    st.markdown("""
#### 참고: 고점 도달 (peak_180d ≥ 200%) vs 매도 실현
- 슈퍼강력매수: **고점 41.5% 도달** → 매도 시 **27.7% 실현** (13.8%p 되돌림)
- 추천매수: 고점 22.6% 도달 → 매도 14.2% 실현 (8.4%p 되돌림)

→ 모델은 슈퍼위너를 잘 찾지만 **고점에서 못 팔고 일부 되돌리는 게 현실**. 그래도 평균 +151% 실현은 충분히 강력.
""")
    st.markdown("""

### 매수 우선순위 & 시점

```
1순위: 슈퍼강력매수 (슈퍼점수 높은 순)
2순위: 추천매수 (슈퍼점수 + 슈퍼위너 확률 종합)

시점:
  · 1순위: 당일 NXT 19:50 시장가 (정규장 닫힌 직후)
  · 2순위: D+1 정규장 시초가
  · 종목당 10만원 (또는 100만원)

[매도]
  매수일 + 180 거래일 후 정규장 종가
  익절/손절 룰 없음
```

### 어떤 걸 사야 할까?

| 자본 | 권장 |
|---|---|
| **1,000만 이하** | 슈퍼강력매수만 (필수 매수) |
| **1,000~3,000만** | 슈퍼강력매수 우선 + 추천매수 일부 |
| **3,000만~1억** | 둘 다 매수 (주 5~10건) |
| **1억 이상** | 무제한 (강력매수 다 매수) |
""")
    st.markdown("---")
    st.markdown("### 키움 HTS 검색식")
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

조건: A AND B AND C AND D AND E AND F AND G AND H AND I
""", language="text")


def page_superscore():
    st.info("좌측 메뉴로 이전됨.")
    cols = st.columns(3)
    if cols[0].button("오늘의 추천", use_container_width=True):
        st.session_state.page = "ss_today"; st.rerun()
    if cols[1].button("이번 주", use_container_width=True):
        st.session_state.page = "ss_week"; st.rerun()
    if cols[2].button("백테스트", use_container_width=True):
        st.session_state.page = "ss_backtest"; st.rerun()
