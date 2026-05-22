"""
디자인 시스템 — 토큰 기반 + 컴포넌트 일괄 처리.

원칙:
- Primary 버튼: 모든 위치(메인/사이드바) 핑크 배경 + 흰색 굵은 글자
- Secondary 버튼: 흰색 배경 + 검정 굵은 글자 + 얇은 보더
- 텍스트 컬러 부모/자식 모두 강제 (Streamlit 내부 p 태그 대응)
- 이모지 ❌, 텍스트만
"""

# =============================================================================
# 디자인 토큰
# =============================================================================
TOKENS = {
    "light": {
        # 표면
        "bg": "#FAFAFA",
        "surface": "#FFFFFF",
        "surface_alt": "#F5F5F5",
        "border": "#EAEAEA",
        "border_strong": "#D0D0D0",
        # 텍스트
        "text": "#1A1A1A",
        "text_2": "#4B5563",
        "text_3": "#9CA3AF",
        "text_on_accent": "#FFFFFF",
        # 액센트
        "accent": "#E91E63",
        "accent_strong": "#C2185B",
        "accent_soft": "#FCE4EC",
        # 시멘틱
        "up": "#E91E63",
        "down": "#3B82F6",
        "warn": "#F59E0B",
        "danger": "#DC2626",
    },
    "dark": {
        "bg": "#0F0F12",
        "surface": "#1A1A1F",
        "surface_alt": "#22222A",
        "border": "#2A2A30",
        "border_strong": "#3A3A42",
        "text": "#FAFAFA",
        "text_2": "#B0B0B8",
        "text_3": "#7A7A82",
        "text_on_accent": "#FFFFFF",
        "accent": "#EC4899",
        "accent_strong": "#DB2777",
        "accent_soft": "#3F1A2F",
        "up": "#EC4899",
        "down": "#60A5FA",
        "warn": "#F59E0B",
        "danger": "#F87171",
    },
}

# 하위 호환 alias
PALETTE = {
    "light": {
        **TOKENS["light"],
        "bg_card": TOKENS["light"]["surface"],
        "text_secondary": TOKENS["light"]["text_2"],
        "text_tertiary": TOKENS["light"]["text_3"],
        "accent_bg": TOKENS["light"]["accent_soft"],
        "accent_soft": TOKENS["light"]["accent_strong"],  # 기존 코드 호환
    },
    "dark": {
        **TOKENS["dark"],
        "bg_card": TOKENS["dark"]["surface"],
        "text_secondary": TOKENS["dark"]["text_2"],
        "text_tertiary": TOKENS["dark"]["text_3"],
        "accent_bg": TOKENS["dark"]["accent_soft"],
        "accent_soft": TOKENS["dark"]["accent_strong"],
    },
}


def get_css(mode: str = "light") -> str:
    t = TOKENS[mode]

    # ────────────────────────────────────────────────────────
    # CSS 토큰 변수 + 글로벌 컴포넌트
    # ────────────────────────────────────────────────────────
    return f"""
<style>
:root {{
    --bg: {t['bg']};
    --surface: {t['surface']};
    --surface-alt: {t['surface_alt']};
    --border: {t['border']};
    --border-strong: {t['border_strong']};
    --text: {t['text']};
    --text-2: {t['text_2']};
    --text-3: {t['text_3']};
    --on-accent: {t['text_on_accent']};
    --accent: {t['accent']};
    --accent-strong: {t['accent_strong']};
    --accent-soft: {t['accent_soft']};
    --up: {t['up']};
    --down: {t['down']};
}}

/* =========================================================
   GLOBAL
   ========================================================= */
.stApp {{ background-color: var(--bg); color: var(--text); }}
* {{ font-family: -apple-system, BlinkMacSystemFont, "Pretendard", "Apple SD Gothic Neo", system-ui, sans-serif; }}

.block-container {{
    padding-top: 36px;
    padding-bottom: 60px;
    max-width: 920px;
}}

/* =========================================================
   TYPOGRAPHY
   ========================================================= */
h1 {{
    color: var(--text) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 8px !important;
    border: none !important;
    padding: 0 !important;
}}
h2 {{
    color: var(--text) !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    margin-top: 28px !important;
    margin-bottom: 12px !important;
}}
h3 {{
    color: var(--text) !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}}
p, .stMarkdown p {{
    color: var(--text-2);
    font-size: 14px;
    line-height: 1.6;
}}
strong, b {{ color: var(--text); }}

/* =========================================================
   SIDEBAR
   ========================================================= */
section[data-testid="stSidebar"] {{
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 24px; padding-left: 20px; padding-right: 20px;
}}

/* =========================================================
   BUTTON SYSTEM (GLOBAL — 모든 위치 적용)
   ========================================================= */

/* ========= Primary 버튼 본체 ========= */
button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {{
    background-color: var(--accent) !important;
    border: none !important;
    box-shadow: none !important;
}}
/* Primary 자식 요소 - 텍스트 색상만 강제, 보더/배경 제거 */
button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div,
[data-testid="baseButton-primary"] *,
[data-testid="baseButton-primary"] p,
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-primary"] p {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    text-shadow: none !important;
    font-weight: 800 !important;
}}
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {{
    background-color: var(--accent-strong) !important;
}}
button[kind="primary"]:hover *,
[data-testid="baseButton-primary"]:hover *,
[data-testid="stBaseButton-primary"]:hover * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

/* ========= Secondary 버튼 본체 ========= */
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="stBaseButton-secondary"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}}
/* Secondary 자식 요소 - 텍스트만, 보더/배경 제거 */
button[kind="secondary"] *,
button[kind="secondary"] p,
button[kind="secondary"] span,
button[kind="secondary"] div,
[data-testid="baseButton-secondary"] *,
[data-testid="baseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-secondary"] p {{
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 700 !important;
}}
button[kind="secondary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {{
    background-color: var(--accent-soft) !important;
    border-color: var(--accent) !important;
}}
button[kind="secondary"]:hover *,
[data-testid="baseButton-secondary"]:hover *,
[data-testid="stBaseButton-secondary"]:hover * {{
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    background-color: transparent !important;
}}

/* 3) 메인 페이지 버튼 - 큰 패딩 */
section.main button,
.main button,
div[data-testid="stMain"] button {{
    border-radius: 12px !important;
    padding: 14px 22px !important;
    font-size: 15px !important;
}}

/* 4) 사이드바 버튼 - 좌측정렬, 풀너비 */
section[data-testid="stSidebar"] button {{
    width: 100% !important;
    text-align: left !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
    margin-bottom: 4px !important;
}}
section[data-testid="stSidebar"] button[kind="secondary"] {{
    background-color: transparent !important;
    border-color: transparent !important;
}}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
    background-color: var(--accent-soft) !important;
    border-color: transparent !important;
}}

/* =========================================================
   CARD
   ========================================================= */
.tcard {{
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: all 0.15s;
}}
.tcard:hover {{
    border-color: var(--accent);
    box-shadow: 0 4px 12px rgba(233, 30, 99, 0.08);
}}

/* =========================================================
   FORM INPUTS
   ========================================================= */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] {{
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    min-height: 44px !important;
}}
div[data-baseweb="select"] *,
div[data-baseweb="input"] input {{
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}}

/* number input 화살표 */
input[type="number"] {{ color: var(--text) !important; }}

/* =========================================================
   METRIC
   ========================================================= */
div[data-testid="stMetric"] {{
    background-color: transparent;
    border: none;
    padding: 0;
}}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label * {{
    color: var(--text-3) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {{
    color: var(--text) !important;
    font-weight: 800 !important;
    font-size: 22px !important;
}}

/* =========================================================
   EXPANDER
   ========================================================= */
.streamlit-expanderHeader,
details > summary,
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] summary {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 12px 16px !important;
}}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary * {{
    color: var(--text) !important;
}}

/* =========================================================
   TAB
   ========================================================= */
button[data-baseweb="tab"] {{
    color: var(--text-2) !important;
    font-weight: 600 !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}}

/* =========================================================
   ALERT
   ========================================================= */
div[data-testid="stAlertContainer"] {{
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 18px !important;
}}
div[data-testid="stAlertContainer"][kind="info"] {{
    background-color: var(--accent-soft) !important;
}}
div[data-testid="stAlertContainer"][kind="success"] {{
    background-color: var(--accent-soft) !important;
}}
div[data-testid="stAlertContainer"][kind="warning"] {{
    background-color: rgba(245, 158, 11, 0.1) !important;
}}
div[data-testid="stAlertContainer"][kind="error"] {{
    background-color: rgba(220, 38, 38, 0.1) !important;
}}
div[data-testid="stAlertContainer"] * {{
    color: var(--text) !important;
}}

/* =========================================================
   DATAFRAME
   ========================================================= */
div[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}}

/* =========================================================
   PROGRESS
   ========================================================= */
.stProgress > div > div {{
    background-color: var(--accent) !important;
    border-radius: 4px;
}}
.stProgress {{
    background-color: var(--border);
    height: 6px;
}}

/* =========================================================
   UTILITY
   ========================================================= */
.up {{ color: var(--up) !important; font-weight: 700; }}
.down {{ color: var(--down) !important; font-weight: 700; }}
.accent {{ color: var(--accent) !important; }}
.muted {{ color: var(--text-2) !important; }}
.subtle {{ color: var(--text-3) !important; }}

.badge {{
    display: inline-block;
    background-color: var(--accent-soft);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 4px;
}}
.badge-secondary {{
    background-color: var(--surface-alt);
    color: var(--text-2);
    border: 1px solid var(--border);
}}

.empty-state {{
    text-align: center;
    padding: 80px 24px;
    color: var(--text-3);
}}
.empty-state .emoji {{
    font-size: 48px;
    margin-bottom: 16px;
}}

.big-number {{
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.2;
}}
.big-number-label {{
    font-size: 13px;
    color: var(--text-2);
    margin-bottom: 4px;
    font-weight: 500;
}}

hr {{ border-color: var(--border); margin: 24px 0; }}

code {{
    background-color: var(--accent-soft);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85em;
    border: 1px solid var(--border);
}}

/* Streamlit chrome 숨김 */
footer, #MainMenu, header[data-testid="stHeader"] {{
    visibility: hidden !important;
    height: 0 !important;
}}
</style>
"""


def get_logo_html(mode: str = "light") -> str:
    t = TOKENS[mode]
    return (
        f'<div style="padding:0 4px 28px 4px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<div style="width:6px;height:28px;background:{t["accent"]};border-radius:3px;"></div>'
        f'<div>'
        f'<div style="font-size:17px;font-weight:800;color:{t["text"]};line-height:1.2;">종가매수</div>'
        f'<div style="font-size:10px;color:{t["text_3"]};letter-spacing:0.5px;font-weight:600;">MARKET CLOSE TRADING</div>'
        f'</div></div></div>'
    )
