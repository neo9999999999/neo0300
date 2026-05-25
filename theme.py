"""
디자인 시스템 v2 — 네이버 그린 그라데이션 + 모바일 반응형

특징:
- 메인 컬러: 강렬 레드 #DC2626 (그라데이션 #EF4444 → #DC2626)
- Pretendard 폰트 우선
- 둥근 모서리 14px
- 부드러운 그림자 (네이버 카드 스타일)
- 모바일 반응형 (768px / 480px breakpoint)
"""

# =============================================================================
# 네이버 디자인 토큰
# =============================================================================
TOKENS = {
    "light": {
        # 표면
        "bg": "#F7F9FA",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F4F6",
        "surface_hover": "#E9EFF2",
        "border": "#E5E8EB",
        "border_strong": "#D4D9DD",
        # 텍스트 (네이버 그레이스케일)
        "text": "#191B1F",
        "text_2": "#4E5559",
        "text_3": "#8B95A1",
        "text_disabled": "#B0B8C1",
        "text_on_accent": "#FFFFFF",
        # 액센트 (한국 시세 빨강)
        "accent": "#DC2626",
        "accent_strong": "#B91C1C",
        "accent_dark": "#991B1B",
        "accent_soft": "#FEF2F2",
        "accent_softer": "#FFF5F5",
        # 그라데이션
        "gradient_start": "#EF4444",
        "gradient_end": "#DC2626",
        # 시멘틱 (한국식 빨강↑/파랑↓)
        "up": "#F04452",
        "up_soft": "#FFEEEF",
        "down": "#1F8FFF",
        "down_soft": "#E8F3FF",
        "warn": "#FF9933",
        "warn_soft": "#FFF4E6",
        "danger": "#E03131",
    },
    "dark": {
        "bg": "#0D1117",
        "surface": "#161B22",
        "surface_alt": "#21262D",
        "surface_hover": "#2D333B",
        "border": "#30363D",
        "border_strong": "#444C56",
        "text": "#E6EDF3",
        "text_2": "#B4BCC4",
        "text_3": "#7D8590",
        "text_disabled": "#484F58",
        "text_on_accent": "#FFFFFF",
        "accent": "#DC2626",
        "accent_strong": "#EF4444",
        "accent_dark": "#B91C1C",
        "accent_soft": "#2A0E0E",
        "accent_softer": "#1A0808",
        "gradient_start": "#EF4444",
        "gradient_end": "#DC2626",
        "up": "#FF6B6B",
        "up_soft": "#3A1E22",
        "down": "#5BA5FF",
        "down_soft": "#1A2F4A",
        "warn": "#FFB951",
        "warn_soft": "#3A2A14",
        "danger": "#FF6B6B",
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
    },
    "dark": {
        **TOKENS["dark"],
        "bg_card": TOKENS["dark"]["surface"],
        "text_secondary": TOKENS["dark"]["text_2"],
        "text_tertiary": TOKENS["dark"]["text_3"],
        "accent_bg": TOKENS["dark"]["accent_soft"],
    },
}


def get_css(mode: str = "light") -> str:
    t = TOKENS[mode]
    return f"""
<style>
/* Pretendard 폰트 로드 */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

:root {{
    --bg: {t['bg']};
    --surface: {t['surface']};
    --surface-alt: {t['surface_alt']};
    --surface-hover: {t['surface_hover']};
    --border: {t['border']};
    --border-strong: {t['border_strong']};
    --text: {t['text']};
    --text-2: {t['text_2']};
    --text-3: {t['text_3']};
    --text-disabled: {t['text_disabled']};
    --on-accent: {t['text_on_accent']};
    --accent: {t['accent']};
    --accent-strong: {t['accent_strong']};
    --accent-dark: {t['accent_dark']};
    --accent-soft: {t['accent_soft']};
    --accent-softer: {t['accent_softer']};
    --gradient-start: {t['gradient_start']};
    --gradient-end: {t['gradient_end']};
    --gradient: linear-gradient(135deg, {t['gradient_start']} 0%, {t['gradient_end']} 100%);
    --up: {t['up']};
    --up-soft: {t['up_soft']};
    --down: {t['down']};
    --down-soft: {t['down_soft']};
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 24px rgba(220,38,38,0.12);
}}

/* =========================================================
   GLOBAL
   ========================================================= */
.stApp {{ background-color: var(--bg); color: var(--text); }}
* {{
    font-family: 'Pretendard', 'Pretendard Variable', -apple-system, BlinkMacSystemFont,
                 'Apple SD Gothic Neo', 'Noto Sans KR', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

.block-container {{
    padding-top: 32px;
    padding-bottom: 80px;
    max-width: 960px;
}}

/* =========================================================
   TYPOGRAPHY (네이버 스타일)
   ========================================================= */
h1 {{
    color: var(--text) !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.6px !important;
    margin-bottom: 8px !important;
    border: none !important;
    padding: 0 !important;
    line-height: 1.3 !important;
}}
h2 {{
    color: var(--text) !important;
    font-size: 19px !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px !important;
    margin-top: 28px !important;
    margin-bottom: 12px !important;
}}
h3 {{
    color: var(--text) !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}}
p, .stMarkdown p {{
    color: var(--text-2);
    font-size: 14px;
    line-height: 1.65;
    letter-spacing: -0.2px;
}}
strong, b {{ color: var(--text); font-weight: 700; }}

/* =========================================================
   SIDEBAR (네이버 미니멀)
   ========================================================= */
section[data-testid="stSidebar"] {{
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 24px; padding-left: 16px; padding-right: 16px;
}}

/* =========================================================
   BUTTON — 네이버 그린 그라데이션
   ========================================================= */

/* Primary 버튼 본체 */
button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {{
    background: var(--gradient) !important;
    background-color: var(--accent) !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(220,38,38,0.20) !important;
    transition: all 0.18s ease !important;
}}
button[kind="primary"] *,
[data-testid="baseButton-primary"] *,
[data-testid="stBaseButton-primary"] * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    text-shadow: none !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}}
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.30) !important;
    filter: brightness(1.05) !important;
}}

/* Secondary 버튼 본체 */
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="stBaseButton-secondary"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}}
button[kind="secondary"] *,
[data-testid="baseButton-secondary"] *,
[data-testid="stBaseButton-secondary"] * {{
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}}
button[kind="secondary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {{
    background-color: var(--accent-softer) !important;
    border-color: var(--accent) !important;
}}
button[kind="secondary"]:hover * {{
    color: var(--accent-dark) !important;
    -webkit-text-fill-color: var(--accent-dark) !important;
}}

/* 메인 페이지 버튼 - 큰 패딩 */
section.main button,
.main button,
div[data-testid="stMain"] button {{
    border-radius: 12px !important;
    padding: 13px 22px !important;
    font-size: 15px !important;
}}

/* 사이드바 버튼 - 좌측정렬 */
section[data-testid="stSidebar"] button {{
    width: 100% !important;
    text-align: left !important;
    border-radius: 10px !important;
    padding: 13px 14px !important;
    font-size: 14px !important;
    margin-bottom: 4px !important;
}}
section[data-testid="stSidebar"] button[kind="secondary"] {{
    background-color: transparent !important;
    border-color: transparent !important;
}}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
    background-color: var(--accent-softer) !important;
    border-color: transparent !important;
}}

/* =========================================================
   CARD — 네이버 부드러운 그림자
   ========================================================= */
.tcard {{
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: all 0.18s ease;
    box-shadow: var(--shadow-sm);
}}
.tcard:hover {{
    border-color: var(--accent);
    box-shadow: var(--shadow-lg);
    transform: translateY(-1px);
}}

/* =========================================================
   FORM INPUTS — 네이버 스타일
   ========================================================= */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] {{
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    min-height: 46px !important;
    transition: border-color 0.15s !important;
}}
div[data-baseweb="select"]:focus-within > div,
div[data-baseweb="input"]:focus-within {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-softer) !important;
}}
div[data-baseweb="select"] *,
div[data-baseweb="input"] input {{
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: 15px !important;
}}

input[type="number"], input[type="text"], input[type="password"] {{
    color: var(--text) !important;
    font-size: 15px !important;
}}

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
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 14px 18px !important;
    transition: all 0.15s !important;
}}
[data-testid="stExpander"] summary:hover {{
    border-color: var(--accent) !important;
    background-color: var(--accent-softer) !important;
}}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary * {{
    color: var(--text) !important;
    font-weight: 600 !important;
}}

/* =========================================================
   TAB — 네이버 언더라인 스타일
   ========================================================= */
button[data-baseweb="tab"] {{
    color: var(--text-2) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 18px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent-dark) !important;
}}
[data-baseweb="tab-highlight"] {{
    background-color: var(--accent) !important;
    height: 3px !important;
    border-radius: 2px !important;
}}

/* =========================================================
   ALERT (Toast)
   ========================================================= */
div[data-testid="stAlertContainer"] {{
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    padding: 14px 18px !important;
}}
div[data-testid="stAlertContainer"][kind="info"] {{
    background-color: var(--accent-softer) !important;
    border-color: var(--accent-soft) !important;
}}
div[data-testid="stAlertContainer"][kind="success"] {{
    background-color: var(--accent-soft) !important;
    border-color: var(--accent) !important;
}}
div[data-testid="stAlertContainer"][kind="warning"] {{
    background-color: {t['warn_soft']} !important;
    border-color: {t['warn']} !important;
}}
div[data-testid="stAlertContainer"][kind="error"] {{
    background-color: var(--up-soft) !important;
    border-color: var(--up) !important;
}}
div[data-testid="stAlertContainer"] * {{
    color: var(--text) !important;
}}

/* =========================================================
   DATAFRAME / TABLE
   ========================================================= */
div[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}}

/* =========================================================
   PROGRESS — 그라데이션
   ========================================================= */
.stProgress > div > div {{
    background: var(--gradient) !important;
    border-radius: 4px;
}}
.stProgress {{
    background-color: var(--border);
    height: 6px;
    border-radius: 4px;
}}

/* =========================================================
   UTILITY
   ========================================================= */
.up {{ color: var(--up) !important; font-weight: 700; }}
.down {{ color: var(--down) !important; font-weight: 700; }}
.accent {{ color: var(--accent-dark) !important; }}
.muted {{ color: var(--text-2) !important; }}
.subtle {{ color: var(--text-3) !important; }}

.badge {{
    display: inline-block;
    background-color: var(--accent-soft);
    color: var(--accent-dark);
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
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.2;
}}
.big-number-label {{
    font-size: 12px;
    color: var(--text-3);
    margin-bottom: 4px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}

hr {{ border-color: var(--border); margin: 24px 0; }}

code {{
    background-color: var(--accent-softer);
    color: var(--accent-dark);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85em;
    border: 1px solid var(--accent-soft);
    font-family: 'JetBrains Mono', 'D2Coding', monospace;
}}

/* Streamlit footer 숨김 (햄버거는 모바일 위해 유지) */
footer, #MainMenu {{
    visibility: hidden !important;
    height: 0 !important;
}}

/* 데스크탑은 헤더 숨김, 모바일은 햄버거 메뉴 보이게 */
@media (min-width: 769px) {{
    header[data-testid="stHeader"] {{
        visibility: hidden !important;
        height: 0 !important;
    }}
}}
@media (max-width: 768px) {{
    header[data-testid="stHeader"] {{
        background-color: var(--surface) !important;
        height: 50px !important;
        border-bottom: 1px solid var(--border);
        z-index: 999 !important;
    }}
    /* 햄버거 메뉴 버튼 (사이드바 토글) — 잘 보이게 */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"] {{
        visibility: visible !important;
        display: flex !important;
        color: var(--accent-dark) !important;
        background: var(--accent-softer) !important;
        border-radius: 8px !important;
        padding: 6px !important;
    }}
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {{
        color: var(--accent-dark) !important;
        fill: var(--accent-dark) !important;
        width: 24px !important;
        height: 24px !important;
    }}
    /* 메인 컨테이너 상단 여백 — 헤더 가리지 않게 */
    .block-container {{
        padding-top: 70px !important;
    }}
}}

/* =========================================================
   📱 모바일 상단 가로 네비게이션 (sidebar fallback)
   ========================================================= */
.mobile-nav {{
    display: none;
    position: sticky;
    top: 50px;
    z-index: 998;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 8px 12px;
    margin: -16px -14px 16px -14px;
    overflow-x: auto;
    white-space: nowrap;
    -webkit-overflow-scrolling: touch;
}}
.mobile-nav-item {{
    display: inline-block;
    padding: 8px 16px;
    margin-right: 6px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-2);
    background: var(--surface-alt);
    border: 1px solid var(--border);
    text-decoration: none;
}}
.mobile-nav-item.active {{
    background: var(--gradient);
    color: #FFFFFF !important;
    border-color: var(--accent-dark);
    box-shadow: 0 2px 6px rgba(220,38,38,0.20);
}}
@media (max-width: 768px) {{
    .mobile-nav {{ display: block; }}
}}

/* =========================================================
   📱 모바일 반응형 (768px 이하 — 태블릿)
   ========================================================= */
@media (max-width: 768px) {{
    .block-container {{
        padding-left: 14px !important;
        padding-right: 14px !important;
        padding-top: 18px !important;
    }}
    h1 {{ font-size: 22px !important; }}
    h2 {{ font-size: 17px !important; }}
    h3 {{ font-size: 15px !important; }}
    .tcard {{ padding: 14px 16px !important; border-radius: 12px !important; }}
    .big-number {{ font-size: 26px !important; }}
    section.main button,
    .main button {{
        padding: 12px 18px !important;
        font-size: 14px !important;
    }}
    /* 모바일에서 데이터프레임 가로 스크롤 가능 */
    div[data-testid="stDataFrame"] {{
        overflow-x: auto !important;
    }}
    table {{
        font-size: 11px !important;
    }}
    table th, table td {{
        padding: 6px 4px !important;
    }}
}}

/* =========================================================
   📱 모바일 반응형 (480px 이하 — 폰)
   ========================================================= */
@media (max-width: 480px) {{
    .block-container {{
        padding-left: 10px !important;
        padding-right: 10px !important;
    }}
    h1 {{ font-size: 20px !important; }}
    .tcard {{ padding: 12px 14px !important; }}
    .big-number {{ font-size: 22px !important; }}
    /* 사이드바 자동 닫힘은 Streamlit 기본 동작 */
    table {{ font-size: 10px !important; }}
    table th, table td {{ padding: 5px 3px !important; }}
    /* 카드 그리드 1열로 */
    [data-testid="column"] {{
        width: 100% !important;
        flex: 0 0 100% !important;
    }}
}}

/* =========================================================
   접근성 — 다크모드 대비
   ========================================================= */
@media (prefers-color-scheme: dark) {{
    /* 다크모드 자동 감지는 사용자 토글로 처리 */
}}

</style>
"""


def get_logo_html(mode: str = "light") -> str:
    t = TOKENS[mode]
    return (
        f'<div style="padding:0 4px 28px 4px;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="width:34px;height:34px;background:linear-gradient(135deg,{t["gradient_start"]} 0%,{t["gradient_end"]} 100%);'
        f'border-radius:10px;display:flex;align-items:center;justify-content:center;'
        f'box-shadow:0 2px 8px rgba(220,38,38,0.25);">'
        f'<span style="color:#FFFFFF;font-weight:900;font-size:18px;letter-spacing:-1px;">N</span>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:16px;font-weight:800;color:{t["text"]};line-height:1.2;letter-spacing:-0.4px;">'
        f'NEO STOCK</div>'
        f'<div style="font-size:10px;color:{t["text_3"]};letter-spacing:0.6px;font-weight:600;text-transform:uppercase;">'
        f'V·S·A·B 등급제</div>'
        f'</div></div></div>'
    )
