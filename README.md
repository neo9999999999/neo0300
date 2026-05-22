# 📈 V/S/A/B 등급제 종가매수 추천 시스템

**🔗 라이브: https://neo0300.streamlit.app** (비밀번호: `123456`)

4 프리셋 앙상블 기반 V/S/A/B 등급제 종가매수 자동 추천 + 백테스트 + 사례 가이드.

---

## 🎯 핵심 — V/S/A/B 등급 시스템

| 등급 | 비중 | 조건 | 빈도 | 평균수익 (180일) | 큰손실률 |
|---|---|---|---|---|---|
| 🏆 V | 50만 | 코스닥 · 등락 7~25% · 점수 ≥ 75 | 연 5회 | **+96.9%** | 7% |
| 💎 S | 30만 | + 4프리셋 만장일치 + 점수 ≥ 65 | 연 17회 | +56.8% | **4%** |
| ⭐ A | 20만 | + 등락 10~18% + 점수 ≥ 65 | 연 22회 | +34.1% | 9% |
| 🟢 B | 10만 | + V1 통과 (1개+) | 매일 | +34.4% | 12% |

**6년 백테스트: 자본 2,100만 → +6,318만 (연 ROI +50.1%)**

---

## 🚀 고급 전략 TOP 3 (NEW)

| 순위 | 전략 | 6년 누적 | 손익비 | 연 ROI | 자본 |
|---|---|---|---|---|---|
| 🏆 TOP 1 | V/S/A/B **365일 보유** | **+9,493만** | 5.96 | 38.3% | 4,130만 |
| 🥈 TOP 2 | 180d **+ 레짐 스케일** (BULL 1.5x/BEAR 0.3x) | +7,900만 | 5.02 | **40.8%** | 3,230만 |
| 🥉 TOP 3 | **240일 보유** | +7,617만 | **5.49** | 46.3% | 2,650만 |

자세한 분석: [`ADVANCED_STRATEGIES.md`](./ADVANCED_STRATEGIES.md)

---

## 🌐 어디서 작동?

| 환경 | 실시간 스캔 | 캐시 데이터 | 백테스트 |
|---|---|---|---|
| **로컬 PC (한국 IP)** | ✅ 1~2분 | ✅ | ✅ |
| **cloudflared 터널** (PC 호스팅) | ✅ 1~2분 | ✅ | ✅ |
| **Streamlit Cloud** (미국 IP) | ❌ KRX 차단 (자동 캐시 fallback) | ✅ | ✅ |
| **+ KIS OpenAPI 키 설정 시** | ✅ 어디서나 | ✅ | ✅ |

**캐시는 GitHub Actions로 매일 한국시간 16:30 자동 갱신** (.github/workflows/daily.yml)

---

## 📦 페이지

1. **🟢 오늘의 종가매수 추천**
   - V/S/A/B 등급별 카드 (매일 조건 만족 종목 모두 표시)
   - 추천사유 자동 생성 (왜 V/S/A/B인지)
   - 총 매수금액 계산 + CSV 다운로드

2. **📊 백테스트 결과**
   - **🆕 보유기간 선택**: 20/30/60/90/120/180/240/365일
   - **🆕 정렬**: 최신순/오래된순/수익률↓↑/점수↓
   - **🆕 필터 적용 버튼** (년/월 다중 선택 후 적용)
   - 등급별 월별 손익 + 누적 + 통계 요약
   - 일자별 종목 리스트 (모든 등급 모두 표시)

3. **📚 사례 & 가이드**
   - V/S/A/B 등급 가이드 (조건/사유/운용)
   - 고급 전략 TOP 3 (NEW)
   - 변동성 분석 + 자본별 추천
   - 실전 사례 35건 (하바로셀/하승훈)

---

## 🔧 로컬 실행

```bash
git clone https://github.com/neo9999999999/neo0300
cd neo0300
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속 → 비밀번호 `123456`.

---

## 🔐 KIS API 키 설정 (선택, 어디서나 실시간)

1. https://apiportal.koreainvestment.com 가입 + 앱 등록
2. APP_KEY, APP_SECRET 발급
3. **Streamlit Cloud**:
   - https://share.streamlit.io → `neo0300` → Settings → Secrets:
   ```toml
   KIS_APP_KEY = "..."
   KIS_APP_SECRET = "..."
   APP_PASSWORD = "123456"
   ```
4. **로컬**:
   ```bash
   export KIS_APP_KEY='...'
   export KIS_APP_SECRET='...'
   ```

---

## 🔄 데이터 자동 갱신

- **GitHub Actions** (`.github/workflows/daily.yml`): 매일 16:30 KST
  - KRX 종목 마스터 + 시세 갱신 (cache/market_snapshot.parquet)
  - 시총 상위 500 OHLCV 증분 업데이트
  - V/S/A/B용 enriched parquet 재계산
  - 자동 커밋 + 푸시 → Streamlit Cloud 자동 재배포

---

## 🧠 사용된 4개 프리셋 (V1 앙상블)

1. **default** — 5대 시그널 균형
2. **box_breakout** — 박스권 돌파 (S1) 강조
3. **habarocell** — 거래량 (S2) + 장대양봉 (S3) 강조
4. **pullback** — 첫눌림 (S7) + 패턴 (S12) 강조

V1 통과 = 4개 중 1개 이상 추천한 종목.

---

## 📊 12 시그널

| | 이름 | 설명 |
|---|---|---|
| S1 | 박스권 돌파 | 직전 60일 횡보 박스 상단 돌파 |
| S2 | 거래량 폭증 | 20일 평균 × 3배 이상 |
| S3 | 장대양봉 | 시가→종가 +5% 이상 |
| S4 | 이평선 정배열 | MA3 > MA5 > MA10 |
| S5 | 전고점 근접/돌파 | 60일 최고가 95% 이상 |
| S6 | 미과열 | 5일 누적 ≤25% (안전) |
| S7 | 첫 눌림목 | 1차 슈팅 후 MA20 근처 조정 |
| S8 | 수급 연속성 | 최근 거래량 추세 ↑ |
| S9 | 장기이평 돌파 | 120/240/480일 돌파 |
| S10 | 상대강도 | 시장 대비 강세 |
| S11 | 갭 + 이평 | 과거 갭 + 이평선 중첩 지지 |
| S12 | 패턴 품질 | 컵앤핸들/역헤드앤숄더 |

---

## 📝 변경 이력

- **2026-05-23**: V/S/A/B 등급제 + 고급 전략 TOP 3 + 보유기간 선택 + 정렬/필터 적용 버튼
- **2026-05-22**: V/S/A/B 등급 시스템 도입 (4 프리셋 앙상블)
- **2026-05-21**: 9 프리셋 OOS 검증 백테스트
- **2026-05-15**: Walk-Forward OOS 검증 시스템
- **2026-05-10**: 단타 + 중장기 비교 (D+1 OHLC 4개, 20~120일)

---

## 📂 디렉토리 구조

```
neo0300/
├── app.py                  # Streamlit 메인 UI
├── grade.py                # V/S/A/B 등급 분류
├── scanner.py              # 시그널 계산 (3중 fallback)
├── kis_api.py              # 한국투자 OpenAPI 모듈
├── backtest_helpers.py     # forward returns
├── precompute_enriched.py  # enriched parquet 사전 계산
├── daily_update.py         # KRX 데이터 자동 갱신
├── telegram_alert.py       # 텔레그램 알림
├── .github/workflows/
│   └── daily.yml           # 매일 자동 갱신 워크플로
├── cache/
│   ├── market_snapshot.parquet  # KRX 종목 마스터 (auto)
│   ├── ohlcv_*.pkl              # OHLCV 캐시 (auto, LFS)
│   └── enriched_*.parquet       # V/S/A/B용 사전 계산 (auto)
├── ADVANCED_STRATEGIES.md  # 분석 보고서
├── VSAB_README.md          # V/S/A/B 시스템 사용 가이드
└── requirements.txt
```
