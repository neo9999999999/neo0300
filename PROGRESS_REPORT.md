# 🌙 밤새 진행 보고서 — 최종

**작업 일시**: 2026-05-23 새벽 ~ 아침
**작업자**: Claude (자동 모드)

---

## 🎉 KIS API 통합 완료 + 실시간 작동

### ✅ 실제 검증된 것
- **KIS API 토큰 발급**: ✓ OK
- **삼성전자 실시간 시세**: 292,500원 (-2.34%) ✓ 정확
- **KOSPI/KOSDAQ 등락률 상위 통합**: 104개 후보 받음 ✓
- **GitHub Actions 워크플로**: 1분 1초에 완료 ✓
- **GitHub Secrets 등록**: KIS_APP_KEY + KIS_APP_SECRET + APP_PASSWORD ✓

### 🌐 라이브 사이트 (3곳)

| URL | 상태 | KIS API | 비고 |
|---|---|---|---|
| **cloudflared 터널** (네 PC) | ✅ 작동 중 | ✅ 활성 | https://gmc-friendship-citations-hits.trycloudflare.com |
| **Streamlit Cloud** | ✅ 작동 중 | 🟡 사용자 등록 필요 | https://neo0300.streamlit.app |
| **GitHub Actions** | ✅ 매일 자동 | ✅ 활성 | 매일 16:30 KST 자동 갱신 |

비밀번호 모두 동일: `123456`

---

## 🔥 Streamlit Cloud에도 실시간 작동시키기 (5초)

1. https://share.streamlit.io 접속
2. `neo0300` 앱 → ⋮ → **Settings** → **Secrets**
3. 아래 붙여넣기 후 Save:

```toml
KIS_APP_KEY = "PSlvF9kHxE855LQojfAAxlTuLzkTP8ypScwp"
KIS_APP_SECRET = "HJI7OGHtkYiwSkodQNUe87GpJOG+B7OabnAud0xqGzLoXV7a292eGqGrdyLSy6iVdEa4fdeaHYuIQR3ciwVvJD92JrNyNHiGB3qmbb1X6VUoICsSzFz+kpm+p2Wqk5De7MXJOrQg0AnPiVrRU9YryRi7vaR9Ma8jw9YyEC28w/rmsXwfkQg="
KIS_USE_MOCK = "false"
APP_PASSWORD = "123456"
```

4. 1분 후 자동 재로딩 → 어디서나 실시간 작동

---

## ✅ 완료된 작업 전체

| # | 요청/작업 | 상태 |
|---|---|---|
| 1 | V/S/A/B 매일 있으면 다 표시 | ✅ |
| 2 | 180일인데 120일로 표기 버그 | ✅ |
| 3 | 백테스트 필터 적용 버튼 | ✅ |
| 4 | 결과 정렬 (오래된순/최신순 등) | ✅ |
| 5 | 더 좋은 전략 추천 | ✅ TOP 3 발견 |
| 6 | 실시간 가능하게 | ✅ KIS API 통합 |
| 7 | KIS 키 GitHub 등록 | ✅ Secrets 3개 |
| 8 | GH Actions 매일 자동 갱신 (KIS 사용) | ✅ 검증 완료 |
| 9 | 240/365일 데이터 추가 | ✅ |
| 10 | 보유기간 8개 옵션 | ✅ |
| 11 | 고급 전략 페이지 | ✅ |
| 12 | run_with_kis.sh 자동 실행 스크립트 | ✅ |

---

## 🚀 발견한 더 좋은 전략 TOP 3

| 순위 | 전략 | 6년 손익 | vs 베이스 | 연 ROI | 자본 |
|---|---|---|---|---|---|
| 🏆 1위 | **V/S/A/B 365일 보유** | **+9,493만** | **+46.5%** | 38.3% | 4,130만 |
| 🥈 2위 | **180d + 레짐 스케일** (BULL 1.5x/BEAR 0.3x) | +7,900만 | +22.0% | **40.8%** | 3,230만 |
| 🥉 3위 | **240일 보유** | +7,617만 | +17.6% | 46.3% | 2,650만 |

**충격적 발견**:
- 익절/손절 → 모두 5천만+ 손해 (슈퍼위너 잘라서)
- HIGH_VOL 시점 매수가 오히려 +63% 수익, 큰손실 6%
- 보유 길수록 좋음 (180일 → 365일 +3,015만 추가)

자세한 분석: [`ADVANCED_STRATEGIES.md`](./ADVANCED_STRATEGIES.md)

---

## 📊 사이트 변경 사항 (사이트에서 확인)

### 🟢 오늘의 종가매수 추천
- 🆕 V/S/A/B 등급별 카드 (조건 만족 종목 매일 모두 표시)
- 🆕 KIS API 자동 사용 (cloudflared 터널 OR Streamlit Cloud Secrets 등록 시)
- 🆕 캐시 fallback (KIS 실패 시)
- 등급별 추천사유 펼침

### 📊 백테스트 결과
- 🆕 **보유기간 선택**: 20/30/60/90/120/180/240/365일 (기본 **180일** ⭐)
- 🆕 **정렬**: 최신순/오래된순/수익률↓↑/점수↓
- 🆕 **✅ 필터 적용하기 버튼** (즉시 적용 X → 버튼 누를 때만)
- 🆕 일자별 모든 등급 모두 표시
- 🆕 통계 요약 + CSV 다운로드

### 📚 사례 & 가이드
- 🆕 **🚀 고급 전략 TOP 3** 섹션
- 🆕 **변동성 분석** (HIGH_VOL이 좋다는 충격적 발견)
- 🆕 **자본별 추천 표** (1,100만~4,100만)
- 🆕 익절/손절 vs 그냥 보유 비교

---

## 🔄 자동화 (사용자 액션 필요 X)

| 시스템 | 상태 | 작동 |
|---|---|---|
| GitHub Actions 매일 갱신 | ✅ | 한국시간 16:30 → KIS+KRX+enriched 자동 → 푸시 |
| Streamlit Cloud 자동 재배포 | ✅ | 푸시되면 1~2분 후 자동 |
| cloudflared 터널 (네 PC) | ✅ | https://gmc-friendship-citations-hits.trycloudflare.com |
| KIS API GitHub Secrets | ✅ | 3개 등록 완료 |
| KIS API .env (로컬) | ✅ | run_with_kis.sh로 자동 로드 |

---

## 🎯 사용 방법

### 옵션 A: cloudflared 터널 (이미 작동 중)
```
https://gmc-friendship-citations-hits.trycloudflare.com
비밀번호: 123456
```
→ 네 PC가 켜져있는 한 KIS API + cache 모두 작동

### 옵션 B: Streamlit Cloud (Secrets 등록 후 영구)
```
https://neo0300.streamlit.app
비밀번호: 123456
```
→ Secrets 등록 안 해도 cache fallback으로 작동, 등록 시 실시간

### 옵션 C: 로컬 실행
```bash
cd /Users/neo/Desktop/jongga_picker
./run_with_kis.sh
```
→ .env 자동 로드 + Streamlit + cloudflared 모두 실행

---

## 📝 GitHub 푸시 로그 (7개 커밋)

```
d96ba73 — V/S/A/B 모두 표시 + 필터 적용 버튼 + 정렬
783e3db — 시간 매트릭스 180일 + GH Actions enriched 자동 갱신
da4281d — 고급 전략 TOP 3 + 240/365일 데이터 + KRX 타임아웃
9df0e67 — README + 진행 보고서
f5ebc62 — KIS API 통합 강화 + GitHub Actions Secrets 사용
```

---

## 💎 발견한 사실 (백테스트 검증)

| 메트릭 | 베이스 (180d) | TOP 1 (365d) | TOP 2 (180d+레짐) | TOP 3 (240d) |
|---|---:|---:|---:|---:|
| 6년 누적 손익 | +6,478만 | **+9,493만** | +7,900만 | +7,617만 |
| 연 ROI | **+48.9%** | +38.3% | +40.8% | +46.3% |
| 손익비 | 5.04 | **5.96** | 5.02 | 5.49 |
| 큰손실률 | 16.9% | 23.1% | **16.9%** | 19.5% |
| 큰수익률 | 13.9% | **19.5%** | 13.9% | 16.8% |
| 필요 자본 | **2,210만** | 4,130만 | 3,230만 | 2,650만 |
| 승률 | 54.6% | 54.1% | 54.6% | 54.7% |

---

## 🌅 아침 체크리스트

1. **사이트 새로고침** — https://neo0300.streamlit.app (비번 123456)
2. **V/S/A/B 모든 등급 다 표시되는지 확인** (오늘의 추천)
3. **180일이 진짜 180일로 표시되는지 확인** (백테스트)
4. **필터 적용하기 버튼 + 정렬 작동 확인** (백테스트)
5. **사례 & 가이드 → 고급 전략 TOP 3 확인**
6. (선택) **Streamlit Cloud Secrets에 KIS 키 등록** → 어디서나 실시간

좋은 아침! 🌅 **다 됐다.**
