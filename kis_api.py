"""
한국투자증권 OpenAPI (KIS Developers) — 클라우드에서도 작동.

[사용법]
1. https://apiportal.koreainvestment.com 가입
2. 모의/실전 앱키 발급 (APP_KEY + APP_SECRET)
3. Streamlit Cloud → Settings → Secrets 에 추가:
   KIS_APP_KEY = "..."
   KIS_APP_SECRET = "..."
   KIS_USE_MOCK = false  # true = 모의서버 (시세는 다소 차이)

[기능]
- get_market_cap_ranking(market): 시총 상위 종목 + 현재가/등락률/거래대금
- get_ohlcv(code, days): 일봉 데이터
- 토큰 자동 캐싱 (24h, hashlib id)
"""
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import requests


# Streamlit secrets 우선, 없으면 환경변수, 없으면 None
def _get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        return str(st.secrets.get(key, os.environ.get(key, default)))
    except Exception:
        return os.environ.get(key, default)


def _is_mock() -> bool:
    v = _get_secret("KIS_USE_MOCK", "false").lower()
    return v in ("true", "1", "yes")


def _base_url() -> str:
    return ("https://openapivts.koreainvestment.com:29443" if _is_mock()
            else "https://openapi.koreainvestment.com:9443")


TOKEN_CACHE_DIR = Path.home() / ".kis_token_cache"
TOKEN_CACHE_DIR.mkdir(exist_ok=True)


def _token_cache_path() -> Path:
    key = _get_secret("KIS_APP_KEY", "")
    if not key:
        return TOKEN_CACHE_DIR / "no_key.json"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    suffix = "_mock" if _is_mock() else "_real"
    return TOKEN_CACHE_DIR / f"token_{h}{suffix}.json"


def get_access_token() -> Optional[str]:
    """OAuth2 토큰. 24h 캐시. 키 없으면 None."""
    app_key = _get_secret("KIS_APP_KEY", "")
    app_secret = _get_secret("KIS_APP_SECRET", "")
    if not app_key or not app_secret:
        return None

    cache_path = _token_cache_path()
    # 캐시 확인
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            expires_at = pd.to_datetime(data.get("expires_at"))
            if expires_at > pd.Timestamp.now() + pd.Timedelta(minutes=10):
                return data["access_token"]
        except Exception:
            pass

    # 새 토큰 발급
    url = f"{_base_url()}/oauth2/tokenP"
    try:
        r = requests.post(url, json={
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        token = data["access_token"]
        expires_at = pd.Timestamp.now() + pd.Timedelta(seconds=int(data.get("expires_in", 86400)))
        with open(cache_path, "w") as f:
            json.dump({"access_token": token, "expires_at": str(expires_at)}, f)
        return token
    except Exception as e:
        print(f"KIS token error: {e}")
        return None


def _headers(tr_id: str) -> Dict[str, str]:
    token = get_access_token()
    return {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": _get_secret("KIS_APP_KEY", ""),
        "appsecret": _get_secret("KIS_APP_SECRET", ""),
        "tr_id": tr_id,
    }


def is_available() -> bool:
    """KIS API 키 설정 + 토큰 발급 OK 여부."""
    return get_access_token() is not None


# ============================================================================
# 시세 / 종목 정보
# ============================================================================

def get_current_price(code: str) -> Optional[Dict]:
    """단일 종목 현재가/등락률/거래대금."""
    tok = get_access_token()
    if not tok: return None
    url = f"{_base_url()}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = _headers("FHKST01010100")
    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = data.get("output", {})
        if not out: return None
        return {
            "Code": code,
            "Close": float(out.get("stck_prpr", 0)),
            "ChangeRatio": float(out.get("prdy_ctrt", 0)),
            "Amount": float(out.get("acml_tr_pbmn", 0)),
            "Volume": float(out.get("acml_vol", 0)),
            "MarketCap": float(out.get("hts_avls", 0)) * 100_000_000,  # 시총(억원→원)
            "Name": out.get("hts_kor_isnm", ""),
            "Market": "KOSPI" if out.get("rprs_mrkt_kor_name", "") == "KOSPI" else "KOSDAQ",
        }
    except Exception as e:
        return None


def get_volume_rank(market: str = "KOSDAQ", top: int = 30) -> pd.DataFrame:
    """거래량/거래대금 순위 조회. KIS API: '거래량순위' (FHPST01710000)."""
    tok = get_access_token()
    if not tok: return pd.DataFrame()
    url = f"{_base_url()}/uapi/domestic-stock/v1/quotations/volume-rank"
    headers = _headers("FHPST01710000")
    market_code = "0000" if market == "전체" else ("0001" if market == "KOSPI" else "1001")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20171",
        "FID_INPUT_ISCD": market_code,
        "FID_DIV_CLS_CODE": "0",
        "FID_BLNG_CLS_CODE": "1",  # 거래대금 순
        "FID_TRGT_CLS_CODE": "111111111",
        "FID_TRGT_EXLS_CLS_CODE": "000000",
        "FID_INPUT_PRICE_1": "0",
        "FID_INPUT_PRICE_2": "0",
        "FID_VOL_CNT": "0",
        "FID_INPUT_DATE_1": "",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = data.get("output", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        # 컬럼 표준화
        keep = pd.DataFrame({
            "Code": df.get("mksc_shrn_iscd", ""),
            "Name": df.get("hts_kor_isnm", ""),
            "Close": pd.to_numeric(df.get("stck_prpr", 0), errors="coerce"),
            "ChangeRatio": pd.to_numeric(df.get("prdy_ctrt", 0), errors="coerce"),
            "Amount": pd.to_numeric(df.get("acml_tr_pbmn", 0), errors="coerce"),
            "Volume": pd.to_numeric(df.get("acml_vol", 0), errors="coerce"),
        })
        keep["Market"] = market
        return keep.head(top).reset_index(drop=True)
    except Exception as e:
        print(f"KIS volume_rank error: {e}")
        return pd.DataFrame()


def get_change_rank(market: str = "KOSDAQ", top: int = 50,
                     direction: str = "up") -> pd.DataFrame:
    """등락률 순위 조회. 등락률 7~25% 종목 필터링에 유용."""
    tok = get_access_token()
    if not tok: return pd.DataFrame()
    url = f"{_base_url()}/uapi/domestic-stock/v1/ranking/fluctuation"
    headers = _headers("FHPST01700000")
    market_code = "0000" if market == "전체" else ("0001" if market == "KOSPI" else "1001")
    params = {
        "FID_RSFL_RATE2": "",
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20170",
        "FID_INPUT_ISCD": market_code,
        "FID_RANK_SORT_CLS_CODE": "0" if direction == "up" else "1",  # 0:상승, 1:하락
        "FID_INPUT_CNT_1": "0",
        "FID_PRC_CLS_CODE": "0",
        "FID_INPUT_PRICE_1": "",
        "FID_INPUT_PRICE_2": "",
        "FID_VOL_CNT": "",
        "FID_TRGT_CLS_CODE": "0",
        "FID_TRGT_EXLS_CLS_CODE": "0",
        "FID_DIV_CLS_CODE": "0",
        "FID_RSFL_RATE1": "",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = data.get("output", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        keep = pd.DataFrame({
            "Code": df.get("stck_shrn_iscd", df.get("mksc_shrn_iscd", "")),
            "Name": df.get("hts_kor_isnm", ""),
            "Close": pd.to_numeric(df.get("stck_prpr", 0), errors="coerce"),
            "ChangeRatio": pd.to_numeric(df.get("prdy_ctrt", 0), errors="coerce"),
            "Amount": pd.to_numeric(df.get("acml_tr_pbmn", 0), errors="coerce"),
            "Volume": pd.to_numeric(df.get("acml_vol", 0), errors="coerce"),
        })
        keep["Market"] = market
        return keep.head(top).reset_index(drop=True)
    except Exception as e:
        print(f"KIS change_rank error: {e}")
        return pd.DataFrame()


def get_ohlcv(code: str, days: int = 100) -> Optional[pd.DataFrame]:
    """일별 OHLCV 조회. period_div='D' (일봉)."""
    tok = get_access_token()
    if not tok: return None
    url = f"{_base_url()}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = _headers("FHKST03010100")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.5))  # 영업일 가산
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start_date.strftime("%Y%m%d"),
        "FID_INPUT_DATE_2": end_date.strftime("%Y%m%d"),
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",  # 0:수정주가
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = data.get("output2", [])
        if not rows: return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["stck_bsop_date"])
        df["Open"] = pd.to_numeric(df["stck_oprc"], errors="coerce")
        df["High"] = pd.to_numeric(df["stck_hgpr"], errors="coerce")
        df["Low"] = pd.to_numeric(df["stck_lwpr"], errors="coerce")
        df["Close"] = pd.to_numeric(df["stck_clpr"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["acml_vol"], errors="coerce")
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_values("Date").set_index("Date")
        return df.tail(days)
    except Exception as e:
        print(f"KIS ohlcv error ({code}): {e}")
        return None


# 빠른 테스트
if __name__ == "__main__":
    if not is_available():
        print("❌ KIS API 키 미설정")
        print("export KIS_APP_KEY='...'")
        print("export KIS_APP_SECRET='...'")
        exit(1)
    print(f"✅ 토큰: {get_access_token()[:30]}...")
    print("\n[삼성전자 현재가]")
    print(get_current_price("005930"))
    print("\n[코스닥 등락률 상위 5]")
    print(get_change_rank("KOSDAQ", top=5))
    print("\n[삼성전자 일봉 5일]")
    df = get_ohlcv("005930", days=5)
    print(df)
