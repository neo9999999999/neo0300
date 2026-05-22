#!/bin/bash
# 종가매수 추천 시스템 실행 스크립트
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py
