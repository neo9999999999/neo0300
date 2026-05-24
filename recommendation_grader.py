"""
종목 등급 분류기
==============
강력매수 / 추천 / 슈퍼위너후보 / 100+가능 / 50+가능 / 손절위험 자동 분류.
StrongScore 모델 출력 사용.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path("cache")


# 임계값 (walk-forward OOS 분석 기반)
THRESHOLDS = {
    "강력매수_StrongScore_pct": 0.80,    # 상위 20%
    "추천_StrongScore_pct": 0.60,         # 상위 40%
    "슈퍼위너후보_p_sw": 0.10,            # 절대 10%+
    "슈퍼위너강한_p_sw": 0.20,            # 절대 20%+
    "백퍼센트_p_100plus": 0.30,           # 30%+
    "오십퍼센트_p_50plus": 0.50,          # 50%+
    "손절위험_p_loss": 0.40,              # 40%+
    "손절고위험_p_loss": 0.55,            # 55%+
}


def assign_grades(df, score_col="StrongScore", scope="day"):
    """등급 부여 (df에 _grade, _tags 컬럼 추가)
    scope: 'day' (당일 내 상대) or 'week' (주 단위) or 'global' (전체)
    """
    if score_col in df.columns and len(df) > 0:
        if scope == "day":
            df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
        elif scope == "week":
            df["bucket"] = df["Date"].dt.strftime("%Y-%U")
        else:
            df["bucket"] = "ALL"

        # StrongScore 상대 분위 (당일/주 내)
        df["_score_pct"] = df.groupby("bucket")[score_col].rank(pct=True)
        df = df.drop(columns=["bucket"])
    else:
        df["_score_pct"] = 0.5

    # 등급 분류 (mutually exclusive)
    grades = []
    for _, r in df.iterrows():
        if r.get("p_loss", 0) >= THRESHOLDS["손절고위험_p_loss"]:
            grades.append("⚠️ 손절위험")
        elif r["_score_pct"] >= THRESHOLDS["강력매수_StrongScore_pct"]:
            grades.append("★ 강력매수")
        elif r["_score_pct"] >= THRESHOLDS["추천_StrongScore_pct"]:
            grades.append("○ 추천")
        else:
            grades.append("- 관망")
    df["등급"] = grades

    # 가능성 태그 (multiple)
    tags = []
    for _, r in df.iterrows():
        t = []
        if r.get("p_sw", 0) >= THRESHOLDS["슈퍼위너강한_p_sw"]:
            t.append("🏆 슈퍼위너 강력후보")
        elif r.get("p_sw", 0) >= THRESHOLDS["슈퍼위너후보_p_sw"]:
            t.append("⭐ 슈퍼위너후보")
        if r.get("p_100plus", 0) >= THRESHOLDS["백퍼센트_p_100plus"]:
            t.append("💯 100%+ 가능")
        if r.get("p_50plus", 0) >= THRESHOLDS["오십퍼센트_p_50plus"]:
            t.append("📈 50%+ 가능")
        if r.get("p_loss", 0) >= THRESHOLDS["손절위험_p_loss"] and r.get("p_loss", 0) < THRESHOLDS["손절고위험_p_loss"]:
            t.append("🔻 손절 주의")
        tags.append(" / ".join(t) if t else "")
    df["가능성태그"] = tags

    return df


def add_predicted_returns(df):
    """예상 수익률 카테고리"""
    cats = []
    for _, r in df.iterrows():
        peak_pred = r.get("peak_pred", 0)
        if peak_pred >= 100:
            cats.append(f"+{peak_pred:.0f}% (대박)")
        elif peak_pred >= 50:
            cats.append(f"+{peak_pred:.0f}% (대상승)")
        elif peak_pred >= 20:
            cats.append(f"+{peak_pred:.0f}% (상승)")
        elif peak_pred >= 0:
            cats.append(f"+{peak_pred:.0f}% (보합)")
        else:
            cats.append(f"{peak_pred:.0f}% (약세)")
    df["예상수익률"] = cats
    return df


if __name__ == "__main__":
    # 테스트
    test = pd.DataFrame({
        "Date": pd.to_datetime(["2025-08-22"]*4),
        "Code": ["A","B","C","D"],
        "StrongScore": [85, 60, 30, 10],
        "p_sw": [0.25, 0.15, 0.05, 0.02],
        "p_100plus": [0.45, 0.30, 0.15, 0.05],
        "p_50plus": [0.65, 0.55, 0.35, 0.15],
        "p_loss": [0.10, 0.20, 0.45, 0.60],
        "peak_pred": [120, 60, 25, -5],
    })
    test = assign_grades(test)
    test = add_predicted_returns(test)
    print(test[["Code","등급","가능성태그","예상수익률"]])
