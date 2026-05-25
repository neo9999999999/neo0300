"""
1) 무제한 vs 주 5건 한도 비교 (2022-2026)
2) ★ 강력매수에서 손절/하락한 종목 패턴 발굴
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path("cache")
ALLOC = 100_000

# 등급 포함 마스터 (28,492건) - 시뮬용
graded = pd.read_csv(CACHE / "MASTER_등급포함_시총300_2023-2026.csv")
graded["Date"] = pd.to_datetime(graded["Date"])

# ★ 강력매수만
strong = graded[graded["등급"] == "★ 강력매수"].copy()
print(f"★ 강력매수 전체: {len(strong):,}건")

# 일자별 + 슈퍼점수 정렬 (StrongScore_v2 ≈ SuperScore)
score_col = "StrongScore_v2" if "StrongScore_v2" in strong.columns else "StrongScore"
strong = strong.sort_values(["Date", score_col], ascending=[True, False])
strong["week"] = strong["Date"].dt.strftime("%Y-%U")
strong["Year"] = strong["Date"].dt.year


def simulate(picks, label):
    picks = picks.dropna(subset=["sell_close"]).copy()
    n = len(picks)
    if n == 0: return None
    invest = n * ALLOC
    profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
    return {
        "전략": label,
        "매수": n,
        "익절": int((picks["ret_180d"]>0).sum()),
        "손절(<=-20%)": int((picks["ret_180d"]<=-20).sum()),
        "보합~-20%": int(((picks["ret_180d"]>-20)&(picks["ret_180d"]<=10)).sum()),
        "SW": int((picks["peak_180d"]>=200).sum()),
        "100+": int((picks["peak_180d"]>=100).sum()),
        "50+": int((picks["peak_180d"]>=50).sum()),
        "SW%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "100+%": round((picks["peak_180d"]>=100).mean()*100, 1),
        "50+%": round((picks["peak_180d"]>=50).mean()*100, 1),
        "손절%": round((picks["ret_180d"]<=-20).mean()*100, 1),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }


# 1) 무제한 (★ 모두)
unlimited = strong.copy()

# 2) 주 5건 한도 (선착순)
def weekly_limit(df, limit):
    df = df.sort_values(["Date", score_col], ascending=[True, False])
    df["week"] = df["Date"].dt.strftime("%Y-%U")
    out = df.groupby("week").head(limit)
    return out

w5 = weekly_limit(strong, 5)
w7 = weekly_limit(strong, 7)
w10 = weekly_limit(strong, 10)
w3 = weekly_limit(strong, 3)

print("\n" + "="*100)
print("주 매수 한도별 비교 (★ 강력매수만, 선착순)")
print("="*100)

results = [simulate(d, lab) for d, lab in [
    (w3, "한도 3건/주"),
    (w5, "한도 5건/주"),
    (w7, "한도 7건/주"),
    (w10, "한도 10건/주"),
    (unlimited, "무제한 (★ 모두)"),
]]
res_df = pd.DataFrame([r for r in results if r])
print(res_df.to_string(index=False))

# 년도별 무제한
print("\n[무제한 — 년도별]")
for y, g in unlimited.groupby("Year"):
    n = len(g.dropna(subset=["sell_close"]))
    inv = n*ALLOC
    prof = ((g["sell_close"]/g["Close"]-1)*ALLOC).sum()
    sw = (g["peak_180d"]>=200).sum()
    loss = (g["ret_180d"]<=-20).sum()
    print(f"  {int(y)}: {n}건 / 투자 {inv/1e4:,.0f}만 → 수익 {prof/1e4:+,.0f}만 ({prof/inv*100:+.1f}%) | SW {sw} / 손절 {loss}")

# ============ 2) ★ 강력매수에서 하락 종목 패턴 ============
print("\n\n" + "="*100)
print("★ 강력매수에서 손절/하락 종목 패턴 분석")
print("="*100)

# 하락 분류
strong["하락"] = (strong["ret_180d"] <= -20).astype(int)
strong["대박실패"] = (strong["peak_180d"] < 10).astype(int)  # 10%도 못 감

n_total = len(strong)
n_loss = strong["하락"].sum()
n_bad = strong["대박실패"].sum()
print(f"\n★ 강력매수 전체: {n_total:,}건")
print(f"  손절 (-20%↓): {n_loss}건 ({n_loss/n_total*100:.1f}%)")
print(f"  대박실패 (peak<10%): {n_bad}건 ({n_bad/n_total*100:.1f}%)")

# 손절 종목 변수 평균 vs 비손절
loss_df = strong[strong["하락"]==1]
ok_df = strong[strong["하락"]==0]

print("\n[손절 vs 비손절 변수 차이]")
cols_check = [
    "Score", "Amount", "past_60", "past_120", "past_240",
    "slope60", "pos_252_high", "drawdown60",
    "p_sw", "p_100plus", "p_50plus", "p_loss",
    "StrongScore_v2",
]
for col in cols_check:
    if col not in strong.columns: continue
    loss_avg = loss_df[col].mean()
    ok_avg = ok_df[col].mean()
    diff = (loss_avg - ok_avg) / (abs(ok_avg)+1e-9) * 100
    print(f"  {col:18s}: 손절 평균 {loss_avg:>10.2f}  비손절 {ok_avg:>10.2f}  차이 {diff:>+6.1f}%")

# 손절 패턴 발굴 - cutoff 단변량
print("\n[손절 회피 룰 후보 (제외시 손절률 감소)]")
candidates = []
for col in cols_check:
    if col not in strong.columns: continue
    for q in [0.05, 0.10, 0.20, 0.80, 0.90, 0.95]:
        s = strong[col].dropna()
        if len(s) < 100: continue
        cutoff = s.quantile(q)
        if q <= 0.20:
            mask = strong[col] <= cutoff; op = "≤"
        else:
            mask = strong[col] >= cutoff; op = "≥"
        n_excl = mask.sum()
        if n_excl < 50 or n_excl > len(strong)*0.3: continue
        excl_loss_rate = strong[mask]["하락"].mean()*100
        rem_loss_rate = strong[~mask]["하락"].mean()*100
        rem_sw_rate = (strong[~mask]["peak_180d"]>=200).mean()*100
        sw_lost = (strong[mask]["peak_180d"]>=200).sum()
        if excl_loss_rate > rem_loss_rate + 5:
            candidates.append({
                "rule": f"{col} {op} {cutoff:.2f}",
                "n_excl": n_excl, "excl_loss%": round(excl_loss_rate,1),
                "rem_loss%": round(rem_loss_rate,1), "rem_sw%": round(rem_sw_rate,1),
                "sw_lost": sw_lost,
            })

if candidates:
    cdf = pd.DataFrame(candidates).sort_values("excl_loss%", ascending=False).head(15)
    print(cdf.to_string(index=False))

# 손절 종목 TOP 10 (peak 낮은 순)
print("\n[★ 강력매수인데 손절된 종목 TOP 10]")
worst = loss_df.sort_values("ret_180d").head(10)
show = worst[[c for c in ["Date","Code","Name","Market","Close","ret_180d","peak_180d","StrongScore_v2","p_sw","p_loss"] if c in worst.columns]]
print(show.to_string(index=False))

# 저장
res_df.to_csv(CACHE / "LIMIT_comparison.csv", index=False)
loss_df.to_csv(CACHE / "STRONG_BUY_LOSS_cases.csv", index=False)
print(f"\n[저장] cache/LIMIT_comparison.csv + STRONG_BUY_LOSS_cases.csv ({len(loss_df)}건)")
