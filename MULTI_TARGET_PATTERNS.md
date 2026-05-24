# 다중 수익 타깃 OOS 패턴 분석

**peak_180d** = 매수일 종가 대비 180 영업일 동안의 최고 종가 수익률

- Train: 2020-04-03 .. 2023-12-31 (9652 rows)

- Test : 2024-01-01 .. 2025-08-22 (4416 rows)

- 통과 기준 (타깃별 적응형):

  - **hit_10** (peak ≥ +10%): Train hit ≥ 85% & lift ≥ 1.2, Test hit ≥ 80% & lift ≥ 1.1
  - **hit_20** (peak ≥ +20%): Train hit ≥ 75% & lift ≥ 1.3, Test hit ≥ 70% & lift ≥ 1.15
  - **hit_30** (peak ≥ +30%): Train hit ≥ 65% & lift ≥ 1.35, Test hit ≥ 60% & lift ≥ 1.2
  - **hit_50** (peak ≥ +50%): Train hit ≥ 50% & lift ≥ 1.5, Test hit ≥ 45% & lift ≥ 1.25
- Train n ≥ 80, Test n ≥ 25

- 시가총액(`marcap*`)은 매수일 종가 × 현재 발행주식수로 근사한 **매수 시점 추정치**임 (생존편향 완화).


## 기준 적중률 (base rates)

| Target | Whole | Train | Test |
|---|---|---|---|
| hit_10 (peak ≥ +10%) | 0.712 | 0.708 | 0.722 |
| hit_20 (peak ≥ +20%) | 0.581 | 0.574 | 0.595 |
| hit_30 (peak ≥ +30%) | 0.481 | 0.476 | 0.493 |
| hit_50 (peak ≥ +50%) | 0.331 | 0.320 | 0.354 |


## 🎯 +10% 도달 베스트 패턴 (OOS 통과: 33개)

| 패턴 | Type | Train n | Train hit | Train lift | Test n | Test hit | Test lift | Mean peak180 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cup_and_handle_detected AND past_60>=30 AND rs<=0.95 | combo3 | 89 | 0.865 | 1.22 | 51 | 0.980 | 1.36 | 63.50% |
| cup_and_handle_detected AND chart=persistent_uptrend AND rs<=0.95 | combo3 | 82 | 0.866 | 1.22 | 43 | 0.977 | 1.35 | 68.34% |
| new_high_240 AND KOSDAQ AND marcap200~500bn | combo3 | 179 | 0.855 | 1.21 | 35 | 0.943 | 1.31 | 82.55% |
| KOSDAQ AND chart=new_high_240 AND marcap200~500bn | combo3 | 179 | 0.855 | 1.21 | 35 | 0.943 | 1.31 | 82.55% |
| new_high_252 AND KOSDAQ AND marcap200~500bn | combo3 | 177 | 0.853 | 1.20 | 34 | 0.941 | 1.30 | 84.13% |
| past_60>=30 AND pos252_top10 AND rs<=0.95 | combo3 | 80 | 0.863 | 1.22 | 41 | 0.927 | 1.28 | 62.56% |
| past_120>=50 AND slope60>=1 AND marcap<=200bn | combo3 | 169 | 0.852 | 1.20 | 39 | 0.923 | 1.28 | 80.80% |
| s10>=70 AND KOSDAQ AND marcap<=200bn | combo3 | 107 | 0.860 | 1.21 | 56 | 0.911 | 1.26 | 183.28% |
| s8>=90 AND new_high_240 AND KOSDAQ | combo3 | 254 | 0.870 | 1.23 | 93 | 0.871 | 1.21 | 87.26% |
| s8>=90 AND KOSDAQ AND chart=new_high_240 | combo3 | 254 | 0.870 | 1.23 | 93 | 0.871 | 1.21 | 87.26% |
| s8>=90 AND new_high_252 AND KOSDAQ | combo3 | 252 | 0.869 | 1.23 | 92 | 0.870 | 1.20 | 87.89% |
| KOSDAQ AND chart=pullback_recovery AND marcap200~500bn | combo3 | 90 | 0.900 | 1.27 | 38 | 0.868 | 1.20 | 62.47% |
| KOSDAQ AND rs>=1.1 AND price_low<=5000won | combo3 | 89 | 0.854 | 1.21 | 38 | 0.868 | 1.20 | 273.88% |
| s8>=70 AND new_high_240 AND KOSDAQ | combo3 | 314 | 0.866 | 1.22 | 110 | 0.864 | 1.20 | 93.16% |
| s8>=70 AND KOSDAQ AND chart=new_high_240 | combo3 | 314 | 0.866 | 1.22 | 110 | 0.864 | 1.20 | 93.16% |

**핵심 비중복 패턴 (Top 8):**

| 패턴 | Train n / hit | Test n / hit | Test lift | Mean peak180 |
|---|---|---|---:|---:|
| `cup_and_handle_detected AND past_60>=30 AND rs<=0.95` | 89 / 86.5% | 51 / 98.0% | 1.36 | 63.5% |
| `cup_and_handle_detected AND chart=persistent_uptrend AND rs<=0.95` | 82 / 86.6% | 43 / 97.7% | 1.35 | 68.3% |
| `new_high_240 AND KOSDAQ AND marcap200~500bn` | 179 / 85.5% | 35 / 94.3% | 1.31 | 82.6% |
| `KOSDAQ AND chart=new_high_240 AND marcap200~500bn` | 179 / 85.5% | 35 / 94.3% | 1.31 | 82.6% |
| `new_high_252 AND KOSDAQ AND marcap200~500bn` | 177 / 85.3% | 34 / 94.1% | 1.30 | 84.1% |
| `past_60>=30 AND pos252_top10 AND rs<=0.95` | 80 / 86.2% | 41 / 92.7% | 1.28 | 62.6% |
| `past_120>=50 AND slope60>=1 AND marcap<=200bn` | 169 / 85.2% | 39 / 92.3% | 1.28 | 80.8% |
| `s10>=70 AND KOSDAQ AND marcap<=200bn` | 107 / 86.0% | 56 / 91.1% | 1.26 | 183.3% |

## 🎯 +20% 도달 베스트 패턴 (OOS 통과: 20개)

| 패턴 | Type | Train n | Train hit | Train lift | Test n | Test hit | Test lift | Mean peak180 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s10>=70 AND KOSDAQ AND marcap<=200bn | combo3 | 107 | 0.766 | 1.33 | 56 | 0.893 | 1.50 | 183.28% |
| KOSDAQ AND rs>=1.1 AND marcap<=200bn | combo3 | 116 | 0.750 | 1.31 | 65 | 0.892 | 1.50 | 172.88% |
| s2>=90 AND chart=V_recovery AND pos252_top10 | combo3 | 115 | 0.791 | 1.38 | 25 | 0.840 | 1.41 | 76.89% |
| s12>=80 AND KOSDAQ AND price_low<=5000won | combo3 | 151 | 0.775 | 1.35 | 30 | 0.833 | 1.40 | 93.64% |
| s12>=80 AND KOSDAQ AND marcap<=200bn | combo3 | 148 | 0.757 | 1.32 | 51 | 0.804 | 1.35 | 118.84% |
| s12>=80 AND past_60>=30 AND marcap<=200bn | combo3 | 105 | 0.771 | 1.34 | 35 | 0.771 | 1.30 | 112.76% |
| cup_and_handle_detected AND chart=persistent_uptrend AND rs<=0.95 | combo3 | 82 | 0.768 | 1.34 | 43 | 0.767 | 1.29 | 68.34% |
| chart=box_breakout AND marcap<=200bn AND price_low<=5000won | combo3 | 118 | 0.788 | 1.37 | 43 | 0.767 | 1.29 | 72.82% |
| KOSDAQ AND chart=pullback_recovery AND marcap200~500bn | combo3 | 90 | 0.778 | 1.35 | 38 | 0.763 | 1.28 | 62.47% |
| KOSDAQ AND rs>=1.1 AND price_low<=5000won | combo3 | 89 | 0.809 | 1.41 | 38 | 0.763 | 1.28 | 273.88% |
| s3>=90 AND s10>=90 AND KOSDAQ | combo3 | 232 | 0.754 | 1.31 | 153 | 0.752 | 1.26 | 107.05% |
| chart=new_high_120 AND price_low<=5000won | combo2 | 100 | 0.760 | 1.32 | 90 | 0.733 | 1.23 | 88.91% |
| s4>=75 AND chart=new_high_120 AND price_low<=5000won | combo3 | 100 | 0.760 | 1.32 | 90 | 0.733 | 1.23 | 88.91% |
| new_high_60 AND chart=new_high_120 AND price_low<=5000won | combo3 | 100 | 0.760 | 1.32 | 90 | 0.733 | 1.23 | 88.91% |
| new_high_120 AND chart=new_high_120 AND price_low<=5000won | combo3 | 100 | 0.760 | 1.32 | 90 | 0.733 | 1.23 | 88.91% |

**핵심 비중복 패턴 (Top 8):**

| 패턴 | Train n / hit | Test n / hit | Test lift | Mean peak180 |
|---|---|---|---:|---:|
| `s10>=70 AND KOSDAQ AND marcap<=200bn` | 107 / 76.6% | 56 / 89.3% | 1.50 | 183.3% |
| `KOSDAQ AND rs>=1.1 AND marcap<=200bn` | 116 / 75.0% | 65 / 89.2% | 1.50 | 172.9% |
| `s2>=90 AND chart=V_recovery AND pos252_top10` | 115 / 79.1% | 25 / 84.0% | 1.41 | 76.9% |
| `s12>=80 AND KOSDAQ AND price_low<=5000won` | 151 / 77.5% | 30 / 83.3% | 1.40 | 93.6% |
| `s12>=80 AND KOSDAQ AND marcap<=200bn` | 148 / 75.7% | 51 / 80.4% | 1.35 | 118.8% |
| `s12>=80 AND past_60>=30 AND marcap<=200bn` | 105 / 77.1% | 35 / 77.1% | 1.30 | 112.8% |
| `cup_and_handle_detected AND chart=persistent_uptrend AND rs<=0.95` | 82 / 76.8% | 43 / 76.7% | 1.29 | 68.3% |
| `chart=box_breakout AND marcap<=200bn AND price_low<=5000won` | 118 / 78.8% | 43 / 76.7% | 1.29 | 72.8% |

## 🎯 +30% 도달 베스트 패턴 (OOS 통과: 86개)

| 패턴 | Type | Train n | Train hit | Train lift | Test n | Test hit | Test lift | Mean peak180 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| KOSDAQ AND rs>=1.1 AND marcap<=200bn | combo3 | 116 | 0.698 | 1.47 | 65 | 0.877 | 1.78 | 172.88% |
| s10>=70 AND KOSDAQ AND marcap<=200bn | combo3 | 107 | 0.710 | 1.49 | 56 | 0.875 | 1.77 | 183.28% |
| s3>=70 AND chart=V_recovery AND pos252_top10 | combo3 | 121 | 0.678 | 1.42 | 50 | 0.840 | 1.70 | 134.68% |
| KOSDAQ AND past_60>=30 AND marcap<=200bn | combo3 | 124 | 0.677 | 1.42 | 37 | 0.838 | 1.70 | 218.64% |
| s2>=70 AND chart=V_recovery AND pos252_top10 | combo3 | 147 | 0.660 | 1.39 | 39 | 0.821 | 1.66 | 125.06% |
| s12>=80 AND KOSDAQ AND marcap<=200bn | combo3 | 148 | 0.703 | 1.48 | 51 | 0.784 | 1.59 | 118.84% |
| s12>=80 AND KOSDAQ AND price_low<=5000won | combo3 | 151 | 0.695 | 1.46 | 30 | 0.767 | 1.56 | 93.64% |
| s10>=90 AND KOSDAQ AND marcap200~500bn | combo3 | 167 | 0.671 | 1.41 | 94 | 0.755 | 1.53 | 134.02% |
| s8>=90 AND chart=V_recovery AND pos252_top10 | combo3 | 109 | 0.670 | 1.41 | 28 | 0.750 | 1.52 | 97.39% |
| new_high_252 AND KOSDAQ AND rs>=1.1 | combo3 | 258 | 0.655 | 1.38 | 103 | 0.748 | 1.52 | 95.11% |
| s12>=80 AND past_60>=30 AND marcap<=200bn | combo3 | 105 | 0.714 | 1.50 | 35 | 0.743 | 1.51 | 112.76% |
| s10>=70 AND new_high_252 AND KOSDAQ | combo3 | 224 | 0.661 | 1.39 | 97 | 0.742 | 1.51 | 96.17% |
| s10>=90 AND new_high_252 AND KOSDAQ | combo3 | 167 | 0.671 | 1.41 | 89 | 0.742 | 1.50 | 96.01% |
| new_high_240 AND KOSDAQ AND rs>=1.1 | combo3 | 258 | 0.655 | 1.38 | 104 | 0.740 | 1.50 | 94.48% |
| KOSDAQ AND chart=new_high_240 AND rs>=1.1 | combo3 | 258 | 0.655 | 1.38 | 104 | 0.740 | 1.50 | 94.48% |

**핵심 비중복 패턴 (Top 8):**

| 패턴 | Train n / hit | Test n / hit | Test lift | Mean peak180 |
|---|---|---|---:|---:|
| `KOSDAQ AND rs>=1.1 AND marcap<=200bn` | 116 / 69.8% | 65 / 87.7% | 1.78 | 172.9% |
| `s10>=70 AND KOSDAQ AND marcap<=200bn` | 107 / 71.0% | 56 / 87.5% | 1.77 | 183.3% |
| `s3>=70 AND chart=V_recovery AND pos252_top10` | 121 / 67.8% | 50 / 84.0% | 1.70 | 134.7% |
| `KOSDAQ AND past_60>=30 AND marcap<=200bn` | 124 / 67.7% | 37 / 83.8% | 1.70 | 218.6% |
| `s2>=70 AND chart=V_recovery AND pos252_top10` | 147 / 66.0% | 39 / 82.1% | 1.66 | 125.1% |
| `s12>=80 AND KOSDAQ AND marcap<=200bn` | 148 / 70.3% | 51 / 78.4% | 1.59 | 118.8% |
| `s12>=80 AND KOSDAQ AND price_low<=5000won` | 151 / 69.5% | 30 / 76.7% | 1.56 | 93.6% |
| `s10>=90 AND KOSDAQ AND marcap200~500bn` | 167 / 67.1% | 94 / 75.5% | 1.53 | 134.0% |

## 🎯 +50% 도달 베스트 패턴 (OOS 통과: 94개)

| 패턴 | Type | Train n | Train hit | Train lift | Test n | Test hit | Test lift | Mean peak180 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s12>=80 AND KOSDAQ AND price_low<=5000won | combo3 | 151 | 0.556 | 1.74 | 30 | 0.733 | 2.07 | 93.64% |
| KOSDAQ AND past_60>=30 AND marcap<=200bn | combo3 | 124 | 0.500 | 1.56 | 37 | 0.730 | 2.06 | 218.64% |
| KOSDAQ AND rs>=1.1 AND marcap<=200bn | combo3 | 116 | 0.509 | 1.59 | 65 | 0.723 | 2.04 | 172.88% |
| s5>=70 AND chart=V_recovery AND pos252_top10 | combo3 | 157 | 0.541 | 1.69 | 46 | 0.717 | 2.03 | 137.63% |
| s3>=70 AND chart=V_recovery AND pos252_top10 | combo3 | 121 | 0.554 | 1.73 | 50 | 0.700 | 1.98 | 134.68% |
| new_high_60 AND KOSDAQ AND price_low<=5000won | combo3 | 133 | 0.511 | 1.60 | 30 | 0.700 | 1.98 | 135.06% |
| s10>=70 AND KOSDAQ AND marcap<=200bn | combo3 | 107 | 0.505 | 1.58 | 56 | 0.696 | 1.97 | 183.28% |
| chart=V_recovery AND pos252_top10 AND turnover_hot>=5pct | combo3 | 167 | 0.527 | 1.65 | 48 | 0.688 | 1.94 | 138.60% |
| chart=V_recovery AND pos252_top10 | combo2 | 195 | 0.544 | 1.70 | 57 | 0.684 | 1.93 | 132.19% |
| s4>=75 AND chart=V_recovery AND pos252_top10 | combo3 | 195 | 0.544 | 1.70 | 57 | 0.684 | 1.93 | 132.19% |
| s10>=90 AND KOSDAQ AND marcap200~500bn | combo3 | 167 | 0.557 | 1.74 | 94 | 0.660 | 1.86 | 134.02% |
| chart=V_recovery AND pos252_top10 AND rs>=1.1 | combo3 | 137 | 0.555 | 1.73 | 45 | 0.644 | 1.82 | 128.71% |
| KOSDAQ AND past_120>=50 AND marcap200~500bn | combo3 | 183 | 0.503 | 1.57 | 66 | 0.636 | 1.80 | 148.06% |
| s12>=80 AND KOSDAQ AND marcap<=200bn | combo3 | 148 | 0.547 | 1.71 | 51 | 0.627 | 1.77 | 118.84% |
| s10>=70 AND chart=V_recovery AND pos252_top10 | combo3 | 124 | 0.540 | 1.69 | 42 | 0.619 | 1.75 | 125.25% |

**핵심 비중복 패턴 (Top 8):**

| 패턴 | Train n / hit | Test n / hit | Test lift | Mean peak180 |
|---|---|---|---:|---:|
| `s12>=80 AND KOSDAQ AND price_low<=5000won` | 151 / 55.6% | 30 / 73.3% | 2.07 | 93.6% |
| `KOSDAQ AND past_60>=30 AND marcap<=200bn` | 124 / 50.0% | 37 / 73.0% | 2.06 | 218.6% |
| `KOSDAQ AND rs>=1.1 AND marcap<=200bn` | 116 / 50.9% | 65 / 72.3% | 2.04 | 172.9% |
| `s5>=70 AND chart=V_recovery AND pos252_top10` | 157 / 54.1% | 46 / 71.7% | 2.03 | 137.6% |
| `s3>=70 AND chart=V_recovery AND pos252_top10` | 121 / 55.4% | 50 / 70.0% | 1.98 | 134.7% |
| `new_high_60 AND KOSDAQ AND price_low<=5000won` | 133 / 51.1% | 30 / 70.0% | 1.98 | 135.1% |
| `s10>=70 AND KOSDAQ AND marcap<=200bn` | 107 / 50.5% | 56 / 69.6% | 1.97 | 183.3% |
| `chart=V_recovery AND pos252_top10 AND turnover_hot>=5pct` | 167 / 52.7% | 48 / 68.8% | 1.94 | 138.6% |

## 📊 시가총액 효과 (매수 시점 추정치, 십억원)

매수일 종가 × 현재 발행주식수로 추정. 절대값은 부정확하지만 상대 순서는 신뢰 가능.


### +10% 도달 (peak 기준)

| Marcap bin | Split | N | Hit rate | Mean peak180 |
|---|---|---:|---:|---:|
| micro <= 100bn (1000억) | train | 589 | 0.767 | 67.7% |
| micro <= 100bn (1000억) | test | 257 | 0.685 | 45.4% |
| small 100~200bn | train | 1857 | 0.751 | 55.4% |
| small 100~200bn | test | 679 | 0.708 | 62.2% |
| small-mid 200~500bn | train | 3364 | 0.696 | 52.9% |
| small-mid 200~500bn | test | 1543 | 0.712 | 59.2% |
| mid 500~3000bn (3조) | train | 3171 | 0.700 | 47.9% |
| mid 500~3000bn (3조) | test | 1504 | 0.719 | 53.1% |
| large 3000~10000bn (10조) | train | 529 | 0.652 | 45.2% |
| large 3000~10000bn (10조) | test | 274 | 0.799 | 62.8% |
| mega > 10000bn | train | 142 | 0.563 | 31.7% |
| mega > 10000bn | test | 159 | 0.824 | 80.7% |

### +20% 도달 (peak 기준)

| Marcap bin | Split | N | Hit rate | Mean peak180 |
|---|---|---:|---:|---:|
| micro <= 100bn (1000억) | train | 589 | 0.642 | 67.7% |
| micro <= 100bn (1000억) | test | 257 | 0.490 | 45.4% |
| small 100~200bn | train | 1857 | 0.615 | 55.4% |
| small 100~200bn | test | 679 | 0.599 | 62.2% |
| small-mid 200~500bn | train | 3364 | 0.573 | 52.9% |
| small-mid 200~500bn | test | 1543 | 0.575 | 59.2% |
| mid 500~3000bn (3조) | train | 3171 | 0.559 | 47.9% |
| mid 500~3000bn (3조) | test | 1504 | 0.598 | 53.1% |
| large 3000~10000bn (10조) | train | 529 | 0.488 | 45.2% |
| large 3000~10000bn (10조) | test | 274 | 0.712 | 62.8% |
| mega > 10000bn | train | 142 | 0.444 | 31.7% |
| mega > 10000bn | test | 159 | 0.704 | 80.7% |

### +30% 도달 (peak 기준)

| Marcap bin | Split | N | Hit rate | Mean peak180 |
|---|---|---:|---:|---:|
| micro <= 100bn (1000억) | train | 589 | 0.587 | 67.7% |
| micro <= 100bn (1000억) | test | 257 | 0.455 | 45.4% |
| small 100~200bn | train | 1857 | 0.523 | 55.4% |
| small 100~200bn | test | 679 | 0.479 | 62.2% |
| small-mid 200~500bn | train | 3364 | 0.483 | 52.9% |
| small-mid 200~500bn | test | 1543 | 0.487 | 59.2% |
| mid 500~3000bn (3조) | train | 3171 | 0.443 | 47.9% |
| mid 500~3000bn (3조) | test | 1504 | 0.484 | 53.1% |
| large 3000~10000bn (10조) | train | 529 | 0.384 | 45.2% |
| large 3000~10000bn (10조) | test | 274 | 0.591 | 62.8% |
| mega > 10000bn | train | 142 | 0.303 | 31.7% |
| mega > 10000bn | test | 159 | 0.591 | 80.7% |

### +50% 도달 (peak 기준)

| Marcap bin | Split | N | Hit rate | Mean peak180 |
|---|---|---:|---:|---:|
| micro <= 100bn (1000억) | train | 589 | 0.448 | 67.7% |
| micro <= 100bn (1000억) | test | 257 | 0.288 | 45.4% |
| small 100~200bn | train | 1857 | 0.373 | 55.4% |
| small 100~200bn | test | 679 | 0.351 | 62.2% |
| small-mid 200~500bn | train | 3364 | 0.327 | 52.9% |
| small-mid 200~500bn | test | 1543 | 0.364 | 59.2% |
| mid 500~3000bn (3조) | train | 3171 | 0.270 | 47.9% |
| mid 500~3000bn (3조) | test | 1504 | 0.338 | 53.1% |
| large 3000~10000bn (10조) | train | 529 | 0.280 | 45.2% |
| large 3000~10000bn (10조) | test | 274 | 0.423 | 62.8% |
| mega > 10000bn | train | 142 | 0.204 | 31.7% |
| mega > 10000bn | test | 159 | 0.403 | 80.7% |

## 🏆 권장 실전 전략 (다양한 타깃별)


### 안전형 (+10%) — 작은 익절·높은 빈도
- **Base rate**: 테스트 구간에서 무작위로 매수 시 72.2% 적중
- **최고 적중률**: `cup_and_handle_detected AND past_60>=30 AND rs<=0.95`
  - Train: n=89, hit=86.5%
  - Test : n=51, hit=98.0% (base 대비 +25.8pp), mean peak180=63.5%
- **최대 빈도**: `s8>=90 AND KOSDAQ AND past_120>=50`
  - Train: n=194, hit=86.6%
  - Test : n=123, hit=82.9% (base 대비 +10.7pp), mean peak180=101.8%

### 균형형 (+20%) — 적당한 익절·높은 적중
- **Base rate**: 테스트 구간에서 무작위로 매수 시 59.5% 적중
- **최고 적중률**: `s10>=70 AND KOSDAQ AND marcap<=200bn`
  - Train: n=107, hit=76.6%
  - Test : n=56, hit=89.3% (base 대비 +29.8pp), mean peak180=183.3%
- **최대 빈도**: `s3>=90 AND KOSDAQ AND past_60>=30`
  - Train: n=238, hit=82.4%
  - Test : n=165, hit=70.9% (base 대비 +11.4pp), mean peak180=92.7%

### 추세형 (+30%) — 큰 익절·여전히 70%+
- **Base rate**: 테스트 구간에서 무작위로 매수 시 49.3% 적중
- **최고 적중률**: `KOSDAQ AND rs>=1.1 AND marcap<=200bn`
  - Train: n=116, hit=69.8%
  - Test : n=65, hit=87.7% (base 대비 +38.4pp), mean peak180=172.9%
- **최대 빈도**: `s3>=70 AND s10>=70 AND KOSDAQ`
  - Train: n=542, hit=66.1%
  - Test : n=360, hit=66.4% (base 대비 +17.1pp), mean peak180=102.1%

### 폭발형 (+50%+) — 큰 익절·신중한 진입
- **Base rate**: 테스트 구간에서 무작위로 매수 시 35.4% 적중
- **최고 적중률**: `s12>=80 AND KOSDAQ AND price_low<=5000won`
  - Train: n=151, hit=55.6%
  - Test : n=30, hit=73.3% (base 대비 +37.9pp), mean peak180=93.6%
- **최대 빈도**: `range120_pct >= 101.342`
  - Train: n=1447, hit=50.2%
  - Test : n=452, hit=47.1% (base 대비 +11.7pp), mean peak180=97.6%

## ⚠️ 해석 주의사항

- `peak_180d`는 180 영업일 내 종가 기준 최고 수익률. 실전에서는 해당 종가에 매도해야 하므로, 실시간 추적/익절 룰이 필수.

- chart_feats 데이터는 9개 프리셋(default/box_breakout/pullback 등) 시그널이 발생한 종목/일자만 포함. 따라서 '시그널 발생 종목 안에서 어떤 패턴이 추가 우위가 있는가'를 답함.

- `marcap*` 빈은 매수일 시점 추정치 (현재 발행주식수 × 매수일 종가). 공모/분할/증자 등으로 인한 오차가 있을 수 있음.

- PER/PBR/EPS 등 본격 펀더멘털은 pykrx KRX API 형식 변경으로 본 분석에서 미반영. 필요 시 별도 DART API 연동 필요.

- 수급(외국인/기관 순매수)은 KIS API 호출량 한계로 본 분석에서 미반영.
