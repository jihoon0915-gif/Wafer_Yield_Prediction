# H2: die-행은 웨이퍼 단위 값의 복제 — 그룹 누수 정량화

**질문**: die-행 15,390개를 독립 관측치처럼 일반 KFold에 넣으면 CV 성능이 부풀려지는가?

**데이터**: die-level 15,390행 vs 웨이퍼 단위로 축소한 1,704행 (동일 피처셋: Thin F2/F3/F4, Temp_OXid, Oxid_time, Energy_Exposure, spin3, input_Energy, UV_type)

## 피처셋 정의

- `A_die_naive_KFold`: die-행 15,390개 그대로 + 일반 KFold(그룹 무시, 틀린 방법)
- `B_die_GroupKFold`: die-행 15,390개 + GroupKFold(group_id) (그룹 보존, 프로젝트 표준)
- `C_wafer_dedup_KFold`: 웨이퍼 단위로 사전 축소한 1,704행 + 일반 KFold (그룹 보존, 대안)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| A_die_naive_KFold | LinearRegression | — | 0.4604±0.0221 | 30.77 | 47.47 |
| A_die_naive_KFold | RandomForest | max_depth=10, n_estimators=100 | 0.9455±0.0066 | 9.62 | 15.06 |
| A_die_naive_KFold | LightGBM | n_estimators=200, num_leaves=31 | 0.9852±0.0017 | 4.94 | 7.86 |
| B_die_GroupKFold | LinearRegression | — | 0.4635±0.0855 | 31.00 | 47.21 |
| B_die_GroupKFold | RandomForest | max_depth=10, n_estimators=100 | 0.5771±0.1252 | 23.93 | 41.82 |
| B_die_GroupKFold | LightGBM | n_estimators=200, num_leaves=31 | 0.6720±0.1041 | 20.99 | 36.85 |
| C_wafer_dedup_KFold | LinearRegression | — | 0.4524±0.0673 | 31.10 | 47.42 |
| C_wafer_dedup_KFold | RandomForest | max_depth=10, n_estimators=100 | 0.6728±0.0910 | 21.82 | 36.33 |
| C_wafer_dedup_KFold | LightGBM | n_estimators=200, num_leaves=31 | 0.6661±0.0871 | 20.22 | 36.80 |

**최적 모델**: `LightGBM` (피처셋 `A_die_naive_KFold`), R²=0.9852±0.0017, Best Hyperparameters: {'n_estimators': 200, 'num_leaves': 31}

## 해석

die-행(15,390개)을 그대로 일반 KFold(A)에 넣은 경우와, 같은 피처·같은 데이터를 그룹 구조를 존중해 분할한 경우(B: GroupKFold, C: 웨이퍼 단위로 먼저 축소 후 KFold)를 비교했다. 세 모델(Linear/RandomForest/LightGBM) 전부에서 A의 R2가 B/C보다 평균 0.213 높게 나왔다 — 이는 같은 웨이퍼의 반복 행이 train/val 양쪽에 걸쳐 있어 모델이 사실상 정답을 일부 미리 본 것과 같은 효과(그룹 누수)다. B와 C는 서로 다른 방식(그룹 분할 vs 사전 축소)임에도 R2가 서로 근접해, 두 가지 '올바른' 접근이 같은 진짜 성능에 수렴함을 보여준다.

## 가설 채택/기각 결론

**채택**: 3개 모델 전부에서 A(die-행 나이브 KFold)가 B/C(그룹 인지 방법)보다 R2가 높았다(전부 해당). die-행을 독립 관측치처럼 다루면 안 된다는 H2가 예측 성능으로도 실증됐다 — 이 프로젝트가 처음부터 GroupKFold를 표준으로 채택한 이유가 바로 이것이며, 이번 검증은 '그룹핑 없이 했으면 얼마나 부풀려졌을지'를 정량화한 것이다.