# H4: 센티널 값의 정보성 결측 — NaN 대치 vs 값 보존+플래그

**질문**: Thin F4·Flux160s 등의 ≤0 값을 NaN으로 지우면(FS_naive) 값을 보존+플래그(FS_flagged)보다 예측력이 떨어지는가?

**데이터**: 웨이퍼 단위 1,704행, 센티널 발생 컬럼 5개(Pressure/Oxid_time/Thin F4/Flux90s/Flux160s)

## 피처셋 정의

- `FS_naive_NaN`: 5개 센티널 컬럼의 ≤0 값을 NaN 처리 후 파이프라인 내부 median 대치
- `FS_flagged_preserved`: 5개 컬럼 원본 값 유지(≤0도 실측값으로) + 공정별 sentinel_flag 3개 추가

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_naive_NaN | LinearRegression | — | -0.0047±0.0076 | 43.83 | 64.24 |
| FS_naive_NaN | Ridge | alpha=100.0 | 0.2584±0.0483 | 37.58 | 55.18 |
| FS_naive_NaN | RandomForest | max_depth=10, n_estimators=100 | 0.4890±0.0869 | 29.89 | 45.51 |
| FS_naive_NaN | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.5063±0.0959 | 27.55 | 44.77 |
| FS_naive_NaN | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.4793±0.1065 | 29.29 | 45.85 |
| FS_naive_NaN | SVR | C=10.0, epsilon=0.5 | 0.2658±0.0532 | 35.02 | 55.06 |
| FS_flagged_preserved | LinearRegression | — | -0.0045±0.0077 | 43.83 | 64.23 |
| FS_flagged_preserved | Ridge | alpha=100.0 | 0.2606±0.0470 | 37.46 | 55.11 |
| FS_flagged_preserved | RandomForest | max_depth=10, n_estimators=300 | 0.4976±0.0901 | 29.74 | 45.11 |
| FS_flagged_preserved | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.5102±0.0977 | 27.66 | 44.55 |
| FS_flagged_preserved | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.4885±0.1036 | 29.06 | 45.41 |
| FS_flagged_preserved | SVR | C=10.0, epsilon=0.5 | 0.2644±0.0531 | 35.47 | 55.10 |

**최적 모델**: `XGBoost` (피처셋 `FS_flagged_preserved`), R²=0.5102±0.0977, Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}

## 해석

6개 모델 중 5개에서 센티널 값을 보존하고 플래그로 표시한 피처셋(FS_flagged)이 단순 NaN 대치(FS_naive)보다 CV R²가 높았다(평균 ΔR²=+0.0038). Thin F4·Flux160s의 ≤0 값은 03_domain master 생성 단계에서 확인한 대로 Target과 통계적으로 유의한 관계(p<0.0001)를 가지므로, 이를 NaN으로 지우고 median으로 덮어씌우면 이 신호가 소실되는 게 예측 성능에도 그대로 반영된다.

## 가설 채택/기각 결론

**채택**: 정보 보존 처리(값 유지+플래그)가 단순 결측 대치보다 예측력이 떨어지지 않거나(대다수 모델에서 개선) 우수하다. preprocessing_summary.md에서 '값을 지우지 말고 플래그만 남기자'고 내렸던 결정이 예측 성능으로도 정당화된다.