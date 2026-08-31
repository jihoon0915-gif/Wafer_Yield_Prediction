# H8: 산화속도(oxidation_rate_nm_per_min) 파생변수 재검증

**질문**: thickness/Oxid_time 비율(산화속도)이 원본 산화 공정변수 대비 추가 예측력을 주는가?

**데이터**: 웨이퍼 단위 1,704행, 산화 공정변수 5개 베이스라인

## 피처셋 정의

- `FS_no_rate`: Temp_OXid, ppm, Pressure, Oxid_time, thickness (원본만)
- `FS_with_rate`: 위 + oxidation_rate_nm_per_min(=thickness/Oxid_time)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_no_rate | LinearRegression | — | 0.0210±0.0214 | 44.05 | 63.37 |
| FS_no_rate | Ridge | alpha=100.0 | 0.0222±0.0193 | 43.95 | 63.34 |
| FS_no_rate | RandomForest | max_depth=10, n_estimators=100 | 0.2216±0.1171 | 39.53 | 56.16 |
| FS_no_rate | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=100 | 0.1929±0.0919 | 40.03 | 57.44 |
| FS_no_rate | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=31 | 0.1783±0.0735 | 40.58 | 57.86 |
| FS_no_rate | SVR | C=10.0, epsilon=1.0 | 0.0127±0.0298 | 41.83 | 63.65 |
| FS_with_rate | LinearRegression | — | 0.0323±0.0266 | 43.74 | 63.00 |
| FS_with_rate | Ridge | alpha=100.0 | 0.0332±0.0224 | 43.66 | 62.98 |
| FS_with_rate | RandomForest | max_depth=10, n_estimators=300 | 0.2219±0.1169 | 39.53 | 56.13 |
| FS_with_rate | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=100 | 0.1859±0.0978 | 40.10 | 57.68 |
| FS_with_rate | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=31 | 0.1832±0.0736 | 40.39 | 57.70 |
| FS_with_rate | SVR | C=10.0, epsilon=1.0 | 0.0217±0.0297 | 41.50 | 63.36 |

**최적 모델**: `RandomForest` (피처셋 `FS_with_rate`), R²=0.2219±0.1169, Best Hyperparameters: {'max_depth': 10, 'n_estimators': 300}

## 해석

6개 모델 중 5개에서 oxidation_rate_nm_per_min 추가가 CV R²를 개선했다(평균 ΔR²=+0.0049). thickness/Oxid_time 원본이 이미 베이스라인에 있는 상태에서, 그 비율(성장속도)이 추가 정보를 주는지 확인한 것이다.

## 가설 채택/기각 결론

**채택** (6개 중 5개 개선). 최종 피처셋에 추가.