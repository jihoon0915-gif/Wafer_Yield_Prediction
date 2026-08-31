# H12: 어닐링 온도차(anneal_temp_diff) 재검증

**질문**: Furance_Temp/RTA_Temp 2개를 온도차 1개로 대체해도 예측력이 유지·개선되는가?

**데이터**: 웨이퍼 단위 1,704행, 이온주입 공정변수 베이스라인

## 피처셋 정의

- `FS_raw_2temps`: Furance_Temp, RTA_Temp 원본 2개 + 나머지 이온주입 변수
- `FS_diff_1`: anneal_temp_diff(=Furance_Temp-RTA_Temp) 1개로 교체 + 나머지 이온주입 변수

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_raw_2temps | LinearRegression | — | -0.0065±0.0082 | 43.89 | 64.29 |
| FS_raw_2temps | Ridge | alpha=100.0 | 0.0347±0.0206 | 43.01 | 62.96 |
| FS_raw_2temps | RandomForest | max_depth=10, n_estimators=300 | 0.4418±0.0731 | 33.21 | 47.68 |
| FS_raw_2temps | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.5333±0.1081 | 25.34 | 43.35 |
| FS_raw_2temps | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.5498±0.0864 | 23.86 | 42.72 |
| FS_raw_2temps | SVR | C=10.0, epsilon=1.0 | 0.0163±0.0322 | 40.24 | 63.62 |
| FS_diff_1 | LinearRegression | — | -0.0065±0.0082 | 43.89 | 64.29 |
| FS_diff_1 | Ridge | alpha=100.0 | 0.0337±0.0195 | 43.11 | 63.00 |
| FS_diff_1 | RandomForest | max_depth=10, n_estimators=300 | 0.4339±0.0769 | 33.47 | 48.02 |
| FS_diff_1 | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.5370±0.1043 | 25.67 | 43.18 |
| FS_diff_1 | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.5389±0.0834 | 24.34 | 43.26 |
| FS_diff_1 | SVR | C=10.0, epsilon=1.0 | 0.0213±0.0314 | 39.94 | 63.47 |

**최적 모델**: `LightGBM` (피처셋 `FS_raw_2temps`), R²=0.5498±0.0864, Best Hyperparameters: {'learning_rate': 0.1, 'n_estimators': 300, 'num_leaves': 31}

## 해석

6개 모델 중 2개에서 Furance_Temp/RTA_Temp 2개를 anneal_temp_diff 1개로 교체했을 때 CV R²가 개선됐다(평균 ΔR²=-0.0019).

## 가설 채택/기각 결론

**기각** (6개 중 2개 개선). 원본 2개 유지.