# H11: 이온주입 누적 플럭스(total_flux_60_480) 재검증

**질문**: Flux60s~480s 4개 원본을 합계 1개로 대체해도 예측력이 유지되는가?

**데이터**: 웨이퍼 단위 1,704행, 이온주입 공정변수 베이스라인. Flux840s는 원본 파이프라인에서 이미 상수로 제외됨.

## 피처셋 정의

- `FS_raw_flux4`: Flux60s/90s/160s/480s 원본 4개 + 나머지 이온주입 변수
- `FS_agg_flux1`: total_flux_60_480(4개 합계) 1개로 교체 + 나머지 이온주입 변수

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_raw_flux4 | LinearRegression | — | -0.0065±0.0082 | 43.89 | 64.29 |
| FS_raw_flux4 | Ridge | alpha=100.0 | 0.0347±0.0206 | 43.01 | 62.96 |
| FS_raw_flux4 | RandomForest | max_depth=10, n_estimators=300 | 0.4418±0.0731 | 33.21 | 47.68 |
| FS_raw_flux4 | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.5333±0.1081 | 25.34 | 43.35 |
| FS_raw_flux4 | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.5498±0.0864 | 23.86 | 42.72 |
| FS_raw_flux4 | SVR | C=10.0, epsilon=1.0 | 0.0163±0.0322 | 40.24 | 63.62 |
| FS_agg_flux1 | LinearRegression | — | -0.0052±0.0049 | 43.84 | 64.26 |
| FS_agg_flux1 | Ridge | alpha=100.0 | 0.0353±0.0184 | 42.99 | 62.96 |
| FS_agg_flux1 | RandomForest | max_depth=10, n_estimators=300 | 0.3713±0.0866 | 35.24 | 50.66 |
| FS_agg_flux1 | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.4548±0.2099 | 28.55 | 46.43 |
| FS_agg_flux1 | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.5027±0.0841 | 27.79 | 45.00 |
| FS_agg_flux1 | SVR | C=1.0, epsilon=1.0 | 0.0010±0.0292 | 41.76 | 64.13 |

**최적 모델**: `LightGBM` (피처셋 `FS_raw_flux4`), R²=0.5498±0.0864, Best Hyperparameters: {'learning_rate': 0.1, 'n_estimators': 300, 'num_leaves': 31}

## 해석

6개 모델 중 2개에서 4개 원본 Flux를 합계 1개로 교체했을 때 CV R²가 개선됐다(평균 ΔR²=-0.0349). 2개 모델은 차이가 0.005 미만으로 사실상 동일 — 정보 손실 없이 피처 수를 4->1로 줄일 수 있는지가 관건이다.

## 가설 채택/기각 결론

**기각** (6개 중 2개 개선).