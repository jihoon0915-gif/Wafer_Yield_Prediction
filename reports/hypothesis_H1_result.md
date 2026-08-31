# H1: UV_type 효과는 Lot 교란을 통제하면 사라지는가

**질문**: UV_type을 Lot_Num에 추가하면 Target 예측력이 개선되는가 (Lot 교란 통제 후 UV_type의 독립 효과)?

**데이터**: 웨이퍼 단위 1,704행. (통계모델 4종 + 강건성 검증은 reports/08_h1_uvtype_model_comparison.md 참고 — 이 스크립트는 ML 변수중요도 벤치마크만 담당)

## 피처셋 정의

- `FS1_Lot_only`: Lot_Num만(범주형)
- `FS2_Lot_plus_UVtype`: Lot_Num + UV_type(범주형)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS1_Lot_only | LinearRegression | — | 0.1635±0.0773 | 40.50 | 58.42 |
| FS1_Lot_only | Ridge | alpha=100.0 | 0.1658±0.0707 | 40.43 | 58.37 |
| FS1_Lot_only | RandomForest | max_depth=10, n_estimators=300 | 0.1668±0.0728 | 40.46 | 58.33 |
| FS1_Lot_only | XGBoost | learning_rate=0.05, max_depth=5, n_estimators=100 | 0.1682±0.0710 | 40.37 | 58.29 |
| FS1_Lot_only | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=15 | 0.1643±0.0765 | 40.45 | 58.40 |
| FS1_Lot_only | SVR | C=10.0, epsilon=1.0 | 0.1267±0.0648 | 39.75 | 59.79 |
| FS2_Lot_plus_UVtype | LinearRegression | — | 0.1633±0.0766 | 40.55 | 58.43 |
| FS2_Lot_plus_UVtype | Ridge | alpha=100.0 | 0.1655±0.0708 | 40.46 | 58.38 |
| FS2_Lot_plus_UVtype | RandomForest | max_depth=10, n_estimators=300 | 0.1661±0.0735 | 40.50 | 58.35 |
| FS2_Lot_plus_UVtype | XGBoost | learning_rate=0.05, max_depth=3, n_estimators=100 | 0.1650±0.0679 | 40.54 | 58.41 |
| FS2_Lot_plus_UVtype | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=15 | 0.1678±0.0756 | 40.45 | 58.28 |
| FS2_Lot_plus_UVtype | SVR | C=10.0, epsilon=0.5 | 0.1225±0.0660 | 39.73 | 59.93 |

**최적 모델**: `XGBoost` (피처셋 `FS1_Lot_only`), R²=0.1682±0.0710, Best Hyperparameters: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}

## 해석

6개 모델 중 1개에서 UV_type 추가가 CV R²를 개선했다(평균 ΔR²=-0.0008). 08번 리포트의 통계모델 4종(나이브 OLS만 유의, Lot 통제 시 전부 비유의 p=0.39~0.94)과 같은 방향 — Lot_Num을 이미 아는 상태에서 UV_type이 주는 추가 정보는 크지 않다.

## 가설 채택/기각 결론

**기각(실무적으로 무의미)**: ΔR²가 전 모델에서 0.005 미만으로 작다. 08번 리포트의 통계적 유의성 검정(OLS 고정효과 2종 + 혼합효과모형, 전부 p>0.39)과 결론이 일치한다 — Executive Summary 시나리오 B(H-line→G-line 전환)를 되살릴 근거는 이번 재검증에서도 나오지 않았다.