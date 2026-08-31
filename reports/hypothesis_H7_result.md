# H7: type/Vapor는 Lot과 교란되지 않는다 — 예측 기여도 재확인

**질문**: type(dry/wet)·Vapor(H2O/O2)를 Lot_Num에 추가하면 예측력이 개선되는가?

**데이터**: 웨이퍼 단위 1,704행

## 피처셋 정의

- `FS1_Lot_only`: Lot_Num만(범주형)
- `FS2_Lot_plus_type_vapor`: Lot_Num + type + Vapor(범주형)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS1_Lot_only | LinearRegression | — | 0.1635±0.0773 | 40.50 | 58.42 |
| FS1_Lot_only | Ridge | alpha=100.0 | 0.1658±0.0707 | 40.43 | 58.37 |
| FS1_Lot_only | RandomForest | max_depth=10, n_estimators=300 | 0.1668±0.0728 | 40.46 | 58.33 |
| FS1_Lot_only | XGBoost | learning_rate=0.05, max_depth=5, n_estimators=100 | 0.1682±0.0710 | 40.37 | 58.29 |
| FS1_Lot_only | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=15 | 0.1643±0.0765 | 40.45 | 58.40 |
| FS1_Lot_only | SVR | C=10.0, epsilon=1.0 | 0.1267±0.0648 | 39.75 | 59.79 |
| FS2_Lot_plus_type_vapor | LinearRegression | — | 0.1652±0.0786 | 40.43 | 58.36 |
| FS2_Lot_plus_type_vapor | Ridge | alpha=100.0 | 0.1676±0.0723 | 40.37 | 58.31 |
| FS2_Lot_plus_type_vapor | RandomForest | max_depth=10, n_estimators=100 | 0.1694±0.0735 | 40.37 | 58.23 |
| FS2_Lot_plus_type_vapor | XGBoost | learning_rate=0.05, max_depth=5, n_estimators=100 | 0.1717±0.0715 | 40.24 | 58.17 |
| FS2_Lot_plus_type_vapor | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=15 | 0.1687±0.0788 | 40.27 | 58.25 |
| FS2_Lot_plus_type_vapor | SVR | C=10.0, epsilon=1.0 | 0.1312±0.0643 | 39.47 | 59.65 |

**최적 모델**: `XGBoost` (피처셋 `FS2_Lot_plus_type_vapor`), R²=0.1717±0.0715, Best Hyperparameters: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}

## 해석

6개 모델 중 6개에서 type/Vapor 추가가 CV R²를 개선했다(평균 ΔR²=+0.0031). H1(UV_type)과 같은 실험 설계를 썼지만, type/Vapor는 애초에 Lot과 교란되지 않는다는 게 EDA에서 이미 확인됐으므로 여기서 보는 델타는 '교란 제거 효과'가 아니라 'type/Vapor 자체의 순수 예측 기여도'다.

## 가설 채택/기각 결론

**참고용(가설 자체는 이미 EDA에서 채택됨 — 여기선 예측 기여도만 추가 확인)**: ΔR² 크기가 H1의 UV_type 델타와 비슷한 수준으로 작다 — type/Vapor는 Lot 교란과 무관하게 그 자체로도 Target에 대한 독립적 설명력이 크지 않은 변수로 보인다.