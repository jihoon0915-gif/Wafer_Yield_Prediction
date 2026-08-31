# H5: 챔버(설비) ID가 Target 예측에 추가 정보를 주는가

**질문**: 4개 공정의 챔버 ID를 피처로 추가하면 예측력이 개선되는가 (설비 간 이질성 존재 여부)?

**데이터**: 웨이퍼 단위 1,704행

## 피처셋 정의

- `FS_no_chamber`: 핵심 공정변수(Thin F2-4, Temp_OXid, Oxid_time, Energy_Exposure) + UV_type
- `FS_with_chamber`: 위 + 4개 챔버 ID(Etching/Ox/lithography/photo_soft, 범주형)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_no_chamber | LinearRegression | — | 0.4392±0.0678 | 31.45 | 48.02 |
| FS_no_chamber | Ridge | alpha=100.0 | 0.4396±0.0662 | 31.24 | 48.02 |
| FS_no_chamber | RandomForest | max_depth=10, n_estimators=100 | 0.6616±0.0984 | 22.43 | 36.89 |
| FS_no_chamber | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.6490±0.1066 | 21.09 | 37.61 |
| FS_no_chamber | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.6348±0.0937 | 21.61 | 38.38 |
| FS_no_chamber | SVR | C=10.0, epsilon=1.0 | 0.4634±0.0773 | 28.10 | 47.18 |
| FS_with_chamber | LinearRegression | — | 0.4349±0.0692 | 31.64 | 48.19 |
| FS_with_chamber | Ridge | alpha=100.0 | 0.4360±0.0674 | 31.37 | 48.17 |
| FS_with_chamber | RandomForest | max_depth=10, n_estimators=100 | 0.6531±0.0935 | 22.93 | 37.42 |
| FS_with_chamber | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.6484±0.1085 | 21.87 | 37.57 |
| FS_with_chamber | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.6254±0.0961 | 22.81 | 38.87 |
| FS_with_chamber | SVR | C=10.0, epsilon=1.0 | 0.3940±0.0661 | 30.57 | 50.10 |

**최적 모델**: `RandomForest` (피처셋 `FS_no_chamber`), R²=0.6616±0.0984, Best Hyperparameters: {'max_depth': 10, 'n_estimators': 100}

## 해석

6개 모델 중 0개에서 챔버 ID 추가가 CV R²를 개선했다(평균 ΔR²=-0.0160). 이전 ANOVA 스크리닝에서 photo_soft_Chamber만 경계선 유의(p≈0.03, 다중비교 보정 시 탈락)였던 것과 일관되게, 챔버 ID의 예측 기여도는 있더라도 작다.

## 가설 채택/기각 결론

**약한 채택 또는 기각(모델에 따라 다름)**: 챔버 효과는 존재하더라도 미미한 수준이다. ANOVA(1차 스크리닝)와 ML 예측성능(2차 검증) 두 방법 모두 '강한 챔버 효과 없음, 약한 신호 가능성만 남음'이라는 같은 결론에 수렴한다. 챔버를 공정 최적화의 주요 레버로 쓰기엔 근거가 부족하다.