# H9: 노광 효율 비율(cd_resolution_ratio, exposure_energy_per_cd_nm) 재검증

**질문**: Line_CD/Resolution 비율과 Energy_Exposure/Line_CD 비율이 원본 노광 공정변수 대비 추가 예측력을 주는가?

**데이터**: 웨이퍼 단위 1,704행, 노광 공정변수 4개 베이스라인

## 피처셋 정의

- `FS_no_ratio`: Line_CD, Wavelength, Resolution, Energy_Exposure (원본만)
- `FS_with_ratio`: 위 + cd_resolution_ratio + exposure_energy_per_cd_nm

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_no_ratio | LinearRegression | — | 0.0448±0.0310 | 43.26 | 62.61 |
| FS_no_ratio | Ridge | alpha=100.0 | 0.0452±0.0286 | 43.20 | 62.60 |
| FS_no_ratio | RandomForest | max_depth=10, n_estimators=300 | 0.3342±0.0830 | 36.26 | 52.11 |
| FS_no_ratio | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.3036±0.1122 | 35.29 | 53.16 |
| FS_no_ratio | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.3342±0.0853 | 35.95 | 52.00 |
| FS_no_ratio | SVR | C=10.0, epsilon=1.0 | 0.0465±0.0371 | 41.06 | 62.62 |
| FS_with_ratio | LinearRegression | — | 0.0476±0.0331 | 43.24 | 62.51 |
| FS_with_ratio | Ridge | alpha=1.0 | 0.0476±0.0321 | 43.21 | 62.52 |
| FS_with_ratio | RandomForest | max_depth=10, n_estimators=300 | 0.3272±0.0813 | 36.40 | 52.41 |
| FS_with_ratio | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.3034±0.1196 | 35.09 | 53.13 |
| FS_with_ratio | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.3388±0.0795 | 35.54 | 51.85 |
| FS_with_ratio | SVR | C=10.0, epsilon=1.0 | 0.0411±0.0377 | 41.21 | 62.80 |

**최적 모델**: `LightGBM` (피처셋 `FS_with_ratio`), R²=0.3388±0.0795, Best Hyperparameters: {'learning_rate': 0.1, 'n_estimators': 300, 'num_leaves': 31}

## 해석

6개 모델 중 3개에서 cd_resolution_ratio + exposure_energy_per_cd_nm 추가가 CV R²를 개선했다(평균 ΔR²=-0.0005).

## 가설 채택/기각 결론

**기각** (6개 중 3개 개선). 최종 피처셋에서 제외 유지.