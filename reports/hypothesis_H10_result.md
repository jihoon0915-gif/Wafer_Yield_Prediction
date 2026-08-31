# H10: Line_CD 스펙이탈도(line_cd_band_gap) 재검증

**질문**: Line_CD의 25~55nm 스펙 이탈도가 원본 노광 공정변수 대비 추가 예측력을 주는가?

**데이터**: 웨이퍼 단위 1,704행, 노광 공정변수 4개 베이스라인

## 피처셋 정의

- `FS_no_gap`: Line_CD, Wavelength, Resolution, Energy_Exposure (원본만)
- `FS_with_gap`: 위 + line_cd_band_gap(스펙 25~55nm 이탈도)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_no_gap | LinearRegression | — | 0.0448±0.0310 | 43.26 | 62.61 |
| FS_no_gap | Ridge | alpha=100.0 | 0.0452±0.0286 | 43.20 | 62.60 |
| FS_no_gap | RandomForest | max_depth=10, n_estimators=300 | 0.3342±0.0830 | 36.26 | 52.11 |
| FS_no_gap | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.3036±0.1122 | 35.29 | 53.16 |
| FS_no_gap | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.3342±0.0853 | 35.95 | 52.00 |
| FS_no_gap | SVR | C=10.0, epsilon=1.0 | 0.0465±0.0371 | 41.06 | 62.62 |
| FS_with_gap | LinearRegression | — | 0.0442±0.0317 | 43.28 | 62.63 |
| FS_with_gap | Ridge | alpha=100.0 | 0.0447±0.0292 | 43.23 | 62.62 |
| FS_with_gap | RandomForest | max_depth=10, n_estimators=300 | 0.3332±0.0834 | 36.27 | 52.15 |
| FS_with_gap | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.3054±0.1001 | 35.55 | 53.13 |
| FS_with_gap | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.3407±0.0824 | 35.64 | 51.76 |
| FS_with_gap | SVR | C=10.0, epsilon=1.0 | 0.0443±0.0378 | 41.08 | 62.70 |

**최적 모델**: `LightGBM` (피처셋 `FS_with_gap`), R²=0.3407±0.0824, Best Hyperparameters: {'learning_rate': 0.1, 'n_estimators': 300, 'num_leaves': 31}

## 해석

6개 모델 중 2개에서 line_cd_band_gap 추가가 CV R²를 개선했다(평균 ΔR²=+0.0007).

## 가설 채택/기각 결론

**기각** (6개 중 2개 개선). 최종 피처셋에서 제외 유지.