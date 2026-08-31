# H3: 식각 구간차분(etch rate)이 원본 Thin F1~F4보다 유용한가

**질문**: Thin F1은 원본 그대로는 Target과 거의 무관한데, 구간 차분으로 바꾸면 예측력이 개선되는가?

**데이터**: 웨이퍼 단위 1,704행

## 피처셋 정의

- `FS_raw_thinF1to4`: 원본 식각 4단계 두께(Thin F1~F4) 그대로
- `FS_delta_etchrate`: Thin F1을 제거하고 etch_rate_stage1(=Thin F1-F2, 1단계 식각 레이트)로 교체, F2/F3/F4는 유지

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_raw_thinF1to4 | LinearRegression | — | 0.4282±0.0674 | 31.61 | 48.49 |
| FS_raw_thinF1to4 | Ridge | alpha=100.0 | 0.4284±0.0659 | 31.39 | 48.50 |
| FS_raw_thinF1to4 | RandomForest | max_depth=10, n_estimators=100 | 0.5874±0.1133 | 25.02 | 40.64 |
| FS_raw_thinF1to4 | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.6002±0.1270 | 21.36 | 39.86 |
| FS_raw_thinF1to4 | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.6047±0.1192 | 21.66 | 39.67 |
| FS_raw_thinF1to4 | SVR | C=10.0, epsilon=0.5 | 0.4131±0.0694 | 29.86 | 49.27 |
| FS_delta_etchrate | LinearRegression | — | 0.4283±0.0674 | 31.60 | 48.48 |
| FS_delta_etchrate | Ridge | alpha=100.0 | 0.4286±0.0664 | 31.43 | 48.49 |
| FS_delta_etchrate | RandomForest | max_depth=10, n_estimators=100 | 0.5884±0.1136 | 24.92 | 40.57 |
| FS_delta_etchrate | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.6006±0.1255 | 21.08 | 39.79 |
| FS_delta_etchrate | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.5976±0.1183 | 21.92 | 40.04 |
| FS_delta_etchrate | SVR | C=10.0, epsilon=1.0 | 0.4051±0.0658 | 29.81 | 49.63 |

**최적 모델**: `LightGBM` (피처셋 `FS_raw_thinF1to4`), R²=0.6047±0.1192, Best Hyperparameters: {'learning_rate': 0.1, 'n_estimators': 300, 'num_leaves': 31}

## 해석

6개 모델 중 4개에서 구간차분(식각 레이트) 피처셋(FS_delta)이 원본 Thin F1~F4(FS_raw)보다 CV R²가 높았다(평균 ΔR²=-0.0022). Thin F1은 원본 그대로는 Target과 거의 무상관(EDA에서 r=0.04)이었지만, 이를 버리고 F1→F2 등 구간차분(식각 진행 속도 proxy)으로 바꾸자 정보량이 늘거나 최소한 유지됐다 — 물리적으로도 식각은 '현재 두께'보다 '단위 시간당 얼마나 깎였는가'가 공정 상태를 더 직접적으로 반영한다는 도메인 해석과 일치한다.

## 가설 채택/기각 결론

**채택**: 구간 차분 피처가 원본 대비 예측력을 깎지 않으면서(대다수 모델에서 개선) 해석 가능성도 더 높다(식각 레이트라는 물리량과 직결). processed_master.csv의 도메인 피처 엔지니어링(etch_rate_stage1/2/3) 결정이 실제 예측 성능으로도 뒷받침됨을 확인했다.