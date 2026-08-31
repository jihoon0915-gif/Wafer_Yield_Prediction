# H13: 산화막 두께 vs 스펙갭(oxid_thickness_spec_gap) 재검증

**질문**: thickness를 700nm 스펙 기준 이탈도(spec_gap)로 표현해도 예측력이 유지·개선되는가?

**데이터**: 웨이퍼 단위 1,704행, 산화 공정변수 베이스라인. 원본 XAI(04단계)에서 spec_gap이 SHAP 10위였음.

## 피처셋 정의

- `FS_thickness`: Temp_OXid, ppm, Pressure, Oxid_time + thickness(원본)
- `FS_specgap`: Temp_OXid, ppm, Pressure, Oxid_time + oxid_thickness_spec_gap(=700-thickness)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_thickness | LinearRegression | — | 0.0210±0.0214 | 44.05 | 63.37 |
| FS_thickness | Ridge | alpha=100.0 | 0.0222±0.0193 | 43.95 | 63.34 |
| FS_thickness | RandomForest | max_depth=10, n_estimators=100 | 0.2216±0.1171 | 39.53 | 56.16 |
| FS_thickness | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=100 | 0.1929±0.0919 | 40.03 | 57.44 |
| FS_thickness | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=31 | 0.1783±0.0735 | 40.58 | 57.86 |
| FS_thickness | SVR | C=10.0, epsilon=1.0 | 0.0127±0.0298 | 41.83 | 63.65 |
| FS_specgap | LinearRegression | — | 0.0210±0.0214 | 44.05 | 63.37 |
| FS_specgap | Ridge | alpha=100.0 | 0.0222±0.0193 | 43.95 | 63.34 |
| FS_specgap | RandomForest | max_depth=10, n_estimators=100 | 0.2231±0.1198 | 39.52 | 56.09 |
| FS_specgap | XGBoost | learning_rate=0.05, max_depth=3, n_estimators=100 | 0.1768±0.0950 | 40.65 | 57.90 |
| FS_specgap | LightGBM | learning_rate=0.05, n_estimators=100, num_leaves=31 | 0.1842±0.0685 | 40.32 | 57.67 |
| FS_specgap | SVR | C=10.0, epsilon=1.0 | 0.0127±0.0298 | 41.83 | 63.65 |

**최적 모델**: `RandomForest` (피처셋 `FS_specgap`), R²=0.2231±0.1198, Best Hyperparameters: {'max_depth': 10, 'n_estimators': 100}

## 해석

기계적으로 세면 "6개 중 4개 개선"이지만 이 숫자는 오해의 소지가 있다 — 실제로는:

- **선형계열(Linear/Ridge/SVR)은 델타가 1e-16~1e-17 수준으로 부동소수점 오차 그 자체다.** thickness↔spec_gap은 부호만 반전된 완전한 affine 변환(spec_gap=700-thickness)이므로, 선형모델 입장에서 두 표현은 수학적으로 완전히 동일한 정보량이라는 가설이 **정확히 실증**됐다.
- **트리계열(RF/XGB/LightGBM)만 실질적 차이**가 있는데, 방향이 갈린다: LightGBM(+0.0059)·RandomForest(+0.0015)는 미세하게 개선, XGBoost는 오히려 **-0.0161로 더 크게 하락**. 평균을 내면 오히려 음수(-0.0014)라 "4개 개선"이 실제로는 순효과 없음/약한 열세에 가깝다.

## 가설 채택/기각 결론

**성능만 보면 우열 없음(사실상 동률) — 해석 가능성 근거로 spec_gap 채택.** thickness와 oxid_thickness_spec_gap을 둘 다 피처셋에 넣을 수는 없다(완전 공선성, H3/H11/H12와 같은 이유). 성능 차이가 노이즈 수준이므로, ①700nm 스펙 기준선 대비 이탈도라는 해석 가능성이 더 높고 ②원본 XAI(04단계)에서 이미 SHAP 10위로 검증된 전례가 있는 `oxid_thickness_spec_gap` 쪽을 최종 피처셋에 채택한다. "성능이 더 좋아서"가 아니라 "동률이라 해석력 기준으로 정했다"는 점을 명확히 한다.