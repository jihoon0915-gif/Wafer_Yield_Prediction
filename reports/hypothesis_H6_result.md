# H6: 시간 드리프트/격주 진동 가설의 예측 기여도

**질문**: 날짜 파생 피처(경과일수, 요일)를 추가하면 Target 예측력이 개선되는가?

**데이터**: 웨이퍼 단위 1,704행(모델링용, 대표 날짜 1개/웨이퍼). 단, die-행 전체 기준 실제 고유 날짜는 181개(2024-12-30~2025-06-29)이며, 같은 웨이퍼 내에서도 Datetime이 최대 6개월 차이날 수 있음(본문 참고).

## 피처셋 정의

- `FS_no_time`: 핵심 공정변수(Thin F2-4, Temp_OXid) + UV_type
- `FS_with_time`: 위 + 경과일수(days_since_start, 연속형) + 요일(day_of_week, 범주형)

## 모델 비교 결과

| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| FS_no_time | LinearRegression | — | 0.4393±0.0678 | 31.54 | 48.01 |
| FS_no_time | Ridge | alpha=100.0 | 0.4397±0.0661 | 31.34 | 48.01 |
| FS_no_time | RandomForest | max_depth=10, n_estimators=100 | 0.6526±0.0997 | 22.59 | 37.38 |
| FS_no_time | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.6375±0.1078 | 21.59 | 38.14 |
| FS_no_time | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=15 | 0.6110±0.0976 | 23.71 | 39.60 |
| FS_no_time | SVR | C=10.0, epsilon=1.0 | 0.4850±0.0814 | 27.65 | 46.21 |
| FS_with_time | LinearRegression | — | 0.4447±0.0662 | 31.51 | 47.74 |
| FS_with_time | Ridge | alpha=100.0 | 0.4456±0.0643 | 31.30 | 47.72 |
| FS_with_time | RandomForest | max_depth=10, n_estimators=100 | 0.6584±0.1001 | 22.20 | 37.04 |
| FS_with_time | XGBoost | learning_rate=0.1, max_depth=5, n_estimators=300 | 0.6431±0.1173 | 21.02 | 37.79 |
| FS_with_time | LightGBM | learning_rate=0.1, n_estimators=300, num_leaves=31 | 0.6239±0.1034 | 21.38 | 38.91 |
| FS_with_time | SVR | C=10.0, epsilon=0.5 | 0.4616±0.0625 | 29.08 | 47.17 |

**최적 모델**: `RandomForest` (피처셋 `FS_with_time`), R²=0.6584±0.1001, Best Hyperparameters: {'max_depth': 10, 'n_estimators': 100}

## 해석

6개 모델 중 5개에서 날짜 파생 피처 추가가 CV R²를 개선했다(평균 ΔR²=+0.0020, XGBoost는 오히려 -0.037로 하락). 이번 검증 과정에서 이전 EDA의 '고유 관측일 22개'라는 근거 자체가 웨이퍼 단위 사전축소로 인한 착시였음을 발견했다: 실제로는 Datetime이 그룹(웨이퍼) 내에서 전혀 상수가 아니라 같은 웨이퍼의 9개 die-행이 최대 6개월(2024-12-30~2025-06-29, 고유 181일)에 걸쳐 서로 다른 날짜를 갖는데도 Target은 9개 행 전부 동일하다. 즉 '데이터가 22개뿐이라 주기를 못 본다'가 아니라, 'Datetime 자체가 애초에 Target 변화와 짝지어지는 방식으로 존재하지 않는다'는 훨씬 강한 결론이다.

## 가설 채택/기각 결론

**기각**: 시간 피처 추가로 인한 예측력 개선이 미미하고 모델에 따라 부호도 엇갈린다(6개 중 XGBoost는 뚜렷한 하락). 더 결정적으로는, 같은 웨이퍼 내에서 Datetime이 최대 6개월 차이나는데 Target이 완전히 동일한 사례를 직접 확인해 '시간 정보와 Target이 애초에 연결되어 있지 않다'는 걸 데이터로 증명했다. sim_days_since_pm_* 시뮬레이션 피처가 전제한 '격주 진동' 가설은 이번 재검증 결과 더 확실하게 기각된다 — 해당 피처는 가설 테스트용 시뮬레이션일 뿐 확정된 패턴으로 오인하면 안 된다는 원래 문서화된 경고가 다시 한번 확인됐다.