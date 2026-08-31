# H1~H7 엔드투엔드 ML 검증 종합 요약

7개 가설 각각을 회귀 ML 6종(LinearRegression/Ridge/RandomForest/XGBoost/LightGBM/SVR,
H2는 대용량 die-행 특성상 3종) x GridSearchCV x 반복 K-Fold로 독립 검증했다.
개별 상세 리포트는 `hypothesis_H{1~7}_result.md`, 원시 결과는
`hypothesis_H{1~7}_results.csv`, 실행 스크립트는 `scripts/h{1~7}_*.py`,
전체를 순서대로 실행하는 오케스트레이터는 `scripts/10_run_all_hypotheses_h1_h7.py`.

## 가설별 최적 모델 및 결론

| 가설 | 질문 | 최적 모델 | R² | Best Hyperparameters | 채택/기각 |
|---|---|---|---|---|---|
| H1 | UV_type이 Lot_Num 대비 추가 정보를 주는가 | LightGBM | 0.168±0.076 | lr=0.05, n=100, leaves=15 | **기각(6개 중 5개 무의미, LightGBM만 미세 신호)** |
| H2 | die-행을 나이브 KFold에 넣으면 성능이 부풀려지는가 | LightGBM | 0.985(부풀려진 값 — 그 자체가 증거) | n_estimators=200, num_leaves=31 | **채택** |
| H3 | 식각 구간차분이 원본 Thin F1보다 유용한가 | LightGBM | 0.605±0.119 | lr=0.1, n=300, leaves=31 | **채택(약하게)** |
| H4 | 센티널 정보보존이 단순 NaN대치보다 나은가 | XGBoost | 0.510±0.098 | lr=0.1, depth=5, n=300 | **채택** |
| H5 | 챔버 ID가 예측에 추가 정보를 주는가 | RandomForest | 0.662±0.098 | depth=10, n=100 | **기각(효과 미미)** |
| H6 | 시간 피처가 예측에 추가 정보를 주는가(격주진동) | RandomForest | 0.658±0.100 | depth=10, n=100 | **기각** |
| H7 | type/Vapor가 Lot 대비 추가 정보를 주는가 | XGBoost | 0.172±0.072 | lr=0.05, depth=5, n=100 | **참고용(작은 신호)** |

## 이 과정에서 발견·수정한 방법론 버그 2건

1. **범주형 전용 피처셋의 스케일링 누락**: 공유 프레임워크(`hypothesis_ml_lib.py`)가
   숫자형 가지에만 StandardScaler를 붙이고 원-핫 출력은 스케일하지 않아, 범주형만 있는
   피처셋(H7의 Lot_Num 단독)에서 Ridge가 alpha=100 정규화를 원-핫 0/1 더미에 그대로
   맞아 R²가 0.090까지 떨어지는 문제가 있었다. ColumnTransformer 전체 출력 뒤에
   StandardScaler를 붙이는 방식(H1과 동일)으로 수정 → Ridge R²가 0.166으로 정상화.
2. **GridSearchCV 내부 CV의 셔플 누락**: `cv=5`(정수)를 그대로 넘기면 셔플 없는
   KFold가 되는데, `processed_master.csv`는 Lot_Num 기준으로 사실상 정렬돼 있어(첫
   20%와 마지막 20%가 겹치지 않는 Lot 집합) 셔플 없는 분할은 특정 Lot이 통째로
   한 fold에 몰리는 비대표적 분할이 됐다. 실측 확인 결과 이 상태의 GridSearchCV
   내부 점수는 R²가 음수로 나올 정도로 왜곡돼 있었다. `KFold(shuffle=True,
   random_state=42)`를 명시하는 것으로 수정 → 트리 모델들의 Best Hyperparameters가
   실제로 바뀌었고(예: XGBoost가 depth=3→5, n=100→300 선택) R²가 전반적으로
   0.02~0.09 상승했다.

두 버그 모두 최종 결론(채택/기각 방향)을 뒤집지는 않았지만, 표에 제시하는
"최적 하이퍼파라미터"와 R² 수치의 정확성에는 실질적 영향을 미쳤다.

**H1 통합 완료**: 위 두 버그를 모두 수정한 공유 프레임워크로 H1도 `h1_uvtype_lot_confound_ml.py`
로 재작성해 이 체계에 포함시켰다(기존 `scripts/09_h1_full_ml_benchmark.py`는 버그가 있던
버전으로, 이 스크립트가 대체한다 — 08번 리포트의 통계모델 4종+강건성 검증은 여전히
별도로 유효하며 이 스크립트는 ML 벤치마크 부분만 담당). 재검증 결과 6개 모델 중
5개는 ΔR²가 사실상 0(-0.0007~-0.0042)이고 LightGBM만 +0.0036의 미세한 양의 델타를
보여, 이전 09번 스크립트의 버그 있는 결과(LightGBM +0.0028, paired p=0.0004)와
방향·크기 모두 일관된다 — 버그 수정이 H1의 최종 결론을 바꾸지 않았음을 재확인했다.
