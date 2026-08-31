# H1 가설 전면 ML 벤치마크: 회귀 모델 6종 x GridSearchCV x 반복 K-Fold

**질문**: UV_type이 Target(결함수) 예측에 Lot_Num 대비 추가 정보를 주는가?
(08번 스크립트가 통계모델 4종+LightGBM 1종으로 검증한 것을, 대표적 회귀 ML 6종으로
전면 확장하고 각각 GridSearchCV로 최적화했다.)

**종속변수**: Target(연속형, 결함수) → **전부 회귀 모델**(분류 모델은 대상 아님).
**데이터**: 웨이퍼 단위 1,704행. **비교 대상**: FS1(Lot_Num만) vs FS2(Lot_Num+UV_type).

## 1. 모델 선정 및 하이퍼파라미터 그리드

| 모델 | 유형 | 그리드 |
|---|---|---|
| LinearRegression | 선형(폐형해) | 없음(튜닝 대상 하이퍼파라미터가 없는 베이스라인) |
| Ridge | 선형+L2 정규화 | alpha ∈ {0.1, 1, 10, 100} |
| RandomForest | 트리 앙상블(배깅) | n_estimators∈{100,300}, max_depth∈{None,5,10}, min_samples_leaf∈{1,5} |
| XGBoost | 트리 앙상블(부스팅) | n_estimators∈{100,300}, max_depth∈{3,5,7}, learning_rate∈{0.05,0.1} |
| LightGBM | 트리 앙상블(부스팅) | n_estimators∈{100,300}, num_leaves∈{15,31}, learning_rate∈{0.05,0.1} |
| SVR | 커널 기반 | C∈{0.1,1,10}, epsilon∈{0.1,1}, kernel∈{rbf,linear} |

전처리: Lot_Num/UV_type 원-핫 인코딩(drop='first'), Ridge·SVR는 추가로 StandardScaler
적용(선형/커널 모델은 스케일에 민감, 트리 모델은 불필요해서 생략).

## 2. 검증 방법론

1. **GridSearchCV**(cv=5, scoring=R²)로 모델별 최적 하이퍼파라미터 탐색.
2. 찾은 최적 하이퍼파라미터로 **RepeatedKFold(5-fold x 10회 반복 = 50 fold)** 재평가 →
   R²/MAE/RMSE 평균±표준편차 산출. (완전한 nested CV보다 가볍지만, 같은 데이터로 고른
   하이퍼파라미터로 반복평가하므로 점추정치가 아주 약간 낙관적일 수 있음을 명시.)
3. GroupKFold가 아니라 일반 KFold 사용 — Lot_Num을 "새 Lot 일반화"가 아니라 "이미
   관측된 Lot들의 평균을 얼마나 잘 추정하는가"를 보는 피처로 쓰기 때문(다른 질문).

## 3. 결과 — 모델 x 피처셋별 성능 비교

| 모델 | 피처셋 | 최적 하이퍼파라미터 | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| LinearRegression | Lot만 | — | 0.1614±0.0815 | 40.57 | 58.67 |
| LinearRegression | Lot+UV | — | 0.1613±0.0807 | 40.63 | 58.67 |
| Ridge | Lot만 | alpha=100 | 0.1640±0.0752 | 40.48 | 58.60 |
| Ridge | Lot+UV | alpha=100 | 0.1638±0.0748 | 40.52 | 58.61 |
| RandomForest | Lot만 | depth=5, leaf=5, n=100 | 0.1556±0.0794 | 40.88 | 58.87 |
| RandomForest | Lot+UV | depth=5, leaf=5, n=100 | 0.1553±0.0794 | 40.89 | 58.88 |
| XGBoost | Lot만 | lr=0.05, depth=3, n=100 | 0.1642±0.0729 | 40.54 | 58.60 |
| XGBoost | Lot+UV | lr=0.05, depth=3, n=100 | 0.1631±0.0733 | 40.57 | 58.64 |
| **LightGBM** | Lot만 | lr=0.05, n=100, leaves=15 | 0.1621±0.0805 | 40.52 | 58.65 |
| **LightGBM** | **Lot+UV** | lr=0.05, n=100, leaves=15 | **0.1649±0.0787** | 40.56 | **58.55** |
| SVR | Lot만 | C=0.1, eps=0.1, rbf | −0.0114±0.0154 | 42.68 | 64.64 |
| SVR | Lot+UV | C=0.1, eps=1.0, rbf | −0.0118±0.0156 | 42.68 | 64.65 |

**최고 성능 모델**: LightGBM(Lot+UV), R²=0.1649 — 다만 Ridge/XGBoost(Lot만)도 R²≈0.164로
사실상 동률(std 0.07~0.08 대비 차이가 훨씬 작음). **SVR은 전 구간에서 R²<0**(평균만
예측하는 것보다 못함) — 32개 범주를 가진 원-핫 피처에 RBF 커널이 적합하지 않은
것으로 보이며, 이는 SVR이라는 알고리즘이 이 피처 표현과 궁합이 안 맞는다는 뜻이지
UV_type 관련 결론에 영향을 주지 않는다.

## 4. UV_type 추가효과(ΔR² = FS2 − FS1)

| 모델 | ΔR² |
|---|---|
| LightGBM | **+0.0028** |
| LinearRegression | −0.0001 |
| Ridge | −0.0002 |
| RandomForest | −0.0003 |
| SVR | −0.0004 |
| XGBoost | −0.0011 |

6개 중 5개 모델은 ΔR²가 ±0.001 수준(사실상 0, fold-std 0.07~0.08에 비해 무시 가능).
**LightGBM만 유일하게 뚜렷한(+0.0028) 양의 델타**를 보여 별도로 조사했다.

### LightGBM 델타의 paired 유의성 검정

RepeatedKFold는 같은 seed면 FS1/FS2에 대해 동일한 fold 분할을 재현하므로, 50개 fold를
1:1로 짝지어 paired t-test/Wilcoxon을 적용할 수 있었다(unpaired 평균±표준편차만으로는
안 보이던 걸 잡아내려는 목적):

- paired t-test: t=3.806, **p=0.0004**
- Wilcoxon: **p=0.0003**
- 50개 fold 중 35개(70%)에서 FS2(Lot+UV)가 FS1(Lot만)보다 우수

**즉 LightGBM 한정으로는 통계적으로 유의한 개선이다.** 하지만 크기가 R² +0.28%p(상대
개선률 약 1.7%)에 불과해 — 통계적 유의성과 실질적 유의성이 갈리는 전형적 사례다.
표본이 1,704개라 paired 설계의 검정력이 매우 커서, "0.3%p짜리 아주 작은 진짜 신호"도
통계적으로는 검출된다. OLS/MixedLM의 UV_type 계수 검정(08번 리포트, p=0.39~0.94)은
선형 주효과만 보는 반면, LightGBM은 비선형/상호작용까지 잡아낼 수 있어 이 미세한
잔여신호를 감지했을 가능성이 있다.

## 5. H1 가설에 대한 최종 해석

- **6개 알고리즘 중 5개(Linear/Ridge/RF/SVR/XGBoost)는 UV_type을 추가해도 예측력이
  개선되지 않는다** — 08번 리포트의 통계모델 4종(OLS 고정효과 2종·혼합효과모형) 결론과
  완전히 일치.
- **LightGBM만 통계적으로 유의하지만 실무적으로는 무시할 수준의 잔여신호**
  (ΔR²=0.3%p)를 찾아낸다 — "완전히 0"은 아니지만 "의미있는 수율 레버"라고 부르기엔
  너무 작다.
- **종합 결론**: 10개 모델(통계 4 + ML 6) 중 9개가 "UV_type 효과 없음/무시 가능"에
  동의하고, 1개(LightGBM)만 통계적으로 검출 가능하지만 실무적으로 무의미한 수준의
  잔여신호를 보고한다. Executive Summary 시나리오 B("H-line→G-line 전환으로 수율
  개선")를 되살릴 근거는 없다 — 08번 리포트의 정정 결론을 그대로 유지한다.

## 재현

`python scripts/09_h1_full_ml_benchmark.py`
원시 결과: `reports/09_h1_full_ml_benchmark.csv`, `reports/09_h1_delta_r2_by_model.csv`
