# 웨이퍼 수율 예측 시스템

6개 반도체 공정(포토소프트베이크·노광·식각·이온주입·산화·검사) 데이터를 통합해 웨이퍼
결함 수를 예측하고, 13개 가설을 통계·머신러닝으로 검증해 실제 수율에 기여하는 변수와
그렇지 않은 변수를 가려낸 프로젝트입니다. 검증된 결과를 FastAPI 백엔드 + Streamlit
대시보드로 패키징해 What-If 시뮬레이션과 SHAP 기반 원인분석을 제공합니다.

## 핵심 발견

- **데이터 누수 실증**: die 단위 반복행(15,390행)을 독립 관측치로 취급해 일반 K-Fold로
  학습하면 R²가 0.945~0.985까지 부풀려집니다. 원인은 반복행의 89.2%가 그룹 내 타깃
  분산 0(같은 웨이퍼가 여러 행에 중복 기록)이기 때문입니다. 웨이퍼 단위(1,704행)로
  축소하고 GroupKFold(Lot_Num)로 재검증한 결과가 실제 신뢰 가능한 성능(R²=0.4699)입니다.
- **가설 재검증으로 통계적 착시 제거**: 노광 파장(UV_type)이 결함수에 영향을 준다는
  초기 결과(H-line vs G/I-line 차이 +20.1%, p<0.0001)는 특정 Lot(불량률 61.1%)이
  UV_type을 단일값으로 쓰는 교란변수 때문이었습니다. 통계모델 4종 + ML 6종, 총 10개
  모델로 Lot을 완전히 통제해 재검증한 결과 유의한 모델은 0개였습니다.
- **정량 관리구간**: 식각 2단계 잔막 두께(Thin F2)가 하위 20% 구간(≤3,638nm)일 때
  불량률 0.0%, 상위 20% 구간(≥3,666nm)일 때 18.4%로 갈립니다. 이 관리구간을 적용한
  시뮬레이션에서 Chip Yield가 80.66% → 81.61%(+0.94%p)로 개선됨을 실측 데이터로
  확인했습니다.
- **계측 결측 사각지대**: 데이터셋 역대 최다 결함 웨이퍼(Target=666)를 SHAP으로
  역추적한 결과, 핵심 예측 변수(Thin F2/F3/F4)가 9개 측정 행 전부에서 완전히
  결측(NaN)이었습니다. 결함이 가장 심각한 웨이퍼일수록 계측 자체가 누락되는 구조적
  사각지대를 발견했고, 이를 결측 발생 자체를 조기경보 신호로 쓰는 프로세스로 제안했습니다.

## 방법론

1. **1차 벤치마크**: 회귀 5종(RF/XGBoost/LightGBM/CatBoost/MLP) + 분류 3종(SMOTE+RF/
   Cost-Sensitive LightGBM/TabNet)을 GroupKFold(웨이퍼 단위)로 비교, SHAP 원인분석,
   베이지안 최적화/유전 알고리즘/SLSQP로 공정 최적화 후보 탐색.
2. **자체 감사(Self-audit)**: 통계적 근거 재검증 과정에서 imputation 누수, 근거 부족한
   "챔피언" 모델 선정, 불공정한 하이퍼파라미터 튜닝, Lot 교란변수 등을 스스로 발견해
   수정.
3. **가설 검증 H1~H7**: 원본 데이터 기반 7개 가설(UV_type 교란, die 반복행 리키지,
   식각 구간차분, 결측 정보 보존, 챔버 효과, 시간 드리프트, type/Vapor)을 회귀 ML 6종
   × GridSearchCV × GroupKFold로 개별 검증.
4. **파생변수 재검증 H8~H13**: 앞서 유의성 검증 없이 만들었던 7개 파생변수를 사후
   검증해 최종 피처셋에 정식 반영(2개 채택, 5개 기각).
5. **최종 모델링**: RFECV 피처 선택 + GridSearchCV 튜닝을 GroupKFold(Lot_Num) —
   신규 Lot에 대한 가장 엄격한 일반화 기준 — 로 평가. LightGBM이 42개 중 22개 피처로
   최고 성능(R²=0.4699, MAE=27.59) 달성.
6. **SHAP 해석 및 서빙**: 배포 모델(LightGBM) + 비교 모델(XGBoost)의 SHAP 전역/의존성/
   로컬 설명을 FastAPI API로 제공하고, Streamlit 대시보드에서 웨이퍼맵 시각화·
   What-If 시뮬레이터·SHAP Waterfall을 실시간 연동.

## 폴더 구조

```
.
├── data/
│   ├── raw/                      # 원본 공정 CSV 6개
│   └── processed/                 # 병합·전처리 산출물 (parquet/csv)
├── src/wafer/                     # 공용 모듈
│   ├── config.py                  # 경로/스펙 상수
│   ├── io.py / integrate.py       # CSV 파싱, No_Die 기준 병합, 웨이퍼맵 복원
│   ├── validate.py                # 병합 결과 회귀 검증(9개 assertion)
│   ├── features.py                # 2단계 도메인 피처 엔지니어링
│   ├── modeling.py                # GroupKFold 기반 CV 러너, 전처리 파이프라인
│   └── final_pipeline.py          # 최종 피처 목록/Top KPI 상수, 커스텀 transformer
├── scripts/
│   ├── 02~06_*.py                 # 1차 벤치마크: 회귀/분류/XAI/최적화/시뮬레이션
│   ├── 07_domain_master_dataset.py    # 도메인 지식 기반 마스터 데이터셋 구축
│   ├── 08~09_h1_*.py              # UV_type 교란 심층 재검증(통계 4종+ML 6종)
│   ├── hypothesis_ml_lib.py       # 가설 검증 공용 프레임워크(GridSearchCV+GroupKFold)
│   ├── h1~h13_*.py                # 가설 H1~H13 개별 검증 스크립트
│   ├── 10_run_all_hypotheses_h1_h13.py  # 전체 가설 순차 실행 오케스트레이터
│   ├── 11_final_preprocessing_pipeline.py  # 가설 검증 결과 반영 최종 전처리
│   ├── 12_final_modeling_pipeline.py       # RFECV+GridSearchCV+GroupKFold(Lot_Num)
│   └── 13_model_interpretation_shap.py     # 최종 모델 SHAP 해석
├── models/
│   ├── export_dashboard_models.py # 웨이퍼맵 조회 테이블(wafer_lookup.json) 생성
│   ├── final_model.pkl            # 배포 파이프라인(전처리+피처선택+LightGBM)
│   ├── final_preprocessor.pkl     # 전처리기(대치+스케일+원핫)
│   ├── final_selected_features.pkl
│   ├── all_candidate_models.pkl   # RF/XGBoost/LightGBM 3종 전체 파이프라인+메타데이터
│   └── wafer_lookup.json          # 실측 웨이퍼맵 조회 테이블(대시보드용)
├── reports/                        # 단계별 분석 리포트(md/csv/png)
│   ├── FINAL_PROJECT_REPORT.md    # 종합 보고서
│   ├── PROCESS_OPTIMIZATION_STRATEGY.md  # 공정 최적화 전략 보고서
│   ├── hypothesis_H{1~13}_result.md      # 가설별 개별 리포트
│   ├── hypothesis_H1_H7_summary.md / hypothesis_H8_H13_summary.md
│   ├── final_modeling_report.md / model_interpretation_report.md
│   └── figures/                   # SHAP summary/dependence/waterfall 시각화
├── app_api.py                      # FastAPI 백엔드(예측/시뮬레이션/SHAP/웨이퍼맵 API)
├── app_ui.py                       # Streamlit 대시보드
└── requirements.txt
```

## 기술 스택

| 계층 | 기술 |
|---|---|
| 언어 | Python |
| 데이터 처리 | pandas, numpy, pyarrow |
| 모델링 | scikit-learn, XGBoost, LightGBM, CatBoost, TabNet |
| 하이퍼파라미터 튜닝 | Optuna, GridSearchCV |
| 불균형 처리 | imbalanced-learn(SMOTE) |
| 통계 검증 | scipy, statsmodels(OLS 고정효과, 혼합효과모형) |
| 모델 해석 | SHAP |
| API 서버 | FastAPI + Uvicorn (비동기 시뮬레이션 엔드포인트) |
| 대시보드 | Streamlit + Plotly |

## 시작하기

### 1. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 분석 파이프라인 재현 (선택 사항 — 산출물은 이미 `models/`, `reports/`에 포함)

```bash
# 가설 검증 H1~H13 전체 실행
python scripts/10_run_all_hypotheses_h1_h13.py

# 가설 검증 결과를 반영한 최종 전처리 -> 모델링 -> SHAP 해석
python scripts/11_final_preprocessing_pipeline.py
python scripts/12_final_modeling_pipeline.py
python scripts/13_model_interpretation_shap.py
```

### 3. 서버 실행

```bash
# 백엔드
uvicorn app_api:app --reload
#  -> http://127.0.0.1:8000/docs 에서 API 스펙 확인

# 프런트엔드 (새 터미널)
streamlit run app_ui.py
#  -> http://localhost:8501 접속
```

## 주요 기능

- **예측**: 단일/배치 웨이퍼의 공정 조건으로 결함 수·수율 예측(`POST /predict`)
- **What-If 시뮬레이션**: 공정 KPI 슬라이더 조정 시 예측치 변화 + SHAP 기여도 변화를
  실시간 반환(`POST /simulate`, 비동기)
- **웨이퍼맵 조회**: 실측 웨이퍼의 26×26 die-level 불량 분포 히트맵(`GET /wafer-map/{id}`)
- **로컬 해석**: 특정 웨이퍼의 예측 근거를 SHAP Waterfall로 분해(`GET /wafer-shap/{id}`)
- **모델 비교**: RandomForest/XGBoost/LightGBM 3개 후보 모델을 대시보드에서 선택 전환

## 라이선스

미정 (개인 프로젝트 — 추후 결정)
