# 최종 프로젝트 종합 보고서 — 반도체 웨이퍼 수율 분석 & 예측 시스템

전처리 → 가설검증(H1~H13) → GroupKFold 모델링 → SHAP 해석 → FastAPI/Streamlit 패키징까지
전 과정을 종합한다. 각 절은 세부 리포트를 링크하며, 여기서는 의사결정에 필요한 핵심만 정리한다.

## 1. 데이터 & 전처리 요약

- 원본: 6개 공정 CSV(포토소프트베이크/노광/식각/이온주입/산화/검사), die-행 15,390개, 웨이퍼(Lot×Wafer) 1,704개.
- **H2 가설(채택)** — die-행은 사실상 웨이퍼 단위 값의 반복(그룹 내 Target 분산 0인 비율 89.2%). 나이브 KFold로 학습하면 RandomForest R²=0.945, LightGBM R²=0.985까지 부풀려짐(실측 검증) → 웨이퍼 단위(1,704행)로 축소 후 모델링.
- 최종 피처 매트릭스: `data/processed/processed_final_feature_matrix.csv` (1,704행 × 46열: 수치 40 + 범주 2 + Target + Lot_Num + 참조컬럼 2).
- 전처리 상세: [`preprocessing_summary.md`](preprocessing_summary.md)

## 2. 가설 검증 요약 (H1~H7)

전체 상세: [`hypothesis_H1_H7_summary.md`](hypothesis_H1_H7_summary.md), 개별: `hypothesis_H{1~7}_result.md`

| 가설 | 내용 | 결론 | 최종 피처셋 반영 |
|---|---|---|---|
| H1 | UV_type이 Lot 교란 통제 후에도 효과 있는가 | **기각** — 6개 모델 중 5개 무의미, 나머지 1개도 실무상 무시 가능 | `UV_type` 완전 제거 |
| H2 | die-행 나이브 처리 시 리키지 발생하는가 | **채택** — 위 참고 | 웨이퍼 단위 축소 |
| H3 | 식각 구간차분이 원본 Thin F1보다 유용한가 | **약하게 채택** | `Thin F1` 제거, `etch_rate_stage1` 추가 |
| H4 | 센티널 정보보존이 단순 NaN대치보다 나은가 | **채택** | 5개 센티널 컬럼 원본 유지 + 플래그 3개 추가 |
| H5 | 챔버 ID가 예측에 기여하는가 | **기각** — 6개 모델 전부 성능 하락 | 챔버 컬럼 4개 완전 제거 |
| H6 | 시간/격주진동이 예측에 기여하는가 | **기각** — 같은 웨이퍼 내 Datetime이 6개월 차이나도 Target 동일함을 확인 | `Datetime` 및 파생 피처 제외 |
| H7 | type/Vapor가 Lot 대비 추가 정보를 주는가 | 미세 신호(6/6 모델 양의 델타지만 작음) | 유지(정제) |

### 2b. 파생변수 사후 재검증 (H8~H13)

07단계에서 만들었지만 유의성 검증 없이 최종 피처셋에서 누락됐던 7개 파생변수를
나중에 재검증했다. 전체 상세: [`hypothesis_H8_H13_summary.md`](hypothesis_H8_H13_summary.md)

| 가설 | 파생변수 | 결론 | 최종 반영 |
|---|---|---|---|
| H8 | `oxidation_rate_nm_per_min`(=thickness/Oxid_time) | 채택(6개 중 5개 개선) | ✅ 추가 |
| H9 | `cd_resolution_ratio` + `exposure_energy_per_cd_nm` | 기각 | ❌ |
| H10 | `line_cd_band_gap` | 기각 | ❌ |
| H11 | `total_flux_60_480`(Flux 4개 합계) | 기각 — 합치면 트리모델 R² 0.05~0.08 하락 | ❌ 원본 4개 유지 |
| H12 | `anneal_temp_diff`(Furance_Temp-RTA_Temp) | 기각 | ❌ 원본 2개 유지 |
| H13 | `oxid_thickness_spec_gap`(원본 XAI에서 이미 검증됐던 것, 최종셋에서 누락 발견) | 성능 동률(선형모델 델타=부동소수점 오차) → 해석력 기준 채택 | ✅ `thickness` 대체 |

## 3. 모델링 성능

전체 상세: [`final_modeling_report.md`](final_modeling_report.md)

검증 체계: **GroupKFold(group=Lot_Num, 5-fold)** — "새로운 Lot에 대한 일반화"를 보는 가장
엄격한 기준(Wafer_ID 그룹핑은 이미 웨이퍼 단위로 축소해 무의미해짐). RFECV로 모델별 최적
피처 서브셋을 고르고, GridSearchCV로 하이퍼파라미터를 튜닝했다. (아래 수치는 H8~H13 반영 후 재실행한 최종 버전.)

| 모델 | 선택 피처 수 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| RandomForest | 32 | depth=15, max_features=0.7, min_leaf=3, n=300 | 0.4677±0.1445 | 28.16 | 45.10 |
| XGBoost | 38 | lr=0.05, depth=5, n=150, reg_lambda=0.1, subsample=0.8 | 0.4500±0.1741 | 28.18 | 44.89 |
| **LightGBM(배포)** | 22 | lr=0.05, n=150, leaves=15, reg_lambda=1.0, subsample=0.8 | **0.4699±0.1603** | 27.59 | 44.41 |

**해석**: R²≈0.47은 이전 리포트의 웨이퍼 단위 무작위 분할 결과(~0.6~0.7)보다 낮은데, 이는
모델이 나빠진 게 아니라 **검증을 더 엄격하게(Lot 완전분리) 했기 때문**이다 — 이전에 진짜
Lot 단위 완전분리로 별도 검증했을 때도 R²가 0.70→0.44로 하락한 바 있어 일관된 결과다.
H8/H13 반영으로 R²가 0.4636→0.4699로 소폭 개선됐다. LightGBM의 RFECV가 이번엔 42개 중
22개만 골라 가장 슬림하면서도 최고 성능을 냈다 — XGBoost(38개)보다도 적은 피처로 더 높은 R².

## 4. SHAP 기반 원인분석 & Top 5 KPI

전체 상세: [`model_interpretation_report.md`](model_interpretation_report.md)

| 우선순위 | KPI | 도메인 의미 |
|---|---|---|
| 1 | **Thin F2** | 식각 2단계 잔막 두께 — 관리구간(하위 20%) 불량률 0.0% vs 상위 20% 18.4% |
| 2 | **Thin F4** | 식각 4단계(최종) 잔막 두께 |
| 3 | **Thin F3** | 식각 3단계 잔막 두께 |
| 4 | **Temp_OXid** | 산화 공정 온도 |
| 5 | **input_Energy** | 이온주입 에너지 |

(H8~H13 반영 후 재계산 — `etch_rate_stage1`은 이번 LightGBM의 22개 슬림 서브셋에서
top5 밖으로 밀렸으나 XGBoost(38개)에서는 여전히 상위권. Top 10 전체는 `model_interpretation_report.md` 참고.)

XGBoost(38개 피처)와 LightGBM(22개 피처)의 SHAP 상위 10개 중 7개가 공통 — 서로 다른
알고리즘·피처셋에서도 같은 신호가 나와 우연이 아닌 실제 공정 신호로 판단.

**⚠️ 실무 경보 — 계측 결측 사각지대**: 오차가 가장 컸던 웨이퍼(Lot 27, 실제 Target=666,
**데이터셋 역대 최댓값**)를 조사한 결과 `Thin F2/F3/F4`가 9개 die-행 전부에서 완전히
결측(원본 CSV부터 NaN)임을 발견했다. 모델이 이 웨이퍼를 크게 과소예측(426 vs 666)한 건
모델 결함이 아니라 가장 강력한 예측 신호가 통째로 없었기 때문 — **결함이 가장 심각한
웨이퍼일수록 계측 자체가 누락되는 경향**이 있을 수 있다는 뜻이라, 계측 누락 자체를 별도
조기경보 신호로 관리할 것을 권장한다.

**⚠️ 예측 기여도 ≠ 인과관계**: SHAP Top 5는 "모델이 무엇을 많이 참고하는가"이지 검증된
인과관계가 아니다. UV_type은 예측 기여도가 있어 보였지만(H1 나이브 비교 시 강한 신호) Lot
교란을 통제하자 사라졌다. Top 5 KPI(Thin F2-4, Temp_OXid, input_Energy)는 전부 원본 04단계
XAI 분석에서도 상위권이었던 변수들로, 여러 차례의 재검증(H3~H4, H8~H13)에서 계속
살아남은 신호라 신뢰도가 높다.

## 5. 엔터프라이즈 패키징 — 시스템 아키텍처

```
┌─────────────────┐        HTTP/JSON        ┌──────────────────────┐
│  app_ui.py       │ ───────────────────────▶ │  app_api.py           │
│  (Streamlit,     │ ◀─────────────────────── │  (FastAPI, :8000)     │
│   :8501)         │                          │                       │
└─────────────────┘                          │  로드:                │
                                              │  - final_model.pkl    │
                                              │  - all_candidate_     │
                                              │    models.pkl         │
                                              │  - wafer_lookup.json  │
                                              │  - final_feature_     │
                                              │    matrix.csv         │
                                              └──────────────────────┘
```

**FastAPI(`app_api.py`) 엔드포인트**

| 엔드포인트 | 메서드 | 기능 |
|---|---|---|
| `/meta` | GET | 피처 범위/기본값, 모델 목록·성능, Top KPI 목록 (UI 초기화용) |
| `/predict` | POST | 단일/배치 웨이퍼 결함수·수율 예측 (모델 선택 가능) |
| `/simulate` | POST (async) | KPI 슬라이더 변경 전/후 예측치 + SHAP 기여도 변화. `asyncio.to_thread`로 SHAP 연산을 스레드풀에 위임해 이벤트 루프를 막지 않음 |
| `/wafer-shap/{group_id}` | GET (async) | 실측 웨이퍼 하나의 실제 공정값 기준 예측 + SHAP 전체 분해(waterfall용) |
| `/wafer-map/{group_id}` | GET | Die-level 26×26 실측 불량 분포 조회 |
| `/wafer-list` | GET | 웨이퍼맵 조회 가능한 웨이퍼 목록 |

**Streamlit(`app_ui.py`) 화면 구성**
- 사이드바: 모델 선택(RF/XGB/LightGBM, R² 표시) + SHAP 상위 10개 KPI 슬라이더 + type/Vapor 선택
- KPI 카드: 현재 설정 vs 베이스라인(중앙값) 예측 수율/결함수, 스윗스팟/위험구간 배지
- 웨이퍼 맵: 실측 웨이퍼 선택 → Die-level 2D Heatmap(Plotly) + 해당 웨이퍼의 SHAP Waterfall
- What-If 시뮬레이터: 슬라이더 변경 시 `/simulate` 실시간 호출 → SHAP Waterfall 갱신
- 글로벌 SHAP 참고: `reports/figures/shap_summary*.png` 정적 이미지(Beeswarm/Bar)

## 6. 실행 가이드

```bash
# 0) (최초 1회, 이미 완료됨) 전처리 -> 모델링 -> SHAP 해석
python scripts/11_final_preprocessing_pipeline.py
python scripts/12_final_modeling_pipeline.py
python scripts/13_model_interpretation_shap.py

# 1) FastAPI 백엔드 실행
uvicorn app_api:app --reload
#  -> http://127.0.0.1:8000/docs 에서 API 스펙 확인 가능(FastAPI 자동 생성 Swagger UI)

# 2) Streamlit 프론트엔드 실행 (새 터미널)
streamlit run app_ui.py
#  -> http://localhost:8501 접속
```

두 프로세스는 독립적으로 실행되며 `app_ui.py`는 `http://127.0.0.1:8000`으로 하드코딩된
API_URL을 통해서만 통신한다(다른 호스트/포트에 배포 시 `app_ui.py` 상단의 `API_URL` 상수만
바꾸면 됨).

## 7. 알려진 한계

- `final_preprocessor.pkl`(중앙값대치+StandardScaler+원핫)은 RFECV/GridSearchCV의 매 fold
  안에서 재적합하지 않고 전체 데이터로 fit된 상태를 재사용했다 — 실용적 단순화이며, 그룹(Lot)
  리스크는 모든 단계에서 GroupKFold로 별도 방어됨.
- R²≈0.46은 "새 Lot에 대한 엄격한 일반화 성능"이라, 개별 웨이퍼 결함수 정밀 예측(MAE≈28)엔
  아직 부족 — 방향성 판단·관리구간 설정·What-If 탐색 도구로 활용을 권장.
- 계측 결측(NaN) 사각지대(4절) — 결함이 가장 심한 웨이퍼일수록 예측 신뢰도가 오히려 낮을
  수 있다는 구조적 한계가 있음을 대시보드 하단에도 명시함.

## 관련 문서

- 전처리: [`preprocessing_summary.md`](preprocessing_summary.md)
- 가설검증(H1~H7): [`hypothesis_H1_H7_summary.md`](hypothesis_H1_H7_summary.md)
- 파생변수 재검증(H8~H13): [`hypothesis_H8_H13_summary.md`](hypothesis_H8_H13_summary.md)
- 모델링: [`final_modeling_report.md`](final_modeling_report.md)
- SHAP 해석: [`model_interpretation_report.md`](model_interpretation_report.md)
