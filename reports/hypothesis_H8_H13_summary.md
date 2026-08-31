# H8~H13 파생변수 재검증 종합 요약

07단계(도메인 마스터 데이터셋)에서 만들었지만 유의성 검증 없이 최종 모델(11~13단계)
피처셋 구성 때 누락됐던 7개 파생변수를 사후 검증했다. 방법론은 H1~H7과 동일한
`hypothesis_ml_lib.py` 프레임워크(회귀 6종 x GridSearchCV x GroupKFold 반복 검증).

**설계 원칙**: 파생변수가 이미 피처셋에 있는 원본 컬럼의 정확한 선형결합(합/차/affine
변환)이면 "추가"가 아니라 "교체" 방식으로 비교해야 한다 — H3에서 `total_etch_removal`을
그냥 추가했다가 무정규화 LinearRegression이 발산한 전례(R²=-51) 때문이다. 이번엔
`total_flux_60_480`(H11), `anneal_temp_diff`(H12), `oxid_thickness_spec_gap`(H13) 
3개가 이 위험군이라 전부 교체 방식으로 안전하게 설계했다.

## 가설별 결과

| 가설 | 파생변수 | 방식 | 개선 모델 수(6개 중) | 결론 | 최종 반영 |
|---|---|---|---|---|---|
| H8 | `oxidation_rate_nm_per_min`(=thickness/Oxid_time) | 추가 | 5/6 | **채택** | ✅ 추가 |
| H9 | `cd_resolution_ratio` + `exposure_energy_per_cd_nm` | 추가 | 3/6 | 기각 | ❌ |
| H10 | `line_cd_band_gap` | 추가 | 2/6 | 기각 | ❌ |
| H11 | `total_flux_60_480` (Flux60~480s 4개 대체) | 교체 | 2/6(원본 4개가 최대 0.08 R² 우세) | 기각 | ❌ 원본 유지 |
| H12 | `anneal_temp_diff` (Furance_Temp/RTA_Temp 대체) | 교체 | 2/6 | 기각 | ❌ 원본 유지 |
| H13 | `oxid_thickness_spec_gap` (thickness 대체) | 교체 | 사실상 동률(선형계열 델타=부동소수점 오차 수준) | 동률 → 해석력 기준 채택 | ✅ 교체 |

## 흥미로운 발견

- **H13에서 부동소수점 수준 델타로 affine 변환 등가성이 정확히 실증됐다**: 
  Linear/Ridge/SVR의 ΔR²가 1e-16~1e-17(사실상 0)로 나와, `oxid_thickness_spec_gap = 
  700 - thickness`가 선형모델 입장에서 완전히 동일한 정보량이라는 수학적 가설을 
  데이터로 직접 확인했다. 트리모델(RF/XGB/LightGBM)만 미세하게 갈렸다(XGBoost는 
  오히려 thickness 선호).
- **H11에서 "정보 압축"의 대가가 뚜렷하게 드러났다**: Flux60~480s 4개를 합계 1개로
  줄이면 트리모델 R²가 0.05~0.08이나 떨어진다 — 개별 성분이 서로 다른 패턴을
  담고 있어 합산 시 정보 손실이 크다는 뜻.

## 최종 반영 결과

최종 피처셋(11~13단계)에 **2개만 변경**: `thickness` → `oxid_thickness_spec_gap` 교체
(H13), `oxidation_rate_nm_per_min` 추가(H8). 수치형 피처 39개 → 40개.

재모델링 결과 LightGBM(배포 모델) GroupKFold(Lot_Num) R²가 **0.4636 → 0.4699**로
소폭 개선됐다(`reports/final_modeling_report.md` 참고). SHAP Top 10도 갱신됐다
(`reports/model_interpretation_report.md`) — 다만 LightGBM의 RFECV가 이번엔 22개
슬림 서브셋을 골라 H8/H13의 신규 변수 자체는 top10 밖으로 밀렸다(모델 성능은
개선됐지만, 그 개선을 이끈 게 정확히 이 두 변수라고 SHAP 순위만으로 단정할 수는
없다 — RFECV 피처선택 단계에서의 기여일 가능성이 더 크다).

## 재현

`python scripts/h8_oxidation_rate.py` ~ `h13_oxid_specgap_vs_thickness.py`
(각각 독립 실행 가능), 개별 리포트는 `hypothesis_H{8~13}_result.md`.
