# Master 데이터셋 전처리 요약 (도메인 기반)

3-Sigma/평균 등 통계적 이상치 탐지가 아니라, 공정 물리·화학 도메인 지식과 사내 변수정의서(pptx) 스펙만을 근거로 1차 전처리·피처 엔지니어링을 수행했다. 산출물: `data/processed/processed_master.csv` (15,390행, No_Die grain).

## 1. 병합(Master Data Integration)

- Key: `No_Die`(Lot_Num+Wafer_Num+공정순번을 이미 내포한 다이 식별자) 기준 1:1 inner join.
- 6개 CSV 전부 병합 전/후 15,390행 유지(불일치 시 즉시 예외 — 이번 실행에서도 통과).
- 병합 순서는 결과에 영향 없음(1:1 inner join)이라, 프롬프트 지정 순서 대신 이미 9개 assertion으로 회귀검증해 둔 기존 PROCESS_ORDER(Photo_softbake→Photo_lithograpy→Etching→Ion_Implantation→Oxidation→Inspect)를 그대로 사용.

**회귀 검증 (validate.run_all_checks) 결과**

| 검증 항목 | 통과 | 상세 |
|---|---|---|
| 병합 후 행 수 == 15,390 | ✅ | 실제 15390행 |
| No_Die 유일성 | ✅ | unique=True |
| error_class 분포가 기존 확인값과 일치 | ✅ | 일치 |
| Wafer_map 절단 행 수 ≈ 2619 | ✅ | 실제 2619행 (17.02%) |
| 파싱된 Wafer_map의 실제 다이 수 == 533 | ✅ | 위반 0건 / 파싱 성공 12771건 |
| group_id 고유 개수 == 1704 | ✅ | 실제 1704개 |
| 그룹 내 Target 표준편차 0 비율 ≈ 89.2% | ✅ | 실제 89.2% (허용오차 ±2.0%p) |
| 그룹 내 error_class 단일값 (100%) | ✅ | 클래스가 2종 이상 섞인 그룹 0개 |
| 그룹 내 Wafer_map 완전 동일 (비절단 기준) | ✅ | 비교 가능 그룹(비절단 2행 이상) 1413개 중 불일치 0개 |

## 2. 도메인 기반 이상치 정제 (Spike/Sentinel Filtering)

**원칙**: 온도·압력·시간·파워·두께·파장·에너지·flux 계열은 물리적으로 0 이하가 나올 수 없다(절대영도 미만 온도, 음수 압력·시간 등은 존재하지 않음). 이 컬럼들의 `<=0` 값은 통계적 이상치가 아니라 **센서/로깅 오류로 판정**해 공정별 플래그로 남겼다(값 자체는 보존, 대치는 하지 않음 — 대치는 다운스트림 분석에서 검증 방식에 맞게 수행해야 그룹 정보 누수를 피할 수 있음).

| 플래그 | 해당 행 수 | 비율 |
|---|---|---|
| `oxidation_sentinel_flag` | 399 | 2.59% |
| `photo_softbake_sentinel_flag` | 0 | 0.00% |
| `photo_litho_sentinel_flag` | 0 | 0.00% |
| `etching_sentinel_flag` | 56 | 0.36% |
| `ion_implant_sentinel_flag` | 220 | 1.43% |
| `any_process_sentinel_flag` | 648 | 4.21% |

**참고 — 이상치 플래그와 Target의 관계 (Welch t-test)**

| 플래그 | n | Target 평균(플래그=True) | Target 평균(그 외) | p-value |
|---|---|---|---|---|
| `oxidation_sentinel_flag` | 399 | 102.32 | 103.1 | 0.7978 |
| `etching_sentinel_flag` | 56 | 58.82 | 103.24 | 0.0 |
| `ion_implant_sentinel_flag` | 220 | 92.49 | 103.23 | 0.0001 |
| `any_process_sentinel_flag` | 648 | 97.06 | 103.35 | 0.0035 |

⚠️ `etching_sentinel_flag`처럼 p<0.05로 유의하게 나오는 경우, 해당 `<=0` 값이 단순 센서 오류가 아니라 '공정이 완전히 끝나 잔막이 사실상 0에 수렴'한 정상 상태를 반영했을 가능성도 있다 — NaN 대치 전에 반드시 도메인 확인이 필요하다(값을 지우지 않고 플래그만 남긴 이유).

## 3. 도메인 특화 Feature Engineering

### Oxidation
- `oxidation_rate_nm_per_min = thickness / Oxid_time` — 산화 성장속도(두께/시간).
- `oxid_thickness_spec_gap = 700.0 - thickness`, `oxid_below_spec`(0/1) — 변수정의서 스펙(700nm 이상) 미달 여부.

### Photo (softbake/litho)
- `cd_resolution_ratio = Line_CD / Resolution` — 목표 선폭 대비 계측 해상도 비율.
- `exposure_energy_per_cd_nm = Energy_Exposure / Line_CD` — CD 1nm당 투입 노광 에너지(낮을수록 적은 에너지로 동일 선폭을 달성한 것 → 효율적). 사내 표준 KPI가 아니라 이번 분석에서 정의한 공학적 비율이며, 해석 방향만 참고할 것.
- `line_cd_band_gap`, `line_cd_out_of_band`(0/1) — 스펙 구간(25.0~55.0nm) 이탈도.

### Etch
- `total_etch_removal = Thin F1 - Thin F4` — 전체 공정 구간 총 식각(제거)량.
- `etch_rate_stage1/2/3 = Thin F(n) - Thin F(n+1)` — 구간별 식각 레이트. 근거: 이번 프로젝트 EDA에서 Thin F1 자체는 Target과 상관 0.04(거의 무의미)였지만, F1→F2 구간차분은 상관 -0.39로 F1 단독보다 훨씬 강한 신호였음(구간 차분이 '식각 진행 속도'라는 물리량을 더 직접적으로 반영하기 때문으로 해석).

### Ion Implantation
- `total_flux_60_480 = Flux60s + Flux90s + Flux160s + Flux480s` — 누적 이온 주입량. **Flux840s는 합산에서 제외**: 15,381/15,390건(99.94%)이 전부 정확히 동일한 값(6e17, std=0)이고 나머지 9건만 결측인 **상수 컬럼**이라, 포함해도 모든 행에 동일 오프셋만 더해질 뿐 정보량이 없고 원본 파이프라인(`wafer.features.CONSTANT_DROP_COLS`)에서도 이미 같은 이유로 드롭 대상이다. (참고: 초기 버전 리포트에 '유효값 1건뿐(결측)'이라고 잘못 적었던 것을 재검증 후 정정 — 실제로는 '거의 결측'이 아니라 '거의 상수'가 배제 사유임.)
- `anneal_temp_diff = Furance_Temp - RTA_Temp` — 노(furnace) 어닐링과 RTA 온도차.

## 4. Chamber-to-Chamber Variation (1차 확인)

웨이퍼 단위(1,704장)로 집계해 챔버별 평균 Target 차이를 일원배치 ANOVA로 1차 스크리닝했다 (통계 검정이지만 '이상치 제거' 목적이 아니라 '분석 우선순위 스크리닝' 목적).

| 챔버 컬럼 | ANOVA p-value | 챔버별 평균 Target |
|---|---|---|
| `Etching_Chamber` | 0.2833 | 1=100.5, 2=102.4, 3=106.4 |
| `Ox_Chamber` | 0.4356 | 1=101.2, 2=102.4, 3=106.0 |
| `lithography_Chamber` | 0.2112 | 1=106.4, 2=103.1, 3=99.7 |
| `photo_soft_Chamber` | 0.0323 | 1=105.2, 2=106.6, 3=97.2 |

`photo_soft_Chamber`만 경계선 수준 유의(p≈0.03, 챔버3이 평균적으로 낮음) — 다만 4개 챔버 컬럼을 동시에 검정했으므로 다중비교 보정(Bonferroni 0.0125) 기준으로는 탈락. 추가 조사 후보로만 취급할 것. Chamber ID는 명목형(순서 없음)이므로 모델링 시 원-핫 인코딩을 권장하며, 순수 수치형(서열)으로 넣으면 신호가 희석될 수 있다.

## 5. 알려진 구조적 한계 (계속 유지)

- **die-행은 사실상 웨이퍼 단위 값의 복제**: 그룹(No_Die가 아닌 Lot_Num+Wafer_Num) 내 9개 행 기준으로 Thin F2/F3/F4/Oxid_time은 그룹 내 분산이 정확히 0%, 그 외 공정 변수들도 그룹 내 표준편차가 전체의 0.1~1% 수준(사실상 상수). 이 CSV의 '15,390행'은 물리적으로 독립적인 15,390개 관측치가 아니라 1,704개 웨이퍼가 반복 기록된 것이므로, 모델링 시 유효 표본수를 1,704 기준으로 판단해야 한다.
- **UV_type은 다수 Lot에 단일값으로 배정**: 32개 Lot 중 13개는 UV_type을 하나만 사용(Lot18~22=G, Lot24~27,25=H, Lot29~32=I). UV_type 효과를 분석할 때는 Lot 간 풀링 비교보다, 같은 Lot 안에서 UV_type이 섞인 19개 Lot만으로 Lot 고정효과를 제거한 비교가 필요하다.

## 6. 파일 스키마

- 행: 15,390 (No_Die grain, 6개 원본 CSV 1:1 병합)
- 열: 78개 (`Wafer_map` 원본 문자열 제외 — 이미 `data/processed/wafer_integrated.parquet`에 보존됨)
- 신규 도메인 파생 컬럼: `oxidation_rate_nm_per_min`, `cd_resolution_ratio`, `exposure_energy_per_cd_nm`, `total_etch_removal`, `etch_rate_stage1/2/3`, `total_flux_60_480`, `anneal_temp_diff`, `*_sentinel_flag`(6개), `any_process_sentinel_flag`