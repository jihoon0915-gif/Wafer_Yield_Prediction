# 최종 통합 예측 모델링 보고서

processed_final_feature_matrix.csv(H1~H7 검증 반영 전처리 완료 데이터, 1,704행 웨이퍼 단위)를 대상으로, 모델별 RFECV 피처 선택 + GridSearchCV 하이퍼파라미터 튜닝을 GroupKFold(group=Lot_Num, 5-fold)로 평가했다.

**검증 체계**: group을 Wafer_ID가 아니라 Lot_Num으로 설정했다 — 전처리 단계에서 이미 die-행을 웨이퍼 단위로 축소해(H2 대응) Wafer_ID 그룹은 전부 크기 1이라 의미가 없고, 실제 남은 리스크는 '새 Lot에 대한 일반화'이기 때문이다(별도 검증에서 진짜 Lot 완전분리 시 R²가 0.70→0.44로 하락함을 이미 확인).

## 모델별 비교

| 모델 | 선택된 피처 수 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |
|---|---|---|---|---|---|
| RandomForest | 32 | max_depth=15, max_features=0.7, min_samples_leaf=3, n_estimators=300 | 0.4677±0.1445 | 28.16 | 45.10 |
| XGBoost | 38 | learning_rate=0.05, max_depth=5, n_estimators=150, reg_lambda=0.1, subsample=0.8 | 0.4500±0.1741 | 28.18 | 44.89 |
| LightGBM | 22 | learning_rate=0.05, n_estimators=150, num_leaves=15, reg_lambda=1.0, subsample=0.8 | 0.4699±0.1603 | 27.59 | 44.41 |

### RandomForest 선택 피처 목록 (32개)

`num__resist_target`, `num__N2_HMDS`, `num__temp_HMDS`, `num__temp_HMDS_bake`, `num__time_HMDS_bake`, `num__spin1`, `num__spin2`, `num__spin3`, `num__photoresist_bake`, `num__temp_softbake`, `num__Line_CD`, `num__Resolution`, `num__Energy_Exposure`, `num__Thin F2`, `num__Thin F3`, `num__Thin F4`, `num__Temp_Etching`, `num__Source_Power`, `num__etch_rate_stage1`, `num__Flux60s`, `num__Flux90s`, `num__Flux160s`, `num__input_Energy`, `num__Temp_implantation`, `num__Furance_Temp`, `num__RTA_Temp`, `num__Temp_OXid`, `num__ppm`, `num__Pressure`, `num__Oxid_time`, `num__oxid_thickness_spec_gap`, `num__oxidation_rate_nm_per_min`

### XGBoost 선택 피처 목록 (38개)

`num__resist_target`, `num__N2_HMDS`, `num__pressure_HMDS`, `num__temp_HMDS`, `num__temp_HMDS_bake`, `num__time_HMDS_bake`, `num__spin1`, `num__spin2`, `num__spin3`, `num__photoresist_bake`, `num__temp_softbake`, `num__time_softbake`, `num__Line_CD`, `num__Wavelength`, `num__Resolution`, `num__Energy_Exposure`, `num__Thin F2`, `num__Thin F3`, `num__Thin F4`, `num__Temp_Etching`, `num__Source_Power`, `num__Selectivity`, `num__etch_rate_stage1`, `num__Flux60s`, `num__Flux90s`, `num__Flux160s`, `num__Flux480s`, `num__input_Energy`, `num__Temp_implantation`, `num__Furance_Temp`, `num__RTA_Temp`, `num__Temp_OXid`, `num__ppm`, `num__Pressure`, `num__Oxid_time`, `num__oxid_thickness_spec_gap`, `num__oxidation_rate_nm_per_min`, `num__oxidation_sentinel_flag`

### LightGBM 선택 피처 목록 (22개)

`num__resist_target`, `num__N2_HMDS`, `num__temp_HMDS`, `num__spin3`, `num__photoresist_bake`, `num__temp_softbake`, `num__Line_CD`, `num__Resolution`, `num__Energy_Exposure`, `num__Thin F2`, `num__Thin F3`, `num__Thin F4`, `num__Temp_Etching`, `num__Source_Power`, `num__Selectivity`, `num__Flux60s`, `num__Flux90s`, `num__input_Energy`, `num__Temp_implantation`, `num__Temp_OXid`, `num__ppm`, `num__oxid_thickness_spec_gap`

## 최종 선정 모델

**LightGBM** (R²=0.4699±0.1603, MAE=27.59, RMSE=44.41) — GroupKFold(Lot_Num) 기준 CV R² 최고.
Best Hyperparameters: {'learning_rate': 0.05, 'n_estimators': 150, 'num_leaves': 15, 'reg_lambda': 1.0, 'subsample': 0.8}

## 한계 및 재현 시 주의

- `final_preprocessor.pkl`(중앙값대치+StandardScaler+원핫)은 전체 1,704행에 fit된 상태로 재사용했다 — RFECV/GridSearchCV의 매 fold 안에서 재적합하지 않는 실용적 단순화다. 그룹(Lot) 리스크는 모든 단계에서 GroupKFold로 방어되지만, 이 단순화로 인해 보고된 R²가 아주 약간 낙관적일 수 있다.
- 여기 보고된 R²는 Lot_Num 완전분리 기준이라, 이전 웨이퍼-단위 GroupKFold 결과(H1~H7 리포트, ~0.6~0.7)보다 낮게 나올 수 있다 — 더 엄격하고 실전에 가까운 추정치라 그렇다.