"""최종 통합 예측 파이프라인(scripts/12_final_modeling_pipeline.py)에서만 쓰는
커스텀 transformer. joblib으로 저장한 파이프라인을 어디서든 다시 불러 쓰려면
클래스가 `__main__`이 아니라 실제 임포트 가능한 모듈 경로에 있어야 한다 —
처음에 scripts/12_...py 안(사실상 __main__)에 정의했다가 다른 프로세스에서
joblib.load()가 `AttributeError: Can't get attribute 'FeatureSelectorSlicer'
on <module '__main__'>`로 실패하는 걸 실측으로 확인하고 여기로 옮겼다.
"""

from __future__ import annotations

import numpy as np

# scripts/11_final_preprocessing_pipeline.py ~ 13_model_interpretation_shap.py가
# 각자 이 목록을 인라인으로 들고 있다(이미 검증되어 돌아가는 코드라 그대로 둠).
# app_api.py 등 새 코드는 여기서 가져다 쓴다 - 5번째 사본을 또 만들지 않기 위함.
#
# H8~H13(07단계에서 만들었지만 애초에 유의성 검증을 안 거쳤던 7개 파생변수 재검증)
# 결과 반영: thickness -> oxid_thickness_spec_gap 교체(H13, 성능 동률+해석력 우위),
# oxidation_rate_nm_per_min 추가(H8, 6개 모델 중 5개 개선). 나머지 5개
# (cd_resolution_ratio/exposure_energy_per_cd_nm/line_cd_band_gap/total_flux_60_480/
# anneal_temp_diff)는 기각되어 반영 안 함 -- 특히 total_flux_60_480과 anneal_temp_diff는
# 원본 성분(Flux60~480s, Furance_Temp/RTA_Temp)이 개별로 더 나은 성능을 보여 원본 유지.
NUMERIC_FEATURES = [
    "resist_target", "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake",
    "time_HMDS_bake", "spin1", "spin2", "spin3", "photoresist_bake",
    "temp_softbake", "time_softbake",
    "Line_CD", "Wavelength", "Resolution", "Energy_Exposure",
    "Thin F2", "Thin F3", "Thin F4", "Temp_Etching", "Source_Power", "Selectivity",
    "etch_rate_stage1",
    "Flux60s", "Flux90s", "Flux160s", "Flux480s", "input_Energy",
    "Temp_implantation", "Furance_Temp", "RTA_Temp",
    "Temp_OXid", "ppm", "Pressure", "Oxid_time", "oxid_thickness_spec_gap",
    "oxidation_rate_nm_per_min",
    "oxidation_sentinel_flag", "etching_sentinel_flag", "ion_implant_sentinel_flag",
]
CATEGORICAL_FEATURES = ["type", "Vapor"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# SHAP mean|SHAP| 상위 10개(reports/model_interpretation_report.md 1절, H8~H13 반영
# 후 13번 스크립트 재실행 결과) - UI 슬라이더로 노출할 "핵심 KPI". LightGBM의 RFECV가
# 이번엔 22개짜리 더 슬림한 서브셋을 골라 etch_rate_stage1/oxid_thickness_spec_gap/
# oxidation_rate_nm_per_min은 top10 밖으로 밀려남(모델 자체 성능은 오히려 개선됐음 -
# H8/H13가 "쓸모없다"는 뜻이 아니라 이 22개 조합 안에서 순위가 밀렸을 뿐).
TOP_KPI_FEATURES = [
    "Thin F2", "Thin F4", "Thin F3", "Temp_OXid", "input_Energy",
    "Source_Power", "ppm", "Resolution", "Temp_implantation", "Temp_Etching",
]


class FeatureSelectorSlicer:
    """RFECV가 고른 위치 기반 마스크로 선택된 컬럼만 남기는 경량 transformer."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.mask]
