"""2단계: 전처리 + 외부 데이터/도메인 스펙 연계로 피처 공간을 확장하는 로직.

원본 소스 및 근거 (모두 명시적 출처 보유, 가상 생성 항목은 SYN_ 접두사로 구분):

1. 물리 스펙 보정 — 사내 과제 변수정의서
   `260320_청년 AI·BigData아카데미_과제수행 변수정의서(B반_반도체).pptx`
   (원본 위치: OneDrive/.../반도체데이터(2026-08-20)/)
   - 산화막 두께(thickness) 이론상 700nm 이상이어야 다음 공정 정상 진행 가능 (slide 2)
   - Line_CD 적정 구간 25~55nm (slide 4)
   이미 `wafer.config.OXIDATION_THICKNESS_MIN_NM`, `LITHOGRAPHY_LINE_CD_RANGE_NM`로
   프로젝트에 반영되어 있던 값을 그대로 재사용.

2. 외부 벤치마크 데이터셋 (비교/검증용, 행 단위 병합은 하지 않음 — 스키마 비호환)
   UCI SECOM Semiconductor Manufacturing Dataset
   http://archive.ics.uci.edu/ml/datasets/secom (Kaggle 미러: kaggle.com/datasets/paresh2047/uci-semcom)
   1,567건 × 590개 익명 센서 변수, Pass/Fail 라벨 1463:104 (불량 비율 6.6%, 불균형비 ≈14.1:1).
   본 프로젝트의 Error_message 불균형(정상 92.7% : 불량 7.3%, ≈12.7:1)과 같은 자릿수 —
   반도체 공정 데이터의 클래스 불균형이 이 데이터셋 고유의 특이값이 아니라 도메인 전반의
   공통 특성임을 뒷받침하는 근거로 사용.

3. 클린룸 환경 표준 범위 (가상 노이즈 피처의 분포를 현실적인 범위로 제한하기 위한 근거)
   ISO 14644 / SEMI 클린룸 표준 요약 (TSI, American Cleanroom Systems, 14644.dk 등 복수 출처
   교차 확인): 리소그래피 베이 온도 20~22℃(±1℃), 상대습도 30~50%RH.
   → 아래 SYN_ENV_* 피처의 정규분포 파라미터로 사용.

SYN_ 접두사가 붙은 컬럼은 원본 데이터에 없는 **가상 생성 피처**이며, 실제 설비 이력이
아니라 (2)에서 인용한 표준 범위로 캘리브레이션한 시뮬레이션이다. 모델링 결과 해석 시
반드시 이 피처들을 "실측 검증된 신호"가 아니라 "가설 테스트용 시뮬레이션"으로 구분해서
다뤄야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wafer.config import (
    LITHOGRAPHY_LINE_CD_RANGE_NM,
    OXIDATION_THICKNESS_MIN_NM,
    SENTINEL_CONVERT_CANDIDATE_COLUMNS,
)

CONSTANT_DROP_COLS = ["Flux840s"]  # 15,381/15,390건이 전부 동일값(6e17, std=0), 9건만 결측 — 상수라 정보량 없음
DUPLICATE_DROP_COLS = ["Chamber_Num"]  # Etching_Chamber와 15,390행 전부 100% 일치(중복 컬럼)

# ISO 14644 / SEMI 클린룸 표준 요약치 (리소그래피 베이 기준)
SYN_ENV_TEMP_MEAN_C = 21.0
SYN_ENV_TEMP_STD_C = 0.35  # 표준상 ±1℃ 이내를 대체로 포괄하도록 설정
SYN_ENV_HUMIDITY_MIN_PCT = 30.0
SYN_ENV_HUMIDITY_MAX_PCT = 50.0

# 기존 EDA에서 발견된 "주별 Target이 격주로 진동" 가설을 테스트하기 위한 가상 PM 주기
SYN_MAINTENANCE_CYCLE_DAYS = 14
SYN_RANDOM_SEED = 42


def drop_constant_and_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CONSTANT_DROP_COLS + DUPLICATE_DROP_COLS if c in df.columns]
    return df.drop(columns=cols)


def convert_physical_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """SENTINEL_CONVERT_CANDIDATE_COLUMNS(물리적으로 0/음수가 불가능한 온도·압력·
    시간·파워·두께·파장·에너지·flux 계열)의 <=0 값을 NaN으로 변환한다.
    실제 대치(imputation)는 GroupKFold train fold 내부에서만 수행해야 하므로 여기서는
    하지 않는다 (전역 대치 시 그룹 정보 누수 발생).
    """
    df = df.copy()
    cols = [c for c in SENTINEL_CONVERT_CANDIDATE_COLUMNS if c in df.columns]
    for col in cols:
        mask = df[col] <= 0
        df.loc[mask, col] = np.nan
    return df


def add_spec_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """변수정의서에 명시된 '적정/이론 스펙'과의 이탈도를 피처로 추가.

    thickness<700nm, Line_CD가 25~55nm 밖인 경우는 '물리적으로 불가능'한 값이
    아니라 '스펙을 벗어난 실제 관측값'이므로 결측 처리하지 않고 이탈도 자체를
    신호로 사용한다 (음수=규격 초과, 양수=규격 미달인 두 경우를 모두 대칭적으로 표현).
    """
    df = df.copy()
    df["oxid_thickness_spec_gap"] = OXIDATION_THICKNESS_MIN_NM - df["thickness"]
    df["oxid_below_spec"] = (df["oxid_thickness_spec_gap"] > 0).astype(int)

    lo, hi = LITHOGRAPHY_LINE_CD_RANGE_NM
    below = (lo - df["Line_CD"]).clip(lower=0)
    above = (df["Line_CD"] - hi).clip(lower=0)
    df["line_cd_band_gap"] = np.maximum(below, above)
    df["line_cd_out_of_band"] = (df["line_cd_band_gap"] > 0).astype(int)
    return df


def add_synthetic_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lot 단위로 가상 클린룸 온습도를 생성해 병합.

    같은 Lot는 동일 시점에 투입되므로 Lot 단위로 하나의 환경값을 부여한다(웨이퍼마다
    독립적으로 흔들리는 것은 비현실적). 시드 고정으로 재현 가능.
    """
    df = df.copy()
    lot_ids = np.sort(df["Lot_Num"].unique())
    rng = np.random.default_rng(SYN_RANDOM_SEED)

    temp_by_lot = rng.normal(SYN_ENV_TEMP_MEAN_C, SYN_ENV_TEMP_STD_C, size=len(lot_ids))
    humid_by_lot = rng.uniform(
        SYN_ENV_HUMIDITY_MIN_PCT, SYN_ENV_HUMIDITY_MAX_PCT, size=len(lot_ids)
    )
    lot_env = pd.DataFrame(
        {
            "Lot_Num": lot_ids,
            "sim_env_temp_c": temp_by_lot,
            "sim_env_humidity_pct": humid_by_lot,
        }
    )
    return df.merge(lot_env, on="Lot_Num", how="left")


def add_synthetic_maintenance_features(df: pd.DataFrame) -> pd.DataFrame:
    """공정×챔버별 가상 PM(예방정비) 주기를 부여하고, 각 행의 Datetime 기준
    '최근 PM 이후 경과일'을 계산한다. 14일 주기는 기존 EDA에서 관찰된 '격주 진동'
    가설을 검증하기 위한 가정값이며 실제 설비 이력이 아니다.
    """
    df = df.copy()
    dt = pd.to_datetime(df["Datetime"])
    df["_dt"] = dt

    chamber_cols = {
        "photo_soft_Chamber": "softbake",
        "lithography_Chamber": "litho",
        "Etching_Chamber": "etch",
        "Ox_Chamber": "oxid",
    }
    rng = np.random.default_rng(SYN_RANDOM_SEED + 1)
    origin = dt.min()

    for col, tag in chamber_cols.items():
        if col not in df.columns:
            continue
        chambers = np.sort(df[col].dropna().unique())
        # 챔버마다 PM 시작 위상(phase)을 무작위로 다르게 주어 챔버 간 동시 정비를 피함
        phase_by_chamber = {
            c: rng.integers(0, SYN_MAINTENANCE_CYCLE_DAYS) for c in chambers
        }
        days_since_origin = (df["_dt"] - origin).dt.days
        phase = df[col].map(phase_by_chamber).fillna(0)
        days_since_pm = (days_since_origin + phase) % SYN_MAINTENANCE_CYCLE_DAYS
        df[f"sim_days_since_pm_{tag}"] = days_since_pm

    df = df.drop(columns=["_dt"])
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """1단계 통합 결과(merged, group_id/error_class 포함)에 2단계 전처리 +
    외부/가상 피처 확장을 순서대로 적용한다."""
    df = drop_constant_and_duplicate(df)
    df = convert_physical_sentinels(df)
    df = add_spec_deviation_features(df)
    df = add_synthetic_environment_features(df)
    df = add_synthetic_maintenance_features(df)
    return df
