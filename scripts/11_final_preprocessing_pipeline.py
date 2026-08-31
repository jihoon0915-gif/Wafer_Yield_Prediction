"""H1~H7 가설 검증 결과(reports/hypothesis_H1_H7_summary.md)를 반영한 최종 통합
전처리 파이프라인.

각 결정의 근거:
  H1(기각) -> UV_type 컬럼 완전 제거 (Lot 통제 시 효과 없음, 6개 모델 중 5개 무의미)
  H2(채택) -> die-행(15,390) 대신 웨이퍼 단위(1,704)로 축소해 그룹 누수 원천 차단.
              그래도 남는 Lot 단위 일반화 리스크에 대비해, 전처리기(스케일러/대치)를
              Lot_Num 기준 GroupShuffleSplit으로 학습/평가 분리해 학습하는 구조를 제공.
              Lot_Num 자체는 새 Lot에 일반화 안 됨(별도 검증에서 R2 0.70->0.44로 하락
              확인)이 이미 알려져 있어, 모델 입력 피처가 아니라 "분할 키"로만 사용한다.
  H3(채택) -> Thin F1(원본 단독 무의미, r=0.04) 제거, etch_rate_stage1(F1-F2 델타,
              r=-0.39) 로 대체. F2/F3/F4는 원본 유지(이들과는 선형독립 확인됨).
  H4(채택) -> 센티널(<=0) 5개 컬럼(Pressure/Oxid_time/Thin F4/Flux90s/Flux160s) 값은
              NaN 대치하지 않고 원본 보존 + 공정별 sentinel_flag 3개(0건인 photo 2개
              제외)를 피처로 추가.
  H5(기각) -> 챔버 ID 4개 컬럼 전부 제거(6개 모델 전부에서 추가 시 성능 하락).
  H6(기각) -> Datetime 및 그로부터 파생 가능한 시간 피처 전부 제외(같은 웨이퍼 내에서
              Datetime이 6개월 차이나도 Target이 동일함을 확인 -> 애초에 무관).
  H7(유지) -> type/Vapor는 미세하지만 방향이 일관된 신호(6/6 모델 양의 델타)라 유지,
              결측 없음 확인된 범주형이라 그대로 원-핫 인코딩.

산출물:
  data/processed/processed_final_feature_matrix.csv  -- 선택·가공된 "원본 스케일" 피처
    (주의: 대치/스케일링처럼 데이터 분포에 의존하는 변환은 여기 미리 구워넣지 않는다.
    이 프로젝트가 이미 한 번 "전역 대치 후 분할"로 리키지를 냈다가 고친 전례가 있어
    -- 그 실수를 반복하지 않기 위해, 실제 대치/스케일은 아래 final_preprocessor로
    학습 시점에 분리해서 적용한다.)
  models/final_preprocessor.pkl  -- fit된 ColumnTransformer(Pipeline). 배포/추론용으로
    전체 데이터에 학습(배포 시점엔 더 이상 보호할 미래 데이터가 없으므로 전체 학습이
    맞는 관행). 모델 "평가"에 재사용하면 안 되고, 평가 시에는 아래 group-aware
    fit/transform 예시처럼 매 fold/분할마다 새로 fit해야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# H1/H5/H6 기각 -> 제거 대상. H3 대체(Thin F1) 포함.
# --------------------------------------------------------------------------
DROP_COLS = [
    "UV_type",                                                   # H1
    "Etching_Chamber", "Ox_Chamber", "lithography_Chamber", "photo_soft_Chamber",  # H5
    "Datetime",                                                   # H6
    "Thin F1",                                                    # H3 (etch_rate_stage1로 대체)
    "photo_softbake_sentinel_flag", "photo_litho_sentinel_flag",  # 항상 0(정보량 없음)
    # 순수 메타/중복 컬럼(모델링 피처 아님, ID/공정명/이미 컬럼명 자체가 상수인 것들)
    "process 2", "Process 2-1", "Process 3", "process4", "process",
    "Wafer_map", "wafer_map_truncated", "Wafer_Num",
]

NUMERIC_FEATURES = [
    # Photo_softbake
    "resist_target", "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake",
    "time_HMDS_bake", "spin1", "spin2", "spin3", "photoresist_bake",
    "temp_softbake", "time_softbake",
    # Photo_lithography (UV_type 제외 - H1)
    "Line_CD", "Wavelength", "Resolution", "Energy_Exposure",
    # Etching (Thin F1 제외 - H3, 대신 etch_rate_stage1)
    "Thin F2", "Thin F3", "Thin F4", "Temp_Etching", "Source_Power", "Selectivity",
    "etch_rate_stage1",
    # Ion_Implantation
    "Flux60s", "Flux90s", "Flux160s", "Flux480s", "input_Energy",
    "Temp_implantation", "Furance_Temp", "RTA_Temp",
    # Oxidation (H13: thickness -> oxid_thickness_spec_gap 교체, 완전 공선성이라 함께 못 씀.
    # H8: oxidation_rate_nm_per_min 추가)
    "Temp_OXid", "ppm", "Pressure", "Oxid_time", "oxid_thickness_spec_gap",
    "oxidation_rate_nm_per_min",
    # H4: 센티널 정보보존 플래그(원본 값은 위 컬럼들에 그대로 남아있음, NaN 변환 안 함)
    "oxidation_sentinel_flag", "etching_sentinel_flag", "ion_implant_sentinel_flag",
]

CATEGORICAL_FEATURES = ["type", "Vapor"]  # H7: 유지

TARGET_COL = "Target"
GROUP_COL = "Lot_Num"          # 분할 키로만 사용, 모델 입력 피처 아님(일반화 리스크 때문)
REFERENCE_COLS = ["group_id", "error_class"]  # 피처 아님, 추적/부가 라벨용으로만 보존


def log(msg: str) -> None:
    print(msg, flush=True)


def load_and_dedup() -> pd.DataFrame:
    """H2 대응: die-행(15,390) 대신 웨이퍼 단위(1,704)로 축소."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_master.csv")
    shape_before = df.shape
    df = df.sort_values("wafer_map_truncated").drop_duplicates(subset="group_id").reset_index(drop=True)
    log(f"[H2] die-행 {shape_before} -> 웨이퍼 단위로 축소 {df.shape}")
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, GROUP_COL] + REFERENCE_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 processed_master.csv에 없습니다: {missing}")
    unexpected_dropped = [c for c in DROP_COLS if c in df.columns]
    log(f"제거 대상 확인: {len(unexpected_dropped)}/{len(DROP_COLS)}개 컬럼이 원본에 존재 (전부 최종 셋에서 제외됨)")
    return df[keep].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def demonstrate_group_safe_fit(df: pd.DataFrame) -> None:
    """H2 요구사항: 전처리기가 Lot_Num 그룹 단위로 안전하게(누수 없이) 학습/평가
    분리될 수 있는 구조임을 실제로 시연한다. 여기서 fit한 전처리기는 저장하지
    않는다 -- 이건 "평가 시 이렇게 써야 한다"는 예시일 뿐, 배포용 아티팩트는
    아래 main()에서 전체 데이터로 별도로 fit한다.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df[GROUP_COL]))
    train_lots = set(df.iloc[train_idx][GROUP_COL])
    test_lots = set(df.iloc[test_idx][GROUP_COL])
    log(f"\n[H2 group-safe 시연] GroupShuffleSplit(groups=Lot_Num): "
        f"train {len(train_idx)}행/{len(train_lots)}개 Lot, "
        f"test {len(test_idx)}행/{len(test_lots)}개 Lot, "
        f"Lot 겹침={len(train_lots & test_lots)}개 (0이어야 진짜 그룹 분리)")

    pre = build_preprocessor()
    X_train = pre.fit_transform(df.iloc[train_idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_test = pre.transform(df.iloc[test_idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    log(f"  train fold에만 fit한 전처리기로 transform -> X_train {X_train.shape}, X_test {X_test.shape}")
    log("  (모델 평가 시에는 이 패턴을 CV의 매 fold마다 반복해야 함 -- "
        "전역으로 한 번 fit한 스케일러를 재사용하면 이 프로젝트가 이미 한 번 겪은 "
        "'전역 대치 리키지'가 재발함)")


def main() -> None:
    df = load_and_dedup()
    final_df = select_final_columns(df)

    log(f"\n=== 컬럼 명세 ===")
    log(f"수치형 피처 {len(NUMERIC_FEATURES)}개: {NUMERIC_FEATURES}")
    log(f"범주형 피처 {len(CATEGORICAL_FEATURES)}개: {CATEGORICAL_FEATURES}")
    log(f"타깃: {TARGET_COL} / 그룹키(비피처): {GROUP_COL} / 참조컬럼(비피처): {REFERENCE_COLS}")
    log(f"최종 데이터 형상: {final_df.shape} (die-행 15,390x60 원본 대비)")

    demonstrate_group_safe_fit(final_df)

    # --- 배포용 전처리기: 전체 데이터로 fit (평가용이 아니라 추론용 아티팩트) ---
    final_preprocessor = build_preprocessor()
    X_full = final_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    transformed = final_preprocessor.fit_transform(X_full)
    feature_names_out = final_preprocessor.get_feature_names_out()
    log(f"\n[배포용] 전체 {len(final_df)}행으로 fit한 최종 전처리기: "
        f"입력 {X_full.shape} -> 변환출력 {transformed.shape} "
        f"({len(feature_names_out)}개 변환후 컬럼, 원-핫으로 인해 입력보다 살짝 늘어남)")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    preproc_path = models_dir / "final_preprocessor.pkl"
    joblib.dump(final_preprocessor, preproc_path)
    log(f"저장 완료: {preproc_path}")

    out_csv = PROJECT_ROOT / "data" / "processed" / "processed_final_feature_matrix.csv"
    final_df.to_csv(out_csv, index=False)
    log(f"저장 완료: {out_csv} ({final_df.shape[0]}행 x {final_df.shape[1]}열, 원본 스케일 유지)")

    log("\n=== 처리 전/후 요약 ===")
    log(f"원본(die-행): (15390, 60)")
    log(f"최종 피처 매트릭스(웨이퍼 단위, 미스케일): {final_df.shape}")
    log(f"전처리기 변환 후(스케일+원핫, 모델 입력용): {transformed.shape}")


if __name__ == "__main__":
    main()
