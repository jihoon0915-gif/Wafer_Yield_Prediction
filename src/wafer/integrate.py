"""6개 공정 CSV를 No_Die 기준으로 통합하고, group_id/error_class/wafer_map/센티널
값을 정리하는 핵심 로직. 노트북과 이후 단계(다이 explode, FastAPI)가 공통으로
이 모듈의 함수를 재사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wafer.config import (
    EXPECTED_ROW_COUNT,
    PROCESS_ORDER,
    RAW_DIR,
    WAFER_MAP_REAL_DIE_COUNT,
)
from wafer.io import is_wafer_map_truncated, parse_error_message, parse_wafer_map

_KEY_COLS_NO_ID = ["Lot_Num", "Wafer_Num", "Datetime"]


def load_raw_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / filename)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """PROCESS_ORDER의 6개 CSV를 로드하고 파일별 기본 무결성을 확인한다."""
    raw: dict[str, pd.DataFrame] = {}
    for filename in PROCESS_ORDER:
        df = load_raw_csv(filename)
        if len(df) != EXPECTED_ROW_COUNT:
            raise ValueError(
                f"{filename} 행 수가 {len(df)}로 예상({EXPECTED_ROW_COUNT})과 다릅니다."
            )
        if not df["No_Die"].is_unique:
            raise ValueError(f"{filename}의 No_Die 값이 유일하지 않습니다.")
        raw[filename] = df
    return raw


def merge_all(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """No_Die 기준 1:1 병합. 공정 순서대로 체이닝하며, 각 단계마다 공통 키 컬럼
    (Lot_Num/Wafer_Num/Datetime)이 이미 병합된 프레임과 일치하는지 검증한 뒤
    중복 컬럼을 정리하고, 병합 후 행 수가 깨지지 않는지 즉시 확인한다.
    """
    merged = raw[PROCESS_ORDER[0]].copy()

    for filename in PROCESS_ORDER[1:]:
        df = raw[filename]
        check_cols = [c for c in _KEY_COLS_NO_ID if c in df.columns]

        consistency = merged[["No_Die"] + check_cols].merge(
            df[["No_Die"] + check_cols],
            on="No_Die",
            how="inner",
            suffixes=("", "_dup"),
        )
        for col in check_cols:
            if not (consistency[col] == consistency[f"{col}_dup"]).all():
                raise ValueError(
                    f"{filename}의 {col} 값이 기존 병합 결과와 일치하지 않습니다."
                )

        df_to_merge = df.drop(columns=check_cols)
        merged = merged.merge(df_to_merge, on="No_Die", how="inner", validate="one_to_one")

        if len(merged) != EXPECTED_ROW_COUNT:
            raise ValueError(
                f"{filename} 병합 후 행 수가 {len(merged)}로 예상({EXPECTED_ROW_COUNT})과 다릅니다."
            )

    return merged


def build_group_id(df: pd.DataFrame) -> pd.Series:
    """(Lot_Num, Wafer_Num) 조합을 문자열 group_id로 변환.

    같은 group_id의 반복 행들은 Target/Error_message/Wafer_map이 사실상 동일하므로,
    이후 모든 train/test 분할·교차검증은 이 컬럼을 기준으로 GroupKFold를 적용해야 한다.
    """
    return df["Lot_Num"].astype(str) + "_" + df["Wafer_Num"].astype(str)


def add_error_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error_class"] = df["Error_message"].map(parse_error_message)
    return df


def process_wafer_maps(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Wafer_map 문자열을 파싱하고, 절단된 행은 같은 group_id 내 다른 행으로 복구를
    시도한다. df에는 이미 'group_id' 컬럼이 있어야 한다.

    Returns
    -------
    df : 'wafer_map_truncated' 컬럼이 추가된 프레임
    maps_by_no_die : 개별 행 단위로 파싱에 성공한 배열
    maps_by_group : group_id 단위로 복구된 배열 (그룹 내 한 행이라도 파싱되면 전파)
    stats : 절단/복구/533다이 검증 통계
    """
    truncated = df["Wafer_map"].map(is_wafer_map_truncated)
    df = df.copy()
    df["wafer_map_truncated"] = truncated

    maps_by_no_die: dict[str, np.ndarray] = {}
    violation_no_dies: list[str] = []
    for no_die, raw, is_trunc in zip(df["No_Die"], df["Wafer_map"], truncated):
        if is_trunc:
            continue
        arr = parse_wafer_map(raw)
        maps_by_no_die[no_die] = arr
        if int((arr != 0).sum()) != WAFER_MAP_REAL_DIE_COUNT:
            violation_no_dies.append(no_die)

    maps_by_group: dict[str, np.ndarray] = {}
    for group_id, no_die in zip(df["group_id"], df["No_Die"]):
        if group_id in maps_by_group:
            continue
        if no_die in maps_by_no_die:
            maps_by_group[group_id] = maps_by_no_die[no_die]

    all_groups = df["group_id"].unique()
    unrecoverable_groups = [g for g in all_groups if g not in maps_by_group]

    stats = {
        "n_truncated_rows": int(truncated.sum()),
        "n_parsed_rows": len(maps_by_no_die),
        "n_real_die_violations": len(violation_no_dies),
        "violation_no_dies": violation_no_dies,
        "n_groups_total": len(all_groups),
        "n_groups_recovered": len(maps_by_group),
        "n_groups_unrecoverable": len(unrecoverable_groups),
        "unrecoverable_group_ids": unrecoverable_groups,
    }
    return df, maps_by_no_die, maps_by_group, stats


def scan_sentinels(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """지정된 컬럼들의 0/음수 값 개수를 보고하는 테이블 (판단은 사용자 확인 후)."""
    rows = []
    for col in columns:
        series = df[col]
        n_zero = int((series == 0).sum())
        n_negative = int((series < 0).sum())
        n_le_zero = int((series <= 0).sum())
        rows.append(
            {
                "column": col,
                "n_zero": n_zero,
                "n_negative": n_negative,
                "n_le_zero": n_le_zero,
                "pct_le_zero": round(100 * n_le_zero / len(series), 3),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("n_le_zero", ascending=False)
        .reset_index(drop=True)
    )


def convert_sentinels_to_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """지정된 컬럼의 <=0 값을 NaN으로 변환한 복사본을 반환. 실제 대치(imputation)는
    이후 모델링 단계에서 GroupKFold train fold 내부에서만 수행한다 (전역 대치로 인한
    그룹 정보 누수를 피하기 위함).
    """
    df = df.copy()
    for col in columns:
        mask = df[col] <= 0
        df.loc[mask, col] = np.nan
    return df
