"""1단계 통합 결과에 대한 검증 assertion 모음.

기존 분석에서 이미 확인된 벤치마크(행 수, 라벨 분포, 그룹 정체성 고정 현상 등)를
서술이 아니라 코드로 재확인하여, 노트북을 재실행할 때마다 회귀를 자동으로 잡아낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wafer.config import (
    EXPECTED_GROUP_COUNT,
    EXPECTED_ROW_COUNT,
    ERROR_CLASSES,
)

EXPECTED_ERROR_CLASS_COUNTS = {
    "none": 14272,
    "Edge-Loc": 515,
    "Loc": 279,
    "Random": 99,
    "Center": 90,
    "Scratch": 63,
    "Near-full": 36,
    "Edge-Ring": 36,
}
EXPECTED_TRUNCATION_COUNT = 2619
EXPECTED_GROUP_TARGET_STD_ZERO_PCT = 89.2


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_row_count(df: pd.DataFrame) -> CheckResult:
    n = len(df)
    return CheckResult(
        "병합 후 행 수 == 15,390",
        n == EXPECTED_ROW_COUNT,
        f"실제 {n}행",
    )


def check_no_die_unique(df: pd.DataFrame) -> CheckResult:
    unique = df["No_Die"].is_unique
    return CheckResult("No_Die 유일성", bool(unique), f"unique={unique}")


def check_error_class_distribution(df: pd.DataFrame) -> CheckResult:
    actual = df["error_class"].value_counts().to_dict()
    mismatches = {
        cls: (actual.get(cls, 0), expected)
        for cls, expected in EXPECTED_ERROR_CLASS_COUNTS.items()
        if actual.get(cls, 0) != expected
    }
    passed = len(mismatches) == 0 and set(actual) == set(EXPECTED_ERROR_CLASS_COUNTS)
    detail = "일치" if passed else f"불일치: {mismatches}"
    return CheckResult("error_class 분포가 기존 확인값과 일치", passed, detail)


def check_truncation_rate(
    df: pd.DataFrame, expected: int = EXPECTED_TRUNCATION_COUNT
) -> CheckResult:
    n = int(df["wafer_map_truncated"].sum())
    return CheckResult(
        f"Wafer_map 절단 행 수 ≈ {expected}",
        n == expected,
        f"실제 {n}행 ({100 * n / len(df):.2f}%)",
    )


def check_real_die_violations(stats: dict) -> CheckResult:
    n = stats["n_real_die_violations"]
    return CheckResult(
        "파싱된 Wafer_map의 실제 다이 수 == 533",
        n == 0,
        f"위반 {n}건 / 파싱 성공 {stats['n_parsed_rows']}건",
    )


def check_group_count(df: pd.DataFrame, expected: int = EXPECTED_GROUP_COUNT) -> CheckResult:
    n = df["group_id"].nunique()
    return CheckResult(f"group_id 고유 개수 == {expected}", n == expected, f"실제 {n}개")


def check_group_target_std_zero_ratio(
    df: pd.DataFrame,
    expected_pct: float = EXPECTED_GROUP_TARGET_STD_ZERO_PCT,
    tolerance_pct: float = 2.0,
) -> CheckResult:
    stds = df.groupby("group_id")["Target"].std(ddof=0).fillna(0)
    pct = round(100 * float((stds == 0).mean()), 2)
    passed = abs(pct - expected_pct) <= tolerance_pct
    return CheckResult(
        f"그룹 내 Target 표준편차 0 비율 ≈ {expected_pct}%",
        passed,
        f"실제 {pct}% (허용오차 ±{tolerance_pct}%p)",
    )


def check_group_error_class_single_valued(df: pd.DataFrame) -> CheckResult:
    n_multi = int((df.groupby("group_id")["error_class"].nunique() > 1).sum())
    return CheckResult(
        "그룹 내 error_class 단일값 (100%)",
        n_multi == 0,
        f"클래스가 2종 이상 섞인 그룹 {n_multi}개",
    )


def check_group_wafer_map_identical(
    df: pd.DataFrame, maps_by_no_die: dict[str, np.ndarray]
) -> CheckResult:
    parsed = df.loc[~df["wafer_map_truncated"]]
    n_checked, n_mismatched = 0, 0
    for _, sub in parsed.groupby("group_id"):
        no_dies = sub["No_Die"].tolist()
        if len(no_dies) < 2:
            continue
        n_checked += 1
        arrays = [maps_by_no_die[nd] for nd in no_dies]
        if not all(np.array_equal(arrays[0], a) for a in arrays[1:]):
            n_mismatched += 1
    passed = n_mismatched == 0
    detail = f"비교 가능 그룹(비절단 2행 이상) {n_checked}개 중 불일치 {n_mismatched}개"
    return CheckResult("그룹 내 Wafer_map 완전 동일 (비절단 기준)", passed, detail)


def run_all_checks(
    df: pd.DataFrame,
    maps_by_no_die: dict[str, np.ndarray],
    wafer_map_stats: dict,
) -> pd.DataFrame:
    checks = [
        check_row_count(df),
        check_no_die_unique(df),
        check_error_class_distribution(df),
        check_truncation_rate(df),
        check_real_die_violations(wafer_map_stats),
        check_group_count(df),
        check_group_target_std_zero_ratio(df),
        check_group_error_class_single_valued(df),
        check_group_wafer_map_identical(df, maps_by_no_die),
    ]
    return pd.DataFrame(
        [{"검증 항목": c.name, "통과": c.passed, "상세": c.detail} for c in checks]
    )
