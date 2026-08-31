"""Inspect.csv의 Error_message / Wafer_map 문자열을 파싱하는 함수 모음.

두 컬럼 모두 표준 JSON/Python 리터럴이 아니라 numpy의 문자열 표현(또는 그와 유사한
커스텀 포맷)이라 ast.literal_eval / json.loads로는 바로 파싱할 수 없다.
"""

from __future__ import annotations

import re

import numpy as np

from wafer.config import WAFER_MAP_GRID_SIZE

_ERROR_CLASS_PATTERN = re.compile(r"\[\['([^']+)'\]\]")
_TRUNCATION_MARKER = "..."
_INT_PATTERN = re.compile(r"-?\d+")


def parse_error_message(raw: str) -> str:
    """"none" 또는 "[['ClassName']]" 형태의 문자열에서 클래스명만 추출한다."""
    raw = raw.strip()
    if raw == "none":
        return "none"
    match = _ERROR_CLASS_PATTERN.match(raw)
    if match is None:
        raise ValueError(f"인식할 수 없는 Error_message 형식: {raw!r}")
    return match.group(1)


def is_wafer_map_truncated(raw: str) -> bool:
    """numpy array2string이 긴 배열을 요약할 때 남기는 '...' 표시가 있는지 확인."""
    return _TRUNCATION_MARKER in raw


def parse_wafer_map(raw: str, grid_size: int = WAFER_MAP_GRID_SIZE) -> np.ndarray:
    """절단되지 않은 Wafer_map 문자열을 (grid_size, grid_size) 정수 배열로 복원.

    numpy array2string 출력은 컬럼 정렬을 위해 공백 폭이 불규칙하므로, 괄호/개행을
    수동으로 나누는 대신 문자열 전체에서 정수 토큰만 순서대로 추출하는 방식이 더 안전하다.
    """
    values = [int(v) for v in _INT_PATTERN.findall(raw)]
    expected = grid_size * grid_size
    if len(values) != expected:
        raise ValueError(
            f"예상 셀 수({expected})와 실제 추출된 정수 개수({len(values)})가 다릅니다."
        )
    return np.array(values, dtype=np.int8).reshape(grid_size, grid_size)
