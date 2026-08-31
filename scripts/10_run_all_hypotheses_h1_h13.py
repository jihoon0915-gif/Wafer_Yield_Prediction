"""H1~H13 엔드투엔드 오케스트레이터: 13개 가설 스크립트를 순차 실행하고 실행 로그를
남긴다. H1~H7은 원본 데이터 기반 가설, H8~H13은 07단계에서 만들었지만 검증 없이
누락됐던 파생변수 사후 재검증이다.

**주의**: 이 스크립트가 만드는 `hypothesis_H1_H13_run_log.md`는 단순 실행 로그(스크립트명
+ 소요시간)다. 사람이 읽는 내용 요약(결론표, 버그 발견 서술 등)은 별도로 손으로 쓴
`reports/hypothesis_H1_H7_summary.md` / `hypothesis_H8_H13_summary.md`에 있고, 이
오케스트레이터가 그 파일들을 덮어쓰지 않는다.

개별 스크립트(h1_*.py ~ h13_*.py)는 각자 단독 실행도 가능하다.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HYPOTHESES = [
    ("H1", "h1_uvtype_lot_confound_ml"),
    ("H2", "h2_die_row_pseudoreplication"),
    ("H3", "h3_etch_delta_features"),
    ("H4", "h4_informative_missingness"),
    ("H5", "h5_chamber_effect"),
    ("H6", "h6_time_drift"),
    ("H7", "h7_type_vapor_confound"),
    ("H8", "h8_oxidation_rate"),
    ("H9", "h9_litho_efficiency"),
    ("H10", "h10_line_cd_specgap"),
    ("H11", "h11_total_flux"),
    ("H12", "h12_anneal_temp_diff"),
    ("H13", "h13_oxid_specgap_vs_thickness"),
]


def main() -> None:
    summary_rows = []
    for hid, module_name in HYPOTHESES:
        print(f"\n{'='*70}\n{hid} 실행: {module_name}.py\n{'='*70}", flush=True)
        t0 = time.perf_counter()
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.perf_counter() - t0
        summary_rows.append((hid, module_name, elapsed))
        print(f"{hid} 완료 ({elapsed:.1f}s)", flush=True)

    print("\n" + "=" * 70)
    print("전체 실행 로그")
    print("=" * 70)
    lines = ["# H1~H13 가설 검증 실행 로그", "", "(내용 요약은 hypothesis_H1_H7_summary.md / hypothesis_H8_H13_summary.md 참고)", ""]
    lines.append("| 가설 | 스크립트 | 결과 파일 | 실행시간 |")
    lines.append("|---|---|---|---|")
    for hid, module_name, elapsed in summary_rows:
        lines.append(f"| {hid} | `scripts/{module_name}.py` | `reports/hypothesis_{hid}_result.md` | {elapsed:.1f}s |")
    out_path = PROJECT_ROOT / "reports" / "hypothesis_H1_H13_run_log.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
