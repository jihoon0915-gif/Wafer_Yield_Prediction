"""H8: 07단계에서 만들었지만 유의성 검증 없이 최종 모델에서 빠졌던 파생변수 재검증 1/6.
oxidation_rate_nm_per_min(=thickness/Oxid_time)이 원본 산화 공정변수(Temp_OXid/ppm/
Pressure/Oxid_time/thickness)만으로 이루어진 베이스라인 대비 추가 정보를 주는가?

thickness/Oxid_time 둘 다 이미 베이스라인에 있으므로 이 파생변수는 그 "비율"이라
정확한 선형종속은 아니다(나눗셈은 덧셈/뺄셈과 달리 선형결합이 아님) — H3/H11/H12와
달리 안전하게 추가(더하기) 방식으로 테스트한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_NUMERIC = ["Temp_OXid", "ppm", "Pressure", "Oxid_time", "thickness"]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    feature_sets = {
        "FS_no_rate": (BASE_NUMERIC, ["type", "Vapor"]),
        "FS_with_rate": (BASE_NUMERIC + ["oxidation_rate_nm_per_min"], ["type", "Vapor"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H8_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_with_rate"] - piv["FS_no_rate"]
    log("\n모델별 산화속도 추가효과(FS_with_rate - FS_no_rate):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 oxidation_rate_nm_per_min 추가가 CV R²를 개선했다"
        f"(평균 ΔR²={piv['delta_r2'].mean():+.4f}). thickness/Oxid_time 원본이 이미 베이스라인에 "
        "있는 상태에서, 그 비율(성장속도)이 추가 정보를 주는지 확인한 것이다."
    )
    verdict = (
        "**채택**" if n_improved >= 4 else "**기각**"
    ) + f" (6개 중 {n_improved}개 개선). {'최종 피처셋에 추가.' if n_improved >= 4 else '최종 피처셋에서 제외 유지.'}"

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H8_result.md",
        hypothesis_id="H8",
        title="산화속도(oxidation_rate_nm_per_min) 파생변수 재검증",
        question="thickness/Oxid_time 비율(산화속도)이 원본 산화 공정변수 대비 추가 예측력을 주는가?",
        data_note="웨이퍼 단위 1,704행, 산화 공정변수 5개 베이스라인",
        feature_set_meaning={
            "FS_no_rate": "Temp_OXid, ppm, Pressure, Oxid_time, thickness (원본만)",
            "FS_with_rate": "위 + oxidation_rate_nm_per_min(=thickness/Oxid_time)",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
