"""H9: 재검증 2/6. cd_resolution_ratio(=Line_CD/Resolution), exposure_energy_per_cd_nm
(=Energy_Exposure/Line_CD)가 원본 노광 공정변수 베이스라인 대비 추가 정보를 주는가?
둘 다 비율이라 원본과 정확한 선형종속 아님 -- 추가 방식으로 안전하게 테스트.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_NUMERIC = ["Line_CD", "Wavelength", "Resolution", "Energy_Exposure"]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    feature_sets = {
        "FS_no_ratio": (BASE_NUMERIC, ["type", "Vapor"]),
        "FS_with_ratio": (BASE_NUMERIC + ["cd_resolution_ratio", "exposure_energy_per_cd_nm"], ["type", "Vapor"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H9_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_with_ratio"] - piv["FS_no_ratio"]
    log("\n모델별 노광효율비율 추가효과(FS_with_ratio - FS_no_ratio):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 cd_resolution_ratio + exposure_energy_per_cd_nm 추가가 "
        f"CV R²를 개선했다(평균 ΔR²={piv['delta_r2'].mean():+.4f})."
    )
    verdict = (
        ("**채택**" if n_improved >= 4 else "**기각**")
        + f" (6개 중 {n_improved}개 개선). {'최종 피처셋에 추가.' if n_improved >= 4 else '최종 피처셋에서 제외 유지.'}"
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H9_result.md",
        hypothesis_id="H9",
        title="노광 효율 비율(cd_resolution_ratio, exposure_energy_per_cd_nm) 재검증",
        question="Line_CD/Resolution 비율과 Energy_Exposure/Line_CD 비율이 원본 노광 공정변수 대비 추가 예측력을 주는가?",
        data_note="웨이퍼 단위 1,704행, 노광 공정변수 4개 베이스라인",
        feature_set_meaning={
            "FS_no_ratio": "Line_CD, Wavelength, Resolution, Energy_Exposure (원본만)",
            "FS_with_ratio": "위 + cd_resolution_ratio + exposure_energy_per_cd_nm",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
