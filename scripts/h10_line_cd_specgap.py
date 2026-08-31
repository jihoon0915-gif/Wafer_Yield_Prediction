"""H10: 재검증 3/6. line_cd_band_gap(25~55nm 스펙 이탈도)이 원본 노광 공정변수
베이스라인 대비 추가 정보를 주는가? line_cd_out_of_band(이진 버전)는 band_gap의
임계값 변환이라 band_gap 하나만 검증한다(원본 XAI 분석에서도 연속형 oxid_thickness_
spec_gap만 유의미했고 이진 버전은 상위권에 없었던 전례를 따름). band_gap은 Line_CD의
구간별 hinge 함수(비선형)라 원본 Line_CD와 정확한 선형종속은 아님 -- 추가 방식 테스트.
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
        "FS_no_gap": (BASE_NUMERIC, ["type", "Vapor"]),
        "FS_with_gap": (BASE_NUMERIC + ["line_cd_band_gap"], ["type", "Vapor"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H10_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_with_gap"] - piv["FS_no_gap"]
    log("\n모델별 line_cd_band_gap 추가효과(FS_with_gap - FS_no_gap):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 line_cd_band_gap 추가가 CV R²를 개선했다"
        f"(평균 ΔR²={piv['delta_r2'].mean():+.4f})."
    )
    verdict = (
        ("**채택**" if n_improved >= 4 else "**기각**")
        + f" (6개 중 {n_improved}개 개선). {'최종 피처셋에 추가.' if n_improved >= 4 else '최종 피처셋에서 제외 유지.'}"
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H10_result.md",
        hypothesis_id="H10",
        title="Line_CD 스펙이탈도(line_cd_band_gap) 재검증",
        question="Line_CD의 25~55nm 스펙 이탈도가 원본 노광 공정변수 대비 추가 예측력을 주는가?",
        data_note="웨이퍼 단위 1,704행, 노광 공정변수 4개 베이스라인",
        feature_set_meaning={
            "FS_no_gap": "Line_CD, Wavelength, Resolution, Energy_Exposure (원본만)",
            "FS_with_gap": "위 + line_cd_band_gap(스펙 25~55nm 이탈도)",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
