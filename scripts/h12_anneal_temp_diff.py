"""H12: 재검증 5/6. anneal_temp_diff(=Furance_Temp-RTA_Temp)가 두 원본 온도 컬럼을
대체할 만한가? Furance_Temp/RTA_Temp 둘 다 이미 베이스라인에 있는 상태로 차이를
"추가"하면 정확한 선형종속(diff=A-B)이 되므로, H11과 같은 이유로 "교체" 방식으로
비교한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OTHER_ION = ["Flux60s", "Flux90s", "Flux160s", "Flux480s", "input_Energy", "Temp_implantation"]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    feature_sets = {
        "FS_raw_2temps": (OTHER_ION + ["Furance_Temp", "RTA_Temp"], []),
        "FS_diff_1": (OTHER_ION + ["anneal_temp_diff"], []),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H12_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_diff_1"] - piv["FS_raw_2temps"]
    log("\n모델별 온도차 1개 대체효과(FS_diff_1 - FS_raw_2temps):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 Furance_Temp/RTA_Temp 2개를 anneal_temp_diff 1개로 "
        f"교체했을 때 CV R²가 개선됐다(평균 ΔR²={piv['delta_r2'].mean():+.4f})."
    )
    verdict = (
        ("**채택(교체)**" if n_improved >= 4 else "**기각**")
        + f" (6개 중 {n_improved}개 개선). {'anneal_temp_diff로 교체.' if n_improved >= 4 else '원본 2개 유지.'}"
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H12_result.md",
        hypothesis_id="H12",
        title="어닐링 온도차(anneal_temp_diff) 재검증",
        question="Furance_Temp/RTA_Temp 2개를 온도차 1개로 대체해도 예측력이 유지·개선되는가?",
        data_note="웨이퍼 단위 1,704행, 이온주입 공정변수 베이스라인",
        feature_set_meaning={
            "FS_raw_2temps": "Furance_Temp, RTA_Temp 원본 2개 + 나머지 이온주입 변수",
            "FS_diff_1": "anneal_temp_diff(=Furance_Temp-RTA_Temp) 1개로 교체 + 나머지 이온주입 변수",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
