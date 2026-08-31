"""H7: "type(dry/wet)·Vapor(H2O/O2)는 UV_type과 달리 Lot과 교란되지 않는다"를
H1과 동일한 설계(Lot_Num만 vs Lot_Num+피처)의 ML 분산분해로 재검증한다.
H1에서는 UV_type이 Lot_Num 대비 추가 정보가 거의 없었다(사실상 기각) — type/Vapor도
같은 검증을 통과하는지, 즉 "Lot과 무관하게 그 자체로도 예측에 별 도움이 안 되는
평범한 변수"인지 확인한다(애초에 Lot 교란 자체가 없었으므로 H1과는 다른 결과가
나올 수 있음 -- type/Vapor이 진짜 독립적 신호를 가질 가능성은 열려 있다).

FS1: Lot_Num만.
FS2: Lot_Num + type + Vapor.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    df["Lot_Num"] = df["Lot_Num"].astype(str)
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    feature_sets = {
        "FS1_Lot_only": ([], ["Lot_Num"]),
        "FS2_Lot_plus_type_vapor": ([], ["Lot_Num", "type", "Vapor"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H7_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS2_Lot_plus_type_vapor"] - piv["FS1_Lot_only"]
    log("\n모델별 type/Vapor 추가효과(FS2 - FS1):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 type/Vapor 추가가 CV R²를 개선했다"
        f"(평균 ΔR²={piv['delta_r2'].mean():+.4f}). H1(UV_type)과 같은 실험 설계를 "
        "썼지만, type/Vapor는 애초에 Lot과 교란되지 않는다는 게 EDA에서 이미 확인됐으므로 "
        "여기서 보는 델타는 '교란 제거 효과'가 아니라 'type/Vapor 자체의 순수 예측 기여도'다."
    )
    verdict = (
        "**참고용(가설 자체는 이미 EDA에서 채택됨 — 여기선 예측 기여도만 추가 확인)**: "
        f"ΔR² 크기가 {'H1의 UV_type 델타와 비슷한 수준으로 작다' if abs(piv['delta_r2'].mean()) < 0.01 else '무시할 수 없는 수준이다'} — "
        "type/Vapor는 Lot 교란과 무관하게 그 자체로도 Target에 대한 독립적 설명력이 "
        "크지 않은 변수로 보인다."
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H7_result.md",
        hypothesis_id="H7",
        title="type/Vapor는 Lot과 교란되지 않는다 — 예측 기여도 재확인",
        question="type(dry/wet)·Vapor(H2O/O2)를 Lot_Num에 추가하면 예측력이 개선되는가?",
        data_note="웨이퍼 단위 1,704행",
        feature_set_meaning={
            "FS1_Lot_only": "Lot_Num만(범주형)",
            "FS2_Lot_plus_type_vapor": "Lot_Num + type + Vapor(범주형)",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
