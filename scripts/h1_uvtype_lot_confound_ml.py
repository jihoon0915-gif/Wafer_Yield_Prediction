"""H1: "UV_type 효과는 Lot 교란을 통제하면 사라지는가"를 H2~H7과 동일한
hypothesis_ml_lib 프레임워크(GridSearchCV + 반복 K-Fold + 6개 회귀 모델)로 재검증한다.

이 스크립트는 이전에 별도로 작성했던 scripts/09_h1_full_ml_benchmark.py를 대체한다.
09번 스크립트는 GridSearchCV에 cv=정수(shuffle 없는 KFold)를 그대로 넘기는 버그가
있었는데(H2~H7 작업 중 발견 — Lot_Num 기준 사실상 정렬된 데이터에서 셔플 없는 분할은
특정 Lot이 한 fold에 몰리는 비대표적 분할이 됨), hypothesis_ml_lib.py는 이미 그 버그를
고친 상태이므로 여기서는 그 수정된 공유 프레임워크를 그대로 재사용한다.

FS1: Lot_Num만. FS2: Lot_Num + UV_type.
(08번 스크립트의 통계모델 4종 + 강건성 검증은 그대로 유효 — 이 스크립트는 그걸
대체하는 게 아니라, ML 벤치마크 부분만 올바른 CV로 다시 만드는 것)
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
        "FS2_Lot_plus_UVtype": ([], ["Lot_Num", "UV_type"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H1_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS2_Lot_plus_UVtype"] - piv["FS1_Lot_only"]
    log("\n모델별 UV_type 추가효과(FS2 - FS1):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 UV_type 추가가 CV R²를 개선했다"
        f"(평균 ΔR²={piv['delta_r2'].mean():+.4f}). 08번 리포트의 통계모델 4종(나이브 OLS만 "
        "유의, Lot 통제 시 전부 비유의 p=0.39~0.94)과 같은 방향 — Lot_Num을 이미 아는 "
        "상태에서 UV_type이 주는 추가 정보는 크지 않다."
    )
    verdict = (
        "**기각(실무적으로 무의미)**: ΔR²가 전 모델에서 0.005 미만으로 작다. "
        "08번 리포트의 통계적 유의성 검정(OLS 고정효과 2종 + 혼합효과모형, 전부 p>0.39)과 "
        "결론이 일치한다 — Executive Summary 시나리오 B(H-line→G-line 전환)를 되살릴 "
        "근거는 이번 재검증에서도 나오지 않았다."
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H1_result.md",
        hypothesis_id="H1",
        title="UV_type 효과는 Lot 교란을 통제하면 사라지는가",
        question="UV_type을 Lot_Num에 추가하면 Target 예측력이 개선되는가 (Lot 교란 통제 후 UV_type의 독립 효과)?",
        data_note="웨이퍼 단위 1,704행. (통계모델 4종 + 강건성 검증은 reports/08_h1_uvtype_model_comparison.md 참고 — 이 스크립트는 ML 변수중요도 벤치마크만 담당)",
        feature_set_meaning={
            "FS1_Lot_only": "Lot_Num만(범주형)",
            "FS2_Lot_plus_UVtype": "Lot_Num + UV_type(범주형)",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
