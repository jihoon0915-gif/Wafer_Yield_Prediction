"""H13: 재검증 6/6. oxid_thickness_spec_gap(=700-thickness)은 원본 XAI 분석(04단계)의
SHAP 상위 20에 이미 있었지만(10위, 1.43) 11단계 최종 피처셋 구성 때 실수로 빠졌다.
그런데 thickness가 이미 베이스라인에 있는 상태로 spec_gap을 "추가"하면 완전한 선형
종속(spec_gap = 700 - thickness, 상수만 다른 부호반전)이라 H3/H11/H12와 같은 문제가
난다 -- 이번엔 "추가"가 아예 안 되고 반드시 "교체"만 가능하다. thickness와 spec_gap
중 어느 쪽이 더 나은 표현인지 비교한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OTHER_OXID = ["Temp_OXid", "ppm", "Pressure", "Oxid_time"]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    feature_sets = {
        "FS_thickness": (OTHER_OXID + ["thickness"], ["type", "Vapor"]),
        "FS_specgap": (OTHER_OXID + ["oxid_thickness_spec_gap"], ["type", "Vapor"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H13_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_specgap"] - piv["FS_thickness"]
    log("\n모델별 spec_gap 대체효과(FS_specgap - FS_thickness):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 thickness를 oxid_thickness_spec_gap으로 교체했을 때 "
        f"CV R²가 개선됐다(평균 ΔR²={piv['delta_r2'].mean():+.4f}). 부호만 반전된 affine 변환이라 "
        "트리 모델(RF/XGB/LightGBM)에는 이론상 차이가 없어야 하고, 선형모델(Linear/Ridge/SVR)에서도 "
        "정보량은 동일해야 한다 -- 차이가 크다면 그 자체로 흥미로운 신호(트리 분할 임계값 근처 효과 등)."
    )
    verdict = (
        f"**{'교체(spec_gap 채택)' if n_improved >= 4 else 'thickness 유지'}** (6개 중 {n_improved}개 개선). "
        "700nm 스펙 기준선 대비 이탈도라는 해석 가능성도 spec_gap 쪽이 더 높음(동률 시 spec_gap 우선)."
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H13_result.md",
        hypothesis_id="H13",
        title="산화막 두께 vs 스펙갭(oxid_thickness_spec_gap) 재검증",
        question="thickness를 700nm 스펙 기준 이탈도(spec_gap)로 표현해도 예측력이 유지·개선되는가?",
        data_note="웨이퍼 단위 1,704행, 산화 공정변수 베이스라인. 원본 XAI(04단계)에서 spec_gap이 SHAP 10위였음.",
        feature_set_meaning={
            "FS_thickness": "Temp_OXid, ppm, Pressure, Oxid_time + thickness(원본)",
            "FS_specgap": "Temp_OXid, ppm, Pressure, Oxid_time + oxid_thickness_spec_gap(=700-thickness)",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
