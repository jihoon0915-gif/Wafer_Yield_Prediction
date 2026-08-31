"""H11: 재검증 4/6. total_flux_60_480(=Flux60s+Flux90s+Flux160s+Flux480s)이 4개 원본
Flux 컬럼을 대체할 만한가? **주의**: 이 4개 원본이 전부 이미 베이스라인에 있는 상태로
합계를 "추가"하면 정확한 선형종속(총합=성분의 합)이 되어 H3의 total_etch_removal과
같은 무정규화 LinearRegression 발산 버그가 재발한다(실측으로 이미 한 번 겪음). 그래서
"추가"가 아니라 "4개 원본을 합계 1개로 교체"하는 방식으로 안전하게 비교한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OTHER_ION = ["input_Energy", "Temp_implantation", "Furance_Temp", "RTA_Temp"]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    feature_sets = {
        "FS_raw_flux4": (["Flux60s", "Flux90s", "Flux160s", "Flux480s"] + OTHER_ION, []),
        "FS_agg_flux1": (["total_flux_60_480"] + OTHER_ION, []),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H11_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_agg_flux1"] - piv["FS_raw_flux4"]
    log("\n모델별 합계 1개 대체효과(FS_agg_flux1 - FS_raw_flux4):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    n_close = int((piv["delta_r2"].abs() < 0.005).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 4개 원본 Flux를 합계 1개로 교체했을 때 CV R²가 개선됐다"
        f"(평균 ΔR²={piv['delta_r2'].mean():+.4f}). {n_close}개 모델은 차이가 0.005 미만으로 "
        "사실상 동일 — 정보 손실 없이 피처 수를 4->1로 줄일 수 있는지가 관건이다."
    )
    if n_close >= 4:
        verdict = f"**부분 채택**: 합계로 교체해도 성능 차이가 미미(6개 중 {n_close}개 거의 동일) -- 피처 수 절감 목적이면 교체, 아니면 원본 유지도 무방. 여기서는 원본 4개 유지(이미 개별적으로 다른 검증을 거친 피처라 보수적으로 유지)."
    else:
        verdict = f"**{'채택' if n_improved >= 4 else '기각'}** (6개 중 {n_improved}개 개선)."

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H11_result.md",
        hypothesis_id="H11",
        title="이온주입 누적 플럭스(total_flux_60_480) 재검증",
        question="Flux60s~480s 4개 원본을 합계 1개로 대체해도 예측력이 유지되는가?",
        data_note="웨이퍼 단위 1,704행, 이온주입 공정변수 베이스라인. Flux840s는 원본 파이프라인에서 이미 상수로 제외됨.",
        feature_set_meaning={
            "FS_raw_flux4": "Flux60s/90s/160s/480s 원본 4개 + 나머지 이온주입 변수",
            "FS_agg_flux1": "total_flux_60_480(4개 합계) 1개로 교체 + 나머지 이온주입 변수",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
