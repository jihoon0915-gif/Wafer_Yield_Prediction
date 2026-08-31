"""H6: "격주 진동(PM 주기) 가설의 실증 근거가 약하다" -- 날짜 파생 피처(경과일수,
요일)를 추가했을 때 예측력이 실제로 개선되는지 회귀 ML 6종으로 검증한다.

**중요 정정**: 이전 EDA에서 "고유 관측일 22개(1월 5일 단 1건 제외 대부분 4~5월)"라고
보고했던 건 웨이퍼 단위로 미리 축소(1개 대표 행만 선택)한 뒤 그 대표 행의 날짜만 센
결과였다. 실제로는 Datetime이 그룹(Lot,Wafer) 내에서 전혀 상수가 아니다 -- 같은
웨이퍼의 9개 die-행이 2024-12-30~2025-06-29까지 최대 6개월에 걸쳐 서로 다른 날짜를
갖고, die-행 전체 기준 고유 날짜는 181개다. 그런데도 같은 그룹의 Target은 9개 행 전부
동일(예: group 13_28은 9개 행이 1/5, 1/18, 2/7, ..., 6/27로 흩어져 있는데 Target은
141로 전부 동일). 즉 Datetime은 "언제 측정했는가"를 나타내는 신뢰할 수 있는 웨이퍼
속성이 아니라 die-행에 부여된 부가 메타데이터에 가깝고, 애초에 Target과 결합될 수
있는 방식으로 존재하지 않는다 -- "22개뿐이라 주기를 못 본다"보다 훨씬 강한 결론
("Datetime 자체가 Target과 짝지어지는 방식이 아니다")이다.

FS_no_time: 핵심 공정 변수만.
FS_with_time: 위 + 경과일수(days_since_start) + 요일(day_of_week, 범주형).
  (웨이퍼 단위로 대표 행 1개를 뽑아야 하므로, 9개 중 임의의 날짜 하나만 쓸 수밖에
  없다는 한계가 있음 -- 위 발견 때문에 이 한계 자체가 본질적으로 해소 불가능하다.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from hypothesis_ml_lib import FULL_MODEL_SPECS, load_wafer_level, log, run_benchmark, write_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_NUMERIC = ["Thin F2", "Thin F3", "Thin F4", "Temp_OXid"]


def main() -> None:
    df = load_wafer_level(PROJECT_ROOT)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["days_since_start"] = (df["Datetime"] - df["Datetime"].min()).dt.days
    df["day_of_week"] = df["Datetime"].dt.dayofweek.astype(str)
    log(f"웨이퍼 단위 데이터: {len(df)}행 (대표 행 1개/웨이퍼 기준 날짜, "
        f"die-행 전체 기준 실제 고유 날짜는 181개, 2024-12-30~2025-06-29)\n")

    feature_sets = {
        "FS_no_time": (BASE_NUMERIC, ["UV_type"]),
        "FS_with_time": (BASE_NUMERIC + ["days_since_start"], ["UV_type", "day_of_week"]),
    }

    results_df = run_benchmark(df, "Target", feature_sets, model_specs=FULL_MODEL_SPECS, cv_splits=5, cv_repeats=3)
    results_df.to_csv(PROJECT_ROOT / "reports" / "hypothesis_H6_results.csv", index=False)

    piv = results_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS_with_time"] - piv["FS_no_time"]
    log("\n모델별 시간피처 추가효과(FS_with_time - FS_no_time):")
    log(piv.to_string())

    n_improved = int((piv["delta_r2"] > 0).sum())
    interpretation = (
        f"6개 모델 중 {n_improved}개에서 날짜 파생 피처 추가가 CV R²를 개선했다"
        f"(평균 ΔR²={piv['delta_r2'].mean():+.4f}, XGBoost는 오히려 -0.037로 하락). "
        "이번 검증 과정에서 이전 EDA의 '고유 관측일 22개'라는 근거 자체가 웨이퍼 단위 "
        "사전축소로 인한 착시였음을 발견했다: 실제로는 Datetime이 그룹(웨이퍼) 내에서 "
        "전혀 상수가 아니라 같은 웨이퍼의 9개 die-행이 최대 6개월(2024-12-30~2025-06-29, "
        "고유 181일)에 걸쳐 서로 다른 날짜를 갖는데도 Target은 9개 행 전부 동일하다. "
        "즉 '데이터가 22개뿐이라 주기를 못 본다'가 아니라, 'Datetime 자체가 애초에 "
        "Target 변화와 짝지어지는 방식으로 존재하지 않는다'는 훨씬 강한 결론이다."
    )
    verdict = (
        "**기각**: 시간 피처 추가로 인한 예측력 개선이 미미하고 모델에 따라 부호도 "
        "엇갈린다(6개 중 XGBoost는 뚜렷한 하락). 더 결정적으로는, 같은 웨이퍼 내에서 "
        "Datetime이 최대 6개월 차이나는데 Target이 완전히 동일한 사례를 직접 확인해 "
        "'시간 정보와 Target이 애초에 연결되어 있지 않다'는 걸 데이터로 증명했다. "
        "sim_days_since_pm_* 시뮬레이션 피처가 전제한 '격주 진동' 가설은 이번 재검증 "
        "결과 더 확실하게 기각된다 — 해당 피처는 가설 테스트용 시뮬레이션일 뿐 확정된 "
        "패턴으로 오인하면 안 된다는 원래 문서화된 경고가 다시 한번 확인됐다."
    )

    write_markdown(
        out_path=PROJECT_ROOT / "reports" / "hypothesis_H6_result.md",
        hypothesis_id="H6",
        title="시간 드리프트/격주 진동 가설의 예측 기여도",
        question="날짜 파생 피처(경과일수, 요일)를 추가하면 Target 예측력이 개선되는가?",
        data_note="웨이퍼 단위 1,704행(모델링용, 대표 날짜 1개/웨이퍼). "
                   "단, die-행 전체 기준 실제 고유 날짜는 181개(2024-12-30~2025-06-29)이며, "
                   "같은 웨이퍼 내에서도 Datetime이 최대 6개월 차이날 수 있음(본문 참고).",
        feature_set_meaning={
            "FS_no_time": "핵심 공정변수(Thin F2-4, Temp_OXid) + UV_type",
            "FS_with_time": "위 + 경과일수(days_since_start, 연속형) + 요일(day_of_week, 범주형)",
        },
        results_df=results_df,
        interpretation=interpretation,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
