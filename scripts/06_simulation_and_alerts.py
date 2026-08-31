"""5단계: 개선안 실행(가상 적용) 및 성과 검증 + Lot 배치이상 조기경보 규칙.

시뮬레이션은 4단계의 모델 기반(R²=0.35) 레시피 최적화 대신, **3단계에서 실측
데이터로 직접 검증한** Thin F2/F3/F4 스윗스팟(N=639, 불량률 0%)과 UV_type 효과
(H vs G/I, p<1e-50)를 기준으로 한다 — 실제 관측된 부분집단의 결과를 그대로
가져다 쓰는 비모수적(non-parametric) counterfactual이라 블랙박스 모델 외삽보다
신뢰도가 높다.

집계 단위: 15,390행은 1,704개 물리적 웨이퍼(group_id)의 반복 측정이므로(그룹 내
Target/error_class는 사실상 동일), 수율 지표는 **그룹당 1행으로 대표추출**한
1,704 웨이퍼 기준으로 계산한다 (반복행 때문에 특정 웨이퍼가 과대가중되는 것을 방지).

Chip Yield% 정의: 웨이퍼당 실제 다이 수 533개(WAFER_MAP_REAL_DIE_COUNT, 기존
검증 완료) 중 정상 다이 비율 = (533 - Target) / 533 * 100.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from wafer import features
from wafer.config import WAFER_MAP_REAL_DIE_COUNT

REAL_DIE_COUNT = WAFER_MAP_REAL_DIE_COUNT  # 533


def log(msg: str) -> None:
    print(msg, flush=True)


def yield_pct(target_series: pd.Series) -> float:
    return float((REAL_DIE_COUNT - target_series).mean() / REAL_DIE_COUNT * 100)


def main():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)

    # 그룹(물리적 웨이퍼) 단위로 대표추출 — Thin F2/F3/F4가 결측 아닌 첫 행 우선
    thin_cols = ["Thin F2", "Thin F3", "Thin F4"]
    df_sorted = df.sort_values(thin_cols, key=lambda s: s.isna(), kind="stable")
    wafers = df_sorted.drop_duplicates(subset="group_id", keep="first").copy()
    log(f"웨이퍼(group_id) 수: {wafers['group_id'].nunique()} (원본 {len(df)}행 -> 대표추출 {len(wafers)}행)")

    is_defect = (wafers["error_class"] != "none")
    baseline_target = wafers["Target"].astype(float).copy()
    baseline_yield = yield_pct(baseline_target)
    baseline_defect_rate = 100 * is_defect.mean()
    log(f"[Baseline] N={len(wafers)}, 평균 Target={baseline_target.mean():.1f}, "
        f"Chip Yield={baseline_yield:.3f}%, 불량패턴 발생률={baseline_defect_rate:.2f}%")

    # -----------------------------------------------------------------
    # Scenario A: 3변수 리스크존(Thin F2/F3/F4 모두 상위20%) -> 스윗스팟 수준 개선
    # -----------------------------------------------------------------
    valid = wafers.dropna(subset=thin_cols)
    q20 = valid[thin_cols].quantile(0.2)
    q80 = valid[thin_cols].quantile(0.8)
    risk_mask = (valid[thin_cols] >= q80).all(axis=1)
    sweet_mask = (valid[thin_cols] <= q20).all(axis=1)
    risk_idx = valid.index[risk_mask]
    sweet_target_mean = valid.loc[sweet_mask, "Target"].mean()
    sweet_defect_rate = 100 * (valid.loc[sweet_mask, "error_class"] != "none").mean()
    log(f"[Scenario A 대상] 리스크존(3변수 모두 상위20%) 웨이퍼 N={len(risk_idx)} "
        f"({100 * len(risk_idx) / len(wafers):.1f}%), 현재 평균 Target={valid.loc[risk_idx, 'Target'].mean():.1f}, "
        f"불량률={100 * (valid.loc[risk_idx, 'error_class'] != 'none').mean():.1f}%")
    log(f"[Scenario A 목표] 스윗스팟(3변수 모두 하위20%) 수준: 평균 Target={sweet_target_mean:.1f}, "
        f"불량률={sweet_defect_rate:.1f}% (N={sweet_mask.sum()})")

    target_A = baseline_target.copy()
    target_A.loc[risk_idx] = sweet_target_mean
    yield_A = yield_pct(target_A)
    defect_A = is_defect.copy()
    defect_A.loc[risk_idx] = False  # 스윗스팟 수준 defect_rate가 0에 가까우므로 근사
    defect_rate_A = 100 * defect_A.mean()

    # -----------------------------------------------------------------
    # Scenario B: UV_type H-line -> G-line 전환
    #
    # ** 자체 검증에서 발견한 confound: Lot 25(웨이퍼 54개, 불량률 61.1%)는
    #    UV_type이 100% H-line이다. H vs G/I의 단순 그룹평균 차이(20.1%)에는
    #    Lot 25 하나의 극단적 배치이상이 통째로 섞여 있어, 실제 "파장 자체의
    #    효과"보다 부풀려져 있다. Lot 25를 제외하고 재계산하면 차이는 20.1%->
    #    7.7%로 줄어든다(그래도 Welch t-test p=4.7e-14로 유의하긴 함).
    #    또한 SHAP에서 UV_type 원핫 변수의 중요도는 57개 피처 중 40위권으로
    #    극히 낮아(Thin F2의 1/400 수준), 다변량 모델은 이 단순 그룹평균
    #    차이만큼 UV_type을 중요하게 보지 않는다 — 즉 "약한/불확실한 신호"로
    #    격하해서 다뤄야 하며, Lot 25는 이 시나리오가 아니라 Lot 경보 규칙으로
    #    별도 대응한다(Lot 25 자체는 U-line 전환 대상에서 제외). **
    # -----------------------------------------------------------------
    non25 = wafers[wafers["Lot_Num"] != 25]
    h_idx = non25.index[non25["UV_type"] == "H"]  # Lot25 제외 — deconfounded
    g_mean_target = non25.loc[non25["UV_type"] == "G", "Target"].mean()
    g_defect_rate = 100 * (non25.loc[non25["UV_type"] == "G", "error_class"] != "none").mean()
    h_before_mean = non25.loc[h_idx, "Target"].mean()
    log(f"[Scenario B 대상 · Lot25 제외(deconfounded)] H-line 웨이퍼 N={len(h_idx)} "
        f"({100 * len(h_idx) / len(wafers):.1f}%), 평균 Target={h_before_mean:.1f} -> "
        f"G-line 평균 Target={g_mean_target:.1f}로 대체 (원래 미보정 차이 20.1%였으나 "
        f"Lot25 제외 후 재확인한 차이는 {100 * (h_before_mean / g_mean_target - 1):.1f}%)")

    target_B = baseline_target.copy()
    target_B.loc[h_idx] = g_mean_target
    yield_B = yield_pct(target_B)
    defect_B = is_defect.copy()
    h_defect_rate_now = 100 * (non25.loc[h_idx, "error_class"] != "none").mean()
    rng = np.random.default_rng(42)
    keep_defect_frac = g_defect_rate / max(h_defect_rate_now, 1e-9)
    for i in h_idx:
        if defect_B.loc[i] and rng.random() > keep_defect_frac:
            defect_B.loc[i] = False
    defect_rate_B = 100 * defect_B.mean()

    # -----------------------------------------------------------------
    # Scenario C: A + B(deconfounded) 동시 적용 — Lot25는 A(식각 리스크존)를
    # 통해서만 개선되고(실제로 Lot25 12/54웨이퍼가 리스크존에 걸림), B의
    # UV_type 전환 대상에서는 제외된 상태를 유지한다.
    # -----------------------------------------------------------------
    target_C = baseline_target.copy()
    target_C.loc[risk_idx] = sweet_target_mean
    h_not_risk = h_idx.difference(risk_idx)
    target_C.loc[h_not_risk] = g_mean_target
    yield_C = yield_pct(target_C)
    defect_C = defect_A.copy()
    for i in h_not_risk:
        if defect_C.loc[i] and rng.random() > keep_defect_frac:
            defect_C.loc[i] = False
    defect_rate_C = 100 * defect_C.mean()

    # -----------------------------------------------------------------
    # Executive summary 표
    # -----------------------------------------------------------------
    exec_rows = [
        {"시나리오": "베이스라인(현재)", "무엇을 바꿨나": "-", "영향 웨이퍼 비중": "-",
         "평균 Target": baseline_target.mean(), "Chip Yield %": baseline_yield,
         "불량패턴 발생률 %": baseline_defect_rate},
        {"시나리오": "A: 식각잔막 리스크존 개선", "무엇을 바꿨나": "Thin F2/F3/F4 모두 상위20% -> 스윗스팟 수준",
         "영향 웨이퍼 비중": f"{100 * len(risk_idx) / len(wafers):.1f}%",
         "평균 Target": target_A.mean(), "Chip Yield %": yield_A, "불량패턴 발생률 %": defect_rate_A},
        {"시나리오": "B: 노광 H-line -> G-line 전환 (Lot25 제외, 약한 신호)",
         "무엇을 바꿨나": "UV_type H-line(Lot25 제외) -> G-line",
         "영향 웨이퍼 비중": f"{100 * len(h_idx) / len(wafers):.1f}%",
         "평균 Target": target_B.mean(), "Chip Yield %": yield_B, "불량패턴 발생률 %": defect_rate_B},
        {"시나리오": "C: A+B(deconfounded) 동시 적용", "무엇을 바꿨나": "위 둘 다",
         "영향 웨이퍼 비중": f"{100 * (len(risk_idx) + len(h_idx.difference(risk_idx))) / len(wafers):.1f}%",
         "평균 Target": target_C.mean(), "Chip Yield %": yield_C, "불량패턴 발생률 %": defect_rate_C},
    ]
    exec_df = pd.DataFrame(exec_rows)
    exec_df["Yield 개선폭(%p)"] = exec_df["Chip Yield %"] - baseline_yield
    exec_df["Target 감소율 %"] = 100 * (baseline_target.mean() - exec_df["평균 Target"]) / baseline_target.mean()
    log("")
    log("=== Executive Summary: 개선 시나리오별 효과 ===")
    log(exec_df.to_string(index=False))
    exec_df.to_csv(PROJECT_ROOT / "reports" / "06_executive_summary.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # Lot 배치이상 조기경보 규칙 (3-sigma 관리도 방식)
    # -----------------------------------------------------------------
    lot_stats = wafers.groupby("Lot_Num").agg(
        n=("group_id", "count"),
        defect_rate=("error_class", lambda s: 100 * (s != "none").mean()),
        mean_target=("Target", "mean"),
        mean_thinF2=("Thin F2", "mean"), mean_thinF3=("Thin F3", "mean"), mean_thinF4=("Thin F4", "mean"),
    ).reset_index()

    mu, sigma = lot_stats["defect_rate"].mean(), lot_stats["defect_rate"].std()
    ucl_3s = mu + 3 * sigma
    ucl_2s = mu + 2 * sigma
    lot_stats["alert_3sigma"] = lot_stats["defect_rate"] > ucl_3s
    lot_stats["alert_2sigma"] = lot_stats["defect_rate"] > ucl_2s
    log("")
    log(f"[Lot 경보규칙] Lot별 불량률 평균={mu:.2f}%, 표준편차={sigma:.2f}%p, "
        f"UCL(mu+2sigma)={ucl_2s:.2f}%, UCL(mu+3sigma)={ucl_3s:.2f}%")
    log(f"[Lot 경보규칙] 2-sigma 규칙 경보 Lot: {lot_stats.loc[lot_stats['alert_2sigma'], 'Lot_Num'].tolist()} "
        f"({lot_stats['alert_2sigma'].sum()}개 / 전체 32개)")
    log(f"[Lot 경보규칙] 3-sigma 규칙 경보 Lot: {lot_stats.loc[lot_stats['alert_3sigma'], 'Lot_Num'].tolist()} "
        f"({lot_stats['alert_3sigma'].sum()}개 / 전체 32개)")

    # Lot 25가 식각잔막 Lot평균 자체도 이상인지 교차검증(원인 추정 보강)
    thin_mu = lot_stats[["mean_thinF2", "mean_thinF3", "mean_thinF4"]].mean()
    thin_sigma = lot_stats[["mean_thinF2", "mean_thinF3", "mean_thinF4"]].std()
    lot25 = lot_stats[lot_stats["Lot_Num"] == 25].iloc[0]
    for col, mu_c, sigma_c in zip(["mean_thinF2", "mean_thinF3", "mean_thinF4"], thin_mu, thin_sigma):
        z = (lot25[col] - mu_c) / sigma_c
        log(f"[Lot25 원인 재검토] {col}: Lot25 평균={lot25[col]:.1f}, 전체Lot평균={mu_c:.1f}"
            f"±{sigma_c:.1f}, z-score={z:.2f}")

    lot_stats.sort_values("defect_rate", ascending=False).to_csv(
        PROJECT_ROOT / "reports" / "06_lot_alert_backtest.csv", index=False, encoding="utf-8-sig"
    )
    log("")
    log("산출물: reports/06_executive_summary.csv, reports/06_lot_alert_backtest.csv")


if __name__ == "__main__":
    main()
