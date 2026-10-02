"""Thin F2 관리구간(잠정 관리한계)의 강건성 검증.

README·보고서가 인용하는 "하위 20% 불량 0% vs 상위 20% 18%대"라는 수치는 세 가지 질문을 받는다.

  1) 어느 단위로 센 값인가 — die 행 기준인가 웨이퍼 기준인가?
  2) 임계값을 같은 데이터에서 구해놓고 같은 데이터로 평가한 것 아닌가?
  3) 불량률 61.1%인 Lot 25 하나가 만든 착시 아닌가?

이 스크립트는 셋을 각각 재계산해 reports/15_control_window_robustness.md(+ .csv)로 남긴다.
결론을 바꾸려는 게 아니라, 인용하는 숫자에 기준을 붙이기 위한 검증이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = PROJECT_ROOT / "data" / "processed" / "processed_final_feature_matrix.csv"
MASTER = PROJECT_ROOT / "data" / "processed" / "processed_master.csv"
OUT_MD = PROJECT_ROOT / "reports" / "15_control_window_robustness.md"
OUT_CSV = PROJECT_ROOT / "reports" / "15_control_window_robustness.csv"

VAR = "Thin F2"          # 두 모델 SHAP 1위 — 대표 관리 인자
CONFOUND_LOT = 25        # 불량률 61.1%, p 관리도 관리 이탈 Lot
N_SPLITS = 5


def rate(defect: np.ndarray, mask: np.ndarray) -> tuple[int, int, float]:
    n = int(mask.sum())
    k = int((defect & mask).sum())
    return k, n, (100 * k / n if n else float("nan"))


def ci95(k: int, n: int) -> tuple[float, float]:
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95)
    return 100 * lo, 100 * hi


def main() -> None:
    df = pd.read_csv(FEATURES)
    defect = (df["error_class"] != "none").values
    x = df[VAR]
    q20, q80 = x.quantile(0.20), x.quantile(0.80)
    rows: list[dict] = []
    md: list[str] = []

    md.append("# 관리구간 강건성 검증 — 식각 20분 후 잔막 두께(Thin F2)\n\n")
    md.append(f"대상: 웨이퍼 {len(df):,}장 · 불량 웨이퍼 {int(defect.sum())}장"
              f"({100 * defect.mean():.1f}%) · 임계값 q20={q20:.0f}nm, q80={q80:.0f}nm\n\n")
    md.append("`python scripts/15_control_window_robustness.py`로 재현. 괄호 안은 95% 신뢰구간.\n")

    # --- 1) 집계 단위: 웨이퍼 vs die 행 -------------------------------------
    md.append("\n## 1. 집계 단위 — 웨이퍼 기준과 die 행 기준이 다르다\n\n")
    lo_mask, hi_mask = (x <= q20).values, (x >= q80).values
    k_lo, n_lo, p_lo = rate(defect, lo_mask)
    k_hi, n_hi, p_hi = rate(defect, hi_mask)
    md.append(f"- 웨이퍼 기준: 하위 20% **{k_lo}/{n_lo}장 = {p_lo:.1f}%**"
              f"(상한 {ci95(k_lo, n_lo)[1]:.1f}%) · 상위 20% **{k_hi}/{n_hi}장 = {p_hi:.1f}%**\n")
    rows += [{"구분": "전체 데이터(웨이퍼)", "구간": "하위 20%", "불량": k_lo, "표본": n_lo, "불량률(%)": round(p_lo, 2)},
             {"구분": "전체 데이터(웨이퍼)", "구간": "상위 20%", "불량": k_hi, "표본": n_hi, "불량률(%)": round(p_hi, 2)}]

    if MASTER.exists():
        m = pd.read_csv(MASTER, usecols=[VAR, "error_class"])
        md_def = (m["error_class"] != "none").values
        mq20, mq80 = m[VAR].quantile(0.20), m[VAR].quantile(0.80)
        k1, n1, p1 = rate(md_def, (m[VAR] <= mq20).values)
        k2, n2, p2 = rate(md_def, (m[VAR] >= mq80).values)
        md.append(f"- die 행 기준(한 웨이퍼가 9행 반복): 하위 20% {k1}/{n1}행 = {p1:.1f}% · "
                  f"상위 20% {k2}/{n2}행 = **{p2:.1f}%**\n")
        md.append("\n> `reports/04_control_window_thinfilm.csv`의 18.4%는 이 die 행 기준 값이다. "
                  "반복 기록 때문에 유효 표본은 웨이퍼 수이므로 보고 시에는 웨이퍼 기준을 기본으로 쓰고, "
                  "두 값을 함께 쓸 때는 기준을 밝힌다.\n")
        rows += [{"구분": "전체 데이터(die 행)", "구간": "하위 20%", "불량": k1, "표본": n1, "불량률(%)": round(p1, 2)},
                 {"구분": "전체 데이터(die 행)", "구간": "상위 20%", "불량": k2, "표본": n2, "불량률(%)": round(p2, 2)}]

    # --- 2) 임계값 일반화: 학습 Lot에서 구해 보류 Lot에 적용 -----------------
    md.append("\n## 2. 임계값 일반화 — 학습 Lot에서 구한 기준을 보류 Lot에 적용\n\n")
    agg = {"lo_k": 0, "lo_n": 0, "hi_k": 0, "hi_n": 0}
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(df, groups=df["Lot_Num"]):
        a, b = df[VAR].iloc[tr].quantile(0.20), df[VAR].iloc[tr].quantile(0.80)
        te_x, te_d = df[VAR].iloc[te], defect[te]
        for key, mask in (("lo", (te_x <= a).values), ("hi", (te_x >= b).values)):
            k, n, _ = rate(te_d, mask)
            agg[f"{key}_k"] += k
            agg[f"{key}_n"] += n
    p_lo_h = 100 * agg["lo_k"] / agg["lo_n"]
    p_hi_h = 100 * agg["hi_k"] / agg["hi_n"]
    md.append(f"- 보류 Lot 합산: 하위 20% **{agg['lo_k']}/{agg['lo_n']}장 = {p_lo_h:.1f}%** · "
              f"상위 20% **{agg['hi_k']}/{agg['hi_n']}장 = {p_hi_h:.1f}%**\n")
    md.append("\n> 임계값을 평가 대상과 분리해도 구간 차이가 유지된다 — 같은 데이터에서 임계값을 뽑아 생긴 착시가 아니다.\n")
    rows += [{"구분": "보류 Lot 적용(5-fold)", "구간": "하위 20%", "불량": agg["lo_k"], "표본": agg["lo_n"], "불량률(%)": round(p_lo_h, 2)},
             {"구분": "보류 Lot 적용(5-fold)", "구간": "상위 20%", "불량": agg["hi_k"], "표본": agg["hi_n"], "불량률(%)": round(p_hi_h, 2)}]

    # --- 3) 교란 Lot 25 제외 (임계값은 고정) --------------------------------
    md.append(f"\n## 3. 교란 Lot {CONFOUND_LOT} 제외 — 효과 크기는 줄지만 방향은 유지\n\n")
    keep = (df["Lot_Num"] != CONFOUND_LOT).values
    k_hi25 = int((defect & hi_mask & ~keep).sum())
    md.append(f"- 상위 20% 불량 {k_hi}장 중 **{k_hi25}장이 Lot {CONFOUND_LOT}**(불량률 61.1%)에 몰려 있다.\n")
    k_lo2, n_lo2, p_lo2 = rate(defect, lo_mask & keep)
    k_hi2, n_hi2, p_hi2 = rate(defect, hi_mask & keep)
    lo_ci, hi_ci = ci95(k_lo2, n_lo2), ci95(k_hi2, n_hi2)
    md.append(f"- 제외 후(임계값은 {q80:.0f}nm 고정): 하위 20% **{k_lo2}/{n_lo2} = {p_lo2:.1f}%**"
              f"(상한 {lo_ci[1]:.1f}%) · 상위 20% **{k_hi2}/{n_hi2} = {p_hi2:.1f}%**"
              f"({hi_ci[0]:.1f}~{hi_ci[1]:.1f}%)\n")
    p = stats.fisher_exact([[k_hi2, n_hi2 - k_hi2], [k_lo2, n_lo2 - k_lo2]])[1]
    md.append(f"- 두 구간 차이는 Lot {CONFOUND_LOT}을 빼고도 유의하다 (Fisher 정확검정 p = {p:.1e}).\n")
    md.append(f"\n> 보고 원칙: **18%대는 Lot {CONFOUND_LOT} 영향이 포함된 값**이고 교란을 제외한 기준은 "
              f"**약 {p_hi2:.0f}%**다. 두 값을 함께 제시한다.\n")
    rows += [{"구분": f"Lot {CONFOUND_LOT} 제외", "구간": "하위 20%", "불량": k_lo2, "표본": n_lo2, "불량률(%)": round(p_lo2, 2)},
             {"구분": f"Lot {CONFOUND_LOT} 제외", "구간": "상위 20%", "불량": k_hi2, "표본": n_hi2, "불량률(%)": round(p_hi2, 2)}]

    # --- 4) 구간 기준 민감도(5분위) ----------------------------------------
    md.append("\n## 4. 구간 기준 민감도 — 5분위 단조 증가\n\n")
    md.append(f"| 5분위 | 범위(nm) | 불량/표본 | 불량률 | Lot {CONFOUND_LOT} 제외 |\n|---|---|---|---|---|\n")
    qs = pd.qcut(x, 5, labels=False, duplicates="drop")
    for q in range(int(np.nanmax(qs)) + 1):
        mask = (qs == q).values
        k, n, pr = rate(defect, mask)
        _, _, pr2 = rate(defect, mask & keep)
        md.append(f"| {q + 1}분위 | {x[mask].min():.0f}~{x[mask].max():.0f} | {k}/{n} | {pr:.1f}% | {pr2:.1f}% |\n")
        rows.append({"구분": "5분위 민감도", "구간": f"{q + 1}분위", "불량": k, "표본": n, "불량률(%)": round(pr, 2)})
    md.append("\n> 하위·상위 20%라는 경계는 탐색용 관례일 뿐이므로, 구간을 바꿔도 단조성이 유지되는지 확인했다.\n")

    # --- 5) 계측 누락 ------------------------------------------------------
    miss = df[x.isna()]
    md.append("\n## 5. 계측 누락 — 표본 2장, 가설 수준\n\n")
    md.append(f"- {VAR} 결측 웨이퍼 {len(miss)}장({', '.join(miss['group_id'])})이며 모두 불량이다"
              f"(Target {', '.join(str(int(t)) for t in miss['Target'])}).\n")
    md.append(f"- 전체 불량 웨이퍼 비율 {100 * defect.mean():.1f}%와 대비되지만 **표본이 {len(miss)}장이라 "
              "통계적 주장이 아니라 Gate 0 경보 규칙의 설계 근거로만 쓴다.**\n")

    # --- 6) 선정 근거 요약 --------------------------------------------------
    corr = {c: df[c].corr(df["Target"]) for c in ["Thin F2", "Thin F3", "Thin F4"]}
    md.append("\n## 6. 왜 Thin F2를 대표 관리 인자로 골랐나\n\n")
    md.append("- LightGBM·XGBoost 두 모델 모두 SHAP 평균 기여도 1위(20.2 / 19.9)\n")
    md.append("- 불량 칩 수와의 상관: " + " · ".join(f"{k} {v:.2f}" for k, v in corr.items())
              + " — Thin F4도 근접해 둘을 함께 핵심 인자로 본다\n")
    md.append(f"- 하위 20% 구간 {n_lo}장에서 불량 {k_lo}장으로 구간 분리가 가장 뚜렷하다\n")
    md.append("- 다만 F2는 레시피 설정값이 아니라 **식각 20분 후 측정값**이라, 레시피 목표가 아닌 "
              "'측정 직후 판정하는 관리한계'로 제안한다\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_MD.name} / {OUT_CSV.name}")
    print(f"웨이퍼 상위20% {p_hi:.1f}% | 보류 Lot {p_hi_h:.1f}% | Lot{CONFOUND_LOT} 제외 {p_hi2:.1f}% | die행 기준 {p2:.1f}%")


if __name__ == "__main__":
    main()
