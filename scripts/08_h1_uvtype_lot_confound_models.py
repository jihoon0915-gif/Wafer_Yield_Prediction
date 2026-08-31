"""H1 가설 검증: "UV_type의 Target 효과는 Lot 교란변수를 통제하면 사라지는가?"

이전 EDA(수동 demeaning)를 정식 모델 여러 개로 재현·비교한다. 전부 같은 질문
(UV_type이 Lot을 통제한 뒤에도 Target에 독립적 효과가 있는가)에 대한 서로 다른
통계적/ML적 접근이며, 결론이 접근법에 상관없이 일관되는지 확인하는 게 목적이다.

핵심 설계 결정: UV_type/Lot_Num/Target은 웨이퍼(그룹) 단위 상수(H2 가설에서 확인)이므로,
die-행 15,390개를 그대로 쓰면 웨이퍼 하나가 ~9번 의사복제(pseudo-replication)되어
표준오차가 인위적으로 작아지고 p-value가 과장된다. 따라서 모든 모델은 웨이퍼 단위로
집계한 1,704행 데이터를 사용한다.

모델 A~D: statsmodels 회귀 (계수/유의성 관점)
모델 E: LightGBM 분산분해 (예측기여도 관점, Lot_Num 단독 vs Lot_Num+UV_type)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def load_wafer_level() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_master.csv")
    df = df.sort_values("wafer_map_truncated").drop_duplicates(subset="group_id").copy()
    df["Lot_Num"] = df["Lot_Num"].astype(int)
    df["UV_type"] = df["UV_type"].astype(str)
    return df.reset_index(drop=True)


def model_A_naive(df: pd.DataFrame) -> dict:
    res = smf.ols("Target ~ C(UV_type, Treatment('G'))", data=df).fit()
    return summarize_ols("A. 나이브 (Lot 통제 없음)", res, df, note="전체 32개 Lot 풀링, 교란 그대로")


def model_B_lot_fe(df: pd.DataFrame) -> dict:
    res = smf.ols(
        "Target ~ C(UV_type, Treatment('G')) + C(Lot_Num)", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["Lot_Num"]})
    return summarize_ols("B. OLS + Lot 고정효과(32개 전체)", res, df, note="Lot을 더미로 통제, SE는 Lot 클러스터robust")


def model_C_lot_fe_mixed_only(df: pd.DataFrame) -> dict:
    ct = pd.crosstab(df["Lot_Num"], df["UV_type"])
    mixed_lots = ct[(ct > 0).sum(axis=1) > 1].index.tolist()
    sub = df[df["Lot_Num"].isin(mixed_lots)].copy()
    res = smf.ols(
        "Target ~ C(UV_type, Treatment('G')) + C(Lot_Num)", data=sub
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["Lot_Num"]})
    return summarize_ols(
        "C. OLS + Lot 고정효과(혼합 19개 Lot만)", res, sub,
        note=f"단일 UV_type Lot({32-len(mixed_lots)}개) 제외 — 가장 깨끗한 식별",
    )


def model_D_mixed_effects(df: pd.DataFrame) -> dict:
    # lbfgs(기본값)는 이 데이터에서 "경계 해"로 수렴해 경고 5건(공분산 특이/Hessian 비양정치)을
    # 내며 다른 값(coef=-2.94)을 준다. cg/bfgs/powell 3개 옵티마이저가 경고 없이 서로
    # 정확히 일치하는 값(coef=-0.34)에 수렴하므로 그쪽을 신뢰하고 사용한다
    # (재현: scripts에서 이 4개를 비교한 로그 참고).
    md = smf.mixedlm("Target ~ C(UV_type, Treatment('G'))", data=df, groups=df["Lot_Num"])
    res = md.fit(method="bfgs")
    rows = []
    for name, coef, se, p in zip(res.params.index, res.params.values, res.bse.values, res.pvalues.values):
        if name.startswith("C(UV_type"):
            rows.append({"term": name, "coef": round(float(coef), 3), "se": round(float(se), 3), "p": round(float(p), 4)})
    return {
        "model": "D. 혼합효과모형(Lot 랜덤절편)",
        "note": "Lot 32개 전체를 부분풀링(partial pooling)으로 통제",
        "n": len(df),
        "r2_or_pseudo": None,
        "aic": round(float(res.aic), 1) if hasattr(res, "aic") else None,
        "uv_type_terms": rows,
    }


def summarize_ols(name: str, res, data: pd.DataFrame, note: str) -> dict:
    rows = []
    for term in res.params.index:
        if term.startswith("C(UV_type"):
            rows.append(
                {
                    "term": term,
                    "coef": round(float(res.params[term]), 3),
                    "se": round(float(res.bse[term]), 3),
                    "p": round(float(res.pvalues[term]), 4),
                }
            )
    return {
        "model": name,
        "note": note,
        "n": int(res.nobs),
        "r2_or_pseudo": round(float(res.rsquared), 4),
        "aic": round(float(res.aic), 1),
        "uv_type_terms": rows,
    }


def model_E_ml_variance_partition(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["Lot_Num_cat"] = d["Lot_Num"].astype("category")
    d["UV_type_cat"] = d["UV_type"].astype("category")
    y = d["Target"].astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def cv_r2(feature_cols):
        X = d[feature_cols]
        model = LGBMRegressor(random_state=RANDOM_STATE, n_estimators=200, verbosity=-1)
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        return scores

    scores_lot_only = cv_r2(["Lot_Num_cat"])
    scores_lot_uv = cv_r2(["Lot_Num_cat", "UV_type_cat"])

    # UV_type의 permutation importance (Lot_Num+UV_type 모델을 전체 데이터로 1회 적합)
    X_full = d[["Lot_Num_cat", "UV_type_cat"]]
    model_full = LGBMRegressor(random_state=RANDOM_STATE, n_estimators=200, verbosity=-1)
    model_full.fit(X_full, y)
    # permutation_importance는 카테고리 dtype을 그대로 다루지 못하므로 코드로 인코딩
    X_enc = X_full.apply(lambda s: s.cat.codes)
    model_enc = LGBMRegressor(random_state=RANDOM_STATE, n_estimators=200, verbosity=-1)
    model_enc.fit(X_enc, y)
    perm = permutation_importance(model_enc, X_enc, y, n_repeats=30, random_state=RANDOM_STATE, scoring="r2")

    return {
        "model": "E. LightGBM 분산분해 (5-fold CV, plain KFold)",
        "note": "Lot_Num만 vs Lot_Num+UV_type 예측기여도 비교",
        "n": len(d),
        "cv_r2_lot_only_mean": round(float(scores_lot_only.mean()), 4),
        "cv_r2_lot_only_std": round(float(scores_lot_only.std()), 4),
        "cv_r2_lot_uv_mean": round(float(scores_lot_uv.mean()), 4),
        "cv_r2_lot_uv_std": round(float(scores_lot_uv.std()), 4),
        "delta_r2_from_uvtype": round(float(scores_lot_uv.mean() - scores_lot_only.mean()), 4),
        "uv_type_permutation_importance_mean": round(float(perm.importances_mean[1]), 5),
        "uv_type_permutation_importance_std": round(float(perm.importances_std[1]), 5),
        "lot_num_permutation_importance_mean": round(float(perm.importances_mean[0]), 5),
    }


def main() -> None:
    df = load_wafer_level()
    log(f"웨이퍼 단위 데이터: {len(df)}행 (die-행 pseudo-replication 제거)")
    ct = pd.crosstab(df["Lot_Num"], df["UV_type"])
    n_pure = int(((ct > 0).sum(axis=1) == 1).sum())
    log(f"단일 UV_type Lot: {n_pure}/32, 혼합 Lot: {32-n_pure}/32\n")

    results = []
    log("[A] 나이브 OLS 적합 중...")
    results.append(model_A_naive(df))
    log("[B] OLS + Lot 고정효과(전체) 적합 중...")
    results.append(model_B_lot_fe(df))
    log("[C] OLS + Lot 고정효과(혼합 Lot만) 적합 중...")
    results.append(model_C_lot_fe_mixed_only(df))
    log("[D] 혼합효과모형 적합 중...")
    results.append(model_D_mixed_effects(df))
    log("[E] LightGBM 분산분해(5-fold CV + permutation importance) 적합 중...")
    results.append(model_E_ml_variance_partition(df))

    log("\n" + "=" * 70)
    for r in results:
        log(f"\n### {r['model']}")
        log(f"  note: {r['note']}, n={r['n']}")
        if "uv_type_terms" in r:
            for t in r["uv_type_terms"]:
                sig = "***" if t["p"] < 0.001 else "**" if t["p"] < 0.01 else "*" if t["p"] < 0.05 else "n.s."
                log(f"    {t['term']}: coef={t['coef']:+.3f}  se={t['se']:.3f}  p={t['p']:.4f}  {sig}")
            log(f"  R2={r['r2_or_pseudo']}  AIC={r['aic']}")
        else:
            log(f"  CV R2 (Lot_Num만) = {r['cv_r2_lot_only_mean']:.4f} +- {r['cv_r2_lot_only_std']:.4f}")
            log(f"  CV R2 (Lot_Num+UV_type) = {r['cv_r2_lot_uv_mean']:.4f} +- {r['cv_r2_lot_uv_std']:.4f}")
            log(f"  UV_type 추가로 인한 delta R2 = {r['delta_r2_from_uvtype']:+.4f}")
            log(f"  UV_type permutation importance = {r['uv_type_permutation_importance_mean']:.5f} "
                f"(+-{r['uv_type_permutation_importance_std']:.5f})  vs Lot_Num={r['lot_num_permutation_importance_mean']:.5f}")

    out_dir = PROJECT_ROOT / "reports"
    import json
    with open(out_dir / "08_h1_uvtype_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\n저장 완료: {out_dir / '08_h1_uvtype_model_comparison.json'}")


if __name__ == "__main__":
    main()
