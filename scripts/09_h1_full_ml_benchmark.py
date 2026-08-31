"""H1 가설: "UV_type 효과는 Lot 교란을 통제하면 사라지는가"를 대표적인 회귀 ML
알고리즘 6종 x GridSearchCV x 반복 K-Fold로 전면 재검증한다.

Target(결함수)은 연속형이므로 전부 회귀 모델이다(분류 모델은 대상이 아님 — 종속변수
특성에 안 맞는 모델을 억지로 끼워넣지 않는다). 08번 스크립트에서 LightGBM 하나로만
했던 "Lot_Num 단독 vs Lot_Num+UV_type" 분산분해를, Linear/Ridge/RandomForest/
XGBoost/LightGBM/SVR 6개 알고리즘으로 확장한다.

방법론:
  1) 각 모델 x 각 feature set(FS1=Lot_Num만, FS2=Lot_Num+UV_type)에 대해
     GridSearchCV(cv=5, scoring=R2)로 최적 하이퍼파라미터를 찾는다.
  2) 찾은 최적 하이퍼파라미터로 RepeatedKFold(5-fold x 10회 반복=50 fold)를 돌려
     R2/MAE/RMSE의 평균±표준편차를 구한다 (튜닝과 최종 안정성 평가를 분리 —
     완전한 nested CV보다 가볍지만, 같은 데이터로 고른 하이퍼파라미터를 그대로
     반복평가에 쓰므로 점추정치가 아주 약간 낙관적일 수 있음을 명시).

CV는 GroupKFold가 아니라 일반 KFold를 쓴다: Lot_Num을 "새 Lot에 대한 일반화"가
아니라 "이미 관측된 Lot들의 평균을 얼마나 잘 추정하는가"를 보는 피처로 쓰기 때문
(이전 조사에서 Lot_Num을 진짜로 held-out하면 R2가 크게 떨어진다는 걸 이미 확인함 —
이번 실험은 그것과 다른 질문).

데이터: 웨이퍼 단위 1,704행 (die-행 반복은 H2에서 확인한 대로 의사복제라 제외).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
INNER_CV = 5          # GridSearchCV 내부 CV
OUTER_SPLITS = 5       # 최종 성능평가 RepeatedKFold
OUTER_REPEATS = 10      # 반복 횟수 (50 fold 평가)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_wafer_level() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_master.csv")
    df = df.sort_values("wafer_map_truncated").drop_duplicates(subset="group_id").copy()
    df["Lot_Num"] = df["Lot_Num"].astype(str)  # 명목형으로 취급(원-핫 대상)
    df["UV_type"] = df["UV_type"].astype(str)
    return df.reset_index(drop=True)


def make_preprocessor(cat_cols: list[str], scale: bool) -> Pipeline:
    ohe = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)]
    )
    steps = [("prep", ohe)]
    if scale:
        steps.append(("scale", StandardScaler(with_mean=True)))
    return Pipeline(steps)


# ---------------------------------------------------------------------
# 모델 정의: (이름, scale 필요여부, estimator, 하이퍼파라미터 그리드)
# 그리드는 "핵심" 파라미터 위주로 작게(2~3단) 잡아 GridSearchCV x RepeatedKFold
# 전체 실행시간을 합리적인 범위로 유지한다.
# ---------------------------------------------------------------------
MODEL_SPECS = [
    ("LinearRegression", False, LinearRegression(), {}),  # 폐형해(closed-form) — 튜닝할 하이퍼파라미터 없음(베이스라인)
    ("Ridge", True, Ridge(random_state=RANDOM_STATE), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
    (
        "RandomForest",
        False,
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 5],
        },
    ),
    (
        "XGBoost",
        False,
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
    (
        "LightGBM",
        False,
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        {
            "model__n_estimators": [100, 300],
            "model__num_leaves": [15, 31],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
    (
        "SVR",
        True,
        SVR(),
        {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.1, 1.0], "model__kernel": ["rbf", "linear"]},
    ),
]

FEATURE_SETS = {
    "FS1_Lot_only": ["Lot_Num"],
    "FS2_Lot_plus_UVtype": ["Lot_Num", "UV_type"],
}

RMSE_SCORER = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)


def run_one(name: str, scale: bool, estimator, grid: dict, feature_cols: list[str], df: pd.DataFrame) -> dict:
    X = df[feature_cols]
    y = df["Target"].astype(float)
    pre = make_preprocessor(feature_cols, scale)
    pipe = Pipeline(pre.steps + [("model", estimator)])

    t0 = time.perf_counter()
    if grid:
        gs = GridSearchCV(pipe, param_grid=grid, cv=INNER_CV, scoring="r2", n_jobs=-1)
        gs.fit(X, y)
        best_pipe = gs.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    else:
        pipe.fit(X, y)
        best_pipe = pipe
        best_params = {}
    tune_s = time.perf_counter() - t0

    rkf = RepeatedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": RMSE_SCORER}
    cv_res = cross_validate(best_pipe, X, y, cv=rkf, scoring=scoring, n_jobs=-1)
    eval_s = time.perf_counter() - t0 - tune_s

    return {
        "model": name,
        "feature_set": [k for k, v in FEATURE_SETS.items() if v == feature_cols][0],
        "best_params": best_params,
        "r2_mean": float(np.mean(cv_res["test_r2"])),
        "r2_std": float(np.std(cv_res["test_r2"])),
        "mae_mean": float(-np.mean(cv_res["test_mae"])),
        "mae_std": float(np.std(cv_res["test_mae"])),
        "rmse_mean": float(-np.mean(cv_res["test_rmse"])),
        "rmse_std": float(np.std(cv_res["test_rmse"])),
        "tune_seconds": round(tune_s, 1),
        "eval_seconds": round(eval_s, 1),
    }


def main() -> None:
    df = load_wafer_level()
    log(f"웨이퍼 단위 데이터: {len(df)}행\n")

    results = []
    for fs_name, cols in FEATURE_SETS.items():
        for name, scale, estimator, grid in MODEL_SPECS:
            log(f"[{fs_name}] {name}: GridSearchCV(grid={len(grid)}개 파라미터) + "
                f"RepeatedKFold({OUTER_SPLITS}x{OUTER_REPEATS}) 실행 중...")
            r = run_one(name, scale, estimator, grid, cols, df)
            results.append(r)
            log(f"  -> best_params={r['best_params']}  R2={r['r2_mean']:.4f}+-{r['r2_std']:.4f}  "
                f"MAE={r['mae_mean']:.2f}  RMSE={r['rmse_mean']:.2f}  "
                f"(tune {r['tune_seconds']}s, eval {r['eval_seconds']}s)")

    res_df = pd.DataFrame(results)
    out_dir = PROJECT_ROOT / "reports"
    res_df.to_csv(out_dir / "09_h1_full_ml_benchmark.csv", index=False)
    log(f"\n저장 완료: {out_dir / '09_h1_full_ml_benchmark.csv'}")

    # 모델별 UV_type 추가효과(ΔR2) 요약
    log("\n=== 모델별 UV_type 추가효과 (FS2 R2 - FS1 R2) ===")
    piv = res_df.pivot(index="model", columns="feature_set", values="r2_mean")
    piv["delta_r2"] = piv["FS2_Lot_plus_UVtype"] - piv["FS1_Lot_only"]
    log(piv.to_string())
    piv.to_csv(out_dir / "09_h1_delta_r2_by_model.csv")


if __name__ == "__main__":
    main()
