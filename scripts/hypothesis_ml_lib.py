"""H2~H7 가설별 엔드투엔드 ML 검증에 공통으로 쓰는 재사용 프레임워크.

각 가설 스크립트(h2_*.py ~ h7_*.py)는 이 모듈의 run_benchmark()/write_markdown()을
불러써서 (1) 여러 모델 x GridSearchCV x 반복 K-Fold 벤치마크를 돌리고,
(2) hypothesis_H{n}_result.md 를 자동 생성한다.

종속변수가 전부 Target(연속형 결함수)이므로 회귀 모델만 다룬다 — H1에서 이미
확인했듯 이 프로젝트의 6개 가설(H2~H7 + H1)은 전부 "연속형 Target을 얼마나 잘
설명/예측하는가"로 귀결되는 구조라, 분류 모델(Logistic/SVM-classifier 등)을 억지로
끼워넣지 않는다 — 종속변수 유형에 맞는 모델만 쓴다는 원칙(사용자 요구사항 1번)을
문자 그대로가 아니라 실제로 지키는 방법이 이거다.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STATE = 42
RMSE_SCORER = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)

# 회귀 모델 6종 표준 세트 (H1과 동일한 라인업 — 일관성 유지)
FULL_MODEL_SPECS = [
    ("LinearRegression", False, LinearRegression(), {}),
    ("Ridge", True, Ridge(random_state=RANDOM_STATE), {"model__alpha": [1.0, 10.0, 100.0]}),
    (
        "RandomForest", False, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 300], "model__max_depth": [5, 10]},
    ),
    (
        "XGBoost", False, XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        {"model__n_estimators": [100, 300], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
    ),
    (
        "LightGBM", False, LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        {"model__n_estimators": [100, 300], "model__num_leaves": [15, 31], "model__learning_rate": [0.05, 0.1]},
    ),
    (
        "SVR", True, SVR(),
        {"model__C": [1.0, 10.0], "model__epsilon": [0.5, 1.0]},
    ),
]

# 대용량(die-행 15,390) 데이터용 경량 세트 — H2에서만 사용 (SVR 등 O(n^2)+ 모델 제외)
LIGHT_MODEL_SPECS = [
    ("LinearRegression", False, LinearRegression(), {}),
    (
        "RandomForest", False, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]},
    ),
    (
        "LightGBM", False, LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        {"model__n_estimators": [100, 200], "model__num_leaves": [15, 31]},
    ),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_wafer_level(project_root: Path) -> pd.DataFrame:
    df = pd.read_csv(project_root / "data" / "processed" / "processed_master.csv")
    df = df.sort_values("wafer_map_truncated").drop_duplicates(subset="group_id").copy()
    return df.reset_index(drop=True)


def load_die_level(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "data" / "processed" / "processed_master.csv")


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str], scale: bool, estimator) -> Pipeline:
    # scale=True는 원-핫 출력까지 포함한 "전체 변환 결과"에 적용해야 한다(Ridge/SVR처럼
    # 정규화 강도가 피처 스케일에 좌우되는 모델용). 이전 버전은 숫자형 가지에만 스케일러를
    # 넣어서, 범주형만 있는 피처셋(H7의 Lot_Num 단독 등)에서 Ridge/SVR가 원-핫 0/1
    # 더미를 스케일 없이 그대로 받는 바람에 alpha=100 정규화가 과도하게 걸려 R2가
    # 크게 낮아지는 버그가 있었다(H1에서 별도 함수로 정확히 처리했던 것과 결과가
    # 달라 발견함). 여기서는 ColumnTransformer 전체 출력 뒤에 StandardScaler를 붙여
    # H1과 동일한 방식으로 통일한다.
    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))
    pre = ColumnTransformer(transformers)
    steps = [("prep", pre)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def run_one_model(
    name: str, scale: bool, estimator, grid: dict,
    numeric_cols: list[str], categorical_cols: list[str],
    df: pd.DataFrame, target_col: str,
    cv_splits: int = 5, cv_repeats: int = 3,
    groups_col: str | None = None,
) -> dict:
    X = df[numeric_cols + categorical_cols]
    y = df[target_col].astype(float)
    pipe = make_pipeline(numeric_cols, categorical_cols, scale, estimator)

    if grid:
        # cv=정수(cv_splits)를 그대로 넘기면 sklearn이 shuffle=False KFold를 쓴다.
        # processed_master.csv는 Lot_Num 기준으로 사실상 정렬돼 있어(그룹 dedup 이후에도
        # 원본 행 순서가 유지됨) 셔플 없는 분할은 특정 Lot이 통째로 한 fold에 몰리는
        # 비대표적 분할이 된다 — 실측으로 R2가 음수까지 떨어지는 것을 확인해 발견한 버그.
        # GridSearchCV 내부 튜닝용 CV도 최종평가(RepeatedKFold, shuffle=True)와 동일하게
        # shuffle=True KFold를 명시한다.
        inner_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, param_grid=grid, cv=inner_cv, scoring="r2", n_jobs=-1)
        gs.fit(X, y)
        best_pipe, best_params = gs.best_estimator_, {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    else:
        pipe.fit(X, y)
        best_pipe, best_params = pipe, {}

    if groups_col is not None:
        cv = GroupKFold(n_splits=cv_splits)
        cv_res = cross_validate(
            best_pipe, X, y, groups=df[groups_col], cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": RMSE_SCORER}, n_jobs=-1,
        )
    else:
        cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=RANDOM_STATE)
        cv_res = cross_validate(
            best_pipe, X, y, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": RMSE_SCORER}, n_jobs=-1,
        )

    return {
        "model": name, "best_params": best_params,
        "r2_mean": float(np.mean(cv_res["test_r2"])), "r2_std": float(np.std(cv_res["test_r2"])),
        "mae_mean": float(-np.mean(cv_res["test_mae"])), "mae_std": float(np.std(cv_res["test_mae"])),
        "rmse_mean": float(-np.mean(cv_res["test_rmse"])), "rmse_std": float(np.std(cv_res["test_rmse"])),
    }


def run_benchmark(
    df: pd.DataFrame, target_col: str,
    feature_sets: dict[str, tuple[list[str], list[str]]],  # name -> (numeric_cols, categorical_cols)
    model_specs=None, cv_splits: int = 5, cv_repeats: int = 3,
    groups_col: str | None = None,
) -> pd.DataFrame:
    model_specs = model_specs or FULL_MODEL_SPECS
    rows = []
    for fs_name, (num_cols, cat_cols) in feature_sets.items():
        for name, scale, estimator, grid in model_specs:
            log(f"  [{fs_name}] {name} 적합 중 (grid={len(grid)}개 파라미터)...")
            r = run_one_model(
                name, scale, estimator, grid, num_cols, cat_cols, df, target_col,
                cv_splits=cv_splits, cv_repeats=cv_repeats, groups_col=groups_col,
            )
            r["feature_set"] = fs_name
            rows.append(r)
            log(f"    -> best_params={r['best_params']}  R2={r['r2_mean']:.4f}+-{r['r2_std']:.4f}  "
                f"MAE={r['mae_mean']:.2f}  RMSE={r['rmse_mean']:.2f}")
    return pd.DataFrame(rows)


def write_markdown(
    out_path: Path, hypothesis_id: str, title: str, question: str, data_note: str,
    feature_set_meaning: dict[str, str], results_df: pd.DataFrame,
    interpretation: str, verdict: str,
) -> None:
    lines = [f"# {hypothesis_id}: {title}", "", f"**질문**: {question}", "", f"**데이터**: {data_note}", ""]

    lines.append("## 피처셋 정의")
    lines.append("")
    for name, meaning in feature_set_meaning.items():
        lines.append(f"- `{name}`: {meaning}")
    lines.append("")

    lines.append("## 모델 비교 결과")
    lines.append("")
    lines.append("| 피처셋 | 모델 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in results_df.iterrows():
        params_str = ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) or "—"
        lines.append(
            f"| {r['feature_set']} | {r['model']} | {params_str} | "
            f"{r['r2_mean']:.4f}±{r['r2_std']:.4f} | {r['mae_mean']:.2f} | {r['rmse_mean']:.2f} |"
        )
    lines.append("")

    best_row = results_df.loc[results_df["r2_mean"].idxmax()]
    lines.append(
        f"**최적 모델**: `{best_row['model']}` (피처셋 `{best_row['feature_set']}`), "
        f"R²={best_row['r2_mean']:.4f}±{best_row['r2_std']:.4f}, "
        f"Best Hyperparameters: {best_row['best_params']}"
    )
    lines.append("")

    lines.append("## 해석")
    lines.append("")
    lines.append(interpretation)
    lines.append("")

    lines.append("## 가설 채택/기각 결론")
    lines.append("")
    lines.append(verdict)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"저장 완료: {out_path}")
