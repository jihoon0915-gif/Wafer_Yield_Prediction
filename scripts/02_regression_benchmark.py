"""2단계: Target(결함 다이 수) 회귀 다중모델 벤치마크.

RandomForest / XGBoost / LightGBM / CatBoost / MLP(NeuralNetwork)를 동일한
GroupKFold(5, group=group_id) 위에서 비교한다. 하이퍼파라미터는 Optuna로 탐색하며,
목적함수는 5-fold GroupKFold CV 평균 R² (nested CV가 아닌 단일 CV 재사용 — 계산
비용/시간 제약을 고려한 실무적 타협, 결과 해석 시 최적치가 다소 낙관적일 수 있음에 유의).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from wafer import features, modeling

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
N_TRIALS = 15
RANDOM_STATE = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def save_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "02_regression_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    cols = ["model", "cv_val_score_mean", "cv_val_score_std", "cv_train_score_mean", "overfit_gap",
            "cv_val_mae_mean", "fit_seconds_mean", "predict_ms_per_1k_mean", "tune_seconds"]
    table = pd.DataFrame(results)[cols]
    table.to_csv(out_dir / "02_regression_benchmark.csv", index=False)


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    return df, numeric_cols, categorical_cols


def make_pipeline(estimator, numeric_cols, categorical_cols, scale: bool) -> Pipeline:
    pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=scale)
    return Pipeline([("pre", pre), ("model", estimator)])


def cv_r2(estimator_factory, X, y, groups, n_splits=N_SPLITS) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        pipe = estimator_factory()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[val_idx])
        scores.append(r2_score(y.iloc[val_idx], pred))
    return float(np.mean(scores))


def tune_and_eval(name, factory_from_params, param_space_fn, X, y, groups, n_trials=N_TRIALS):
    trial_times = []

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        t0 = time.perf_counter()
        score = cv_r2(lambda: factory_from_params(params), X, y, groups)
        trial_times.append(time.perf_counter() - t0)
        log(f"  [{name}] trial {trial.number + 1}/{n_trials} r2={score:.4f} "
            f"({trial_times[-1]:.1f}s/trial)")
        return score

    log(f"[{name}] tuning start ({n_trials} trials x {N_SPLITS}-fold)")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tune_seconds = time.perf_counter() - t0

    best_params = study.best_params
    cv_result = modeling.run_group_cv(
        name,
        lambda: factory_from_params(best_params),
        X,
        y,
        groups,
        scorer=r2_score,
        mae_fn=mean_absolute_error,
        n_splits=N_SPLITS,
    )
    summary = cv_result.summary()
    summary["best_params"] = best_params
    summary["optuna_best_cv_r2"] = study.best_value
    summary["tune_seconds"] = tune_seconds
    summary["n_trials"] = n_trials
    log(f"[{name}] DONE — val R2={summary['cv_val_score_mean']:.4f} "
        f"(std {summary['cv_val_score_std']:.4f}), overfit_gap={summary['overfit_gap']:.4f}, "
        f"fit={summary['fit_seconds_mean']:.2f}s, tune={tune_seconds:.1f}s")
    return summary


def main():
    df, numeric_cols, categorical_cols = load_data()
    X = df[numeric_cols + categorical_cols]
    y = df["Target"]
    groups = df["group_id"]
    log(f"X shape {X.shape}, numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")

    out_dir = PROJECT_ROOT / "reports"
    results = []

    # 0) RandomForest 기본값 baseline (기존 대화정리 문서의 R²=0.892 재현/검증용, 튜닝 없음, GroupKFold만 적용)
    baseline = modeling.run_group_cv(
        "RandomForest (default, no tuning)",
        lambda: make_pipeline(
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            numeric_cols, categorical_cols, scale=False,
        ),
        X, y, groups, scorer=r2_score, mae_fn=mean_absolute_error, n_splits=N_SPLITS,
    )
    baseline_summary = baseline.summary()
    baseline_summary["best_params"] = {}
    baseline_summary["optuna_best_cv_r2"] = None
    baseline_summary["tune_seconds"] = 0.0
    baseline_summary["n_trials"] = 0
    log(f"[baseline RF] DONE — val R2={baseline_summary['cv_val_score_mean']:.4f} "
        f"(std {baseline_summary['cv_val_score_std']:.4f}), overfit_gap={baseline_summary['overfit_gap']:.4f}")
    results.append(baseline_summary)
    save_results(results, out_dir)

    # 1) RandomForest (Optuna 튜닝)
    def rf_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
        }

    def rf_factory(p):
        return make_pipeline(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **p),
            numeric_cols, categorical_cols, scale=False,
        )

    results.append(tune_and_eval("RandomForest (Optuna)", rf_factory, rf_space, X, y, groups))
    save_results(results, out_dir)

    # 2) XGBoost
    def xgb_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def xgb_factory(p):
        return make_pipeline(
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **p),
            numeric_cols, categorical_cols, scale=False,
        )

    results.append(tune_and_eval("XGBoost (Optuna)", xgb_factory, xgb_space, X, y, groups))
    save_results(results, out_dir)

    # 3) LightGBM
    def lgbm_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def lgbm_factory(p):
        return make_pipeline(
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **p),
            numeric_cols, categorical_cols, scale=False,
        )

    results.append(tune_and_eval("LightGBM (Optuna)", lgbm_factory, lgbm_space, X, y, groups))
    save_results(results, out_dir)

    # 4) CatBoost
    def cat_space(trial):
        return {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

    def cat_factory(p):
        return make_pipeline(
            CatBoostRegressor(
                random_state=RANDOM_STATE, verbose=False, allow_writing_files=False, thread_count=-1, **p
            ),
            numeric_cols, categorical_cols, scale=False,
        )

    results.append(tune_and_eval("CatBoost (Optuna)", cat_factory, cat_space, X, y, groups))
    save_results(results, out_dir)

    # 5) MLP (Neural Network) — 이전 실행에서 극소 alpha/lr 조합이 800 iter까지 수렴하지
    # 못해 사실상 멈춘 것처럼 보였음. 탐색 범위를 좁히고 max_iter/patience를 낮춰
    # 각 fit이 수 초~수십 초 안에 끝나도록 제한.
    def mlp_space(trial):
        # study.best_params returns the RAW suggested value (the string), not the
        # tuple derived below — so keep the string here and convert in mlp_factory
        # instead, otherwise re-instantiating from best_params breaks.
        return {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", ["64", "64,32"]),
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-4, 5e-3, log=True),
        }

    def mlp_factory(p):
        p = dict(p)
        hidden = tuple(int(x) for x in p.pop("hidden_layer_sizes").split(","))
        return make_pipeline(
            MLPRegressor(
                hidden_layer_sizes=hidden, random_state=RANDOM_STATE, max_iter=200,
                early_stopping=True, n_iter_no_change=8, **p,
            ),
            numeric_cols, categorical_cols, scale=True,
        )

    results.append(tune_and_eval("MLP NeuralNetwork (Optuna)", mlp_factory, mlp_space, X, y, groups, n_trials=8))
    save_results(results, out_dir)

    log("")
    table = pd.DataFrame(results)[
        ["model", "cv_val_score_mean", "cv_val_score_std", "cv_train_score_mean", "overfit_gap",
         "cv_val_mae_mean", "fit_seconds_mean", "predict_ms_per_1k_mean", "tune_seconds"]
    ]
    log(table.to_string(index=False))


if __name__ == "__main__":
    main()
