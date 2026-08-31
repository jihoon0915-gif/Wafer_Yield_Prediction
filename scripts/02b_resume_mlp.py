"""02_regression_benchmark.py가 MLP 단계에서 죽었을 때, 이미 체크포인트된
RF/XGBoost/LightGBM/CatBoost 결과는 재계산하지 않고 MLP만 이어서 실행한다."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

spec = importlib.util.spec_from_file_location("reg_bench", PROJECT_ROOT / "scripts" / "02_regression_benchmark.py")
reg_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg_bench)  # 함수 정의만 로드됨 (main()은 실행 안 함)


def main():
    out_dir = PROJECT_ROOT / "reports"
    with open(out_dir / "02_regression_benchmark.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    reg_bench.log(f"loaded checkpoint with {len(results)} models: {[r['model'] for r in results]}")

    df, numeric_cols, categorical_cols = reg_bench.load_data()
    X = df[numeric_cols + categorical_cols]
    y = df["Target"]
    groups = df["group_id"]

    def mlp_space(trial):
        return {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", ["64", "64,32"]),
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-4, 5e-3, log=True),
        }

    def mlp_factory(p):
        p = dict(p)
        hidden = tuple(int(x) for x in p.pop("hidden_layer_sizes").split(","))
        return reg_bench.make_pipeline(
            reg_bench.MLPRegressor(
                hidden_layer_sizes=hidden, random_state=reg_bench.RANDOM_STATE, max_iter=200,
                early_stopping=True, n_iter_no_change=8, **p,
            ),
            numeric_cols, categorical_cols, scale=True,
        )

    mlp_result = reg_bench.tune_and_eval(
        "MLP NeuralNetwork (Optuna)", mlp_factory, mlp_space, X, y, groups, n_trials=8
    )
    results.append(mlp_result)
    reg_bench.save_results(results, out_dir)

    reg_bench.log("")
    import pandas as pd
    table = pd.DataFrame(results)[
        ["model", "cv_val_score_mean", "cv_val_score_std", "cv_train_score_mean", "overfit_gap",
         "cv_val_mae_mean", "fit_seconds_mean", "predict_ms_per_1k_mean", "tune_seconds"]
    ]
    reg_bench.log(table.to_string(index=False))


if __name__ == "__main__":
    main()
