"""한계 ⑤ 수정(회귀 쪽): MLP가 8 trials로 트리 모델들(15 trials)보다 적게 튜닝된
공정성 문제를 고친다. 15 trials로 재실행해 체크포인트의 기존 MLP 항목을 교체."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

spec = importlib.util.spec_from_file_location("reg_bench", PROJECT_ROOT / "scripts" / "02_regression_benchmark.py")
reg_bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg_bench)


def main():
    out_dir = PROJECT_ROOT / "reports"
    with open(out_dir / "02_regression_benchmark.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    results = [r for r in results if not r["model"].startswith("MLP")]
    reg_bench.log(f"loaded checkpoint, kept: {[r['model'] for r in results]}")

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
        "MLP NeuralNetwork (Optuna, 15 trials)", mlp_factory, mlp_space, X, y, groups, n_trials=15
    )
    results.append(mlp_result)
    reg_bench.save_results(results, out_dir)

    reg_bench.log("")
    import pandas as pd
    table = pd.DataFrame(results)[
        ["model", "cv_val_score_mean", "cv_val_score_std", "overfit_gap", "tune_seconds"]
    ]
    reg_bench.log(table.to_string(index=False))


if __name__ == "__main__":
    main()
