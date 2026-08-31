"""한계 ⑤ 수정: TabNet이 Optuna 튜닝을 전혀 안 받은 채(고정 하이퍼파라미터) 다른
모델(12~20회 튜닝)과 비교된 공정성 문제를 고친다. TabNet은 fold당 학습이 무거워
(fold당 ~45-49s) 전체 Optuna 루프가 매우 느리므로, 탐색 단계는 축소된 epoch로
가볍게 하고 최종 평가만 원래 epoch로 재확인한다.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import optuna
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from wafer import features, modeling

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42
N_SPLITS = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def load_data():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    return df, numeric_cols, categorical_cols


def tabnet_cv_f1(params, X, y_enc, groups, numeric_cols, categorical_cols, max_epochs, patience):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(y_enc), -1, dtype=int)
    for tr_idx, val_idx in gkf.split(X, y_enc, groups):
        pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=True)
        Xtr = np.asarray(pre.fit_transform(X.iloc[tr_idx], y_enc[tr_idx]), dtype=np.float32)
        Xval = np.asarray(pre.transform(X.iloc[val_idx]), dtype=np.float32)
        ytr = y_enc[tr_idx]
        clf = TabNetClassifier(
            n_d=params["n_d"], n_a=params["n_d"], n_steps=params["n_steps"],
            gamma=params["gamma"], seed=RANDOM_STATE, verbose=0, device_name="cpu",
            optimizer_params={"lr": params["lr"]},
        )
        clf.fit(
            Xtr, ytr, max_epochs=max_epochs, patience=patience, batch_size=512, virtual_batch_size=128,
            eval_set=[(Xtr, ytr)], eval_metric=["balanced_accuracy"],
        )
        oof[val_idx] = clf.predict(Xval)
    return f1_score(y_enc, oof, average="macro", zero_division=0), oof


def main():
    df, numeric_cols, categorical_cols = load_data()
    X = df[numeric_cols + categorical_cols]
    le = LabelEncoder()
    y_enc = le.fit_transform(df["error_class"])
    classes = list(le.classes_)
    groups = df["group_id"]

    out_path = PROJECT_ROOT / "reports" / "03_classification_benchmark.json"
    results = json.load(open(out_path, encoding="utf-8"))
    results = [r for r in results if r["model"] != "TabNet"]  # 기존 미튜닝 TabNet 결과 제거
    log(f"checkpoint loaded, kept models: {[r['model'] for r in results]}")

    def space(trial):
        return {
            "n_d": trial.suggest_categorical("n_d", [8, 16, 24]),
            "n_steps": trial.suggest_int("n_steps", 3, 5),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lr": trial.suggest_float("lr", 5e-3, 3e-2, log=True),
        }

    n_trials = 6
    log(f"[TabNet] Optuna 튜닝 시작 ({n_trials} trials x {N_SPLITS}-fold, 탐색 단계 max_epochs=20/patience=6 축소)")
    trial_times = []

    def objective(trial):
        params = space(trial)
        t0 = time.perf_counter()
        f1, _ = tabnet_cv_f1(params, X, y_enc, groups, numeric_cols, categorical_cols, max_epochs=20, patience=6)
        trial_times.append(time.perf_counter() - t0)
        log(f"  [TabNet] trial {trial.number + 1}/{n_trials} macro-F1={f1:.4f} ({trial_times[-1]:.1f}s)")
        return f1

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tune_seconds = time.perf_counter() - t0
    best_params = study.best_params
    log(f"[TabNet] 탐색 완료, best_params={best_params}, best_cv_macro_f1={study.best_value:.4f}, {tune_seconds:.1f}s")

    log("[TabNet] 최종 평가 (원래 epoch budget: max_epochs=40/patience=8)")
    t0 = time.perf_counter()
    final_f1, oof_pred = tabnet_cv_f1(
        best_params, X, y_enc, groups, numeric_cols, categorical_cols, max_epochs=40, patience=8
    )
    final_seconds = time.perf_counter() - t0

    report = classification_report(y_enc, oof_pred, target_names=classes, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    minority_f1 = float(np.mean([report[c]["f1-score"] for c in classes if c != "none"]))
    summary = {
        "model": "TabNet (Optuna, 6 trials)",
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "minority_avg_f1": minority_f1,
        "none_f1": report["none"]["f1-score"],
        "fit_seconds_mean": final_seconds / N_SPLITS,
        "predict_ms_per_1k_mean": None,
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
        "best_params": best_params,
        "tune_seconds": tune_seconds,
        "optuna_best_macro_f1": study.best_value,
        "n_trials": n_trials,
    }
    log(f"[TabNet] DONE (튜닝 후) — macro-F1={macro_f1:.4f} (튜닝 전 0.2617 대비 "
        f"{'+' if macro_f1 > 0.2617 else ''}{macro_f1 - 0.2617:.4f}), minority-F1={minority_f1:.4f}")

    results.append(summary)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "per_class_f1"} for r in results])
    cols = ["model", "macro_f1", "weighted_f1", "minority_avg_f1", "none_f1", "fit_seconds_mean"]
    table[[c for c in cols if c in table.columns]].to_csv(
        PROJECT_ROOT / "reports" / "03_classification_benchmark.csv", index=False
    )
    log("")
    log(table[[c for c in cols if c in table.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
