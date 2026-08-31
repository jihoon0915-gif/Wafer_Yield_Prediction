"""2단계: Error_message(8개 클래스, 심각한 불균형) 다중분류 벤치마크.

SMOTE+RandomForest / Cost-Sensitive LightGBM(class_weight=balanced) / TabNet을
동일한 GroupKFold(5, group=group_id) 위에서 비교한다.

클래스 불균형이 심해(Near-full/Edge-Ring 등 36건) 개별 fold에 특정 소수 클래스가
아예 없을 수 있으므로, per-fold 지표 평균 대신 5-fold의 **out-of-fold(OOF) 예측을
모아 한 번에** classification_report를 계산한다 (macro-F1이 fold별 결측 클래스로
왜곡되는 것을 피하기 위함). 학습/검증 분할 자체는 여전히 GroupKFold로 leak-free.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from wafer import features, modeling

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42
SMOTE_CAP = 1500  # 클래스당 오버샘플링 상한 (majority 'none'까지 완전 매칭하면 학습셋이 8배로 불어나 매 fold가 느려짐)


def log(msg: str) -> None:
    print(msg, flush=True)


def save_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "03_classification_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "per_class_f1"} for r in results])
    cols = ["model", "macro_f1", "weighted_f1", "minority_avg_f1", "none_f1", "fit_seconds_mean", "predict_ms_per_1k_mean"]
    table = table[[c for c in cols if c in table.columns]]
    table.to_csv(out_dir / "03_classification_benchmark.csv", index=False)


def smote_strategy_and_k(y):
    """각 소수 클래스를 SMOTE_CAP까지만 오버샘플링 (majority 완전매칭 방지).
    표본이 1개뿐인 클래스는 SMOTE 대상에서 제외(원본 그대로 유지) — SMOTE는
    최소 2개 이상 있어야 보간 가능. k_neighbors는 대상 클래스 중 가장 적은
    표본 수에 안전하게 맞춰 동적으로 낮춘다(최대 2, 최소 1)."""
    counts = Counter(y)
    majority = max(counts.values())
    targets = {cls: min(SMOTE_CAP, majority) for cls, cnt in counts.items() if 1 < cnt < SMOTE_CAP}
    if not targets:
        return {}, 1
    min_count = min(counts[c] for c in targets)
    k = max(1, min(2, min_count - 1))
    return targets, k


def load_data():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    return df, numeric_cols, categorical_cols


def oof_predict(pipeline_factory, X, y_enc, groups, n_splits=N_SPLITS, name=""):
    """pipeline_factory(y_train) -> sklearn/imblearn Pipeline (y_train을 받아 SMOTE
    타깃처럼 fold별로 달라지는 구성을 만들 수 있게 함; 안 쓰면 인자를 무시하면 됨)."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(y_enc), -1, dtype=int)
    fit_times, predict_times_per_1k = [], []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_enc, groups)):
        pipe = pipeline_factory(y_enc[tr_idx])
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr_idx], y_enc[tr_idx])
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred = pipe.predict(X.iloc[val_idx])
        dt = time.perf_counter() - t0
        predict_times_per_1k.append(dt / max(len(val_idx), 1) * 1000)

        oof_pred[val_idx] = pred
        log(f"  [{name}] final-eval fold {fold + 1}/{n_splits} fit={fit_times[-1]:.1f}s")
    assert (oof_pred >= 0).all(), "일부 행이 OOF 예측에서 누락됨"
    return oof_pred, float(np.mean(fit_times)), float(np.mean(predict_times_per_1k))


def summarize(name, y_true_enc, oof_pred, classes, fit_s, pred_ms, extra=None):
    report = classification_report(
        y_true_enc, oof_pred, target_names=classes, output_dict=True, zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    minority_classes = [c for c in classes if c != "none"]
    minority_f1 = float(np.mean([report[c]["f1-score"] for c in minority_classes]))
    summary = {
        "model": name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "minority_avg_f1": minority_f1,
        "none_f1": report["none"]["f1-score"],
        "fit_seconds_mean": fit_s,
        "predict_ms_per_1k_mean": pred_ms,
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
    }
    if extra:
        summary.update(extra)
    log(
        f"[{name}] DONE — macro-F1={macro_f1:.4f} weighted-F1={weighted_f1:.4f} "
        f"minority-avg-F1={minority_f1:.4f} fit={fit_s:.2f}s"
    )
    return summary


def main():
    df, numeric_cols, categorical_cols = load_data()
    X = df[numeric_cols + categorical_cols]
    le = LabelEncoder()
    y_enc = le.fit_transform(df["error_class"])
    classes = list(le.classes_)
    groups = df["group_id"]
    log(f"X shape {X.shape}, classes={classes}")
    log(f"class counts: {pd.Series(df['error_class']).value_counts().to_dict()}")

    out_dir = PROJECT_ROOT / "reports"
    results = []

    # 1) SMOTE + RandomForest (Optuna 튜닝, macro-F1 목적함수)
    def rf_smote_factory(p, y_train):
        strategy, k = smote_strategy_and_k(y_train)
        pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p)
        if not strategy:
            return Pipeline([("pre", pre), ("model", model)])
        return ImbPipeline(
            [
                ("pre", pre),
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k, sampling_strategy=strategy)),
                ("model", model),
            ]
        )

    def rf_smote_cv_f1(params, trial_no, n_trials):
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof = np.full(len(y_enc), -1, dtype=int)
        t0 = time.perf_counter()
        for tr_idx, val_idx in gkf.split(X, y_enc, groups):
            pipe = rf_smote_factory(params, y_enc[tr_idx])
            pipe.fit(X.iloc[tr_idx], y_enc[tr_idx])
            oof[val_idx] = pipe.predict(X.iloc[val_idx])
        f1 = f1_score(y_enc, oof, average="macro", zero_division=0)
        log(f"  [SMOTE+RF] trial {trial_no}/{n_trials} macro-F1={f1:.4f} ({time.perf_counter() - t0:.1f}s)")
        return f1

    def rf_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }

    n_trials_rf = 12
    log(f"[SMOTE+RF] tuning start ({n_trials_rf} trials x {N_SPLITS}-fold)")
    study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(
        lambda t: rf_smote_cv_f1(rf_space(t), t.number + 1, n_trials_rf), n_trials=n_trials_rf, show_progress_bar=False
    )
    best_rf = study_rf.best_params
    oof_pred, fit_s, pred_ms = oof_predict(
        lambda ytr: rf_smote_factory(best_rf, ytr), X, y_enc, groups, name="SMOTE+RF"
    )
    results.append(
        summarize(
            "SMOTE+RandomForest (Optuna)", y_enc, oof_pred, classes, fit_s, pred_ms,
            extra={"best_params": best_rf, "optuna_best_macro_f1": study_rf.best_value},
        )
    )
    save_results(results, out_dir)

    # 2) Cost-Sensitive LightGBM (class_weight=balanced, Optuna 튜닝)
    def lgbm_factory(p):
        pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
        return Pipeline(
            [
                ("pre", pre),
                ("model", LGBMClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, class_weight="balanced", **p
                )),
            ]
        )

    def lgbm_cv_f1(params, trial_no, n_trials):
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof = np.full(len(y_enc), -1, dtype=int)
        t0 = time.perf_counter()
        for tr_idx, val_idx in gkf.split(X, y_enc, groups):
            pipe = lgbm_factory(params)
            pipe.fit(X.iloc[tr_idx], y_enc[tr_idx])
            oof[val_idx] = pipe.predict(X.iloc[val_idx])
        f1 = f1_score(y_enc, oof, average="macro", zero_division=0)
        log(f"  [Cost-Sensitive LightGBM] trial {trial_no}/{n_trials} macro-F1={f1:.4f} ({time.perf_counter() - t0:.1f}s)")
        return f1

    def lgbm_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

    n_trials_lgbm = 15
    log(f"[Cost-Sensitive LightGBM] tuning start ({n_trials_lgbm} trials x {N_SPLITS}-fold)")
    study_lgbm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgbm.optimize(
        lambda t: lgbm_cv_f1(lgbm_space(t), t.number + 1, n_trials_lgbm), n_trials=n_trials_lgbm, show_progress_bar=False
    )
    best_lgbm = study_lgbm.best_params
    oof_pred, fit_s, pred_ms = oof_predict(
        lambda ytr: lgbm_factory(best_lgbm), X, y_enc, groups, name="Cost-Sensitive LightGBM"
    )
    results.append(
        summarize(
            "Cost-Sensitive LightGBM (Optuna)", y_enc, oof_pred, classes, fit_s, pred_ms,
            extra={"best_params": best_lgbm, "optuna_best_macro_f1": study_lgbm.best_value},
        )
    )
    save_results(results, out_dir)

    # 3) TabNet (딥러닝 tabular 모델, 고정 하이퍼파라미터 — CPU 환경 학습시간 고려해 epoch 축소)
    log(f"[TabNet] start ({N_SPLITS}-fold, no Optuna tuning — fixed architecture)")
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(y_enc), -1, dtype=int)
    fit_times, pred_times = [], []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_enc, groups)):
        pre_fold = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=True)
        Xtr = pre_fold.fit_transform(X.iloc[tr_idx], y_enc[tr_idx])
        Xval = pre_fold.transform(X.iloc[val_idx])
        Xtr = np.asarray(Xtr, dtype=np.float32)
        Xval = np.asarray(Xval, dtype=np.float32)
        ytr = y_enc[tr_idx]

        clf = TabNetClassifier(
            n_d=16, n_a=16, n_steps=3, seed=RANDOM_STATE, verbose=1,
            device_name="cpu",
        )
        t0 = time.perf_counter()
        clf.fit(
            Xtr, ytr, max_epochs=40, patience=8, batch_size=512, virtual_batch_size=128,
            eval_set=[(Xtr, ytr)], eval_metric=["balanced_accuracy"],
        )
        fit_s = time.perf_counter() - t0
        fit_times.append(fit_s)

        t0 = time.perf_counter()
        pred = clf.predict(Xval)
        dt = time.perf_counter() - t0
        pred_times.append(dt / max(len(val_idx), 1) * 1000)
        oof_pred[val_idx] = pred
        log(f"  [TabNet] fold {fold + 1}/{N_SPLITS} done, fit={fit_s:.1f}s, "
            f"best_epoch={clf.best_epoch if hasattr(clf, 'best_epoch') else 'n/a'}")

    results.append(
        summarize(
            "TabNet", y_enc, oof_pred, classes, float(np.mean(fit_times)), float(np.mean(pred_times)),
        )
    )
    save_results(results, out_dir)

    log("")
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "per_class_f1"} for r in results])
    cols = ["model", "macro_f1", "weighted_f1", "minority_avg_f1", "none_f1", "fit_seconds_mean", "predict_ms_per_1k_mean"]
    table = table[[c for c in cols if c in table.columns]]
    log(table.to_string(index=False))
    log("")
    log("per-class F1:")
    for r in results:
        log(f"{r['model']} {r['per_class_f1']}")


if __name__ == "__main__":
    main()
