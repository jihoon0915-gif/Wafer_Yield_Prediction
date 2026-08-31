"""2단계 보강: 챔피언 분류 모델(Cost-Sensitive LightGBM)의 혼동행렬.

03_classification_benchmark.py는 OOF 예측으로 macro-F1 등 요약 지표만 저장하고
개별 예측/정답 쌍은 남기지 않았음. 이미 찾아둔 best_params로 동일한
GroupKFold(5, group=group_id) OOF 예측을 재현해 8x8 혼동행렬을 만든다
(하이퍼파라미터 재탐색 없음 — 02d_champion_significance_test.py와 동일한 원칙).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from wafer import features, modeling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    X = df[numeric_cols + categorical_cols]

    le = LabelEncoder()
    y_enc = le.fit_transform(df["error_class"])
    classes = list(le.classes_)
    groups = df["group_id"]

    bench = json.load(open(PROJECT_ROOT / "reports" / "03_classification_benchmark.json", encoding="utf-8"))
    champion = next(r for r in bench if r["model"] == "Cost-Sensitive LightGBM (Optuna)")
    best_params = champion["best_params"]
    print(f"champion best_params (reused, no retuning): {best_params}")

    def make_pipeline():
        pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
        model = LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, class_weight="balanced", **best_params
        )
        return Pipeline([("pre", pre), ("model", model)])

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(y_enc), -1, dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_enc, groups)):
        pipe = make_pipeline()
        pipe.fit(X.iloc[tr_idx], y_enc[tr_idx])
        oof_pred[val_idx] = pipe.predict(X.iloc[val_idx])
        print(f"fold {fold + 1}/{N_SPLITS} done")
    assert (oof_pred >= 0).all()

    macro_f1 = f1_score(y_enc, oof_pred, average="macro", zero_division=0)
    print(f"macro-F1 sanity check: {macro_f1:.4f} (03_classification_benchmark.json: {champion['macro_f1']:.4f})")

    acc = accuracy_score(y_enc, oof_pred)
    report = classification_report(y_enc, oof_pred, target_names=classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.loc["accuracy", ["precision", "recall", "f1-score"]] = [acc, acc, acc]
    report_df.loc["accuracy", "support"] = len(y_enc)
    out_dir = PROJECT_ROOT / "reports"
    report_df.to_csv(out_dir / "03_classification_report.csv")
    print(f"\noverall OOF accuracy: {acc:.4f}\n")
    print(report_df.round(4).to_string())

    cm = confusion_matrix(y_enc, oof_pred, labels=range(len(classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true \\ predicted"

    out_dir = PROJECT_ROOT / "reports"
    cm_df.to_csv(out_dir / "03_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    cm_norm_df = pd.DataFrame(np.round(cm_norm, 4), index=classes, columns=classes)
    cm_norm_df.index.name = "true \\ predicted"
    cm_norm_df.to_csv(out_dir / "03_confusion_matrix_normalized.csv")

    print("\n[raw counts]")
    print(cm_df.to_string())
    print("\n[row-normalized = recall per true class]")
    print(cm_norm_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized) — Cost-Sensitive LightGBM, OOF GroupKFold(5)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="recall (row-normalized)")
    fig.tight_layout()
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "03_confusion_matrix.png", dpi=150)
    print(f"\nsaved: reports/03_confusion_matrix.csv, reports/03_confusion_matrix_normalized.csv, reports/figures/03_confusion_matrix.png")


if __name__ == "__main__":
    main()
