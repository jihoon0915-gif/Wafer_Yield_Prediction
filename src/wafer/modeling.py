"""회귀(Target)/분류(Error_message) 벤치마크에 공통으로 쓰는 피처 선택, 전처리,
GroupKFold 기반 CV 러너.

모든 train/test 분할·CV는 반드시 group_id(Lot_Num_Wafer_Num) 기준 GroupKFold로
수행한다 (원 프로젝트 CLAUDE.md 규칙 — 같은 웨이퍼의 반복 행이 train/val에 나뉘면
데이터 누수).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABEL_COLS = ["Target", "Error_message", "error_class", "Wafer_map", "wafer_map_truncated"]
ID_COLS = ["No_Die", "Lot_Num", "Wafer_Num", "Datetime", "group_id"]
CONSTANT_CATEGORICAL = ["process 2", "Process 2-1", "Process 3", "process4", "process"]
CATEGORICAL_FEATURES = ["UV_type", "type", "Vapor"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = set(LABEL_COLS + ID_COLS + CONSTANT_CATEGORICAL)
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    numeric = [
        c
        for c in df.columns
        if c not in exclude and c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric, categorical


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)]
    )


@dataclass
class FoldResult:
    fold: int
    train_score: float
    val_score: float
    val_mae: float | None
    fit_seconds: float
    predict_seconds_per_1k: float
    n_train: int
    n_val: int


@dataclass
class CVResult:
    model_name: str
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict:
        val_scores = [f.val_score for f in self.folds]
        train_scores = [f.train_score for f in self.folds]
        maes = [f.val_mae for f in self.folds if f.val_mae is not None]
        return {
            "model": self.model_name,
            "cv_train_score_mean": float(np.mean(train_scores)),
            "cv_val_score_mean": float(np.mean(val_scores)),
            "cv_val_score_std": float(np.std(val_scores)),
            "overfit_gap": float(np.mean(train_scores) - np.mean(val_scores)),
            "cv_val_mae_mean": float(np.mean(maes)) if maes else None,
            "fit_seconds_mean": float(np.mean([f.fit_seconds for f in self.folds])),
            "predict_ms_per_1k_mean": float(
                np.mean([f.predict_seconds_per_1k for f in self.folds]) * 1000
            ),
        }


def run_group_cv(
    model_name: str,
    pipeline_factory,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scorer,
    n_splits: int = 5,
    mae_fn=None,
) -> CVResult:
    gkf = GroupKFold(n_splits=n_splits)
    result = CVResult(model_name=model_name)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        pipe = pipeline_factory()
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        fit_seconds = time.perf_counter() - t0

        train_pred = pipe.predict(X.iloc[tr_idx])
        t0 = time.perf_counter()
        val_pred = pipe.predict(X.iloc[val_idx])
        predict_seconds = time.perf_counter() - t0
        predict_seconds_per_1k = predict_seconds / max(len(val_idx), 1) * 1000

        train_score = scorer(y.iloc[tr_idx], train_pred)
        val_score = scorer(y.iloc[val_idx], val_pred)
        val_mae = mae_fn(y.iloc[val_idx], val_pred) if mae_fn else None

        result.folds.append(
            FoldResult(
                fold=fold,
                train_score=train_score,
                val_score=val_score,
                val_mae=val_mae,
                fit_seconds=fit_seconds,
                predict_seconds_per_1k=predict_seconds_per_1k,
                n_train=len(tr_idx),
                n_val=len(val_idx),
            )
        )
    return result
