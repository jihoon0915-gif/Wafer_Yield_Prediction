"""검증: 기존 GroupKFold(group=Lot_Num+Wafer_Num, 웨이퍼 단위 분리)를
Lot_Num 단독(로트 단위 완전 분리)으로 바꿔도 챔피언 LightGBM의 R²가 유지되는지 확인.

동기: 자기소개서 초안에서 "Lot 단위로 완전히 분리"라고 썼는데, 실제 group_id는
Lot_Num+Wafer_Num(웨이퍼 단위)라 각 fold에 32개 Lot이 전부 train/val 양쪽에
나타남(실측 확인). 이 스크립트는 그 초안 문구를 사실로 만들 수 있는지 —
즉 진짜 Lot 단위로 쪼개도 성능이 버티는지 — 재튜닝 없이(기존 best_params 그대로)
검증한다.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from wafer import features, modeling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SPLITS = 5
RANDOM_STATE = 42

BEST_PARAMS = {
    "n_estimators": 517,
    "num_leaves": 66,
    "learning_rate": 0.01855998084649059,
    "subsample": 0.6733618039413735,
    "colsample_bytree": 0.7216968971838151,
    "reg_lambda": 0.12561043700013558,
}


def main() -> None:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    X = df[numeric_cols + categorical_cols]
    y = df["Target"].astype(float)

    def make_pipeline():
        pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
        model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **BEST_PARAMS)
        return Pipeline([("pre", pre), ("model", model)])

    for label, groups in [
        ("기존: group=Lot_Num+Wafer_Num (웨이퍼 단위)", df["group_id"]),
        ("검증대상: group=Lot_Num 단독 (로트 단위 완전분리)", df["Lot_Num"].astype(str)),
    ]:
        gkf = GroupKFold(n_splits=N_SPLITS)
        r2s, maes = [], []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            # sanity: 로트 단위 분리 시 train/val Lot이 겹치지 않는지 확인
            tr_lots = set(df.iloc[tr_idx]["Lot_Num"].unique())
            val_lots = set(df.iloc[val_idx]["Lot_Num"].unique())
            overlap = len(tr_lots & val_lots)

            pipe = make_pipeline()
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = pipe.predict(X.iloc[val_idx])
            r2 = r2_score(y.iloc[val_idx], pred)
            mae = mean_absolute_error(y.iloc[val_idx], pred)
            r2s.append(r2)
            maes.append(mae)
            print(f"  [{label}] fold {fold+1}: R2={r2:.4f} MAE={mae:.2f} "
                  f"val_lots={len(val_lots)} train/val Lot 겹침={overlap}")
        print(f"[{label}] mean R2={np.mean(r2s):.4f} (std={np.std(r2s):.4f})  mean MAE={np.mean(maes):.2f}")
        print()


if __name__ == "__main__":
    main()
