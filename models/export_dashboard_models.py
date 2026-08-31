"""대시보드용 산출물 생성: 챔피언 회귀·분류 파이프라인을 전체 데이터로 재학습해
joblib로 저장하고, 슬라이더 범위/기본값/스윗스팟 임계값/웨이퍼맵 조회 테이블을
만든다. FastAPI(api/main.py)가 이 산출물만 로드해서 서빙한다.

슬라이더 10개 설계 근거:
- Thin F2/F3/F4: SHAP 1·2·4위. "레시피"가 아니라 식각 결과 실측값이지만,
  Temp_Etching/Source_Power만으로는 이 값을 전혀 못 맞춰서(R²≈0, 자체 확인)
  레시피 슬라이더로 대체할 근거가 없었다 — 대신 실측값을 직접 조작해 3절
  스윗스팟 스토리를 체험하게 한다.
- Temp_OXid/Oxid_time/ppm: 4단계 최적화에서 세 알고리즘이 공통으로 움직인
  레버리지 포인트, SHAP 3위(Temp_OXid).
- Energy_Exposure/UV_type: 노광, 3절 confound 스토리(재보정된 약한 신호).
- input_Energy/spin3: SHAP 5·11위, 이온주입·포토 대표.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder

from wafer import features, integrate, modeling

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).resolve().parent

SLIDER_FEATURES = [
    "Thin F2", "Thin F3", "Thin F4",
    "Temp_OXid", "Oxid_time", "ppm",
    "Energy_Exposure", "input_Energy", "spin3",
]
SLIDER_CATEGORICAL = ["UV_type"]


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    X = df[numeric_cols + categorical_cols]

    # ---------------------------------------------------------------
    # 1) 챔피언 회귀 파이프라인 (LightGBM, 전체데이터 재학습 — 배포용)
    # ---------------------------------------------------------------
    reg_ckpt = json.load(open(PROJECT_ROOT / "reports" / "02_regression_benchmark.json", encoding="utf-8"))
    reg_best = max(reg_ckpt, key=lambda r: r["cv_val_score_mean"])
    log(f"regression champion: {reg_best['model']} val_R2={reg_best['cv_val_score_mean']:.4f}")

    reg_pipe = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
    from sklearn.pipeline import Pipeline
    reg_pipeline = Pipeline([
        ("pre", reg_pipe),
        ("model", LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **reg_best["best_params"])),
    ])
    reg_pipeline.fit(X, df["Target"])
    joblib.dump(reg_pipeline, MODELS_DIR / "regression_champion.joblib")
    log("saved regression_champion.joblib")

    # ---------------------------------------------------------------
    # 2) 챔피언 분류 파이프라인 (Cost-Sensitive LightGBM)
    # ---------------------------------------------------------------
    clf_ckpt = json.load(open(PROJECT_ROOT / "reports" / "03_classification_benchmark.json", encoding="utf-8"))
    clf_best = max(clf_ckpt, key=lambda r: r["macro_f1"])
    log(f"classification champion: {clf_best['model']} macro_F1={clf_best['macro_f1']:.4f}")

    le = LabelEncoder()
    y_enc = le.fit_transform(df["error_class"])
    clf_pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
    clf_pipeline = Pipeline([
        ("pre", clf_pre),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
                                  class_weight="balanced", **clf_best["best_params"])),
    ])
    clf_pipeline.fit(X, y_enc)
    joblib.dump(clf_pipeline, MODELS_DIR / "classification_champion.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    log("saved classification_champion.joblib + label_encoder.joblib")

    # ---------------------------------------------------------------
    # 3) 슬라이더 범위(5~95th pct, 외삽 방지) + 스윗스팟 임계값 + 기본값
    # ---------------------------------------------------------------
    slider_ranges = {}
    for col in SLIDER_FEATURES:
        s = df[col].dropna()
        slider_ranges[col] = {
            "min": round(float(s.quantile(0.05)), 2),
            "max": round(float(s.quantile(0.95)), 2),
            "median": round(float(s.median()), 2),
            "q20": round(float(s.quantile(0.20)), 2),
            "q80": round(float(s.quantile(0.80)), 2),
        }
    slider_ranges["UV_type"] = {"options": sorted(df["UV_type"].dropna().unique().tolist()), "default": "H"}

    # 슬라이더 밖 나머지 피처는 데이터셋 중앙값/최빈값으로 고정(그대로 held-out)
    feature_defaults = {}
    for col in numeric_cols:
        if col in SLIDER_FEATURES:
            continue
        feature_defaults[col] = round(float(df[col].median()), 4) if df[col].notna().any() else 0.0
    for col in categorical_cols:
        if col in SLIDER_CATEGORICAL:
            continue
        feature_defaults[col] = df[col].mode().iloc[0]

    with open(MODELS_DIR / "slider_ranges.json", "w", encoding="utf-8") as f:
        json.dump(slider_ranges, f, ensure_ascii=False, indent=2)
    with open(MODELS_DIR / "feature_defaults.json", "w", encoding="utf-8") as f:
        json.dump(feature_defaults, f, ensure_ascii=False, indent=2)
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"numeric": numeric_cols, "categorical": categorical_cols}, f, ensure_ascii=False, indent=2)
    log(f"saved slider_ranges.json ({len(slider_ranges)} sliders), "
        f"feature_defaults.json ({len(feature_defaults)} held-out features)")

    # ---------------------------------------------------------------
    # 4) 웨이퍼맵 조회 테이블 (실측 매칭용) — group_id별 대표 1행 + 파싱된 맵
    # ---------------------------------------------------------------
    merged = df.copy()
    merged, maps_by_no_die, maps_by_group, wm_stats = integrate.process_wafer_maps(merged)
    log(f"wafer_map parse stats: {wm_stats['n_groups_recovered']}/{wm_stats['n_groups_total']} groups recovered")

    df_sorted = merged.sort_values(SLIDER_FEATURES[:3], key=lambda s: s.isna(), kind="stable")
    wafers = df_sorted.drop_duplicates(subset="group_id", keep="first").copy()

    lookup_rows = []
    for _, row in wafers.iterrows():
        gid = row["group_id"]
        if gid not in maps_by_group:
            continue
        lookup_rows.append({
            "group_id": gid,
            "Lot_Num": int(row["Lot_Num"]),
            "Wafer_Num": int(row["Wafer_Num"]),
            "Target": float(row["Target"]),
            "error_class": row["error_class"],
            "wafer_map": maps_by_group[gid].tolist(),
        })
    with open(MODELS_DIR / "wafer_lookup.json", "w", encoding="utf-8") as f:
        json.dump(lookup_rows, f)
    log(f"saved wafer_lookup.json ({len(lookup_rows)} wafers with parseable maps)")

    log("DONE")


if __name__ == "__main__":
    main()
