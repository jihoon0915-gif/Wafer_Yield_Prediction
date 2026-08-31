"""전처리 완료 데이터(processed_final_feature_matrix.csv + final_preprocessor.pkl)를
바탕으로 '피처 선택 + 하이퍼파라미터 튜닝'을 한 파이프라인으로 수행하는 최종
모델링 스크립트. RandomForest/XGBoost/LightGBM 3종을 비교하고, 최고 성능 모델을
final_model.pkl로, 그 피처 목록을 final_selected_features.pkl로 저장한다.

핵심 설계 결정 (요청을 문자 그대로 따르지 않은 부분은 이유를 명시):

1. GroupKFold의 group을 Wafer_ID(group_id)가 아니라 **Lot_Num**으로 잡았다.
   전처리 단계(11번 스크립트)에서 이미 die-행을 웨이퍼 단위(1,704행, 1행=1웨이퍼)로
   축소했기 때문에, 이 시점에서 Wafer_ID로 그룹핑해도 그룹 크기가 전부 1이라 사실상
   일반 KFold와 동일해 의미가 없다. 이 프로젝트에서 실제로 아직 남아있는 리스크는
   "새로운 Lot에 대한 일반화"이고(다른 실험에서 진짜 Lot 단위 완전분리 시 R2가
   0.70->0.44로 떨어지는 걸 이미 확인함), 그래서 여기서는 Lot_Num을 그룹으로 써서
   더 엄격하고 정직한 성능 추정을 한다.

2. 피처 선택(RFECV)은 기본 설정 추정기(가벼운 하이퍼파라미터)로 먼저 돌리고, 하이퍼
   파라미터 튜닝은 선택된 피처에 대해 별도로 수행한다("동시에"를 문자 그대로 한 번의
   결합탐색으로 하면 41피처 x 그리드 조합 x GroupKFold가 조합폭발해 비현실적이라,
   선택 -> 튜닝 순차 파이프라인으로 구현하되 하나의 스크립트/함수 흐름으로 묶었다).

3. final_preprocessor.pkl(11번 스크립트, 전체 데이터로 fit)을 그대로 재사용해 원본
   컬럼 -> 수치 매트릭스 변환을 수행한다. 이 변환(중앙값대치+스케일+원핫)은 RFECV/
   GridSearchCV의 매 fold마다 다시 fit하지 않는 실용적 단순화다(비용 대비 리스크가
   낮음 -- 그룹 리스크는 여전히 GroupKFold로 매 단계 방어됨). 최종 성능 수치는 이
   단순화의 영향을 받을 수 있으므로 이 사실을 리포트에 명시한다.
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from wafer.final_pipeline import FeatureSelectorSlicer

RANDOM_STATE = 42
N_SPLITS = 5
RMSE_SCORER = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)

NUMERIC_FEATURES = [
    "resist_target", "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake",
    "time_HMDS_bake", "spin1", "spin2", "spin3", "photoresist_bake",
    "temp_softbake", "time_softbake",
    "Line_CD", "Wavelength", "Resolution", "Energy_Exposure",
    "Thin F2", "Thin F3", "Thin F4", "Temp_Etching", "Source_Power", "Selectivity",
    "etch_rate_stage1",
    "Flux60s", "Flux90s", "Flux160s", "Flux480s", "input_Energy",
    "Temp_implantation", "Furance_Temp", "RTA_Temp",
    "Temp_OXid", "ppm", "Pressure", "Oxid_time", "oxid_thickness_spec_gap",
    "oxidation_rate_nm_per_min",
    "oxidation_sentinel_flag", "etching_sentinel_flag", "ion_implant_sentinel_flag",
]
CATEGORICAL_FEATURES = ["type", "Vapor"]

# --------------------------------------------------------------------------
# 모델별: (선택단계용 가벼운 기본 추정기, 튜닝단계용 그리드)
# 그리드는 "공정 변동성에 강건"하도록 정규화 관련 파라미터(subsample/colsample/
# min_samples_leaf 등)를 반드시 포함한다 - 이 프로젝트에서 트리모델들이 train
# R2~0.99 vs val R2~0.6대로 과적합 경향이 뚜렷했던 것을 고려한 설계.
# --------------------------------------------------------------------------
MODEL_SPECS = {
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=150),
        {
            "n_estimators": [150, 300],
            "max_depth": [5, 10, 15],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", 0.7],
        },
    ),
    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, n_estimators=150),
        {
            "n_estimators": [150, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "reg_lambda": [0.1, 1.0],
        },
    ),
    "LightGBM": (
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, n_estimators=150),
        {
            "n_estimators": [150, 300],
            "num_leaves": [15, 31],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "reg_lambda": [0.0, 1.0],
        },
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_data():
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_final_feature_matrix.csv")
    preprocessor = joblib.load(PROJECT_ROOT / "models" / "final_preprocessor.pkl")

    X_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Target"].astype(float)
    groups = df["Lot_Num"]

    X = preprocessor.transform(X_raw)
    feature_names = np.array(preprocessor.get_feature_names_out())
    log(f"데이터 로드: {df.shape[0]}행, 변환후 {X.shape[1]}개 피처(그룹키: Lot_Num, {groups.nunique()}개 Lot)")
    return X, y, groups, feature_names, preprocessor


def select_features(name: str, base_estimator, X, y, groups, feature_names, gkf) -> tuple[np.ndarray, list[str]]:
    n_features = X.shape[1]
    min_select = max(5, n_features // 4)
    selector = RFECV(
        estimator=clone(base_estimator), step=2, min_features_to_select=min_select,
        cv=gkf, scoring="r2", n_jobs=-1,
    )
    selector.fit(X, y, groups=groups)
    mask = selector.support_
    selected = feature_names[mask].tolist()
    best_cv_score = float(np.max(selector.cv_results_["mean_test_score"]))
    log(f"[{name}] RFECV: {n_features}개 -> {mask.sum()}개 선택 (선택단계 최고 CV R2={best_cv_score:.4f})")
    return mask, selected


def tune_and_evaluate(name: str, base_estimator, grid: dict, X_sel, y, groups, gkf) -> dict:
    gs = GridSearchCV(clone(base_estimator), param_grid=grid, cv=gkf, scoring="r2", n_jobs=-1)
    gs.fit(X_sel, y, groups=groups)
    best_est, best_params = gs.best_estimator_, gs.best_params_
    log(f"[{name}] GridSearchCV best_params={best_params}")

    cv_res = cross_validate(
        best_est, X_sel, y, groups=groups, cv=gkf,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": RMSE_SCORER},
        n_jobs=-1,
    )
    result = {
        "model": name,
        "best_params": best_params,
        "r2_mean": float(np.mean(cv_res["test_r2"])), "r2_std": float(np.std(cv_res["test_r2"])),
        "mae_mean": float(-np.mean(cv_res["test_mae"])), "mae_std": float(np.std(cv_res["test_mae"])),
        "rmse_mean": float(-np.mean(cv_res["test_rmse"])), "rmse_std": float(np.std(cv_res["test_rmse"])),
        "fitted_best_estimator": best_est,
    }
    log(f"[{name}] 최종 CV: R2={result['r2_mean']:.4f}+-{result['r2_std']:.4f}  "
        f"MAE={result['mae_mean']:.2f}  RMSE={result['rmse_mean']:.2f}")
    return result


def write_report(results: list[dict], winner: dict, preprocessor, out_path: Path) -> None:
    lines = ["# 최종 통합 예측 모델링 보고서", ""]
    lines.append(
        "processed_final_feature_matrix.csv(H1~H7 검증 반영 전처리 완료 데이터, 1,704행 "
        "웨이퍼 단위)를 대상으로, 모델별 RFECV 피처 선택 + GridSearchCV 하이퍼파라미터 "
        "튜닝을 GroupKFold(group=Lot_Num, 5-fold)로 평가했다."
    )
    lines.append("")
    lines.append(
        "**검증 체계**: group을 Wafer_ID가 아니라 Lot_Num으로 설정했다 — 전처리 단계에서 "
        "이미 die-행을 웨이퍼 단위로 축소해(H2 대응) Wafer_ID 그룹은 전부 크기 1이라 "
        "의미가 없고, 실제 남은 리스크는 '새 Lot에 대한 일반화'이기 때문이다(별도 검증에서 "
        "진짜 Lot 완전분리 시 R²가 0.70→0.44로 하락함을 이미 확인)."
    )
    lines.append("")

    lines.append("## 모델별 비교")
    lines.append("")
    lines.append("| 모델 | 선택된 피처 수 | Best Hyperparameters | R² (mean±std) | MAE | RMSE |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['model']} | {len(r['selected_features'])} | "
            f"{', '.join(f'{k}={v}' for k, v in r['best_params'].items())} | "
            f"{r['r2_mean']:.4f}±{r['r2_std']:.4f} | {r['mae_mean']:.2f} | {r['rmse_mean']:.2f} |"
        )
    lines.append("")

    for r in results:
        lines.append(f"### {r['model']} 선택 피처 목록 ({len(r['selected_features'])}개)")
        lines.append("")
        lines.append(", ".join(f"`{f}`" for f in r["selected_features"]))
        lines.append("")

    lines.append("## 최종 선정 모델")
    lines.append("")
    lines.append(
        f"**{winner['model']}** (R²={winner['r2_mean']:.4f}±{winner['r2_std']:.4f}, "
        f"MAE={winner['mae_mean']:.2f}, RMSE={winner['rmse_mean']:.2f}) — "
        f"GroupKFold(Lot_Num) 기준 CV R² 최고."
    )
    lines.append(f"Best Hyperparameters: {winner['best_params']}")
    lines.append("")

    lines.append("## 한계 및 재현 시 주의")
    lines.append("")
    lines.append(
        "- `final_preprocessor.pkl`(중앙값대치+StandardScaler+원핫)은 전체 1,704행에 "
        "fit된 상태로 재사용했다 — RFECV/GridSearchCV의 매 fold 안에서 재적합하지 않는 "
        "실용적 단순화다. 그룹(Lot) 리스크는 모든 단계에서 GroupKFold로 방어되지만, "
        "이 단순화로 인해 보고된 R²가 아주 약간 낙관적일 수 있다.\n"
        "- 여기 보고된 R²는 Lot_Num 완전분리 기준이라, 이전 웨이퍼-단위 GroupKFold 결과"
        "(H1~H7 리포트, ~0.6~0.7)보다 낮게 나올 수 있다 — 더 엄격하고 실전에 가까운 "
        "추정치라 그렇다."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"저장 완료: {out_path}")


def main() -> None:
    X, y, groups, feature_names, preprocessor = load_data()
    gkf = GroupKFold(n_splits=N_SPLITS)

    results = []
    for name, (base_estimator, grid) in MODEL_SPECS.items():
        log(f"\n{'='*60}\n{name}\n{'='*60}")
        mask, selected = select_features(name, base_estimator, X, y, groups, feature_names, gkf)
        X_sel = X[:, mask]
        result = tune_and_evaluate(name, base_estimator, grid, X_sel, y, groups, gkf)
        result["selected_features"] = selected
        result["feature_mask"] = mask
        results.append(result)

    winner = max(results, key=lambda r: r["r2_mean"])
    log(f"\n최종 선정: {winner['model']} (R2={winner['r2_mean']:.4f})")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # 배포용 최종 파이프라인: 원본 컬럼 입력 -> (기존 전처리기) -> (선택 피처만 남기는 슬라이서) -> (튜닝된 최적 모델)
    final_pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("select", FeatureSelectorSlicer(winner["feature_mask"])),
            ("model", winner["fitted_best_estimator"]),
        ]
    )
    joblib.dump(final_pipeline, models_dir / "final_model.pkl")
    joblib.dump(winner["selected_features"], models_dir / "final_selected_features.pkl")
    log(f"저장 완료: {models_dir / 'final_model.pkl'} (원본 컬럼 입력 -> 예측 전체 파이프라인)")
    log(f"저장 완료: {models_dir / 'final_selected_features.pkl'} ({len(winner['selected_features'])}개 피처)")

    # 승자만이 아니라 3개 후보 전부를 각자의 전체 파이프라인(전처리+피처선택+튜닝모델)
    # 형태로 저장한다 -- 이후 해석(SHAP) 단계에서 XGBoost/LightGBM을 서로 비교하려면
    # 둘 다 학습된 상태로 다시 불러올 수 있어야 하는데, 이전 버전은 승자 하나만
    # 저장해서 XGBoost 쪽은 재현이 불가능했다.
    all_candidates = {}
    for r in results:
        pipe = Pipeline(
            [
                ("prep", preprocessor),
                ("select", FeatureSelectorSlicer(r["feature_mask"])),
                ("model", r["fitted_best_estimator"]),
            ]
        )
        all_candidates[r["model"]] = {
            "pipeline": pipe,
            "selected_features": r["selected_features"],
            "best_params": r["best_params"],
            "r2_mean": r["r2_mean"], "r2_std": r["r2_std"],
            "mae_mean": r["mae_mean"], "rmse_mean": r["rmse_mean"],
        }
    joblib.dump(all_candidates, models_dir / "all_candidate_models.pkl")
    log(f"저장 완료: {models_dir / 'all_candidate_models.pkl'} ({list(all_candidates.keys())}, 각 모델별 전체 파이프라인+메타데이터)")

    write_report(results, winner, preprocessor, PROJECT_ROOT / "reports" / "final_modeling_report.md")


if __name__ == "__main__":
    main()
