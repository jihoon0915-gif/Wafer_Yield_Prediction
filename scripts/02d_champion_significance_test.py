"""한계 ④ 수정: '챔피언' 선정이 fold 표준편차보다 작은 차이(R² 0.685~0.699)로
결정된 것이 통계적으로 근거가 약하다는 지적을 검증한다. GroupKFold(5)는
동일한 X/y/groups에 대해 결정적(deterministic)이므로 — 셔플을 안 하기 때문에
같은 입력이면 항상 같은 fold 분할이 나온다 — 모든 모델이 정확히 같은 5개
fold를 공유한다. 따라서 독립표본이 아니라 **대응표본(paired)** 비교가 맞다:
fold별 R² 차이에 paired t-test / Wilcoxon을 적용해 LightGBM이 다른 상위
모델들과 통계적으로 구별되는지 확인한다. 재튜닝 없이 이미 찾은 best_params로
5-fold 재평가만 수행(빠름).
"""

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

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

from wafer import modeling


def main():
    df, numeric_cols, categorical_cols = reg_bench.load_data()
    X = df[numeric_cols + categorical_cols]
    y = df["Target"]
    groups = df["group_id"]

    checkpoint = json.load(open(PROJECT_ROOT / "reports" / "02_regression_benchmark.json", encoding="utf-8"))
    by_name = {r["model"]: r for r in checkpoint}

    factories = {
        "RandomForest (Optuna)": lambda p: reg_bench.make_pipeline(
            reg_bench.RandomForestRegressor(random_state=reg_bench.RANDOM_STATE, n_jobs=-1, **p),
            numeric_cols, categorical_cols, scale=False),
        "XGBoost (Optuna)": lambda p: reg_bench.make_pipeline(
            reg_bench.XGBRegressor(random_state=reg_bench.RANDOM_STATE, n_jobs=-1, tree_method="hist", **p),
            numeric_cols, categorical_cols, scale=False),
        "LightGBM (Optuna)": lambda p: reg_bench.make_pipeline(
            reg_bench.LGBMRegressor(random_state=reg_bench.RANDOM_STATE, n_jobs=-1, verbosity=-1, **p),
            numeric_cols, categorical_cols, scale=False),
        "CatBoost (Optuna)": lambda p: reg_bench.make_pipeline(
            reg_bench.CatBoostRegressor(random_state=reg_bench.RANDOM_STATE, verbose=False,
                                         allow_writing_files=False, thread_count=-1, **p),
            numeric_cols, categorical_cols, scale=False),
    }

    per_fold = {}
    for name, factory in factories.items():
        params = by_name[name]["best_params"]
        cv = reg_bench.modeling.run_group_cv(
            name, lambda p=params, f=factory: f(p), X, y, groups,
            scorer=r2_score, mae_fn=mean_absolute_error, n_splits=5,
        )
        scores = [fr.val_score for fr in cv.folds]
        per_fold[name] = scores
        reg_bench.log(f"{name}: per-fold R2 = {[round(s, 4) for s in scores]}, mean={np.mean(scores):.4f}")

    champion = max(per_fold, key=lambda k: np.mean(per_fold[k]))
    reg_bench.log(f"\n최고 평균 R2 모델: {champion} (mean={np.mean(per_fold[champion]):.4f})")
    reg_bench.log("\n=== paired 비교 (동일 GroupKFold 5-fold, champion 대비) ===")

    sig_results = []
    for name, scores in per_fold.items():
        if name == champion:
            continue
        champ_scores = per_fold[champion]
        diffs = np.array(champ_scores) - np.array(scores)
        t_stat, p_t = stats.ttest_rel(champ_scores, scores)
        try:
            w_stat, p_w = stats.wilcoxon(champ_scores, scores)
        except ValueError:
            w_stat, p_w = float("nan"), float("nan")
        reg_bench.log(
            f"{champion} vs {name}: mean diff={diffs.mean():+.4f} (fold별: {[round(d,4) for d in diffs]}), "
            f"paired t p={p_t:.3f}, wilcoxon p={p_w:.3f} "
            f"-> {'통계적으로 유의한 차이' if p_t < 0.05 else '유의하지 않음(사실상 동률)'}"
        )
        sig_results.append({
            "champion": champion, "competitor": name, "mean_diff": float(diffs.mean()),
            "fold_diffs": diffs.tolist(), "paired_t_p": float(p_t), "wilcoxon_p": float(p_w),
            "significant_at_0.05": bool(p_t < 0.05),
        })

    with open(PROJECT_ROOT / "reports" / "02d_champion_significance.json", "w", encoding="utf-8") as f:
        json.dump({"champion": champion, "per_fold_scores": per_fold, "comparisons": sig_results}, f,
                   ensure_ascii=False, indent=2)
    reg_bench.log("\n산출물: reports/02d_champion_significance.json")


if __name__ == "__main__":
    main()
