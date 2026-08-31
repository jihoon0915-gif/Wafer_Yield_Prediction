"""3단계: XAI 기반 원인 분석 — SHAP, PDP, ICE로 Target(결함 다이 수)에 영향을 주는
변수의 방향성/문턱값을 규명한다. 특히 식각 잔막 두께(Thin F2/F3/F4)와 노광 파장
(UV_type: H vs G/I)의 영향을 정량화한다.

- SHAP/PDP/ICE는 회귀 챔피언 LightGBM(2단계 벤치마크 1위, best_params 재사용)을
  전체 데이터로 재학습한 모델을 설명한다. 이 모델의 '일반화 성능'은 이미
  02_regression_benchmark에서 GroupKFold로 정직하게 측정했으므로(val R²≈0.70),
  여기서는 그 모델이 데이터를 '어떻게' 쓰는지 해석하는 데 집중한다.
- 분류(Error_message) 챔피언은 gain 기반 feature importance만 가볍게 곁들여
  회귀 결과와 같은 변수가 패턴 분류에도 지배적인지 교차 확인한다.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

from wafer import features, modeling

RANDOM_STATE = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def load_champion_params():
    reg = json.load(open(PROJECT_ROOT / "reports" / "02_regression_benchmark.json", encoding="utf-8"))
    reg_best = max(reg, key=lambda r: r["cv_val_score_mean"])
    clf = json.load(open(PROJECT_ROOT / "reports" / "03_classification_benchmark.json", encoding="utf-8"))
    clf_best = max(clf, key=lambda r: r["macro_f1"])
    log(f"regression champion: {reg_best['model']} (val R2={reg_best['cv_val_score_mean']:.4f}) "
        f"params={reg_best['best_params']}")
    log(f"classification champion: {clf_best['model']} (macro-F1={clf_best['macro_f1']:.4f}) "
        f"params={clf_best['best_params']}")
    return reg_best["best_params"], clf_best["best_params"]


def main():
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)
    numeric_cols, categorical_cols = modeling.get_feature_columns(df)
    X = df[numeric_cols + categorical_cols]
    y_target = df["Target"]

    reg_params, clf_params = load_champion_params()

    # ---------------------------------------------------------------
    # 1) 회귀 챔피언(LightGBM) 전체 데이터 재학습 + SHAP
    # ---------------------------------------------------------------
    pre = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
    Xt = pre.fit_transform(X, y_target)
    feat_names = pre.get_feature_names_out()
    Xt_df = pd.DataFrame(np.asarray(Xt), columns=feat_names, index=X.index)

    reg_model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **reg_params)
    reg_model.fit(Xt_df, y_target)
    log(f"[SHAP] regression model refit on full data (n={len(X)}), computing TreeExplainer...")

    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer.shap_values(Xt_df)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_names).sort_values(ascending=False)
    top20 = mean_abs_shap.head(20)
    log("[SHAP] top 20 features by mean|SHAP|:")
    for name, val in top20.items():
        log(f"  {name}: {val:.3f}")
    top20.to_csv(PROJECT_ROOT / "reports" / "04_shap_top_features.csv", header=["mean_abs_shap"])

    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, Xt_df, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "04_shap_summary.png", dpi=120)
    plt.close()

    # 상위 3개 식각 잔막 변수 dependence plot
    for col in ["Thin F2", "Thin F3", "Thin F4"]:
        if col in feat_names:
            plt.figure(figsize=(6, 4))
            shap.dependence_plot(col, shap_values, Xt_df, interaction_index=None, show=False)
            plt.tight_layout()
            safe = col.replace(" ", "_")
            plt.savefig(fig_dir / f"04_shap_dependence_{safe}.png", dpi=120)
            plt.close()

    # ---------------------------------------------------------------
    # 2) PDP + ICE (Thin F2/F3/F4, Wavelength) — 파이프라인 전체를 estimator로 사용
    # ---------------------------------------------------------------
    from sklearn.pipeline import Pipeline
    full_pipe = Pipeline([("pre", modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)),
                           ("model", LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **reg_params))])
    full_pipe.fit(X, y_target)

    pdp_features = [c for c in ["Thin F2", "Thin F3", "Thin F4", "Wavelength"] if c in X.columns]
    fig, ax = plt.subplots(1, len(pdp_features), figsize=(5 * len(pdp_features), 4))
    PartialDependenceDisplay.from_estimator(
        full_pipe, X, features=pdp_features, kind="both",
        subsample=300, random_state=RANDOM_STATE, ax=ax if len(pdp_features) > 1 else [ax],
        ice_lines_kw={"alpha": 0.1, "linewidth": 0.5}, pd_line_kw={"color": "red", "linewidth": 2},
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "04_pdp_ice_thinfilm_wavelength.png", dpi=120)
    plt.close()
    log("[PDP/ICE] saved reports/figures/04_pdp_ice_thinfilm_wavelength.png")

    # ---------------------------------------------------------------
    # 3) 잔막(Thin F2/F3/F4) Sweet-spot / Control Window 수치화
    #    (원 대화정리 문서의 방법론을 센티널 보정된 데이터로 재현)
    # ---------------------------------------------------------------
    thin_cols = ["Thin F2", "Thin F3", "Thin F4"]
    sub = df[thin_cols + ["Target", "error_class"]].dropna(subset=thin_cols)
    is_defect = (sub["error_class"] != "none").astype(int)

    window_rows = []
    for col in thin_cols:
        q20, q80 = sub[col].quantile([0.2, 0.8])
        bottom = sub[sub[col] <= q20]
        top = sub[sub[col] >= q80]
        window_rows.append({
            "variable": col,
            "q20": q20, "q80": q80,
            "bottom20_mean_target": bottom["Target"].mean(),
            "bottom20_defect_rate_pct": 100 * is_defect[bottom.index].mean(),
            "bottom20_n": len(bottom),
            "top20_mean_target": top["Target"].mean(),
            "top20_defect_rate_pct": 100 * is_defect[top.index].mean(),
            "top20_n": len(top),
        })
    window_df = pd.DataFrame(window_rows)
    log("[Control Window] Thin F2/F3/F4 하위20%(sweet spot) vs 상위20%(위험구간):")
    log(window_df.to_string(index=False))
    window_df.to_csv(PROJECT_ROOT / "reports" / "04_control_window_thinfilm.csv", index=False)

    # 세 변수 모두 양호(하위20%) vs 모두 불량(상위20%) 동시 구간
    all_good = sub[(sub["Thin F2"] <= sub["Thin F2"].quantile(0.2)) &
                   (sub["Thin F3"] <= sub["Thin F3"].quantile(0.2)) &
                   (sub["Thin F4"] <= sub["Thin F4"].quantile(0.2))]
    all_bad = sub[(sub["Thin F2"] >= sub["Thin F2"].quantile(0.8)) &
                  (sub["Thin F3"] >= sub["Thin F3"].quantile(0.8)) &
                  (sub["Thin F4"] >= sub["Thin F4"].quantile(0.8))]
    log(f"[Control Window] 3개 변수 모두 하위20% (N={len(all_good)}): "
        f"평균 Target={all_good['Target'].mean():.1f}, 불량률={100*is_defect[all_good.index].mean():.1f}%")
    log(f"[Control Window] 3개 변수 모두 상위20% (N={len(all_bad)}): "
        f"평균 Target={all_bad['Target'].mean():.1f}, 불량률={100*is_defect[all_bad.index].mean():.1f}%")

    # ---------------------------------------------------------------
    # 4) UV_type(H-line vs G/I-line) 통계 검정
    # ---------------------------------------------------------------
    groups_uv = [df.loc[df["UV_type"] == t, "Target"] for t in ["H", "G", "I"]]
    f_stat, p_anova = stats.f_oneway(*groups_uv)
    h = df.loc[df["UV_type"] == "H", "Target"]
    gi = df.loc[df["UV_type"].isin(["G", "I"]), "Target"]
    t_stat, p_ttest = stats.ttest_ind(h, gi, equal_var=False)
    log(f"[UV_type] ANOVA(H/G/I) F={f_stat:.2f}, p={p_anova:.2e}")
    log(f"[UV_type] H(n={len(h)}, mean={h.mean():.1f}) vs G/I(n={len(gi)}, mean={gi.mean():.1f}): "
        f"Welch t={t_stat:.2f}, p={p_ttest:.2e}, 차이={100*(h.mean()/gi.mean()-1):.1f}%")

    plt.figure(figsize=(5, 4))
    df.boxplot(column="Target", by="UV_type", ax=plt.gca())
    plt.title("Target by UV_type")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_target_by_uvtype.png", dpi=120)
    plt.close()

    # ---------------------------------------------------------------
    # 5) 분류 챔피언(Cost-Sensitive LightGBM) gain-based importance (경량 교차확인)
    # ---------------------------------------------------------------
    le = LabelEncoder()
    y_err = le.fit_transform(df["error_class"])
    pre_c = modeling.build_preprocessor(numeric_cols, categorical_cols, scale=False)
    Xt_c = pre_c.fit_transform(X, y_err)
    feat_names_c = pre_c.get_feature_names_out()
    clf_model = LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, class_weight="balanced", **clf_params
    )
    clf_model.fit(Xt_c, y_err)
    gain_imp = pd.Series(clf_model.booster_.feature_importance(importance_type="gain"), index=feat_names_c)
    gain_imp = (gain_imp / gain_imp.sum() * 100).sort_values(ascending=False)
    log("[Classifier] Cost-Sensitive LightGBM gain-based importance top 15:")
    for name, val in gain_imp.head(15).items():
        log(f"  {name}: {val:.2f}%")
    gain_imp.head(20).to_csv(PROJECT_ROOT / "reports" / "04_classifier_gain_importance.csv", header=["gain_pct"])

    log("")
    log("XAI 분석 완료. 산출물: reports/04_shap_top_features.csv, "
        "04_control_window_thinfilm.csv, 04_classifier_gain_importance.csv, "
        "reports/figures/04_*.png")


if __name__ == "__main__":
    main()
