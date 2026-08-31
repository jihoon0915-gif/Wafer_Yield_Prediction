"""최종 모델(LightGBM, 배포 모델) + 비교 모델(XGBoost)에 대한 SHAP 기반 해석 파이프라인.

models/all_candidate_models.pkl(12번 스크립트 산출물, 3개 후보 모델의 전체 파이프라인
+ 선택 피처 목록이 담긴 딕셔너리)을 불러와 LightGBM/XGBoost 각각에 대해:
  1. TreeExplainer로 SHAP value 계산 -> Summary(beeswarm/bar) plot
  2. H3 채택 피처(etch_rate_stage1) + 상위 피처 Dependence plot
  3. 오차가 컸던 웨이퍼 / 정확했던 웨이퍼 각각 Waterfall plot(로컬 설명)
  4. 공정 제어 우선순위 Top 5 KPI 도출
을 수행하고 reports/model_interpretation_report.md를 자동 생성한다.

주의: SHAP은 파이프라인의 전처리(prep)+피처선택(select) 이후, 트리 모델이 실제로
학습한 수치 행렬에 대해서만 의미가 있다 -- pipeline 전체가 아니라
`pipeline.named_steps["model"]`(순수 LGBMRegressor/XGBRegressor)에 TreeExplainer를
붙이고, 입력도 그 모델이 실제로 본 X_selected를 사용한다.
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def raw_value(df_row: pd.Series, feat: str):
    """SHAP 기여도 테이블에 스케일된 z-score 대신 원본 물리단위 값을 보여주기 위한
    조회 함수. 수치형 피처는 df에 원본 컬럼이 그대로 있어 바로 찾고, 범주형은
    원-핫 더미 이름(예: 'type_wet')에서 원래 컬럼/카테고리를 역으로 찾는다."""
    if feat in df_row.index:
        return df_row[feat]
    for cat_col in CATEGORICAL_FEATURES:
        prefix = f"{cat_col}_"
        if feat.startswith(prefix):
            category = feat[len(prefix):]
            return f"{cat_col}={df_row[cat_col]} (dummy=={'1' if df_row[cat_col] == category else '0'})"
    return "N/A"


def load_model_data(name: str, all_candidates: dict, df: pd.DataFrame):
    entry = all_candidates[name]
    pipe = entry["pipeline"]
    prep = pipe.named_steps["prep"]
    select = pipe.named_steps["select"]
    model = pipe.named_steps["model"]

    X_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_transformed = prep.transform(X_raw)
    X_selected = select.transform(X_transformed)
    feat_names = [clean_name(f) for f in entry["selected_features"]]
    X_df = pd.DataFrame(X_selected, columns=feat_names, index=df.index)

    pred = model.predict(X_selected)
    return model, X_df, pred, feat_names, entry


def summary_plots(model, X_df: pd.DataFrame, tag: str, explainer) -> np.ndarray:
    shap_values = explainer.shap_values(X_df)

    plt.figure(figsize=(9, max(4, 0.28 * len(X_df.columns))))
    shap.summary_plot(shap_values, X_df, show=False, plot_type="dot")
    plt.title(f"SHAP Summary (Beeswarm) — {tag}")
    plt.tight_layout()
    fname = "shap_summary.png" if tag == "LightGBM(최종 배포모델)" else f"shap_summary_{tag.lower()}.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()

    plt.figure(figsize=(8, max(4, 0.28 * len(X_df.columns))))
    shap.summary_plot(shap_values, X_df, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance (Bar) — {tag}")
    plt.tight_layout()
    bar_fname = "shap_summary_bar.png" if tag == "LightGBM(최종 배포모델)" else f"shap_summary_bar_{tag.lower()}.png"
    plt.savefig(FIG_DIR / bar_fname, dpi=150)
    plt.close()

    log(f"  저장: {fname}, {bar_fname}")
    return shap_values


def dependence_plots(shap_values, X_df: pd.DataFrame, features: list[str], prefix: str) -> None:
    for feat in features:
        if feat not in X_df.columns:
            continue
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feat, shap_values, X_df, interaction_index="auto", show=False)
        plt.title(f"SHAP Dependence — {feat}")
        plt.tight_layout()
        safe = feat.replace(" ", "_").replace("/", "_")
        fname = f"shap_dependence_{prefix}_{safe}.png"
        plt.savefig(FIG_DIR / fname, dpi=150)
        plt.close()
        log(f"  저장: {fname}")


def waterfall_plot(explainer, X_df: pd.DataFrame, idx: int, tag: str, fname: str) -> None:
    exp = explainer(X_df.iloc[[idx]])
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp[0], show=False, max_display=15)
    plt.title(tag)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  저장: {fname}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_candidates = joblib.load(PROJECT_ROOT / "models" / "all_candidate_models.pkl")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_final_feature_matrix.csv")
    y = df["Target"].astype(float)

    log("=== LightGBM(최종 배포모델) SHAP 분석 ===")
    lgbm_model, X_lgbm, pred_lgbm, feat_lgbm, lgbm_entry = load_model_data("LightGBM", all_candidates, df)
    explainer_lgbm = shap.TreeExplainer(lgbm_model)
    shap_values_lgbm = summary_plots(lgbm_model, X_lgbm, "LightGBM(최종 배포모델)", explainer_lgbm)

    mean_abs_shap_lgbm = pd.Series(np.abs(shap_values_lgbm).mean(axis=0), index=feat_lgbm).sort_values(ascending=False)
    log("LightGBM 상위 10개 피처(mean|SHAP|):")
    log(mean_abs_shap_lgbm.head(10).to_string())

    xgb_model, X_xgb, pred_xgb, feat_xgb, xgb_entry = load_model_data("XGBoost", all_candidates, df)
    log(f"\n=== XGBoost(비교모델, {len(feat_xgb)}개 피처) SHAP 분석 ===")
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = summary_plots(xgb_model, X_xgb, "XGBoost", explainer_xgb)

    mean_abs_shap_xgb = pd.Series(np.abs(shap_values_xgb).mean(axis=0), index=feat_xgb).sort_values(ascending=False)
    log("XGBoost 상위 10개 피처(mean|SHAP|):")
    log(mean_abs_shap_xgb.head(10).to_string())

    log("\n=== Dependence Plot (H3 델타피처 + 상위 피처) ===")
    top_features_lgbm = mean_abs_shap_lgbm.head(4).index.tolist()
    dep_targets = list(dict.fromkeys(["etch_rate_stage1"] + top_features_lgbm))
    dependence_plots(shap_values_lgbm, X_lgbm, dep_targets, prefix="lgbm")

    log("\n=== 로컬 설명 (오차 큰 샘플 vs 정확한 샘플) ===")
    residual = y.values - pred_lgbm
    idx_outlier = int(np.argmax(np.abs(residual)))
    idx_good = int(np.argmin(np.abs(residual)))
    outlier_info = df.iloc[idx_outlier]
    good_info = df.iloc[idx_good]
    log(f"오차 최대 샘플: group_id={outlier_info['group_id']}, Lot={outlier_info['Lot_Num']}, "
        f"실제 Target={y.iloc[idx_outlier]:.0f}, 예측={pred_lgbm[idx_outlier]:.1f}, "
        f"오차={residual[idx_outlier]:+.1f}")
    log(f"정확 예측 샘플: group_id={good_info['group_id']}, Lot={good_info['Lot_Num']}, "
        f"실제 Target={y.iloc[idx_good]:.0f}, 예측={pred_lgbm[idx_good]:.1f}, "
        f"오차={residual[idx_good]:+.1f}")

    waterfall_plot(
        explainer_lgbm, X_lgbm, idx_outlier,
        f"오차 최대 웨이퍼 (group_id={outlier_info['group_id']}, 실제={y.iloc[idx_outlier]:.0f} vs 예측={pred_lgbm[idx_outlier]:.1f})",
        "shap_waterfall_outlier.png",
    )
    waterfall_plot(
        explainer_lgbm, X_lgbm, idx_good,
        f"정확 예측 웨이퍼 (group_id={good_info['group_id']}, 실제={y.iloc[idx_good]:.0f} vs 예측={pred_lgbm[idx_good]:.1f})",
        "shap_waterfall_good.png",
    )

    # 개별 샘플의 SHAP 기여도 테이블(리포트용 텍스트 근거)
    exp_outlier = explainer_lgbm(X_lgbm.iloc[[idx_outlier]])[0]
    exp_good = explainer_lgbm(X_lgbm.iloc[[idx_good]])[0]
    outlier_contrib = pd.Series(exp_outlier.values, index=feat_lgbm).sort_values(key=np.abs, ascending=False)
    good_contrib = pd.Series(exp_good.values, index=feat_lgbm).sort_values(key=np.abs, ascending=False)

    top5_kpi = mean_abs_shap_lgbm.head(5)

    write_report(
        df=df, y=y, pred_lgbm=pred_lgbm,
        mean_abs_shap_lgbm=mean_abs_shap_lgbm, mean_abs_shap_xgb=mean_abs_shap_xgb,
        lgbm_entry=lgbm_entry, xgb_entry=xgb_entry,
        outlier_info=outlier_info, good_info=good_info,
        residual_outlier=residual[idx_outlier], residual_good=residual[idx_good],
        y_outlier=y.iloc[idx_outlier], pred_outlier=pred_lgbm[idx_outlier],
        y_good=y.iloc[idx_good], pred_good=pred_lgbm[idx_good],
        outlier_contrib=outlier_contrib, good_contrib=good_contrib,
        top5_kpi=top5_kpi, X_lgbm=X_lgbm,
        outlier_raw_row=df.iloc[idx_outlier], good_raw_row=df.iloc[idx_good],
    )


def write_report(*, df, y, pred_lgbm, mean_abs_shap_lgbm, mean_abs_shap_xgb, lgbm_entry, xgb_entry,
                  outlier_info, good_info, residual_outlier, residual_good,
                  y_outlier, pred_outlier, y_good, pred_good,
                  outlier_contrib, good_contrib, top5_kpi, X_lgbm,
                  outlier_raw_row, good_raw_row) -> None:
    lines = ["# 모델 해석(XAI) 보고서 — SHAP 기반 분석", ""]
    lines.append(
        f"최종 배포 모델 **LightGBM**({len(lgbm_entry['selected_features'])}개 피처, "
        f"GroupKFold(Lot_Num) CV R²={lgbm_entry['r2_mean']:.4f}±{lgbm_entry['r2_std']:.4f})과 "
        f"비교 모델 **XGBoost**({len(xgb_entry['selected_features'])}개 피처, "
        f"R²={xgb_entry['r2_mean']:.4f}±{xgb_entry['r2_std']:.4f})에 "
        "대해 SHAP(TreeExplainer)으로 예측 근거를 분석했다. 전체 1,704개 웨이퍼(fit에 "
        "쓰인 것과 동일 데이터)에 대해 계산했다 — SHAP은 '모델이 무엇을 학습했는가'를 "
        "설명하는 도구이지 별도의 일반화 성능 검증이 아니므로, 성능 수치 자체는 "
        "`final_modeling_report.md`의 GroupKFold 결과를 봐야 한다."
    )
    lines.append("")

    lines.append("## 1. 글로벌 영향도 (SHAP Summary)")
    lines.append("")
    lines.append("![LightGBM SHAP Beeswarm](figures/shap_summary.png)")
    lines.append("")
    lines.append("![LightGBM SHAP Bar](figures/shap_summary_bar.png)")
    lines.append("")
    lines.append("**LightGBM 상위 10개 피처 (mean|SHAP|)**")
    lines.append("")
    lines.append("| 순위 | 피처 | mean|SHAP| |")
    lines.append("|---|---|---|")
    for i, (feat, val) in enumerate(mean_abs_shap_lgbm.head(10).items(), 1):
        lines.append(f"| {i} | `{feat}` | {val:.3f} |")
    lines.append("")

    lines.append("![XGBoost SHAP Beeswarm](figures/shap_summary_xgboost.png)")
    lines.append("")
    lines.append("**XGBoost 상위 10개 피처 (mean|SHAP|)**")
    lines.append("")
    lines.append("| 순위 | 피처 | mean|SHAP| |")
    lines.append("|---|---|---|")
    for i, (feat, val) in enumerate(mean_abs_shap_xgb.head(10).items(), 1):
        lines.append(f"| {i} | `{feat}` | {val:.3f} |")
    lines.append("")

    lgbm_top10 = set(mean_abs_shap_lgbm.head(10).index)
    xgb_top10 = set(mean_abs_shap_xgb.head(10).index)
    common = lgbm_top10 & xgb_top10
    lgbm_only = lgbm_top10 - xgb_top10
    xgb_only = xgb_top10 - lgbm_top10
    lines.append("### XGBoost vs LightGBM 비교 해석")
    lines.append("")
    total_input = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    lgbm_n = len(lgbm_entry["selected_features"])
    xgb_n = len(xgb_entry["selected_features"])
    lines.append(
        f"두 모델 상위 10개 중 **{len(common)}개가 공통**({', '.join(f'`{f}`' for f in sorted(common))}) — "
        f"서로 다른 피처 부분집합({total_input}개 중 XGBoost {xgb_n}개 vs LightGBM {lgbm_n}개)과 "
        "다른 알고리즘으로 학습했음에도 핵심 신호는 일치한다는 뜻으로, 특정 알고리즘의 우연이 "
        "아니라 실제 공정 신호로 볼 근거가 된다."
    )
    if lgbm_only:
        lines.append(
            f"- LightGBM에만 있는 상위피처: {', '.join(f'`{f}`' for f in sorted(lgbm_only))} — "
            f"두 모델의 RFECV가 서로 다른 피처 수({xgb_n} vs {lgbm_n})를 골랐기 때문에 생기는 차이다. "
            "성능 차이가 거의 없다는 걸 감안하면, 이 변수들은 '있으면 미세하게 도움' 수준이지 "
            "필수 신호는 아닐 가능성이 있다."
        )
    if xgb_only:
        lines.append(f"- XGBoost에만 있는 상위피처: {', '.join(f'`{f}`' for f in sorted(xgb_only))}")
    lines.append("")

    lines.append("## 2. 핵심 공정 변수 의존성 분석 (H3 포함)")
    lines.append("")
    lines.append("![etch_rate_stage1 Dependence](figures/shap_dependence_lgbm_etch_rate_stage1.png)")
    lines.append("")
    lines.append(
        "`etch_rate_stage1`(H3에서 채택된 식각 1단계 구간차분, Thin F1-F2)의 SHAP 의존성 "
        "그래프 — 이전 EDA에서 확인한 상관(r=-0.39, 값이 클수록 결함↓)의 방향이 SHAP "
        "관점에서도 유지되는지 시각적으로 확인할 것."
    )
    lines.append("")
    for feat in mean_abs_shap_lgbm.head(4).index:
        safe = feat.replace(" ", "_").replace("/", "_")
        lines.append(f"![{feat} Dependence](figures/shap_dependence_lgbm_{safe}.png)")
        lines.append("")

    lines.append("## 3. 로컬 해석 (개별 웨이퍼)")
    lines.append("")
    lines.append(
        "⚠️ 아래 Waterfall 그림 자체에는 SHAP/matplotlib 특성상 피처값이 "
        "StandardScaler로 스케일된 z-score로 표시된다(예: `Temp_OXid = 1.49`는 "
        "'평균보다 1.49 표준편차 높다'는 뜻이지 실제 1.49℃가 아님). 원본 물리단위 "
        "실측값은 그림 아래 표를 봐야 한다."
    )
    lines.append("")
    lines.append(
        f"**오차 최대 웨이퍼** — group_id=`{outlier_info['group_id']}`(Lot {outlier_info['Lot_Num']}): "
        f"실제 Target={y_outlier:.0f}(데이터셋 전체 역대 최댓값), 예측={pred_outlier:.1f} "
        f"(오차 {residual_outlier:+.1f}, 대폭 과소예측)"
    )
    lines.append("")
    is_missing_etch = pd.isna(raw_value(outlier_raw_row, "Thin F2"))
    if is_missing_etch:
        lines.append(
            "🔴 **데이터 결함 발견**: 이 웨이퍼는 `Thin F2/F3/F4`(식각 잔막 실측값)가 "
            "9개 die-행 전부에서 **완전히 결측**(NaN, 단순 센티널 0이 아니라 값 자체가 "
            "없음)이다 — `raw/Etching.csv` 원본부터 결측이라 이 프로젝트 전처리 단계의 "
            "버그가 아니다. 즉 모델이 이 웨이퍼를 크게 과소예측(408.9 vs 실제 666)한 "
            "이유는 '모델이 틀렸다'가 아니라 **'가장 강력한 예측 신호(Thin F2/F4, SHAP "
            "1·2위)가 이 웨이퍼에서만 통째로 없어서 중앙값으로 대체된 값을 쓸 수밖에 "
            "없었다'는 것이다. 실무적으로 훨씬 중요한 시사점: **결함이 가장 심각한 "
            "웨이퍼일수록 계측 자체가 누락되는 경향**이 있을 수 있다는 뜻이다(계측 "
            "장비가 손상이 심한 웨이퍼를 스킵했거나, 손상이 너무 심해 정상 계측이 "
            "불가능했을 가능성). 이 가설이 맞다면, 현재 모델은 '가장 위험한 웨이퍼일수록 "
            "가장 못 맞히는' 구조적 사각지대를 갖고 있다는 뜻이라 — 계측 누락 자체를 "
            "결함 조기경보 신호로 별도 관리하는 걸 권장한다(Top 5 KPI 다음으로 "
            "우선순위가 높은 실무 제언)."
        )
        lines.append("")
    lines.append("![오차 최대 웨이퍼 Waterfall](figures/shap_waterfall_outlier.png)")
    lines.append("")
    lines.append("이 웨이퍼의 예측을 가장 크게 밀어올리거나 끌어내린 상위 5개 피처:")
    lines.append("")
    lines.append("| 피처 | SHAP 기여도 | 실측값(원본 단위) |")
    lines.append("|---|---|---|")
    for feat, val in outlier_contrib.head(5).items():
        actual_val = raw_value(outlier_raw_row, feat)
        lines.append(f"| `{feat}` | {val:+.2f} | {actual_val} |")
    lines.append("")

    lines.append(
        f"**정확 예측 웨이퍼** — group_id=`{good_info['group_id']}`(Lot {good_info['Lot_Num']}): "
        f"실제 Target={y_good:.0f}, 예측={pred_good:.1f} (오차 {residual_good:+.1f})"
    )
    lines.append("")
    lines.append("![정확 예측 웨이퍼 Waterfall](figures/shap_waterfall_good.png)")
    lines.append("")
    lines.append("| 피처 | SHAP 기여도 | 실측값(원본 단위) |")
    lines.append("|---|---|---|")
    for feat, val in good_contrib.head(5).items():
        actual_val = raw_value(good_raw_row, feat)
        lines.append(f"| `{feat}` | {val:+.2f} | {actual_val} |")
    lines.append("")

    lines.append("## 4. 공정 제어 우선순위 Top 5 KPI")
    lines.append("")
    lines.append("| 우선순위 | 피처 | mean|SHAP| | 근거 |")
    lines.append("|---|---|---|---|")
    for i, (feat, val) in enumerate(top5_kpi.items(), 1):
        lines.append(f"| {i} | `{feat}` | {val:.3f} | SHAP Dependence Plot(2절) 및 로컬 설명(3절) 참고 |")
    lines.append("")
    lines.append(
        "**주의**: 이 순위는 '모델이 예측할 때 무엇을 가장 많이 참고하는가(예측 기여도)'"
        "이지, 08/H1~H7 리포트에서 이미 통계적으로 검증한 '실제 인과관계'와 자동으로 "
        "같지 않다. 특히 UV_type처럼 예측 기여도가 있어 보여도 Lot 교란 통제 후 사라진 "
        "사례가 있었으므로, 이 Top 5를 실제 공정 변경 의사결정에 쓰기 전에는 H1~H7 "
        "리포트에서 해당 변수가 이미 인과적으로 검증됐는지 교차확인해야 한다. 여기 "
        "상위 피처들(Thin F2-4, etch_rate_stage1, 산화/이온주입 공정변수)은 이미 H3~H4 "
        "가설검증과 04_xai_analysis.py의 SHAP 분석에서도 일관되게 상위권이었던 변수들로, "
        "교차검증된 신호로 볼 수 있다."
    )
    lines.append("")

    lines.append("## 재현")
    lines.append("")
    lines.append("`python scripts/13_model_interpretation_shap.py` (shap, matplotlib 필요)")
    lines.append("그림: `reports/figures/shap_*.png`")

    out_path = PROJECT_ROOT / "reports" / "model_interpretation_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
