"""GitHub Pages 정적 대시보드(docs/)용 산출물 내보내기.

서버 없이 브라우저에서 예측·SHAP·품질 분석을 돌리기 위해 필요한 것을 JSON으로 만든다.

  docs/assets/model.json    LightGBM 파이프라인 전체(대치 → 스케일 → 원핫 → 피처선택 → 트리 150개)
  docs/assets/quality.json  SPC 관리도 · Cp/Cpk · Pareto · PFMEA · 8D · ML vs SPC 비교 데이터
  docs/assets/wafers.json   웨이퍼맵(26x26을 문자열 1개로 압축) + 웨이퍼별 원본 피처값
  docs/sample_wafers.csv    업로드 기능 시연용 샘플

ML vs SPC 비교는 반드시 out-of-fold 예측으로 한다 — 학습에 쓴 웨이퍼를 그대로 예측하면
ML이 부당하게 좋아 보인다. GroupKFold(Lot_Num) 5-fold, 전처리도 fold 안에서 재적합한다.
(RFECV가 고른 22개 피처 마스크 자체는 전체 데이터로 선택된 것이라 소폭 낙관적일 수 있음 —
final_modeling_report.md의 R²와 동일한 조건.)
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
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from wafer.final_pipeline import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

DOCS = PROJECT_ROOT / "docs"
ASSETS = DOCS / "assets"
REAL_DIE_COUNT = 533
OXIDE_LSL_NM = 700.0
LINE_CD_SPEC = (25.0, 55.0)

# 공정 순서(물리적 흐름)와 모델이 실제로 쓰는 22개 피처의 소속 — 비전공자용 라벨 포함
PROCESSES = [
    {"id": "oxidation", "name": "산화", "desc": "웨이퍼 표면에 절연막(산화막)을 키우는 공정",
     "params": [("Temp_OXid", "산화 온도", "°C"), ("ppm", "가스 농도", "ppm"),
                ("thickness", "산화막 두께", "nm")]},
    {"id": "coat", "name": "감광액 도포", "desc": "빛에 반응하는 감광액을 얇게 바르고 굽는 공정",
     "params": [("resist_target", "감광액 두께 목표비", ""), ("N2_HMDS", "접착제 처리 질소 유량", ""),
                ("temp_HMDS", "접착제 처리 온도", "°C"), ("spin3", "도포 회전 속도(3단계)", "rpm"),
                ("photoresist_bake", "감광액 굽는 시간", ""), ("temp_softbake", "굽는 온도", "°C")]},
    {"id": "litho", "name": "노광", "desc": "빛으로 회로 패턴을 새기는 공정",
     "params": [("Line_CD", "회로 선폭", "nm"), ("Resolution", "해상도", ""),
                ("Energy_Exposure", "빛 에너지", "")]},
    {"id": "etch", "name": "식각", "desc": "패턴대로 막을 깎아내는 공정 — 결함에 가장 큰 영향",
     "params": [("Thin F2", "2단계 후 남은 막 두께", "Å"), ("Thin F3", "3단계 후 남은 막 두께", "Å"),
                ("Thin F4", "최종 남은 막 두께", "Å"), ("Temp_Etching", "식각 온도", "°C"),
                ("Source_Power", "플라즈마 출력", ""), ("Selectivity", "선택비", "")]},
    {"id": "implant", "name": "이온주입", "desc": "불순물 이온을 넣어 전기적 성질을 만드는 공정",
     "params": [("Flux60s", "60초 주입량", ""), ("Flux90s", "90초 주입량", ""),
                ("input_Energy", "주입 에너지", ""), ("Temp_implantation", "주입 온도", "°C")]},
    {"id": "inspect", "name": "검사", "desc": "완성된 칩의 불량 여부를 판정",
     "params": []},
]

DEFECT_KO = {
    "Edge-Loc": "가장자리 국부 불량", "Loc": "국부 불량", "Random": "무작위 불량",
    "Center": "중앙 집중 불량", "Scratch": "긁힘", "Near-full": "전면 불량", "Edge-Ring": "가장자리 링 불량",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ---------------------------------------------------------------------
# 1) 모델 내보내기
# ---------------------------------------------------------------------
def export_tree(node: dict, nodes: list) -> int:
    idx = len(nodes)
    if "leaf_value" in node:
        nodes.append({"v": node["leaf_value"], "c": node.get("leaf_count", 0)})
        return idx
    nodes.append(None)
    left = export_tree(node["left_child"], nodes)
    right = export_tree(node["right_child"], nodes)
    if node["decision_type"] != "<=":
        raise ValueError(f"지원하지 않는 분기 유형: {node['decision_type']}")
    nodes[idx] = {
        "f": node["split_feature"], "t": node["threshold"], "l": left, "r": right,
        "d": bool(node["default_left"]), "m": node["missing_type"],
        "v": node["internal_value"], "c": node["internal_count"],
    }
    return idx


def export_model(entry: dict) -> dict:
    pipe = entry["pipeline"]
    prep = pipe.named_steps["prep"]
    num_pipe = prep.named_transformers_["num"]
    cat_pipe = prep.named_transformers_["cat"]
    imputer, scaler = num_pipe.named_steps["impute"], num_pipe.named_steps["scale"]
    cat_imp, onehot = cat_pipe.named_steps["impute"], cat_pipe.named_steps["onehot"]

    mask = np.asarray(pipe.named_steps["select"].mask)
    selected_idx = np.flatnonzero(mask).tolist()
    booster = pipe.named_steps["model"].booster_
    dump = booster.dump_model()

    trees = []
    for t in dump["tree_info"]:
        nodes: list = []
        export_tree(t["tree_structure"], nodes)
        trees.append(nodes)

    return {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "median": [float(v) for v in imputer.statistics_],
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "cat_mode": [str(v) for v in cat_imp.statistics_],
        # drop='first' — 각 범주형의 첫 범주는 더미가 없음
        "cat_categories": [[str(c) for c in cats] for cats in onehot.categories_],
        "selected_index": selected_idx,
        "selected_names": [f.replace("num__", "").replace("cat__", "") for f in entry["selected_features"]],
        "trees": trees,
        "metrics": {"r2_mean": entry["r2_mean"], "r2_std": entry["r2_std"], "mae_mean": entry["mae_mean"],
                    "n_features": len(selected_idx), "n_trees": len(trees)},
    }


# ---------------------------------------------------------------------
# 2) Out-of-fold 예측 (ML vs SPC 공정 비교용)
# ---------------------------------------------------------------------
def oof_predictions(entry: dict, df: pd.DataFrame) -> np.ndarray:
    pipe = entry["pipeline"]
    mask = np.asarray(pipe.named_steps["select"].mask)
    params = {k.replace("model__", ""): v for k, v in entry["best_params"].items()}
    X, y, groups = df[ALL_FEATURES], df["Target"].astype(float).values, df["Lot_Num"].values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        prep = clone(pipe.named_steps["prep"])
        Xtr = prep.fit_transform(X.iloc[tr])[:, mask]
        Xte = prep.transform(X.iloc[te])[:, mask]
        m = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1, **params).fit(Xtr, y[tr])
        oof[te] = np.maximum(0.0, m.predict(Xte))
    return oof


def recall_at_fpr(scores: np.ndarray, is_defect: np.ndarray, fpr_target: float) -> dict:
    neg = np.sort(scores[~is_defect])
    thr = np.quantile(neg, 1 - fpr_target)
    flagged = scores > thr
    tp = int((flagged & is_defect).sum())
    fp = int((flagged & ~is_defect).sum())
    return {"threshold": r(thr, 3), "recall": r(tp / is_defect.sum(), 4),
            "precision": r(tp / max(tp + fp, 1), 4), "tp": tp, "fp": fp,
            "fpr": r(fp / (~is_defect).sum(), 4)}


# ---------------------------------------------------------------------
# 3) 품질 분석 데이터
# ---------------------------------------------------------------------
def capability(values: pd.Series, lsl: float | None, usl: float | None) -> dict:
    s = values.dropna().astype(float)
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    cp = (usl - lsl) / (6 * sd) if (lsl is not None and usl is not None) else None
    cpu = (usl - mu) / (3 * sd) if usl is not None else None
    cpl = (mu - lsl) / (3 * sd) if lsl is not None else None
    cpk = min(v for v in (cpu, cpl) if v is not None)
    below = int((s < lsl).sum()) if lsl is not None else 0
    above = int((s > usl).sum()) if usl is not None else 0
    hist, edges = np.histogram(s, bins=30)
    return {"n": int(len(s)), "mean": r(mu), "sd": r(sd), "lsl": lsl, "usl": usl,
            "cp": r(cp, 3), "cpk": r(cpk, 3), "cpl": r(cpl, 3), "cpu": r(cpu, 3),
            "out_of_spec": below + above, "out_of_spec_pct": r(100 * (below + above) / len(s), 2),
            "hist": hist.tolist(), "edges": [r(e, 3) for e in edges]}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cands = joblib.load(PROJECT_ROOT / "models" / "all_candidate_models.pkl")
    entry = cands["LightGBM"]
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_final_feature_matrix.csv")
    df["thickness"] = OXIDE_LSL_NM - df["oxid_thickness_spec_gap"]
    df["wafer_num"] = df["group_id"].str.split("_").str[1].astype(int)
    df = df.sort_values(["Lot_Num", "wafer_num"]).reset_index(drop=True)
    is_defect = (df["error_class"] != "none").values

    # --- 모델 ---
    model = export_model(entry)
    (ASSETS / "model.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    log(f"model.json: {len(model['trees'])} trees, {model['metrics']['n_features']} features")

    # 검증용 기준값 — 브라우저 구현을 Node로 대조하기 위해
    pipe = entry["pipeline"]
    probe = df.iloc[[0, 5, 100, 400, 900, 1500]].copy()
    probe_pred = np.maximum(0.0, pipe.predict(probe[ALL_FEATURES]))
    import shap
    X_sel = pipe.named_steps["select"].transform(pipe.named_steps["prep"].transform(probe[ALL_FEATURES]))
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(X_sel)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    ref = {"rows": probe[ALL_FEATURES].replace({np.nan: None}).to_dict(orient="records"),
           "pred": probe_pred.tolist(), "shap": np.asarray(sv).tolist(), "base": base}
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "models" / "_web_reference.json"
    ref_path.write_text(json.dumps(ref, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)),
                        encoding="utf-8")
    log(f"reference: {ref_path}")

    # --- 파라미터 범위(슬라이더) ---
    param_meta = {}
    for proc in PROCESSES:
        for key, label, unit in proc["params"]:
            s = df[key].astype(float).dropna()
            dfd = df.loc[df[key].notna()]
            param_meta[key] = {
                "label": label, "unit": unit, "process": proc["id"],
                "min": r(s.quantile(0.01), 3), "max": r(s.quantile(0.99), 3),
                "median": r(s.median(), 3), "q20": r(s.quantile(0.2), 3), "q80": r(s.quantile(0.8), 3),
                "defect_rate_low": r(100 * (dfd.loc[dfd[key] <= s.quantile(0.2), "error_class"] != "none").mean(), 1),
                "defect_rate_high": r(100 * (dfd.loc[dfd[key] >= s.quantile(0.8), "error_class"] != "none").mean(), 1),
            }

    # --- SPC 1: Lot별 불량률 p-chart ---
    lot = df.groupby("Lot_Num").agg(n=("group_id", "count"), d=("error_class", lambda s: int((s != "none").sum())),
                                    mean_target=("Target", "mean")).reset_index()
    p_bar = lot["d"].sum() / lot["n"].sum()
    lot["p"] = lot["d"] / lot["n"]
    lot["ucl"] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / lot["n"])
    lot["lcl"] = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / lot["n"]))
    p_chart = {"p_bar": r(p_bar), "lots": [
        {"lot": int(x.Lot_Num), "n": int(x.n), "d": int(x.d), "p": r(x.p), "ucl": r(x.ucl), "lcl": r(x.lcl),
         "ooc": bool(x.p > x.ucl), "mean_target": r(x.mean_target, 1)} for x in lot.itertuples()]}

    # --- SPC 2: 웨이퍼 순서대로 I-MR 관리도 (핵심 인자) ---
    imr = {}
    for key in ["Thin F2", "Thin F4", "thickness", "Temp_OXid"]:
        s = df[key].astype(float)
        mr = s.diff().abs()
        x_bar, mr_bar = float(s.mean()), float(mr.mean())
        sigma = mr_bar / 1.128  # d2(n=2)
        ucl, lcl = x_bar + 3 * sigma, x_bar - 3 * sigma
        # 8점 연속 한쪽 규칙(Nelson rule 2)
        side = np.sign(s - x_bar).fillna(0).values
        run_flag = np.zeros(len(s), dtype=bool)
        cnt = 0
        for i in range(len(side)):
            cnt = cnt + 1 if i > 0 and side[i] == side[i - 1] and side[i] != 0 else 1
            if cnt >= 8:
                run_flag[i - 7:i + 1] = True
        imr[key] = {"label": param_meta[key]["label"], "unit": param_meta[key]["unit"],
                    "center": r(x_bar, 3), "ucl": r(ucl, 3), "lcl": r(lcl, 3), "mr_bar": r(mr_bar, 3),
                    "mr_ucl": r(3.267 * mr_bar, 3),
                    "values": [r(v, 2) for v in s], "mr": [r(v, 2) for v in mr],
                    "defect": is_defect.tolist(), "lot": df["Lot_Num"].astype(int).tolist(),
                    "n_ooc": int(((s > ucl) | (s < lcl)).sum()), "n_run": int(run_flag.sum()),
                    "missing": int(s.isna().sum())}

    # --- 공정능력 ---
    cap = {
        "thickness": {"label": "산화막 두께", "unit": "nm", "spec_source": "변수정의서: 700nm 이상",
                      "has_spec": True, **capability(df["thickness"], OXIDE_LSL_NM, None)},
        "Line_CD": {"label": "회로 선폭", "unit": "nm", "spec_source": "변수정의서: 25~55nm",
                    "has_spec": True, **capability(df["Line_CD"], *LINE_CD_SPEC)},
    }
    # 규격이 없는 식각 잔막 — 불량률이 0%인 하위 20% 구간 경계를 '잠정 관리 상한'으로 둔 참고치
    f2 = df["Thin F2"]
    cap["Thin F2"] = {"label": "2단계 후 남은 막 두께", "unit": "Å", "has_spec": False,
                      "spec_source": f"공식 규격 없음 — 불량률이 {param_meta['Thin F2']['defect_rate_high']}%로 뛰는 "
                                     "상위 20% 구간의 경계(q80)를 잠정 상한으로 사용",
                      **capability(f2, None, float(f2.quantile(0.8)))}

    # --- Pareto ---
    pareto_counts = df.loc[is_defect, "error_class"].value_counts()
    cum = pareto_counts.cumsum() / pareto_counts.sum()
    pareto = [{"code": k, "name": DEFECT_KO.get(k, k), "count": int(v), "cum_pct": r(100 * cum[k], 1),
               "mean_target": r(df.loc[df["error_class"] == k, "Target"].mean(), 1)}
              for k, v in pareto_counts.items()]

    # --- ML vs SPC (out-of-fold) ---
    log("OOF 예측 계산 중 (GroupKFold 5-fold)…")
    oof = oof_predictions(entry, df)
    key_params = ["Thin F2", "Thin F3", "Thin F4", "Temp_OXid", "input_Energy"]
    z = pd.DataFrame({k: (df[k] - df[k].mean()) / df[k].std() for k in key_params})
    spc_score = z.abs().max(axis=1).fillna(0).values  # 5개 인자 중 가장 벗어난 정도(|z|)
    spc_flag = spc_score > 3
    spc_fpr = float(spc_flag[~is_defect].mean())
    spc_tp = int((spc_flag & is_defect).sum())
    spc_fp = int((spc_flag & ~is_defect).sum())
    ml_auc = roc_auc_score(is_defect, oof)
    spc_auc = roc_auc_score(is_defect, spc_score)
    compare = {
        "n_wafers": int(len(df)), "n_defect": int(is_defect.sum()),
        "spc_rule": "핵심 5개 인자(Thin F2/F3/F4·산화 온도·주입 에너지) 중 하나라도 평균±3σ 이탈",
        "spc": {"recall": r(spc_tp / is_defect.sum(), 4), "precision": r(spc_tp / max(spc_tp + spc_fp, 1), 4),
                "tp": spc_tp, "fp": spc_fp, "fpr": r(spc_fpr, 4), "auc": r(spc_auc, 4)},
        "ml_same_fpr": {**recall_at_fpr(oof, is_defect, spc_fpr), "auc": r(ml_auc, 4)},
        "ml_5pct_fpr": recall_at_fpr(oof, is_defect, 0.05),
        "spc_5pct_fpr": recall_at_fpr(spc_score, is_defect, 0.05),
        "oof_r2": r(1 - ((df["Target"] - oof) ** 2).sum() / ((df["Target"] - df["Target"].mean()) ** 2).sum(), 4),
        "missing_metrology_defect": {
            "n_missing": int(df[["Thin F2", "Thin F3", "Thin F4"]].isna().any(axis=1).sum()),
            "n_missing_defect": int((df[["Thin F2", "Thin F3", "Thin F4"]].isna().any(axis=1) & is_defect).sum())},
    }
    # ROC 곡선(표시용, 50점)
    def roc_points(score):
        pts = []
        for q in np.linspace(0, 1, 51):
            thr = np.quantile(score, q)
            f = score >= thr
            pts.append([r(f[~is_defect].mean(), 4), r(f[is_defect].mean(), 4)])
        return sorted(pts)
    compare["roc_ml"], compare["roc_spc"] = roc_points(oof), roc_points(spc_score)
    log(f"ML AUC={ml_auc:.3f}  SPC AUC={spc_auc:.3f}  "
        f"SPC recall={compare['spc']['recall']} @FPR={spc_fpr:.3f}  ML recall={compare['ml_same_fpr']['recall']} @same FPR")

    # --- 참고 수치 ---
    ex = pd.read_csv(PROJECT_ROOT / "reports" / "06_executive_summary.csv")
    quality = {
        "processes": [{k: v for k, v in p.items() if k != "params"} | {"params": [k for k, _, _ in p["params"]]}
                      for p in PROCESSES],
        "params": param_meta, "p_chart": p_chart, "imr": imr, "capability": cap, "pareto": pareto,
        "compare": compare,
        "overall": {"n_wafers": int(len(df)), "n_lots": int(df["Lot_Num"].nunique()),
                    "defect_rate_pct": r(100 * is_defect.mean(), 2),
                    "mean_target": r(df["Target"].mean(), 2), "real_die_count": REAL_DIE_COUNT,
                    "baseline_yield": r(ex.iloc[0]["Chip Yield %"], 2),
                    "scenario_a_yield": r(ex.iloc[1]["Chip Yield %"], 2)},
        "target_ucl": {"mean": r(df["Target"].mean(), 2), "sd": r(df["Target"].std(), 2),
                       "ucl2": r(df["Target"].mean() + 2 * df["Target"].std(), 2),
                       "ucl3": r(df["Target"].mean() + 3 * df["Target"].std(), 2)},
    }
    (ASSETS / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, separators=(",", ":")),
                                         encoding="utf-8")
    log("quality.json 저장")

    # --- 웨이퍼 (맵 + 피처) ---
    lookup = {w["group_id"]: w for w in json.load(open(PROJECT_ROOT / "models" / "wafer_lookup.json", encoding="utf-8"))}
    keys = sorted({k for p in PROCESSES for k, _, _ in p["params"]})
    wafers = []
    for i, row in df.iterrows():
        w = lookup.get(row["group_id"])
        wafers.append({
            "id": row["group_id"], "lot": int(row["Lot_Num"]), "target": int(row["Target"]),
            "cls": row["error_class"], "oof": r(oof[i], 1),
            "map": "".join(str(c) for rr in w["wafer_map"] for c in rr) if w else None,
            "x": {k: r(row[k], 3) for k in keys},
        })
    (ASSETS / "wafers.json").write_text(json.dumps(wafers, ensure_ascii=False, separators=(",", ":")),
                                        encoding="utf-8")
    log(f"wafers.json: {len(wafers)} wafers ({sum(w['map'] is not None for w in wafers)} with maps)")

    # --- 업로드 시연용 샘플 CSV: Lot 25(이상 Lot) + Lot 13 일부 ---
    sample = df[df["Lot_Num"].isin([25, 13])].groupby("Lot_Num").head(12)
    cols = ["group_id"] + keys + ["Target"]
    sample[cols].rename(columns={"group_id": "wafer_id", "Target": "actual_defects"}).to_csv(
        DOCS / "sample_wafers.csv", index=False, encoding="utf-8")
    log(f"sample_wafers.csv: {len(sample)} rows")


if __name__ == "__main__":
    main()
