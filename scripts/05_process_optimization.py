"""4단계: 공정 최적화 — 실제로 엔지니어가 설정 가능한 '레시피 set-point'만으로
Target(결함 다이 수)을 최소화하는 최적 조합을 Bayesian Optimization / Genetic
Algorithm / 제약최적화(SLSQP) 3가지 방식으로 각각 탐색하고 비교한다.

핵심 설계 원칙 — "실행 가능한(actionable)" 최적화가 되려면 Thin F1-4/thickness/
Line_CD/Resolution/Flux* 같은 **중간 측정치(공정 결과)**를 결정변수로 넣으면 안 된다
(엔지니어가 "잔막 두께를 X로 설정"할 수는 없음). 이 변수들은 3단계 SHAP에서 Target을
가장 잘 설명했지만, 그건 원인이 아니라 원인(레시피)과 결과(Target) 사이의 중간 결과물
이기 때문이다. 따라서:
  - 목적함수 모델은 변수정의서(pptx) 기준 '진짜 set-point'만 입력으로 사용해 재학습한다
    (Thin F1-4 등 제외 → 3단계 챔피언보다 R²가 낮아지는 게 정상이며, 이는 필요한 대가다).
  - 산화막 두께 ≥700nm 물리 제약은 별도 보조모델(산화 레시피→thickness)로 근사해
    SLSQP의 명시적 부등식 제약, GA/Optuna의 페널티항으로 사용한다.
  - 범주형 선택(UV_type/type/Vapor)은 그룹평균 비교로 사전에 고정한다(G-line, wet/H2O
    가 각각 H-line, dry/O2보다 유리함이 3단계에서 통계적으로 확인됨) — 연속 최적화는
    남은 21개 수치형 set-point에 대해서만 수행.
  - 탐색 범위는 실제 이 공정이 과거에 가동된 관측 범위([min,max])로 제한한다 — 모델이
    한 번도 보지 못한 영역을 추천하는 외삽을 막기 위함.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

from wafer import features

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42

CONTROLLABLE_NUMERIC = [
    "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake", "time_HMDS_bake",
    "spin1", "spin2", "spin3", "photoresist_bake", "temp_softbake", "time_softbake",
    "Energy_Exposure",
    "Source_Power", "Temp_Etching",
    "input_Energy", "Temp_implantation", "Furance_Temp", "RTA_Temp",
    "Temp_OXid", "ppm", "Pressure", "Oxid_time",
]
FIXED_CATEGORICAL = {"UV_type": "G", "type": "wet", "Vapor": "H2O"}  # 3단계에서 그룹평균 최선으로 확인됨


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "wafer_integrated.parquet")
    df = features.build_feature_frame(df)

    cols = [c for c in CONTROLLABLE_NUMERIC if c in df.columns]
    log(f"controllable numeric set-points ({len(cols)}): {cols}")

    X = df[cols]
    y = df["Target"]
    groups = df["group_id"]

    # -----------------------------------------------------------------
    # 1) 목적함수 모델: set-point만으로 Target 예측 (3단계 챔피언과 정직하게 비교)
    #
    # 한계 ③ 수정: 이전 버전은 imputer를 GroupKFold 분할 '이전에' 전체 X에
    # fit해서 검증 fold의 정보가 median 계산에 살짝 섞이는 원칙 위반이 있었다
    # (2·3단계는 fold 내부에서 처리했음). 아래 CV 루프는 imputer도 매 fold
    # train 데이터에만 fit한다. 최적화 탐색에 실제로 쓰는 objective_model은
    # 배포용 최종 모델이라 전체데이터로 학습(Xi, 아래)하는 것이 맞다 — 이건
    # 3단계 SHAP에서도 쓴 것과 동일한, 검증(CV)과 배포(전체 재학습)를 분리하는
    # 표준적 설계다.
    # -----------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    val_scores = []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        fold_imputer = SimpleImputer(strategy="median")
        Xtr_i = pd.DataFrame(fold_imputer.fit_transform(X.iloc[tr_idx]), columns=cols, index=X.iloc[tr_idx].index)
        Xval_i = pd.DataFrame(fold_imputer.transform(X.iloc[val_idx]), columns=cols, index=X.iloc[val_idx].index)
        m = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, n_estimators=400, num_leaves=31)
        m.fit(Xtr_i, y.iloc[tr_idx])
        val_scores.append(r2_score(y.iloc[val_idx], m.predict(Xval_i)))
    log(f"[objective model] set-point-only GroupKFold val R2 = {np.mean(val_scores):.4f} "
        f"(fold-내부 imputation으로 재검증, 이전 버전은 전역 imputation이라 원칙 위반이었음) "
        f"(참고: 3단계 챔피언은 Thin F1-4 등 중간측정치 포함 R2=0.699 — 그 차이가 "
        f"'중간측정치가 알려주는 정보량'이며, 최적화는 이 차이를 감수하고 진짜 조작변수만 쓴다)")

    # 배포용 최종 모델(탐색에 실제 사용) — 전체 데이터로 재학습, CV와 별개
    imputer = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imputer.fit_transform(X), columns=cols, index=X.index)
    objective_model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, n_estimators=400, num_leaves=31)
    objective_model.fit(Xi, y)

    # -----------------------------------------------------------------
    # 2) 제약 보조모델: 산화 레시피(Temp_OXid, ppm, Pressure, Oxid_time) -> thickness
    #    실제 물리 제약(변수정의서: thickness >= 700nm) 근사
    # -----------------------------------------------------------------
    ox_cols = ["Temp_OXid", "ppm", "Pressure", "Oxid_time"]
    Xox = df[ox_cols].copy()
    Xox["type_wet"] = (df["type"] == "wet").astype(int)
    yox = df["thickness"]
    thickness_model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, n_estimators=300, num_leaves=31)
    thickness_model.fit(Xox, yox)
    ox_gkf_scores = []
    for tr_idx, val_idx in gkf.split(Xox, yox, groups):
        m = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, n_estimators=300, num_leaves=31)
        m.fit(Xox.iloc[tr_idx], yox.iloc[tr_idx])
        ox_gkf_scores.append(r2_score(yox.iloc[val_idx], m.predict(Xox.iloc[val_idx])))
    log(f"[constraint model] oxidation recipe -> thickness GroupKFold val R2 = {np.mean(ox_gkf_scores):.4f}")

    def predict_thickness(x_dict):
        row = pd.DataFrame([{
            "Temp_OXid": x_dict["Temp_OXid"], "ppm": x_dict["ppm"],
            "Pressure": x_dict["Pressure"], "Oxid_time": x_dict["Oxid_time"],
            "type_wet": 1,  # FIXED_CATEGORICAL: wet
        }])
        return float(thickness_model.predict(row)[0])

    # -----------------------------------------------------------------
    # 3) 탐색 범위 = 관측 5~95퍼센타일 (min/max 전체를 쓰면 개별 변수는 관측범위
    #    안이어도 '조합' 자체가 한 번도 없었던 지점이 나올 수 있음 — 아래 novelty
    #    페널티와 함께 이중으로 외삽을 억제한다)
    # -----------------------------------------------------------------
    bounds = {c: (float(Xi[c].quantile(0.05)), float(Xi[c].quantile(0.95))) for c in cols}
    x0 = {c: float(Xi[c].median()) for c in cols}

    def predict_target(x_dict):
        row = pd.DataFrame([x_dict])[cols]
        return float(objective_model.predict(row)[0])

    # novelty(외삽) 페널티: 후보 레시피가 표준화 공간에서 실제 관측 데이터와 얼마나
    # 떨어져 있는지(1-NN 거리)를 측정해 페널티로 더한다 — GA가 대리모델이 한 번도
    # 보지 못한 변수 '조합'을 파고들어 음수 Target 같은 물리적으로 불가능한 예측을
    # 뽑아내는 것을 막기 위함.
    #
    # 한계 ⑥ 수정: 이전 버전은 가중치를 40.0으로 고정하고 "한 번 돌려서 결과가
    # 그럴듯하면 채택"하는 식이었다(체계적 근거 없음). 이번엔 (a) 임계값을
    # 데이터 자체의 밀도에서 유도하고 — 관측 데이터 안에서 각 점이 자기 자신을
    # 뺀 가장 가까운 다른 점까지의 거리(leave-one-out 1-NN) 분포의 90th
    # percentile을 "정상적인 국소 성김(sparsity)"으로 보고, 그보다 가까우면
    # 페널티 0(정상적인 보간), 그보다 멀면 초과분만 페널티 — (b) 가중치는
    # "임계값의 2배 지점에서 페널티가 Target 평균(103)만큼 되도록" 데이터
    # 스케일에 맞춰 역산한다. 마지막으로 이 가중치를 0.5x/2x로 흔들어도 GA
    # 결과가 안정적인지(음수·병적 해가 다시 나오지 않는지) 확인한다.
    Xi_mean, Xi_std = Xi.mean(), Xi.std().replace(0, 1)
    Xi_scaled = (Xi - Xi_mean) / Xi_std
    nn = NearestNeighbors(n_neighbors=2).fit(Xi_scaled.values)  # k=2: 0번째는 자기 자신
    loo_dist, _ = nn.kneighbors(Xi_scaled.values)
    typical_dist = float(np.quantile(loo_dist[:, 1], 0.90))
    NOVELTY_WEIGHT = float(y.mean() / typical_dist)  # 임계값의 2배 지점 penalty == mean(Target)
    log(f"[novelty penalty 보정] 관측 데이터 자체의 90th-pct 1-NN 거리(임계값)={typical_dist:.3f}, "
        f"이를 근거로 역산한 NOVELTY_WEIGHT={NOVELTY_WEIGHT:.2f} (이전 버전은 40.0을 임의 고정)")

    nn1 = NearestNeighbors(n_neighbors=1).fit(Xi_scaled.values)

    def novelty_dist(x_dict):
        vec = np.array([[(x_dict[c] - Xi_mean[c]) / Xi_std[c] for c in cols]])
        dist, _ = nn1.kneighbors(vec)
        return float(dist[0, 0])

    def novelty_penalty(x_dict, weight):
        return weight * max(0.0, novelty_dist(x_dict) - typical_dist)

    THICKNESS_MIN = 700.0
    PENALTY_WEIGHT = 2.0  # thickness 1nm 부족당 예측 Target에 더하는 페널티

    def penalized_objective(x_dict, novelty_weight=None):
        weight = NOVELTY_WEIGHT if novelty_weight is None else novelty_weight
        target = predict_target(x_dict)
        thick = predict_thickness(x_dict)
        shortfall = max(0.0, THICKNESS_MIN - thick)
        novelty = novelty_dist(x_dict)
        penalized = target + PENALTY_WEIGHT * shortfall + novelty_penalty(x_dict, weight)
        return penalized, target, thick, novelty

    results = {}

    # ---- (a) Bayesian Optimization (Optuna TPE) ----
    log("[Bayesian/Optuna] start (80 trials)")

    def optuna_objective(trial):
        x = {c: trial.suggest_float(c, bounds[c][0], bounds[c][1]) for c in cols}
        penalized, target, thick, novelty = penalized_objective(x)
        return penalized

    t0 = time.perf_counter()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(optuna_objective, n_trials=80, show_progress_bar=False)
    bo_seconds = time.perf_counter() - t0
    bo_x = study.best_params
    bo_pen, bo_target, bo_thick, bo_novelty = penalized_objective(bo_x)
    log(f"[Bayesian/Optuna] DONE — predicted Target={bo_target:.1f}, thickness={bo_thick:.1f}nm, "
        f"novelty(1-NN dist)={bo_novelty:.2f}, {bo_seconds:.1f}s, {len(study.trials)} evals")
    results["Bayesian Optimization (Optuna/TPE)"] = {
        "predicted_target": bo_target, "predicted_thickness": bo_thick, "novelty": bo_novelty,
        "seconds": bo_seconds, "n_evals": len(study.trials), "recipe": bo_x,
    }

    # ---- (b) Genetic Algorithm (수작업 구현: 실수 인코딩, 토너먼트선택, BLX교차, 가우시안돌연변이) ----
    n_dim = len(cols)
    lo = np.array([bounds[c][0] for c in cols])
    hi = np.array([bounds[c][1] for c in cols])
    pop_size, n_gen = 40, 60

    def decode(vec):
        return {c: float(v) for c, v in zip(cols, vec)}

    def run_ga(novelty_weight, seed, verbose=True):
        rng = np.random.default_rng(seed)

        def fitness(vec):
            pen, target, thick, novelty = penalized_objective(decode(vec), novelty_weight=novelty_weight)
            return pen

        pop = rng.uniform(lo, hi, size=(pop_size, n_dim))
        t0 = time.perf_counter()
        n_evals = 0
        for gen in range(n_gen):
            fits = np.array([fitness(ind) for ind in pop])
            n_evals += pop_size
            order = np.argsort(fits)
            pop = pop[order]
            fits = fits[order]
            elites = pop[:4].copy()

            children = [elites[0], elites[1], elites[2], elites[3]]
            while len(children) < pop_size:
                def tournament():
                    idx = rng.integers(0, pop_size, size=3)
                    best = idx[np.argmin(fits[idx])]
                    return pop[best]

                p1, p2 = tournament(), tournament()
                alpha = 0.3
                lo_c = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
                hi_c = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
                child = rng.uniform(lo_c, hi_c)
                if rng.random() < 0.2:
                    child += rng.normal(0, 0.05 * (hi - lo))
                child = np.clip(child, lo, hi)
                children.append(child)
            pop = np.array(children)
            if verbose and (gen + 1) % 20 == 0:
                log(f"  [GA w={novelty_weight:.1f}] gen {gen + 1}/{n_gen} best_penalized={fits[0]:.1f}")
        seconds = time.perf_counter() - t0
        final_fits = np.array([fitness(ind) for ind in pop])
        n_evals += pop_size
        best_x = decode(pop[np.argmin(final_fits)])
        pen, target, thick, novelty = penalized_objective(best_x, novelty_weight=novelty_weight)
        return best_x, target, thick, novelty, seconds, n_evals

    log(f"[Genetic Algorithm] start (pop={pop_size}, gen={n_gen}, novelty_weight={NOVELTY_WEIGHT:.2f})")
    ga_x, ga_target, ga_thick, ga_novelty, ga_seconds, n_evals = run_ga(NOVELTY_WEIGHT, RANDOM_STATE)
    log(f"[Genetic Algorithm] DONE — predicted Target={ga_target:.1f}, thickness={ga_thick:.1f}nm, "
        f"novelty(1-NN dist)={ga_novelty:.2f}, {ga_seconds:.1f}s, {n_evals} evals")

    # 한계 ⑥ 민감도 점검: 가중치를 0.5x/2x로 흔들어도 병적 해(음수 Target 등)가
    # 다시 나오지 않는지 확인 (시간 절약을 위해 세대 수를 30으로 줄인 축소판)
    log("[Genetic Algorithm] novelty_weight 민감도 점검 (0.5x / 2x, 세대수 축소)")
    n_gen_sens = 30
    sens_results = {}
    for mult in (0.5, 2.0):
        n_gen_backup = n_gen
        n_gen = n_gen_sens
        w = NOVELTY_WEIGHT * mult
        _, sens_target, sens_thick, sens_novelty, sens_s, _ = run_ga(w, RANDOM_STATE + 1, verbose=False)
        n_gen = n_gen_backup
        log(f"  [GA 민감도] weight={w:.2f}({mult}x) -> predicted Target={sens_target:.1f}, "
            f"thickness={sens_thick:.1f}nm, novelty={sens_novelty:.2f} "
            f"{'[!! 음수/병적]' if sens_target < 0 else '[정상 범위]'}")
        sens_results[f"{mult}x"] = {"weight": w, "target": sens_target, "thickness": sens_thick, "novelty": sens_novelty}
    results["Genetic Algorithm (custom)"] = {
        "predicted_target": ga_target, "predicted_thickness": ga_thick, "novelty": ga_novelty,
        "seconds": ga_seconds, "n_evals": n_evals, "recipe": ga_x,
        "novelty_weight_used": NOVELTY_WEIGHT, "novelty_threshold": typical_dist,
        "sensitivity": sens_results,
    }

    # ---- (c) 제약 최적화 (SLSQP, 명시적 부등식 제약: thickness>=700) ----
    # 트리 모델(LightGBM)의 목적함수는 계단형이라 원래 스케일에서 유한차분을 하면
    # 스텝이 분기점을 못 넘어 그래디언트가 거의 항상 0이 됨(SLSQP가 시작점에서
    # 못 움직임). 각 변수를 [0,1]로 정규화하고 finite-diff 스텝(eps)을 넉넉히
    # (정규화 범위의 3%) 줘서 분기점을 실제로 넘나들게 만든다.
    log("[Constrained/SLSQP] start (정규화 공간 + eps=0.03로 계단형 목적함수 대응)")
    rng_arr = hi - lo
    x0_norm = np.array([(x0[c] - bounds[c][0]) / rng_arr[i] for i, c in enumerate(cols)])

    def denorm(vec_norm):
        return decode(lo + np.clip(vec_norm, 0, 1) * rng_arr)

    n_evals_slsqp = [0]

    def slsqp_obj_counting(vec_norm):
        n_evals_slsqp[0] += 1
        return predict_target(denorm(vec_norm))

    def slsqp_constraint(vec_norm):
        return predict_thickness(denorm(vec_norm)) - THICKNESS_MIN  # >=0

    t0 = time.perf_counter()
    res = minimize(
        slsqp_obj_counting, x0_norm, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_dim,
        constraints=[{"type": "ineq", "fun": slsqp_constraint}],
        options={"maxiter": 100, "ftol": 1e-2, "eps": 0.03},
    )
    slsqp_seconds = time.perf_counter() - t0
    slsqp_x = denorm(res.x)
    slsqp_pen, slsqp_target, slsqp_thick, slsqp_novelty = penalized_objective(slsqp_x)
    moved = float(np.linalg.norm(res.x - x0_norm))
    log(f"[Constrained/SLSQP] DONE — success={res.success}, predicted Target={slsqp_target:.1f}, "
        f"thickness={slsqp_thick:.1f}nm, novelty={slsqp_novelty:.2f}, 정규화공간 이동거리={moved:.3f}, "
        f"{slsqp_seconds:.1f}s, {n_evals_slsqp[0]} evals")
    results["Constrained Optimization (SLSQP)"] = {
        "predicted_target": slsqp_target, "predicted_thickness": slsqp_thick, "novelty": slsqp_novelty,
        "seconds": slsqp_seconds, "n_evals": n_evals_slsqp[0], "recipe": slsqp_x,
        "converged": bool(res.success), "moved_in_normalized_space": moved,
    }

    # -----------------------------------------------------------------
    # 4) 현재 운영 평균(baseline) 대비 비교
    # -----------------------------------------------------------------
    baseline_x = x0  # 관측 데이터의 중앙값 레시피를 '현재 평균 운영 상태' 근사치로 사용
    baseline_pen, baseline_target, baseline_thick, baseline_novelty = penalized_objective(baseline_x)
    log(f"[Baseline] 관측 데이터 중앙값 레시피 predicted Target={baseline_target:.1f}, "
        f"thickness={baseline_thick:.1f}nm (실측 평균 Target={y.mean():.1f})")
    results["Baseline (관측 중앙값 레시피)"] = {
        "predicted_target": baseline_target, "predicted_thickness": baseline_thick, "novelty": baseline_novelty,
        "seconds": 0.0, "n_evals": 0, "recipe": baseline_x,
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "05_optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log("")
    log("=== 최적화 방법 비교 ===")
    summary = pd.DataFrame([
        {"method": k, "predicted_target": v["predicted_target"], "predicted_thickness": v["predicted_thickness"],
         "novelty": v.get("novelty"), "seconds": v["seconds"], "n_evals": v["n_evals"]}
        for k, v in results.items()
    ])
    log(summary.to_string(index=False))
    summary.to_csv(out_dir / "05_optimization_summary.csv", index=False)


if __name__ == "__main__":
    main()
