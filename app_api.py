"""엔터프라이즈 패키징 — FastAPI 추론/시뮬레이션 백엔드.

models/final_model.pkl(배포 모델, LightGBM) + models/all_candidate_models.pkl
(RandomForest/XGBoost/LightGBM 3종 전부)을 로드해 예측·What-If 시뮬레이션·
SHAP 기여도·웨이퍼맵 조회를 REST API로 제공한다.

엔드포인트:
  GET  /meta            대시보드 초기화용 메타데이터(피처 범위/기본값, 모델 목록, 성능)
  POST /predict         단일/배치 웨이퍼 결함수 예측 (+ 이상 탐지 경보)
  POST /simulate        (비동기) KPI 변경 전/후 예측치 + SHAP 기여도 변화 (+ 경보)
  GET  /wafer-map/{id}   group_id(예: "13_28") 기준 die-level 26x26 실측 불량 맵 조회
  GET  /alerts          최근 경보 이력
  GET  /alerts/config   경보 임계값 + Slack 연동 상태
  POST /alerts/test     Slack 웹훅 설정 검증용 테스트 메시지

실행: uvicorn app_api:app --reload
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import json

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wafer import alerting, slack
from wafer.final_pipeline import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TOP_KPI_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REAL_DIE_COUNT = 533  # wafer.config.WAFER_MAP_REAL_DIE_COUNT

load_dotenv(PROJECT_ROOT / ".env")  # SLACK_WEBHOOK_URL 등 — 없으면 조용히 넘어감

app = FastAPI(
    title="Wafer Yield — Final Model Inference & Simulation API",
    description="GroupKFold(Lot_Num)로 검증한 웨이퍼 결함수 예측 모델의 추론·What-If 시뮬레이션·"
    "SHAP 원인분석·이상 탐지 경보 API",
    version="1.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------
# 시작 시 1회 로드
# ---------------------------------------------------------------------
all_candidates: dict = joblib.load(MODELS_DIR / "all_candidate_models.pkl")
final_pipeline = joblib.load(MODELS_DIR / "final_model.pkl")  # == all_candidates["LightGBM"]["pipeline"]
final_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_final_feature_matrix.csv")

with open(MODELS_DIR / "wafer_lookup.json", encoding="utf-8") as f:
    wafer_lookup = {w["group_id"]: w for w in json.load(f)}

DEFAULT_MODEL = "LightGBM"

# 피처 기본값(중앙값/최빈값)·범위 -- 슬라이더 초기화 및 override 병합용
FEATURE_DEFAULTS: dict[str, float | str] = {}
FEATURE_RANGES: dict[str, dict[str, float]] = {}
for col in NUMERIC_FEATURES:
    s = final_df[col].astype(float)  # oxidation/etching/ion_implant_sentinel_flag는 bool dtype -> quantile 계산 불가라 캐스팅
    FEATURE_DEFAULTS[col] = float(s.median())
    FEATURE_RANGES[col] = {
        "min": float(s.min()), "max": float(s.max()),
        "q05": float(s.quantile(0.05)), "q95": float(s.quantile(0.95)),
        "q20": float(s.quantile(0.20)), "q80": float(s.quantile(0.80)),
        "median": float(s.median()),
    }
for col in CATEGORICAL_FEATURES:
    mode_val = final_df[col].mode().iloc[0]
    FEATURE_DEFAULTS[col] = mode_val
    FEATURE_RANGES[col] = {"options": sorted(final_df[col].dropna().unique().tolist())}

# 모델별 SHAP TreeExplainer(순수 트리 모델에만 적용 가능 -- Pipeline이 아니라
# pipeline.named_steps["model"]에 붙여야 함)
EXPLAINERS: dict[str, shap.TreeExplainer] = {
    name: shap.TreeExplainer(entry["pipeline"].named_steps["model"])
    for name, entry in all_candidates.items()
}

# 경보 임계값은 하드코딩하지 않고 실제 학습 데이터에서 계산한다
ALERT_THRESHOLDS = alerting.AlertThresholds.from_frame(final_df)
ALERT_LOG = alerting.AlertLog(maxlen=100, cooldown_seconds=300.0)


def yield_pct(target: float) -> float:
    return round((REAL_DIE_COUNT - max(0.0, target)) / REAL_DIE_COUNT * 100, 3)


def validate_model_name(name: str) -> str:
    if name not in all_candidates:
        raise HTTPException(400, f"알 수 없는 모델: {name}. 사용 가능: {list(all_candidates)}")
    return name


def validate_overrides(overrides: dict) -> None:
    unknown = [k for k in overrides if k not in ALL_FEATURES]
    if unknown:
        raise HTTPException(422, f"알 수 없는 피처: {unknown}. /meta에서 유효한 피처 목록을 확인하세요.")


def build_row_df(overrides: dict) -> pd.DataFrame:
    """override에 명시적으로 null을 넣으면 NaN으로 남긴다 — 계측 결측(Gate 0)을
    '중앙값으로 조용히 메우지 않고' 결측 그대로 모델과 경보 규칙에 전달하기 위함."""
    row = dict(FEATURE_DEFAULTS)
    row.update(overrides)
    ordered = {c: (np.nan if row[c] is None else row[c]) for c in ALL_FEATURES}
    return pd.DataFrame([ordered])


def queue_alerts(background: BackgroundTasks, alerts: list[alerting.Alert]) -> list[dict]:
    """쿨다운을 통과한 경보만 이력에 남기고 Slack 전송은 응답 이후로 미룬다
    (웹훅 왕복시간이 추론 응답을 붙잡지 않도록)."""
    fresh = ALERT_LOG.record(alerts)
    if fresh:
        background.add_task(slack.send_alerts, fresh)
    return [a.to_dict() for a in alerts]


def selected_matrix(model_name: str, X_raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    entry = all_candidates[model_name]
    pipe = entry["pipeline"]
    X_transformed = pipe.named_steps["prep"].transform(X_raw)
    X_selected = pipe.named_steps["select"].transform(X_transformed)
    feat_names = [f.replace("num__", "").replace("cat__", "") for f in entry["selected_features"]]
    return X_selected, feat_names


def raw_value_lookup(row_df: pd.DataFrame, feat: str):
    """SHAP은 스케일된(z-score) 값을 다루지만, 사람이 읽을 땐 원본 물리단위가
    필요하다(13번 스크립트에서 한 번 겪은 실수 -- 여기서도 똑같이 재발할 뻔했다).
    row_df는 build_row_df()/원본 wafer row처럼 스케일 전 원본 컬럼을 가진 1행 DataFrame."""
    r = row_df.iloc[0]
    if feat in r.index:
        val = r[feat]
        if isinstance(val, (bool, np.bool_)):
            # sentinel_flag 계열 컬럼 -- numpy.bool_은 int/float 서브클래스가 아니라
            # 아래 분기를 그냥 타면 float(bool)로 잘못 바뀌고, 그렇다고 그대로
            # 반환하면 FastAPI의 jsonable_encoder가 numpy.bool_을 직렬화하지 못해
            # 500 에러가 난다(XGBoost의 oxidation_sentinel_flag에서 실제 발생).
            return bool(val)
        if isinstance(val, (int, float, np.floating, np.integer)):
            # NaN은 JSON 규격상 표현 불가(Starlette JSONResponse는 allow_nan=False) —
            # 계측 결측을 null로 내보낸다.
            return None if np.isnan(float(val)) else float(val)
        return val
    for cat_col in CATEGORICAL_FEATURES:
        prefix = f"{cat_col}_"
        if feat.startswith(prefix):
            category = feat[len(prefix):]
            return f"{cat_col}={r[cat_col]}"
    return None


class MetaResponse(BaseModel):
    models: dict
    feature_defaults: dict
    feature_ranges: dict
    top_kpi_features: list[str]
    categorical_features: list[str]
    real_die_count: int


class PredictRequest(BaseModel):
    wafers: list[dict[str, float | str | None]] = Field(
        ..., description="웨이퍼별 피처 override dict 목록(1개=단일, N개=배치). "
        "지정 안 한 피처는 데이터셋 중앙값/최빈값으로 채워짐. "
        "값을 null로 보내면 '계측 결측'으로 처리되어 Gate 0 경보 대상이 된다."
    )
    model: str = DEFAULT_MODEL


class SimulateRequest(BaseModel):
    overrides: dict[str, float | str | None] = Field(default_factory=dict, description="베이스라인 대비 변경할 KPI")
    model: str = DEFAULT_MODEL
    top_n_shap: int = 10


@app.get("/")
def health():
    return {"status": "ok", "models": list(all_candidates), "n_wafer_lookup": len(wafer_lookup)}


@app.get("/meta", response_model=MetaResponse)
def meta():
    return MetaResponse(
        models={
            name: {
                "n_features": len(entry["selected_features"]),
                "r2_mean": round(entry["r2_mean"], 4),
                "r2_std": round(entry["r2_std"], 4),
                "mae_mean": round(entry["mae_mean"], 2),
                "best_params": entry["best_params"],
            }
            for name, entry in all_candidates.items()
        },
        feature_defaults=FEATURE_DEFAULTS,
        feature_ranges=FEATURE_RANGES,
        top_kpi_features=TOP_KPI_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        real_die_count=REAL_DIE_COUNT,
    )


@app.post("/predict")
def predict(req: PredictRequest, background: BackgroundTasks):
    validate_model_name(req.model)
    for w in req.wafers:
        validate_overrides(w)

    pipe = all_candidates[req.model]["pipeline"]
    row_frames = [build_row_df(w) for w in req.wafers]
    rows = pd.concat(row_frames, ignore_index=True)
    preds = pipe.predict(rows)
    preds = np.maximum(0.0, preds)  # 물리적 하한(4단계에서 확립된 원칙)

    predictions = []
    for i, (frame, pred) in enumerate(zip(row_frames, preds)):
        pred = float(pred)
        alerts = alerting.evaluate(
            frame.iloc[0].to_dict(), pred, ALERT_THRESHOLDS,
            context=f"predict:{req.model}", subject=f"예측 요청 #{i + 1}",
        )
        predictions.append({
            "predicted_target": round(pred, 1),
            "predicted_yield_pct": yield_pct(pred),
            "alerts": queue_alerts(background, alerts),
        })

    return {"model": req.model, "predictions": predictions}


def explain_row(model_name: str, row_df: pd.DataFrame) -> dict:
    """한 행에 대한 예측 + SHAP 전체 분해(base_value 포함, waterfall 구성용)."""
    pipe = all_candidates[model_name]["pipeline"]
    explainer = EXPLAINERS[model_name]
    pred = max(0.0, float(pipe.predict(row_df)[0]))
    X_sel, feat_names = selected_matrix(model_name, row_df)
    shap_vals = explainer.shap_values(X_sel)[0]
    order = np.argsort(-np.abs(shap_vals))
    contributions = [
        {
            "feature": feat_names[i],
            "shap_value": round(float(shap_vals[i]), 3),
            "feature_value_scaled": round(float(X_sel[0, i]), 3),  # SHAP 계산에 실제 쓰인 z-score
            "feature_value_raw": raw_value_lookup(row_df, feat_names[i]),  # 사람이 읽을 원본 물리단위
        }
        for i in order
    ]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]  # 회귀인데도 1-원소 배열로 오는 SHAP 버전 대응
    return {
        "predicted_target": round(pred, 1),
        "predicted_yield_pct": yield_pct(pred),
        "base_value": round(float(base_value), 3),
        "shap_contributions": contributions,
    }


def _simulate_sync(overrides: dict, model_name: str, top_n_shap: int) -> dict:
    baseline_row = build_row_df({})
    modified_row = build_row_df(overrides)

    baseline = explain_row(model_name, baseline_row)
    modified = explain_row(model_name, modified_row)

    base_shap = {c["feature"]: c["shap_value"] for c in baseline["shap_contributions"]}
    mod_shap = {c["feature"]: c["shap_value"] for c in modified["shap_contributions"]}
    deltas = sorted(
        ({"feature": f, "shap_baseline": base_shap[f], "shap_modified": mod_shap[f],
          "shap_delta": round(mod_shap[f] - base_shap[f], 3)} for f in base_shap),
        key=lambda d: -abs(d["shap_delta"]),
    )[:top_n_shap]

    return {
        "model": model_name,
        "overrides_applied": overrides,
        "baseline": {"predicted_target": baseline["predicted_target"], "predicted_yield_pct": baseline["predicted_yield_pct"]},
        "modified": {"predicted_target": modified["predicted_target"], "predicted_yield_pct": modified["predicted_yield_pct"]},
        "delta": {
            "target_delta": round(modified["predicted_target"] - baseline["predicted_target"], 2),
            "yield_pct_delta": round(modified["predicted_yield_pct"] - baseline["predicted_yield_pct"], 3),
        },
        "shap_contribution_changes": deltas,
        "base_value": modified["base_value"],
        "modified_shap_full": modified["shap_contributions"],
    }


@app.post("/simulate")
async def simulate(req: SimulateRequest, background: BackgroundTasks):
    validate_model_name(req.model)
    validate_overrides(req.overrides)
    # SHAP 계산은 CPU-bound라 이벤트 루프를 막지 않도록 스레드풀로 오프로드한다
    # (async def만 붙이고 동기 계산을 그대로 두면 진짜 비동기가 아니라 껍데기만
    # 비동기인 것 -- 그 차이를 실제로 지키기 위해 asyncio.to_thread 사용).
    result = await asyncio.to_thread(_simulate_sync, req.overrides, req.model, req.top_n_shap)

    alerts = alerting.evaluate(
        build_row_df(req.overrides).iloc[0].to_dict(),
        result["modified"]["predicted_target"],
        ALERT_THRESHOLDS,
        context=f"simulate:{req.model}", subject="What-If 슬라이더 설정",
    )
    result["alerts"] = queue_alerts(background, alerts)
    return result


@app.get("/wafer-shap/{group_id}")
async def wafer_shap(group_id: str, background: BackgroundTasks, model: str = DEFAULT_MODEL):
    """실측 웨이퍼 하나의 실제 공정 피처값으로 예측 + SHAP 전체 분해(waterfall 구성용).
    /simulate가 슬라이더 What-If용이라면, 이건 '실제 이 웨이퍼는 왜 이렇게 나왔나'용."""
    validate_model_name(model)
    match = final_df[final_df["group_id"] == group_id]
    if match.empty:
        raise HTTPException(404, f"group_id={group_id!r}의 공정 피처 데이터가 없습니다.")
    row = match.iloc[[0]][ALL_FEATURES]
    result = await asyncio.to_thread(explain_row, model, row)
    result["group_id"] = group_id
    result["actual_target"] = float(match.iloc[0]["Target"])
    result["actual_error_class"] = match.iloc[0]["error_class"]
    result["model"] = model
    result["alerts"] = queue_alerts(
        background,
        alerting.evaluate(
            row.iloc[0].to_dict(), result["predicted_target"], ALERT_THRESHOLDS,
            context=f"wafer:{group_id}", subject=f"실측 웨이퍼 {group_id}",
        ),
    )
    return result


@app.get("/wafer-map/{group_id}")
def wafer_map(group_id: str):
    if group_id not in wafer_lookup:
        raise HTTPException(404, f"group_id={group_id!r} 를 찾을 수 없습니다(실측 웨이퍼맵 파싱 성공분 {len(wafer_lookup)}개 중 없음).")
    w = wafer_lookup[group_id]
    return {
        "group_id": w["group_id"],
        "Lot_Num": w["Lot_Num"],
        "Wafer_Num": w["Wafer_Num"],
        "Target": w["Target"],
        "yield_pct": yield_pct(w["Target"]),
        "error_class": w["error_class"],
        "wafer_map": w["wafer_map"],
    }


@app.get("/wafer-list")
def wafer_list(limit: int = 200):
    items = list(wafer_lookup.values())[:limit]
    return [{"group_id": w["group_id"], "Lot_Num": w["Lot_Num"], "Target": w["Target"], "error_class": w["error_class"]} for w in items]


# ---------------------------------------------------------------------
# 이상/불량 탐지 경보
# ---------------------------------------------------------------------
@app.get("/alerts")
def alerts(limit: int = 20):
    """최근 경보 이력. Slack 웹훅이 없어도 경보 판정 자체는 항상 기록되므로,
    이 엔드포인트만으로 규칙이 동작하는지 확인할 수 있다."""
    return {"slack": slack.config_status(), "alerts": ALERT_LOG.recent(limit)}


@app.get("/alerts/config")
def alerts_config():
    return {"thresholds": ALERT_THRESHOLDS.to_dict(), "slack": slack.config_status()}


@app.post("/alerts/test")
def alerts_test():
    """Slack 웹훅 설정 검증용 — 실제 경보와 구분되는 테스트 메시지를 보낸다."""
    result = slack.send_test_message()
    if not result["sent"] and result.get("reason") == "webhook_not_configured":
        raise HTTPException(
            409,
            f"Slack 웹훅이 설정되지 않았습니다. 환경변수 {slack.WEBHOOK_ENV}에 Incoming Webhook URL을 넣고 재시작하세요.",
        )
    return result
