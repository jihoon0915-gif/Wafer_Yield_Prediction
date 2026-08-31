"""엔터프라이즈 패키징 — Streamlit 인터랙티브 대시보드 프론트엔드.

app_api.py(FastAPI, :8000)와 HTTP로만 통신한다(모델/피처 로직은 백엔드에만 있음 —
프론트는 순수 뷰). 실행: streamlit run app_ui.py
"""

from __future__ import annotations

import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="웨이퍼 수율 — 최종 모델 대시보드", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: "IBM Plex Sans KR", "Noto Sans KR", sans-serif; }
      div[data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; }
      .badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }
      .badge-good { background:#e5f6ec; color:#0c8a3e; }
      .badge-bad { background:#fbe9e9; color:#c93a3a; }
      .badge-warn { background:#fff4e0; color:#a56100; }
      .note { color:#838d9e; font-size:12.5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def get_meta() -> dict:
    r = requests.get(f"{API_URL}/meta", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def get_wafer_list(limit: int = 300) -> list[dict]:
    r = requests.get(f"{API_URL}/wafer-list", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()


def call_simulate(overrides: dict, model: str) -> dict:
    r = requests.post(f"{API_URL}/simulate", json={"overrides": overrides, "model": model, "top_n_shap": 15}, timeout=30)
    r.raise_for_status()
    return r.json()


def call_wafer_map(group_id: str) -> dict | None:
    r = requests.get(f"{API_URL}/wafer-map/{group_id}", timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def call_wafer_shap(group_id: str, model: str) -> dict | None:
    r = requests.get(f"{API_URL}/wafer-shap/{group_id}", params={"model": model}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def wafer_map_figure(grid: list[list[int]], title: str) -> go.Figure:
    colorscale = [[0.0, "#e8e9ec"], [0.34, "#e8e9ec"], [0.34, "#2fae66"], [0.67, "#2fae66"], [0.67, "#d64545"], [1.0, "#d64545"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=grid, colorscale=colorscale, zmin=0, zmax=2, showscale=False,
            xgap=1.5, ygap=1.5, hovertemplate="row %{y}, col %{x}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    fig.update_layout(title=dict(text=title, font=dict(size=13)), margin=dict(l=0, r=0, t=32, b=0), height=360,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def shap_waterfall_figure(base_value: float, contributions: list[dict], predicted: float, title: str, top_n: int = 12) -> go.Figure:
    top = contributions[:top_n]
    rest = contributions[top_n:]
    rest_sum = sum(c["shap_value"] for c in rest)

    labels = [c["feature"] for c in top]
    values = [c["shap_value"] for c in top]
    if rest and abs(rest_sum) > 0.01:
        labels.append(f"기타 {len(rest)}개 피처")
        values.append(rest_sum)

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(labels) + ["total"],
            x=["기준값(base)"] + labels + ["예측값"],
            y=[base_value] + values + [0],
            connector={"line": {"color": "rgba(150,150,150,0.5)"}},
            increasing={"marker": {"color": "#d64545"}},
            decreasing={"marker": {"color": "#2fae66"}},
            totals={"marker": {"color": "#4a4fe0"}},
            textposition="outside",
        )
    )
    fig.update_layout(title=dict(text=title, font=dict(size=13)), showlegend=False, height=460,
                       margin=dict(l=10, r=10, t=40, b=80))
    return fig


try:
    meta = get_meta()
except Exception:
    st.error(f"FastAPI 백엔드({API_URL})에 연결할 수 없습니다. `uvicorn app_api:app --reload`를 먼저 실행하세요.")
    st.stop()

sr = meta["feature_ranges"]

st.title("🔬 웨이퍼 수율 — 최종 모델 대시보드")
st.caption(
    "H1–H7 가설 검증 → 전처리 → GroupKFold(Lot_Num) 모델링 → SHAP 해석까지 반영한 최종 배포 모델의 "
    "예측/시뮬레이션/원인분석 대시보드입니다. GroupKFold(Lot_Num) 기준 R²≈0.46은 '완전히 새로운 Lot'에 "
    "대한 엄격한 추정치라, 웨이퍼 단위 무작위 분할(R²≈0.6–0.7, 이전 리포트)보다 보수적입니다."
)

with st.sidebar:
    st.header("모델 선택")
    model_names = list(meta["models"].keys())
    model = st.selectbox(
        "예측 모델", model_names, index=model_names.index("LightGBM") if "LightGBM" in model_names else 0,
        format_func=lambda m: f"{m} (R²={meta['models'][m]['r2_mean']:.3f}, {meta['models'][m]['n_features']}개 피처)",
    )
    st.caption(f"Best Hyperparameters: {meta['models'][model]['best_params']}")

    st.divider()
    st.header("공정 KPI (What-If)")
    st.caption("SHAP 상위 10개 KPI만 슬라이더로 노출. 나머지 29개 피처는 데이터셋 중앙값/최빈값 고정.")

    overrides: dict[str, float | str] = {}
    for feat in meta["top_kpi_features"]:
        rng = sr[feat]
        overrides[feat] = st.slider(
            feat, min_value=round(rng["q05"], 2), max_value=round(rng["q95"], 2),
            value=round(rng["median"], 2),
        )

    st.divider()
    st.subheader("범주형")
    for feat in meta["categorical_features"]:
        options = sr[feat]["options"]
        overrides[feat] = st.selectbox(feat, options, index=options.index(meta["feature_defaults"][feat]))

sim = call_simulate(overrides, model)

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("예측 Chip Yield (현재 설정)", f"{sim['modified']['predicted_yield_pct']:.2f}%",
          f"{sim['delta']['yield_pct_delta']:+.2f}%p vs 베이스라인(중앙값)")
c2.metric("예측 결함 다이 수", f"{sim['modified']['predicted_target']:.0f}",
          f"{sim['delta']['target_delta']:+.1f} vs 베이스라인", delta_color="inverse")
c3.metric("베이스라인 Yield(중앙값 설정)", f"{sim['baseline']['predicted_yield_pct']:.2f}%")
c4.metric("사용 모델", model, f"GroupKFold R²={meta['models'][model]['r2_mean']:.3f}")

badges = ""
for var in ["Thin F2", "Thin F3", "Thin F4"]:
    if var not in overrides:
        continue
    v = overrides[var]
    if v <= sr[var]["q20"]:
        badges += f'<span class="badge badge-good">{var} 스윗스팟</span>'
    elif v >= sr[var]["q80"]:
        badges += f'<span class="badge badge-bad">{var} 위험구간</span>'
if badges:
    st.markdown(badges, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------
# Wafer Map
# ---------------------------------------------------------------------
st.subheader("🗺️ 웨이퍼 맵 (실측)")
wafer_list = get_wafer_list()
wafer_options = {f"{w['group_id']} (Lot {w['Lot_Num']}, Target={w['Target']:.0f}, {w['error_class']})": w["group_id"] for w in wafer_list}
selected_label = st.selectbox("실측 웨이퍼 선택 (Die-level 불량 분포 조회)", list(wafer_options.keys()))
selected_group_id = wafer_options[selected_label]

col_map, col_shap_local = st.columns([1, 1.2])
wmap = call_wafer_map(selected_group_id)
with col_map:
    if wmap:
        st.plotly_chart(
            wafer_map_figure(wmap["wafer_map"], f"{wmap['group_id']} — 실제 Target={wmap['Target']:.0f}, Yield={wmap['yield_pct']:.2f}%"),
            width="stretch",
        )
        st.caption("🟩 정상 다이 · 🟥 결함 다이 · 회색 = 웨이퍼 영역 밖. 실제 다이 533개 기준.")
    else:
        st.warning("이 웨이퍼는 맵이 절단되어 복구 불가(원본 데이터 특성상 17% 발생) — 다른 웨이퍼를 선택하세요.")

with col_shap_local:
    wshap = call_wafer_shap(selected_group_id, model)
    if wshap:
        st.plotly_chart(
            shap_waterfall_figure(
                wshap["base_value"], wshap["shap_contributions"], wshap["predicted_target"],
                f"이 웨이퍼의 SHAP 근거 (실제={wshap['actual_target']:.0f} vs 예측={wshap['predicted_target']:.1f})",
            ),
            width="stretch",
        )
        st.caption(
            f"실제 불량패턴: **{wshap['actual_error_class']}**. Waterfall의 각 막대는 이 웨이퍼의 "
            f"실측 공정값이 예측을 기준값({wshap['base_value']:.1f}, 전체 평균)에서 얼마나 밀어올리거나 "
            "끌어내렸는지를 나타냅니다."
        )

st.divider()

# ---------------------------------------------------------------------
# What-If SHAP (현재 슬라이더 설정)
# ---------------------------------------------------------------------
st.subheader("🎛️ What-If 시뮬레이션 — 현재 슬라이더 설정의 SHAP 근거")
st.plotly_chart(
    shap_waterfall_figure(
        sim["base_value"], sim["modified_shap_full"], sim["modified"]["predicted_target"],
        f"현재 슬라이더 설정 예측 근거 (예측={sim['modified']['predicted_target']:.1f})",
    ),
    width="stretch",
)

with st.expander("베이스라인 대비 SHAP 기여도 변화 (슬라이더로 무엇이 얼마나 움직였나)"):
    st.dataframe(sim["shap_contribution_changes"], width="stretch")

st.divider()

# ---------------------------------------------------------------------
# 글로벌 SHAP 참고 이미지
# ---------------------------------------------------------------------
st.subheader("📊 참고 — 전체 데이터 기준 SHAP 글로벌 중요도")
img_col1, img_col2 = st.columns(2)
try:
    img_col1.image("reports/figures/shap_summary.png", caption="Beeswarm (LightGBM, 전체 1,704웨이퍼)", width="stretch")
    img_col2.image("reports/figures/shap_summary_bar.png", caption="Bar (LightGBM)", width="stretch")
except Exception:
    st.info("scripts/13_model_interpretation_shap.py를 먼저 실행해 reports/figures/shap_summary*.png를 생성하세요.")

st.caption(
    "모델: GroupKFold(Lot_Num) 기준 R²="
    f"{meta['models'][model]['r2_mean']:.3f}±{meta['models'][model]['r2_std']:.3f}, "
    f"MAE={meta['models'][model]['mae_mean']:.1f}. "
    "⚠️ 예측 기여도(SHAP)는 인과관계와 다릅니다 — UV_type처럼 Lot 교란 통제 후 사라진 신호도 있었으니 "
    "실제 공정 변경 전 reports/hypothesis_H1_H7_summary.md를 교차확인하세요. "
    "⚠️ 결함이 극심한 웨이퍼일수록 계측 자체가 결측인 사례가 있어(reports/model_interpretation_report.md 3절), "
    "해당 구간 예측은 특히 보수적으로 해석해야 합니다."
)
