"""웨이퍼 수율 예측 — Streamlit 대시보드 (프론트엔드).

모델/피처/경보 로직은 전부 app_api.py(FastAPI)에 있고 이 파일은 순수 뷰다.
HTTP로만 통신하므로 백엔드를 어디에 두든(로컬·별도 호스트·같은 컨테이너) 동일하게 동작한다.

실행 방식 두 가지:
  1) 로컬 2-프로세스 — `uvicorn app_api:app --reload` + `streamlit run app_ui.py`
  2) 단일 프로세스 — `streamlit run app_ui.py` 만 실행. API가 안 떠 있으면
     같은 프로세스 안에서 백그라운드 스레드로 기동한다(Streamlit Community Cloud처럼
     포트를 하나만 열어주는 호스팅용).

환경변수:
  WAFER_API_URL   외부에 배포한 API를 쓸 때 지정(지정 시 내장 기동 안 함)
  WAFER_API_PORT  내장 기동 포트 (기본 8000)
  SLACK_WEBHOOK_URL  설정 시 경보를 Slack으로 전송
"""

from __future__ import annotations

import html
import os
import threading
import time

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="웨이퍼 수율 예측 시스템",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_URL = "https://github.com/jihoon0915-gif/Wafer_Yield_Prediction"
EMBEDDED_API_PORT = int(os.environ.get("WAFER_API_PORT", "8000"))
API_URL = os.environ.get("WAFER_API_URL", f"http://127.0.0.1:{EMBEDDED_API_PORT}").rstrip("/")

SEVERITY_STYLE = {
    "critical": ("🔴", "심각", "#c0392b", "#fdecea"),
    "warning": ("🟠", "경고", "#a56100", "#fff6e5"),
}


# ---------------------------------------------------------------------
# 백엔드 부트스트랩
# ---------------------------------------------------------------------
def bridge_secrets() -> None:
    """Streamlit Cloud의 Secrets를 환경변수로 옮긴다.

    app_api / wafer.slack 은 streamlit에 의존하지 않고 os.environ만 읽도록 유지하고
    싶어서(별도 호스트에 API만 따로 배포할 수 있어야 함) 여기서 한 번 연결해 준다.
    secrets.toml이 아예 없는 로컬 실행에서는 st.secrets 접근 자체가 예외라 감싼다.
    """
    for key in ("SLACK_WEBHOOK_URL", "SLACK_ALERT_MIN_SEVERITY"):
        try:
            value = st.secrets[key]
        except Exception:
            continue
        os.environ.setdefault(key, str(value))


def api_alive(timeout: float = 1.5) -> bool:
    try:
        return requests.get(f"{API_URL}/", timeout=timeout).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="추론 API를 기동하는 중입니다…")
def ensure_api() -> str:
    """API에 연결할 수 있는 상태를 보장하고, 어떤 방식으로 연결됐는지 반환한다."""
    bridge_secrets()

    if os.environ.get("WAFER_API_URL"):
        return "external" if api_alive(timeout=5.0) else "unreachable"
    if api_alive():
        return "separate-process"

    import uvicorn

    import app_api  # 무거운 로드(모델 12MB + SHAP explainer)라 필요할 때만 import

    def serve() -> None:
        config = uvicorn.Config(app_api.app, host="127.0.0.1", port=EMBEDDED_API_PORT, log_level="warning")
        # uvicorn은 메인 스레드가 아니면 시그널 핸들러 등록을 알아서 건너뛴다
        uvicorn.Server(config).run()

    threading.Thread(target=serve, daemon=True, name="wafer-api").start()

    for _ in range(60):
        if api_alive(timeout=1.0):
            return "embedded"
        time.sleep(0.5)
    return "unreachable"


# ---------------------------------------------------------------------
# API 클라이언트
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_meta() -> dict:
    r = requests.get(f"{API_URL}/meta", timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def get_alert_config() -> dict:
    r = requests.get(f"{API_URL}/alerts/config", timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def get_wafer_list(limit: int = 300) -> list[dict]:
    r = requests.get(f"{API_URL}/wafer-list", params={"limit": limit}, timeout=15)
    r.raise_for_status()
    return r.json()


def call_simulate(overrides: dict, model: str) -> dict:
    r = requests.post(
        f"{API_URL}/simulate",
        json={"overrides": overrides, "model": model, "top_n_shap": 15},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def call_wafer_map(group_id: str) -> dict | None:
    r = requests.get(f"{API_URL}/wafer-map/{group_id}", timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def call_wafer_shap(group_id: str, model: str) -> dict | None:
    r = requests.get(f"{API_URL}/wafer-shap/{group_id}", params={"model": model}, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def get_alert_history(limit: int = 15) -> dict:
    r = requests.get(f"{API_URL}/alerts", params={"limit": limit}, timeout=15)
    r.raise_for_status()
    return r.json()


def send_slack_test() -> tuple[bool, str]:
    r = requests.post(f"{API_URL}/alerts/test", timeout=20)
    if r.ok:
        return True, "Slack 채널로 테스트 메시지를 전송했습니다."
    return False, r.json().get("detail", r.text)


# ---------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------
def wafer_map_figure(grid: list[list[int]], title: str) -> go.Figure:
    colorscale = [
        [0.0, "#eceef2"], [0.34, "#eceef2"],
        [0.34, "#2f9e6e"], [0.67, "#2f9e6e"],
        [0.67, "#d64545"], [1.0, "#d64545"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=grid, colorscale=colorscale, zmin=0, zmax=2, showscale=False,
            xgap=1.5, ygap=1.5, hovertemplate="row %{y}, col %{x}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=0, r=0, t=34, b=0), height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_waterfall_figure(base_value: float, contributions: list[dict], title: str, top_n: int = 12) -> go.Figure:
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
            connector={"line": {"color": "rgba(150,150,150,0.45)"}},
            increasing={"marker": {"color": "#d64545"}},
            decreasing={"marker": {"color": "#2f9e6e"}},
            totals={"marker": {"color": "#3d5afe"}},
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)), showlegend=False, height=470,
        margin=dict(l=10, r=10, t=40, b=90),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def format_metric_value(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def alert_card_html(alert: dict) -> str:
    icon, label, fg, bg = SEVERITY_STYLE.get(alert["severity"], ("⚪", alert["severity"], "#555", "#f2f2f2"))
    metrics = " · ".join(
        f"{html.escape(k)}: {html.escape(format_metric_value(v))}" for k, v in alert["metrics"].items()
    )
    return f"""
    <div class="alert-card" style="border-left:4px solid {fg}; background:{bg};">
      <div class="alert-head" style="color:{fg};">{icon} [{label}] {html.escape(alert['title'])}</div>
      <div class="alert-body">{html.escape(alert['detail'])}</div>
      <div class="alert-meta">측정값 — {metrics}</div>
      <div class="alert-meta">판정 근거 — {html.escape(alert['evidence'])} · 규칙 <code>{html.escape(alert['rule'])}</code></div>
    </div>
    """


# ---------------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: "IBM Plex Sans KR", "Pretendard", "Noto Sans KR", sans-serif; }
      div[data-testid="stMetricValue"] { font-family: "IBM Plex Mono", ui-monospace, monospace; }

      .hero { border:1px solid #e3e6ec; border-radius:14px; padding:22px 26px; margin-bottom:18px;
              background:linear-gradient(135deg,#f7f9fc 0%,#eef2f9 100%); }
      .hero h1 { margin:0 0 6px 0; font-size:27px; letter-spacing:-0.5px; }
      .hero p { margin:0; color:#5b6577; font-size:14px; line-height:1.65; }
      .hero .links { margin-top:12px; font-size:13px; }
      .hero .links a { color:#3d5afe; text-decoration:none; margin-right:16px; font-weight:600; }

      .pill { display:inline-block; padding:3px 11px; border-radius:999px; font-size:12px;
              font-weight:600; margin:0 6px 6px 0; }
      .pill-good { background:#e5f6ec; color:#0c8a3e; }
      .pill-bad  { background:#fbe9e9; color:#c0392b; }
      .pill-neutral { background:#eef1f6; color:#4a5568; }

      .alert-card { border-radius:9px; padding:13px 16px; margin-bottom:10px; }
      .alert-head { font-weight:700; font-size:14.5px; margin-bottom:5px; }
      .alert-body { font-size:13.5px; color:#2d3340; line-height:1.6; }
      .alert-meta { font-size:12px; color:#6b7280; margin-top:5px; }
      .alert-meta code { background:rgba(0,0,0,0.05); padding:1px 5px; border-radius:4px; }

      .ok-box { border:1px dashed #cfd6e0; border-radius:9px; padding:14px 16px;
                color:#5b6577; font-size:13.5px; background:#fbfcfe; }
      .section-note { color:#78839a; font-size:12.5px; margin:-6px 0 10px 0; }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 부트스트랩 + 메타데이터
# ---------------------------------------------------------------------
api_mode = ensure_api()
if api_mode == "unreachable":
    st.error(
        f"추론 API({API_URL})에 연결할 수 없습니다.\n\n"
        "로컬에서는 `uvicorn app_api:app --reload`를 먼저 실행하거나, "
        "`WAFER_API_URL` 환경변수를 비우고 `streamlit run app_ui.py`만 실행하세요."
    )
    st.stop()

meta = get_meta()
alert_config = get_alert_config()
sr = meta["feature_ranges"]
slack_status = alert_config["slack"]
thresholds = alert_config["thresholds"]

st.markdown(
    f"""
    <div class="hero">
      <h1>🔬 웨이퍼 수율 예측 및 이상 탐지 시스템</h1>
      <p>
        6개 반도체 공정 데이터를 통합해 웨이퍼 결함 다이 수를 예측하고, 13개 가설 검증으로
        선별한 피처만 사용해 <b>GroupKFold(Lot_Num)</b> — 학습에 없던 새 Lot 기준 — 으로 성능을 확정한 모델입니다.
        예측·What-If 시뮬레이션·SHAP 원인분석·이상 탐지 경보를 하나의 인터페이스로 제공합니다.
      </p>
      <div class="links">
        <a href="{REPO_URL}" target="_blank">GitHub 저장소</a>
        <a href="{API_URL}/docs" target="_blank">API 문서 (OpenAPI)</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader("예측 모델")
    model_names = list(meta["models"].keys())
    model = st.selectbox(
        "모델 선택", model_names,
        index=model_names.index("LightGBM") if "LightGBM" in model_names else 0,
        format_func=lambda m: f"{m} · R²={meta['models'][m]['r2_mean']:.3f} · {meta['models'][m]['n_features']}개 피처",
        label_visibility="collapsed",
    )
    m = meta["models"][model]
    st.caption(f"R² {m['r2_mean']:.4f} ± {m['r2_std']:.4f} · MAE {m['mae_mean']:.2f} · GroupKFold(Lot_Num)")
    with st.expander("하이퍼파라미터"):
        st.json(m["best_params"], expanded=False)

    st.divider()
    st.subheader("공정 조건 (What-If)")
    st.caption(
        f"SHAP 상위 {len(meta['top_kpi_features'])}개 KPI만 조작 가능하며, "
        "나머지 피처는 데이터셋 중앙값/최빈값으로 고정됩니다."
    )

    gate0_demo = st.toggle(
        "계측 결측 상황 재현 (Gate 0)",
        value=False,
        help="Thin F2/F3/F4를 결측(null)으로 전송합니다. 모델은 결측을 중앙값으로 대치해 "
             "예측값이 정상처럼 보이지만, 경보 규칙은 결측 자체를 잡아냅니다.",
    )

    overrides: dict[str, float | str | None] = {}
    for feat in meta["top_kpi_features"]:
        rng = sr[feat]
        overrides[feat] = st.slider(
            feat,
            min_value=round(rng["q05"], 2),
            max_value=round(rng["q95"], 2),
            value=round(rng["median"], 2),
        )

    st.markdown("**범주형 조건**")
    for feat in meta["categorical_features"]:
        options = sr[feat]["options"]
        overrides[feat] = st.selectbox(feat, options, index=options.index(meta["feature_defaults"][feat]))

    if gate0_demo:
        for feat in thresholds["metrology_watch_features"]:
            overrides[feat] = None

    st.divider()
    st.subheader("Slack 연동")
    if slack_status["enabled"]:
        st.success(f"활성 · 최소 심각도 `{slack_status['min_severity']}`")
        if st.button("테스트 메시지 전송", width="stretch"):
            ok, msg = send_slack_test()
            (st.success if ok else st.error)(msg)
    else:
        st.info(
            "미설정 — 경보는 아래 이력에만 기록됩니다.\n\n"
            f"`{slack_status['webhook_env_var']}` 환경변수에 Incoming Webhook URL을 넣으면 "
            "Slack 채널로 전송됩니다."
        )

# ---------------------------------------------------------------------
# KPI + 경보
# ---------------------------------------------------------------------
sim = call_simulate(overrides, model)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "예측 Chip Yield", f"{sim['modified']['predicted_yield_pct']:.2f}%",
    f"{sim['delta']['yield_pct_delta']:+.2f}%p vs 베이스라인",
)
c2.metric(
    "예측 결함 다이 수", f"{sim['modified']['predicted_target']:.0f}",
    f"{sim['delta']['target_delta']:+.1f} vs 베이스라인", delta_color="inverse",
)
c3.metric("베이스라인 Yield", f"{sim['baseline']['predicted_yield_pct']:.2f}%", "전 피처 중앙값 설정")
c4.metric("관리상한 (μ+3σ)", f"{thresholds['predicted_target']['ucl_3sigma']:.0f}", "결함 다이 수 기준")

pills = ""
for var in thresholds["metrology_watch_features"]:
    value = overrides.get(var)
    if value is None:
        pills += f'<span class="pill pill-neutral">{var} 계측 결측</span>'
    elif value <= sr[var]["q20"]:
        pills += f'<span class="pill pill-good">{var} 스윗스팟</span>'
    elif value >= sr[var]["q80"]:
        pills += f'<span class="pill pill-bad">{var} 리스크존</span>'
if pills:
    st.markdown(pills, unsafe_allow_html=True)

st.subheader("🔔 이상 탐지 경보")
st.markdown(
    '<div class="section-note">계측 결측(Gate 0) · 식각 잔막 리스크존 · 예측 관리상한 초과 '
    '세 규칙을 매 요청마다 평가합니다. 임계값은 학습 데이터에서 계산된 값입니다.</div>',
    unsafe_allow_html=True,
)
if sim["alerts"]:
    for alert in sim["alerts"]:
        st.markdown(alert_card_html(alert), unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="ok-box">현재 설정에서 발생한 경보가 없습니다. '
        '사이드바에서 <b>계측 결측 상황 재현(Gate 0)</b>을 켜거나 Thin F2/F3/F4를 상위 구간으로 올리면 '
        '경보가 발생합니다.</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------
# 웨이퍼 맵 + 로컬 SHAP
# ---------------------------------------------------------------------
st.subheader("🗺️ 실측 웨이퍼 분석")
st.markdown(
    '<div class="section-note">실제 웨이퍼의 die 단위 불량 분포와, 그 웨이퍼의 예측 근거를 SHAP으로 분해합니다.</div>',
    unsafe_allow_html=True,
)

wafer_list = get_wafer_list()
wafer_options = {
    f"{w['group_id']} · Lot {w['Lot_Num']} · 결함 {w['Target']:.0f}개 · {w['error_class']}": w["group_id"]
    for w in wafer_list
}
selected_label = st.selectbox("웨이퍼 선택", list(wafer_options.keys()))
selected_group_id = wafer_options[selected_label]

col_map, col_shap = st.columns([1, 1.2])
wmap = call_wafer_map(selected_group_id)
with col_map:
    if wmap:
        st.plotly_chart(
            wafer_map_figure(
                wmap["wafer_map"],
                f"{wmap['group_id']} — 실측 결함 {wmap['Target']:.0f}개 · Yield {wmap['yield_pct']:.2f}%",
            ),
            width="stretch",
        )
        st.caption("🟩 정상 다이 · 🟥 결함 다이 · 회색은 웨이퍼 영역 밖. 실제 다이 533개 기준.")
    else:
        st.warning("이 웨이퍼는 맵 데이터가 절단되어 복구할 수 없습니다(원본 데이터 특성상 17% 발생).")

with col_shap:
    wshap = call_wafer_shap(selected_group_id, model)
    if wshap:
        st.plotly_chart(
            shap_waterfall_figure(
                wshap["base_value"], wshap["shap_contributions"],
                f"예측 근거 — 실측 {wshap['actual_target']:.0f}개 vs 예측 {wshap['predicted_target']:.1f}개",
            ),
            width="stretch",
        )
        st.caption(
            f"실측 불량 패턴 **{wshap['actual_error_class']}**. 각 막대는 이 웨이퍼의 실측 공정값이 "
            f"예측을 기준값({wshap['base_value']:.1f}, 전체 평균)에서 얼마나 밀어올리거나 끌어내렸는지를 나타냅니다."
        )

if wshap and wshap.get("alerts"):
    st.markdown("**이 웨이퍼에서 발생한 경보**")
    for alert in wshap["alerts"]:
        st.markdown(alert_card_html(alert), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------
# What-If SHAP
# ---------------------------------------------------------------------
st.subheader("🎛️ What-If 시뮬레이션")
st.markdown(
    '<div class="section-note">사이드바에서 설정한 공정 조건이 예측값을 어떻게 만들어내는지 분해합니다.</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    shap_waterfall_figure(
        sim["base_value"], sim["modified_shap_full"],
        f"현재 공정 조건의 예측 근거 — 예측 {sim['modified']['predicted_target']:.1f}개",
    ),
    width="stretch",
)
with st.expander("베이스라인 대비 SHAP 기여도 변화"):
    st.dataframe(sim["shap_contribution_changes"], width="stretch")

st.divider()

# ---------------------------------------------------------------------
# 글로벌 SHAP + 경보 이력
# ---------------------------------------------------------------------
tab_shap, tab_alerts, tab_rules = st.tabs(["📊 전역 변수 중요도", "📋 경보 이력", "⚙️ 경보 규칙"])

with tab_shap:
    img1, img2 = st.columns(2)
    try:
        img1.image("reports/figures/shap_summary.png", caption="Beeswarm — LightGBM, 전체 1,704 웨이퍼", width="stretch")
        img2.image("reports/figures/shap_summary_bar.png", caption="평균 절대 기여도 — LightGBM", width="stretch")
    except Exception:
        st.info("`python scripts/13_model_interpretation_shap.py`를 실행해 SHAP 시각화를 생성하세요.")

with tab_alerts:
    history = get_alert_history()
    if history["alerts"]:
        st.caption(
            f"최근 {len(history['alerts'])}건 · 동일 경보는 5분간 중복 억제됩니다"
            + (" · Slack 전송 활성" if history["slack"]["enabled"] else " · Slack 미설정(기록만)")
        )
        for alert in history["alerts"]:
            st.markdown(alert_card_html(alert), unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-box">기록된 경보가 없습니다.</div>', unsafe_allow_html=True)

with tab_rules:
    st.markdown(
        f"""
| 규칙 | 심각도 | 발동 조건 | 판정 근거 |
|---|---|---|---|
| `METROLOGY_MISSING` | 심각 | Thin F2/F3/F4 중 하나라도 결측 | 역대 최다 결함 웨이퍼(결함 666개)가 9개 측정행 전부에서 이 값들이 결측이었음 |
| `ETCH_RISK_ZONE` | 경고 | 세 값이 모두 상위 20% 구간<br/>(F2≥{thresholds['etch_risk_zone_q80']['Thin F2']:.0f} · F3≥{thresholds['etch_risk_zone_q80']['Thin F3']:.0f} · F4≥{thresholds['etch_risk_zone_q80']['Thin F4']:.0f}) | 이 구간 실측 불량률 18.4% vs 하위 20% 구간 0.0% |
| `PREDICTED_TARGET_UCL` | 경고(2σ) / 심각(3σ) | 예측 결함수 > μ+kσ<br/>(2σ={thresholds['predicted_target']['ucl_2sigma']:.1f} · 3σ={thresholds['predicted_target']['ucl_3sigma']:.1f}) | 웨이퍼 단위 SPC 관리도 (μ={thresholds['predicted_target']['mean']:.1f}, σ={thresholds['predicted_target']['std']:.1f}) |

Lot 단위 배치 이상 규칙(불량률 관리도 μ+2σ={thresholds['lot_defect_rate_pct']['ucl_2sigma']:.1f}%)은
`reports/06_lot_alert_backtest.csv`에서 32개 Lot 백테스트를 거쳤으며, 실제 불량률 61.1%였던 Lot 25만
탐지하고 오탐은 0건이었습니다.
        """
    )

with st.expander("모델 한계 및 해석 시 주의사항"):
    st.markdown(
        f"""
- **성능 기준**: R² {m['r2_mean']:.4f} ± {m['r2_std']:.4f}, MAE {m['mae_mean']:.2f}는 GroupKFold(Lot_Num) 기준입니다.
  학습에 전혀 등장하지 않은 새 Lot을 맞히는 가장 엄격한 조건이라, 웨이퍼 단위 무작위 분할(R² 0.6–0.7)보다
  보수적입니다. 반대로 die 단위 반복행을 그대로 쓰면 R²가 0.945–0.985까지 부풀려지는데, 이는 데이터 누수입니다.
- **SHAP은 인과관계가 아닙니다**: 노광 파장(UV_type)은 초기 분석에서 유의해 보였으나 특정 Lot 교란을
  통제하자 통계·ML 10개 모델 전부에서 유의성이 사라졌습니다. 공정 변경 전 `reports/hypothesis_H1_H7_summary.md`를
  반드시 교차 확인하세요.
- **결측 구간 예측은 신뢰도가 낮습니다**: 결함이 극심한 웨이퍼일수록 계측 자체가 누락되는 경향이 있어,
  모델은 이를 중앙값으로 대치합니다. 예측값이 정상 범위로 보여도 Gate 0 경보가 떴다면 예측이 아니라
  계측 로그를 먼저 확인해야 합니다.
        """
    )

st.caption(
    f"백엔드 연결 방식: `{api_mode}` · 사용 모델: {model} · "
    f"데이터 1,704 웨이퍼 / 42개 피처 중 {m['n_features']}개 선택 · "
    f"[전체 분석 보고서]({REPO_URL}/blob/main/reports/FINAL_PROJECT_REPORT.md)"
)
