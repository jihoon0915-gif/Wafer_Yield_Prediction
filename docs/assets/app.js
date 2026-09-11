import { createModel } from "./model.js";
import { renderLine } from "./equipment.js";
import { renderQuality } from "./quality.js";
import { shapBars, histogram, waferMap, fmt } from "./charts.js";

const $ = (s) => document.querySelector(s);
const METROLOGY = ["Thin F2", "Thin F3", "Thin F4"];

const [spec, Q, WAFERS] = await Promise.all(
  ["assets/model.json", "assets/quality.json", "assets/wafers.json"].map((u) => fetch(u).then((r) => r.json())),
);
const model = createModel(spec);
const PARAMS = Q.params;
const PARAM_KEYS = Object.keys(PARAMS);
const WAFER_BY_ID = Object.fromEntries(WAFERS.map((w) => [w.id, w]));
const DIES = Q.overall.real_die_count;
const yieldOf = (t) => (100 * (DIES - t)) / DIES;

// 모델 피처명 → 사람이 읽는 라벨 / 원래 단위 값
const PROC_NAME = Object.fromEntries(Q.processes.map((p) => [p.id, p.name]));
// 공정 패널 밖에서는 "식각 · 2단계 후 남은 막 두께"처럼 공정명을 붙여 어느 공정의 값인지 보이게 한다
function labelOf(name) {
  const key = name === "oxid_thickness_spec_gap" ? "thickness" : name;
  const m = PARAMS[key];
  return m ? `${PROC_NAME[m.process]} · ${m.label}` : name;
}
function rawOf(name, row) {
  const key = name === "oxid_thickness_spec_gap" ? "thickness" : name;
  const v = row[key];
  if (v === null || v === undefined) return "계측 누락";
  const unit = PARAMS[key]?.unit ? ` ${PARAMS[key].unit}` : "";
  return `${fmt(v, 2)}${unit}`;
}
function processOf(name) {
  const key = name === "oxid_thickness_spec_gap" ? "thickness" : name;
  return PARAMS[key]?.process;
}

// ---------------------------------------------------------------- 경보 규칙 (app_api와 동일한 3종)
function evaluate(row, pred) {
  const out = [];
  const missing = METROLOGY.filter((k) => row[k] === null || row[k] === undefined || Number.isNaN(row[k]));
  if (missing.length) {
    out.push({ level: "crit", title: "핵심 계측값 누락 (Gate 0)",
      body: `${missing.map(labelOf).join(", ")} 값이 비어 있습니다. 모델은 빈 값을 평균적인 값으로 채워 예측하므로 결과가 정상처럼 보일 수 있습니다 — 예측보다 계측 기록을 먼저 확인하세요.`,
      evidence: "역대 최다 불량 웨이퍼(불량 칩 666개)가 이 세 값이 모두 비어 있었음" });
  }
  const risk = METROLOGY.every((k) => row[k] !== null && row[k] !== undefined && row[k] >= PARAMS[k].q80);
  if (risk) {
    out.push({ level: "warn", title: "식각 후 남은 막 두께 리스크존",
      body: "2·3단계와 최종 남은 막 두께가 모두 상위 20% 구간입니다. 식각 조건 보정을 검토하세요.",
      evidence: "세 값이 모두 상위 20%인 웨이퍼 37장 중 54.1%가 불량" });
  }
  const u = Q.target_ucl;
  if (pred > u.ucl3) out.push({ level: "crit", title: "예측 불량 칩 수 관리 상한(3σ) 초과", body: `예측 ${pred.toFixed(0)}개 > 상한 ${u.ucl3.toFixed(0)}개`, evidence: `전체 평균 ${u.mean}개, 표준편차 ${u.sd}` });
  else if (pred > u.ucl2) out.push({ level: "warn", title: "예측 불량 칩 수 관리 상한(2σ) 초과", body: `예측 ${pred.toFixed(0)}개 > 상한 ${u.ucl2.toFixed(0)}개`, evidence: `전체 평균 ${u.mean}개, 표준편차 ${u.sd}` });
  return out;
}
// 예측 근거 막대 읽는 법 — 실제 숫자로 "기준값 + 막대 합 = 예측"을 보여준다
function shapNote(ex, pred) {
  const top = [...ex].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const shown = top.slice(0, 10).reduce((s, e) => s + e.value, 0);
  const rest = top.slice(10).reduce((s, e) => s + e.value, 0);
  const sg = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
  return `<b>읽는 법</b> — 학습 데이터의 평균적인 웨이퍼는 불량 칩 <b>${model.baseValue.toFixed(1)}개</b>로 예측됩니다.
    각 막대는 지금 설정된 그 변수 값이 이 평균에서 불량 칩을 몇 개 늘렸는지(빨강)·줄였는지(초록)입니다.
    ${model.baseValue.toFixed(1)} ${sg(shown)}(표시한 10개) ${sg(rest)}(나머지 ${top.length - 10}개) = 예측 <b>${Math.max(0, model.baseValue + shown + rest).toFixed(1)}개</b>
    ${Math.abs(model.baseValue + shown + rest - pred) > 0.05 ? "(음수는 0으로 표시)" : ""}. 인과관계가 아니라 모델이 그렇게 판단한 근거입니다.`;
}
const levelOf = (alerts) => (alerts.some((a) => a.level === "crit") ? "crit" : alerts.length ? "warn" : "ok");
const BADGE = { ok: '<span class="badge ok">정상</span>', warn: '<span class="badge warn">▲ 주의</span>', crit: '<span class="badge crit">● 심각</span>' };
const alertHtml = (alerts) => alerts.length
  ? alerts.map((a) => `<div class="alert ${a.level}"><div class="t">${a.level === "crit" ? "●" : "▲"} ${a.title}</div><div class="b">${a.body}</div><div class="e">판정 근거: ${a.evidence}</div></div>`).join("")
  : '<div class="ok-box">발생한 경보가 없습니다.</div>';

// ---------------------------------------------------------------- 탭
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
  document.querySelectorAll(".view").forEach((v) => v.dataset.active = v.id === `view-${b.dataset.view}` ? "true" : "false");
  if (b.dataset.view === "quality" && !$("#q-nav").childElementCount) renderQuality($("#q-nav"), $("#q-body"), { quality: Q, wafers: WAFERS, model });
  window.scrollTo({ top: 0 });
}));

// ---------------------------------------------------------------- 1. 데이터 · 예측
$("#model-stats").innerHTML = [
  ["예측 모델", "LightGBM", `공정 변수 ${spec.metrics.n_features}개 사용`],
  ["검증 성능 (R²)", spec.metrics.r2_mean.toFixed(3), "학습에 없던 Lot 기준"],
  ["평균 오차", `±${spec.metrics.mae_mean.toFixed(1)}개`, "웨이퍼당 불량 칩 수"],
  ["학습 데이터", `${Q.overall.n_wafers.toLocaleString()}장`, `${Q.overall.n_lots}개 Lot · 6개 공정`],
].map(([k, v, d]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
$("#fmt-cols").textContent = ["wafer_id", ...PARAM_KEYS, "actual_defects"].join(", ");

let uploaded = null;
function parseCSV(text) {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim());
  const split = (l) => { const out = []; let cur = "", q = false;
    for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; }
    out.push(cur); return out.map((s) => s.trim()); };
  const head = split(lines[0]);
  return lines.slice(1).map((l) => {
    const cells = split(l), row = {};
    head.forEach((k, i) => { const v = cells[i]; row[k] = v === undefined || v === "" || /^nan$/i.test(v) ? null : Number.isNaN(Number(v)) ? v : Number(v); });
    return row;
  });
}
function normalize(row, i) {
  const id = row.wafer_id ?? row.group_id ?? row.id ?? `#${i + 1}`;
  const actual = row.actual_defects ?? row.Target ?? null;
  const x = {};
  PARAM_KEYS.forEach((k) => { x[k] = k in row ? row[k] : PARAMS[k].median; });
  if (!("thickness" in row) && row.oxid_thickness_spec_gap !== undefined && row.oxid_thickness_spec_gap !== null) x.thickness = 700 - row.oxid_thickness_spec_gap;
  return { id: String(id), actual, x };
}
function loadText(text, name) {
  const rows = parseCSV(text);
  if (!rows.length) { $("#file-status").textContent = "데이터 행이 없습니다."; return; }
  uploaded = rows.map(normalize);
  const known = PARAM_KEYS.filter((k) => k in rows[0]).length;
  $("#file-status").innerHTML = `<b>${name}</b> · 웨이퍼 ${uploaded.length}장 · 인식된 공정 변수 ${known}/${PARAM_KEYS.length}개` +
    (known < PARAM_KEYS.length ? ' <span class="muted">(없는 변수는 중앙값으로 채움)</span>' : "");
  $("#btn-predict").disabled = false;
}
const dz = $("#dropzone"), fi = $("#file-input");
fi.addEventListener("change", () => fi.files[0] && fi.files[0].text().then((t) => loadText(t, fi.files[0].name)));
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) f.text().then((t) => loadText(t, f.name)); });
$("#btn-sample").addEventListener("click", async () => {
  loadText(await fetch("sample_wafers.csv").then((r) => r.text()), "sample_wafers.csv (이상 Lot 25 + 정상 Lot 13)");
  runPrediction();
});
$("#btn-predict").addEventListener("click", runPrediction);

let results = [];
function runPrediction() {
  results = uploaded.map((w) => {
    const pred = model.predict(w.x);
    const ex = model.explain(w.x);
    const top = [...ex].sort((a, b) => b.value - a.value)[0];
    return { ...w, pred, yield: yieldOf(pred), ex, top, alerts: evaluate(w.x, pred) };
  });
  const n = results.length;
  const mean = results.reduce((s, r) => s + r.pred, 0) / n;
  const nCrit = results.filter((r) => levelOf(r.alerts) === "crit").length;
  const nWarn = results.filter((r) => levelOf(r.alerts) === "warn").length;
  const withActual = results.filter((r) => r.actual !== null);
  const mae = withActual.length ? withActual.reduce((s, r) => s + Math.abs(r.pred - r.actual), 0) / withActual.length : null;
  $("#summary-empty").hidden = true; $("#summary").hidden = false;
  $("#summary-stats").innerHTML = [
    ["분석한 웨이퍼", `${n}장`, ""],
    ["평균 예측 수율", `${yieldOf(mean).toFixed(2)}%`, `불량 칩 평균 ${mean.toFixed(1)}개`],
    ["경보 웨이퍼", `${nCrit + nWarn}장`, `심각 ${nCrit} · 주의 ${nWarn}`, nCrit ? "bad" : ""],
    ...(mae !== null ? [["실제값 대비 평균 오차", `±${mae.toFixed(1)}개`, `실제값 있는 ${withActual.length}장 기준`]] : []),
  ].map(([k, v, d, c]) => `<div class="stat ${c || ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
  histogram($("#summary-hist"), { values: results.map((r) => r.pred), height: 200, xLabel: "예측 불량 칩 수 (웨이퍼당)", label: "예측 분포",
    lines: [{ x: Q.target_ucl.ucl2, label: "주의 2σ", color: "#fab219", anchor: "end" }, { x: Q.target_ucl.ucl3, label: "심각 3σ" }] });

  const hasActual = withActual.length > 0;
  $("#result-table").innerHTML = `<thead><tr><th>웨이퍼</th><th class="num">예측 불량 칩</th><th class="num">예측 수율</th>
    ${hasActual ? '<th class="num">실제 불량 칩</th>' : ""}<th>상태</th><th>불량을 가장 늘린 요인</th></tr></thead><tbody>${
    results.map((r, i) => `<tr data-i="${i}" tabindex="0"><td class="mono">${r.id}</td><td class="num">${r.pred.toFixed(1)}</td>
      <td class="num">${r.yield.toFixed(2)}%</td>${hasActual ? `<td class="num">${r.actual ?? "—"}</td>` : ""}
      <td>${BADGE[levelOf(r.alerts)]}</td><td>${r.top.value > 0 ? `${labelOf(r.top.name)} <span class="muted">+${r.top.value.toFixed(1)}</span>` : '<span class="muted">—</span>'}</td></tr>`).join("")}</tbody>`;
  $("#result-card").hidden = false;
  const rows = [...document.querySelectorAll("#result-table tbody tr")];
  rows.forEach((tr) => { const go = () => { rows.forEach((x) => x.classList.toggle("sel", x === tr)); showDetail(results[+tr.dataset.i]); };
    tr.addEventListener("click", go); tr.addEventListener("keydown", (e) => e.key === "Enter" && go()); });
  const worst = results.reduce((a, b) => (b.pred > a.pred ? b : a));
  rows[results.indexOf(worst)].click();
}

function showDetail(r) {
  $("#detail-card").hidden = false;
  $("#detail-title").textContent = `웨이퍼 ${r.id} — 예측 불량 칩 ${r.pred.toFixed(1)}개 · 수율 ${r.yield.toFixed(2)}%`;
  const known = WAFER_BY_ID[r.id];
  waferMap($("#detail-map"), known?.map, { title: known ? `실측 불량 지도 (실제 불량 칩 ${known.target}개)` : "업로드 데이터 — 실측 지도 없음" });
  $("#detail-shap").innerHTML = '<h3 style="margin-bottom:6px">예측 근거 — 평균 대비 무엇이 불량을 늘리고 줄였나</h3><div></div>';
  shapBars($("#detail-shap div"), r.ex.map((e) => ({ ...e, label: labelOf(e.name), raw: rawOf(e.name, r.x) })));
  $("#detail-shap").insertAdjacentHTML("beforeend", `<p class="note">${shapNote(r.ex, r.pred)}</p>`);
  $("#detail-alerts").innerHTML = alertHtml(r.alerts);
}

// ---------------------------------------------------------------- 2. 공정 라인
const initial = () => Object.fromEntries(PARAM_KEYS.map((k) => [k, PARAMS[k].median]));
let state = initial();
let gate0 = false;
let active = "etch";
const basePred = model.predict(initial());
const line = renderLine($("#line-svg"), Q.processes, { onSelect: (id) => { active = id; drawPanel(); line.setActive(id); } });

const current = () => (gate0 ? { ...state, ...Object.fromEntries(METROLOGY.map((k) => [k, null])) } : state);
const stepOf = (m) => { const span = m.max - m.min; return span > 1e6 ? span / 200 : +(span / 200).toPrecision(2); };
const valId = (k) => `v-${k.replace(/\W/g, "_")}`; // "Thin F2"처럼 공백 있는 변수명을 안전한 id로

function drawPanel() {
  const p = Q.processes.find((x) => x.id === active);
  const panel = $("#param-panel");
  if (!p.params.length) {
    const pred = model.predict(current());
    panel.innerHTML = `<h2>${p.name}</h2><p class="desc">${p.desc}. 앞 공정의 조건으로 계산한 최종 판정입니다.</p>
      <div class="stat-row"><div class="stat"><div class="k">예측 불량 칩</div><div class="v">${pred.toFixed(1)}개</div></div>
      <div class="stat"><div class="k">예측 수율</div><div class="v">${yieldOf(pred).toFixed(2)}%</div></div></div>${alertHtml(evaluate(current(), pred))}`;
    return;
  }
  panel.innerHTML = `<h2>${p.name}</h2><p class="desc">${p.desc}</p>` +
    (p.id === "etch" ? `<label class="toggle"><input type="checkbox" id="gate0" ${gate0 ? "checked" : ""}> 막 두께 측정값이 빠진 상황 가정하기</label>
      <p class="note" style="margin:-10px 0 14px">식각 후 막 두께가 측정·기록되지 않은 웨이퍼를 재현합니다. 실제 최다 불량 웨이퍼(불량 칩 666개)가
        이런 경우였는데, 모델은 빈 값을 평균으로 채워 예측이 정상처럼 나옵니다. 켜보면 예측값은 거의 그대로지만
        <b>Gate 0 경보</b>가 이를 잡아내는 것을 확인할 수 있습니다.</p>` : "") +
    p.params.map((k) => {
      const m = PARAMS[k], v = state[k], st = stepOf(m);
      const risk = m.defect_rate_high - m.defect_rate_low;
      const riskTxt = Math.abs(risk) >= 4
        ? `<div class="risk" style="color:${risk > 0 ? "#ff8a8a" : "#4cc24c"}">${risk > 0 ? "높을수록" : "낮을수록"} 불량 ↑ (하위 20%: ${m.defect_rate_low}% · 상위 20%: ${m.defect_rate_high}%)</div>` : "";
      return `<div class="slider"><div class="row"><span class="name">${m.label}</span><span class="val" id="${valId(k)}">${fmt(v, 2)}${m.unit ? ` ${m.unit}` : ""}</span></div>
        <input type="range" data-k="${k}" min="${m.min}" max="${m.max}" step="${st}" value="${v}" aria-label="${m.label}" ${gate0 && METROLOGY.includes(k) ? "disabled" : ""}>
        <div class="hint"><span>${fmt(m.min, 1)}</span><span>중앙값 ${fmt(m.median, 1)}</span><span>${fmt(m.max, 1)}</span></div>${riskTxt}</div>`;
    }).join("");
  panel.querySelectorAll("input[type=range]").forEach((inp) => inp.addEventListener("input", () => {
    const k = inp.dataset.k; state[k] = Number(inp.value);
    document.getElementById(valId(k)).textContent = `${fmt(state[k], 2)}${PARAMS[k].unit ? ` ${PARAMS[k].unit}` : ""}`;
    update();
  }));
  $("#gate0")?.addEventListener("change", (e) => { gate0 = e.target.checked; drawPanel(); update(); });
}

function update() {
  const row = current();
  const pred = model.predict(row), ex = model.explain(row), alerts = evaluate(row, pred);
  const d = pred - basePred;
  $("#line-stats").innerHTML = [
    ["예측 불량 칩", `${pred.toFixed(1)}개`, `초기 조건 대비 ${d >= 0 ? "+" : ""}${d.toFixed(1)}개`, d > 0.5 ? "bad" : d < -0.5 ? "good" : ""],
    ["예측 수율", `${yieldOf(pred).toFixed(2)}%`, `초기 ${yieldOf(basePred).toFixed(2)}%`],
    ["관리 상한 (3σ)", `${Q.target_ucl.ucl3.toFixed(0)}개`, `주의 2σ ${Q.target_ucl.ucl2.toFixed(0)}개`],
  ].map(([k, v, dd, c]) => `<div class="stat ${c || ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${dd}</div></div>`).join("");
  $("#line-alerts").innerHTML = alertHtml(alerts);
  $("#line-shap").innerHTML = '<div></div><p class="note"></p>';
  shapBars($("#line-shap div"), ex.map((e) => ({ ...e, label: labelOf(e.name), raw: rawOf(e.name, row) })));
  $("#line-shap .note").innerHTML = shapNote(ex, pred);
  // 장비 표시등: 그 공정 변수들이 예측을 얼마나 밀어올렸는지
  const byProc = {};
  ex.forEach((e) => { const p = processOf(e.name); if (p) byProc[p] = (byProc[p] || 0) + e.value; });
  Q.processes.forEach((p) => {
    let lv = (byProc[p.id] || 0) > 25 ? "crit" : (byProc[p.id] || 0) > 8 ? "warn" : "ok";
    if (p.id === "etch" && alerts.some((a) => a.title.includes("계측") || a.title.includes("막 두께"))) lv = levelOf(alerts.filter((a) => !a.title.includes("관리 상한")));
    if (p.id === "inspect") lv = levelOf(alerts);
    line.setStatus(p.id, lv);
  });
}

function optimize(keys) {
  if (gate0) { $("#opt-result").innerHTML = '<div class="alert warn"><div class="t">계측 누락 재현을 끄고 실행하세요</div></div>'; return; }
  // 22개를 한꺼번에 극단값으로 밀면 데이터에 없던 조합이 되어 예측이 0으로 붙는다(외삽).
  // 그래서 관측 데이터의 중심 60%(q20~q80) 안에서, 효과가 큰 순서로 최대 3개 변수만 바꾼다.
  const MAX_CHANGES = 3;
  const start = { ...state }, before = model.predict(state);
  const used = new Set();
  for (let step = 0; step < MAX_CHANGES; step++) {
    let best = null, bestP = model.predict(state);
    keys.filter((k) => !used.has(k)).forEach((k) => {
      const m = PARAMS[k];
      for (let g = 0; g <= 20; g++) {
        const v = m.q20 + ((m.q80 - m.q20) * g) / 20;
        const pr = model.predict({ ...state, [k]: v });
        if (pr < bestP - 0.05) { bestP = pr; best = [k, v]; }
      }
    });
    if (!best) break;
    state[best[0]] = best[1];
    used.add(best[0]);
  }
  const after = model.predict(state);
  const changed = keys.filter((k) => Math.abs(state[k] - start[k]) > 1e-9);
  $("#opt-result").innerHTML = `<div class="stat-row">
      <div class="stat"><div class="k">최적화 전</div><div class="v">${before.toFixed(1)}개</div><div class="d">수율 ${yieldOf(before).toFixed(2)}%</div></div>
      <div class="stat good"><div class="k">최적화 후</div><div class="v">${after.toFixed(1)}개</div><div class="d">수율 ${yieldOf(after).toFixed(2)}%</div></div></div>
    ${changed.length ? `<div class="table-wrap"><table class="data plain"><thead><tr><th>변수</th><th class="num">변경 전</th><th class="num">추천값</th></tr></thead><tbody>
      ${changed.map((k) => `<tr><td>${labelOf(k)}</td><td class="num">${fmt(start[k], 2)}</td><td class="num"><b>${fmt(state[k], 2)}</b></td></tr>`).join("")}</tbody></table></div>`
      : '<div class="ok-box">현재 조건이 이미 이 범위 안에서 최적입니다.</div>'}
    <p class="note">예측 모델 기준 추천입니다. 모델은 변수 간 상관을 학습한 것이지 인과를 보장하지 않으므로, 실제 적용 전 소량 시험 생산으로 확인해야 합니다. 추천값은 공정 라인에 이미 반영되었습니다.</p>`;
  drawPanel(); update();
}
$("#btn-optimize").addEventListener("click", () => optimize(Q.processes.find((p) => p.id === active).params));
$("#btn-optimize-all").addEventListener("click", () => optimize(PARAM_KEYS));
$("#btn-reset").addEventListener("click", () => { state = initial(); gate0 = false; $("#opt-result").innerHTML = ""; drawPanel(); update(); });

line.setActive(active);
drawPanel();
update();
