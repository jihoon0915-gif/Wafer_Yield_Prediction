// 품질 분석 6종 — 모두 이 프로젝트의 실제 데이터(quality.json, wafers.json)에서 계산된 값을 쓴다.
// 심각도(S)·검출도(D)처럼 데이터로 정할 수 없는 값은 화면에 '가정'으로 명시한다.
import { controlChart, histogram, rocChart, pareto, fmt, COLORS as C } from "./charts.js";

const h = (html) => { const d = document.createElement("div"); d.innerHTML = html.trim(); return d; };
// 변수명 앞에 공정명을 붙여 어느 공정의 값인지 바로 보이게 (예: "식각 · 2단계 후 남은 막 두께")
const full = (q, key, label) => {
  const proc = q.processes.find((p) => p.id === q.params[key]?.process);
  return proc ? `${proc.name} · ${label}` : label;
};
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

const MODULES = [
  { id: "spc", n: "①", title: "SPC 관리도" },
  { id: "cpk", n: "②", title: "공정능력 Cp/Cpk" },
  { id: "fish", n: "③", title: "Fishbone 원인 분석" },
  { id: "fmea", n: "④", title: "PFMEA" },
  { id: "d8", n: "⑤", title: "8D Report" },
  { id: "ml", n: "⑥", title: "ML vs SPC 비교" },
];

export function renderQuality(nav, body, ctx) {
  nav.innerHTML = "";
  MODULES.forEach((m, i) => {
    const b = document.createElement("button");
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === 0 ? "true" : "false");
    b.innerHTML = `<span class="n">${m.n}</span>${m.title}`;
    b.addEventListener("click", () => {
      nav.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
      show(m.id);
    });
    nav.appendChild(b);
  });
  const show = (id) => { body.innerHTML = ""; RENDER[id](body, ctx); };
  show("spc");
}

// ---------------------------------------------------------------- ① SPC
function spc(body, { quality: q }) {
  const pc = q.p_chart;
  const ooc = pc.lots.filter((l) => l.ooc);
  body.append(h(`
    <p class="q-intro">통계적 공정관리(SPC)는 공정이 '평소처럼' 돌고 있는지 관리 한계선으로 감시하는 방법입니다.
      생산 묶음(Lot)별 불량률은 p 관리도로, 개별 공정 변수는 웨이퍼 순서대로 I-MR 관리도로 그렸습니다.</p>
    <div class="card"><div class="card-head"><h2>Lot별 불량률 — p 관리도</h2>
      <span class="muted small">평균 불량률 ${pct(pc.p_bar, 2)} · 관리 한계는 Lot 크기에 따라 달라짐</span></div>
      <div id="pchart"></div>
      <div class="legend"><span><i style="background:${C.s1}"></i>Lot 불량률</span><span><i style="background:${C.critical}"></i>관리 상·하한 (±3σ)</span>
        <span><i class="dot" style="background:${C.critical}"></i>관리 이탈 Lot</span></div>
      <div class="takeaway"><b>관리 이탈 ${ooc.length}개 Lot:</b> ${ooc.map((l) => `Lot ${l.lot} (불량률 ${pct(l.p)} · 상한 ${pct(l.ucl)})`).join(", ")}.
        특히 Lot 25는 상한의 3배가 넘어 즉시 원인 조사 대상입니다 — ⑤ 8D Report에서 이어집니다.</div>
    </div>
    <div class="card"><div class="card-head"><h2>공정 변수 I-MR 관리도</h2><div class="seg" id="imr-seg"></div></div>
      <div id="ichart"></div><div id="mrchart"></div>
      <div class="legend"><span><i class="dot" style="background:${C.critical}"></i>관리 한계 이탈</span>
        <span><i class="dot" style="background:${C.warning}"></i>실제 불량 웨이퍼</span></div>
      <div id="imr-note"></div>
    </div>`));
  controlChart(body.querySelector("#pchart"), {
    values: pc.lots.map((l) => l.p), center: pc.p_bar, uclSeries: pc.lots.map((l) => l.ucl), lclSeries: pc.lots.map((l) => l.lcl),
    flags: pc.lots.map((l) => l.ooc), xs: pc.lots.map((l) => `L${l.lot}`), height: 240, label: "Lot별 불량률 p 관리도",
    tipFn: (i) => { const l = pc.lots[i]; return `<b>Lot ${l.lot}</b><br>불량 ${l.d}/${l.n}장 · ${pct(l.p)}<br><span class="tt-k">관리 상한</span> ${pct(l.ucl)}${l.ooc ? "<br><b style='color:#ff7a7a'>관리 이탈</b>" : ""}`; },
  });

  const keys = Object.keys(q.imr);
  const seg = body.querySelector("#imr-seg");
  const draw = (key) => {
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.k === key ? "true" : "false"));
    const m = { ...q.imr[key], label: full(q, key, q.imr[key].label) };
    const flags = m.values.map((v) => v !== null && (v > m.ucl || v < m.lcl));
    const unit = m.unit ? ` (${m.unit})` : "";
    controlChart(body.querySelector("#ichart"), {
      values: m.values, center: m.center, ucl: m.ucl, lcl: m.lcl, flags, marks: m.defect, height: 250,
      label: `${m.label} I 관리도`, yLabel: `${m.label}${unit}`, xLabel: "웨이퍼 순서 (Lot 순) →",
      tipFn: (i) => `Lot ${m.lot[i]} · ${i + 1}번째<br><b>${fmt(m.values[i], 2)}</b>${unit}${m.defect[i] ? "<br><span style='color:#fab219'>실제 불량 웨이퍼</span>" : ""}`,
    });
    controlChart(body.querySelector("#mrchart"), {
      values: m.mr, center: m.mr_bar, ucl: m.mr_ucl, lcl: 0, height: 150, label: `${m.label} MR 관리도`, yLabel: "이동범위(MR)",
      flags: m.mr.map((v) => v !== null && v > m.mr_ucl), tipFn: (i) => `이동범위 <b>${fmt(m.mr[i], 2)}</b>`,
    });
    const hit = flags.filter((f, i) => f && m.defect[i]).length;
    const nDef = m.defect.filter(Boolean).length;
    let extra = "";
    if (key === "Temp_OXid") {
      extra = `<div class="alert warn"><div class="t">⚠ 관리 한계 이탈 ${m.n_ooc}건 — 실제 이상이 아니라 관리도 설계 문제</div>
        <div class="b">이 데이터에는 건식(평균 993°C)과 습식(평균 1,175°C) 두 산화 방식이 섞여 있어, 하나의 관리도로 그리면
          정상 웨이퍼 절반이 '이탈'로 찍힙니다. 방식별로 나눠 그리면 이탈이 건식 130/990장, 습식 62/714장으로 줄어듭니다.
          → <b>같은 조건끼리 묶어(합리적 부분군) 관리도를 따로 운영</b>해야 한다는 근거입니다.</div></div>`;
    }
    body.querySelector("#imr-note").innerHTML = `
      <div class="takeaway">관리 한계 이탈 <b>${m.n_ooc}장</b> 중 실제 불량 웨이퍼는 <b>${hit}장</b> — 전체 불량 ${nDef}장의 ${pct(hit / nDef)}만 이 관리도 하나로 잡힙니다.
        '8점 연속 한쪽' 규칙 해당 ${m.n_run}장은 Lot 단위로 평균이 이동하기 때문에 생긴 것으로, Lot 간 산포가 크다는 신호입니다.</div>${extra}`;
  };
  keys.forEach((k) => {
    const b = document.createElement("button");
    b.dataset.k = k; b.textContent = full(q, k, q.imr[k].label);
    b.addEventListener("click", () => draw(k));
    seg.appendChild(b);
  });
  draw(keys[0]);
}

// ---------------------------------------------------------------- ② Cp/Cpk
function grade(cpk) {
  if (cpk >= 1.67) return ["매우 우수", "ok"];
  if (cpk >= 1.33) return ["양호", "ok"];
  if (cpk >= 1.0) return ["주의", "warn"];
  return ["능력 부족", "crit"];
}
function cpk(body, { quality: q }) {
  const cap = q.capability;
  body.append(h(`
    <p class="q-intro">공정능력지수는 공정의 산포가 규격 폭 안에 얼마나 여유 있게 들어오는지를 나타냅니다.
      <b>Cp</b>는 산포만, <b>Cpk</b>는 평균이 규격 중심에서 치우친 정도까지 반영합니다. 일반적으로 Cpk 1.33 이상을 양호로 봅니다.</p>
    <div class="grid-3" id="cap-cards"></div>`));
  const wrap = body.querySelector("#cap-cards");
  const insight = {
    thickness: `규격(700nm) 미달 웨이퍼가 ${cap.thickness.out_of_spec_pct}%인데, 정작 이 웨이퍼들의 불량률은 1.2%로 규격 내(7.9%)보다 낮습니다.
      <b>규격과 실제 불량이 연결되지 않는다</b>는 뜻이라, 규격 한계의 타당성 재검토가 먼저입니다.`,
    Line_CD: `규격(25~55nm) 이탈이 ${cap.Line_CD.out_of_spec_pct}%로 가장 심각합니다. 평균은 규격 중앙(40nm) 근처라
      <b>치우침보다 산포가 문제</b>(Cp ${cap.Line_CD.cp} ≈ Cpk ${cap.Line_CD.cpk}) — 노광 조건의 편차를 줄여야 합니다. 이상 Lot 25는 54장 중 30장이 이탈했습니다.`,
    "Thin F2": `공식 규격이 없어, 불량률이 0%인 하위 20% 구간과 18.3%로 뛰는 상위 20% 구간의 경계(3,666Å)를 잠정 상한으로 두고 계산했습니다.
      <b>규격 제정이 필요한 핵심 변수</b>임을 보여주는 참고치입니다.`,
  };
  Object.entries(cap).forEach(([key, c]) => {
    const [g, cls] = grade(c.cpk);
    const card = h(`<div class="card">
      <div class="card-head"><h2>${full(q, key, c.label)}${c.unit ? ` <span class="muted small">(${c.unit})</span>` : ""}</h2>
        <span class="badge ${cls}">${g}</span></div>
      <div class="muted small">${c.spec_source}</div>
      <div class="stat-row">
        ${c.cp !== null ? `<div class="stat"><div class="k">Cp</div><div class="v">${c.cp}</div></div>` : ""}
        <div class="stat ${cls === "crit" ? "bad" : ""}"><div class="k">Cpk${c.has_spec ? "" : " (잠정)"}</div><div class="v">${c.cpk}</div></div>
        <div class="stat"><div class="k">규격 이탈</div><div class="v">${c.out_of_spec_pct}%</div><div class="d">${c.out_of_spec} / ${c.n}장</div></div>
      </div>
      <div class="hist"></div>
      <p class="note">평균 ${fmt(c.mean, 2)} · 표준편차 ${fmt(c.sd, 2)}</p>
      <div class="takeaway">${insight[key]}</div></div>`).firstElementChild;
    wrap.appendChild(card);
    const lines = [{ x: c.mean, label: "평균", color: C.ink2, anchor: "middle" }];
    if (c.lsl !== null) lines.push({ x: c.lsl, label: `LSL ${c.lsl}`, anchor: "end" });
    if (c.usl !== null) lines.push({ x: c.usl, label: `${c.has_spec ? "USL" : "잠정 상한"} ${fmt(c.usl, 0)}` });
    histogram(card.querySelector(".hist"), { bins: c.hist, edges: c.edges, lines, height: 200, width: 400, label: `${c.label} 분포와 규격` });
  });
  body.append(h(`<div class="card"><h2>판정 기준</h2><div class="table-wrap"><table class="data plain">
    <thead><tr><th>Cpk</th><th>판정</th><th>의미</th></tr></thead><tbody>
    <tr><td>1.67 이상</td><td><span class="badge ok">매우 우수</span></td><td>규격 대비 산포 여유가 충분</td></tr>
    <tr><td>1.33 ~ 1.67</td><td><span class="badge ok">양호</span></td><td>일반적인 양산 목표 수준</td></tr>
    <tr><td>1.00 ~ 1.33</td><td><span class="badge warn">주의</span></td><td>규격을 겨우 만족 — 관리 강화 필요</td></tr>
    <tr><td>1.00 미만</td><td><span class="badge crit">능력 부족</span></td><td>규격 이탈이 상시 발생 — 공정 개선 필요</td></tr>
    </tbody></table></div></div>`));
}

// ---------------------------------------------------------------- ③ Fishbone
const BONES = [
  { cat: "방법 (공정 조건)", side: "top", causes: [
    ["v", "식각 후 남은 막 두께 상위 20% → 불량률 18.3%", "예측 기여도 1위 · 하위 20% 구간은 0%"],
    ["v", "식각 온도 상위 20% → 불량률 12.6%", "하위 20%는 3.8%"],
    ["v", "건식 산화 불량률 9.4% (습식 4.3%)", "산화 방식별 2배 차이"]] },
  { cat: "측정 (계측)", side: "top", causes: [
    ["v", "핵심 계측값(막 두께) 누락 웨이퍼 3장 중 2장이 최다 불량(666개)", "계측 누락 자체가 선행 신호 → Gate 0 경보"],
    ["w", "산화 온도 관리도 오경보 과다", "두 산화 방식 혼합으로 이탈 851건 — 관리도 설계 문제"]] },
  { cat: "자재 (재료)", side: "top", causes: [
    ["v", "감광액 두께 목표비 상위 20% → 불량률 11.7%", "하위 20%는 3.5%"],
    ["x", "산화막 두께 규격(700nm) 미달", "미달 웨이퍼 불량률 1.2%로 오히려 낮음 — 원인 아님"]] },
  { cat: "설비", side: "bottom", causes: [
    ["x", "장비(챔버)별 차이", "가설 H5 검증 결과 기각 — 6개 모델 모두 성능 하락"],
    ["x", "노광 파장(UV 종류) 차이", "가설 H1: Lot 25 교란으로 판명, 통제 후 유의 모델 0개"]] },
  { cat: "Lot 운영", side: "bottom", causes: [
    ["v", "Lot 25 불량률 61.1% · Lot 23 20.4%", "p 관리도 관리 한계 이탈"],
    ["v", "Lot 25: 막 두께 Lot 평균이 전체보다 2.4σ 높음", "회로 선폭 규격 이탈 30/54장"]] },
  { cat: "환경 (시간)", side: "bottom", causes: [
    ["x", "시간 경과에 따른 변화(드리프트)", "가설 H6 기각 — 같은 웨이퍼 내 6개월 차이에도 결과 동일"],
    ["w", "굽는 온도 하위 20% → 불량률 14.7%", "예측 모델에는 영향 작음 — 추가 확인 필요"]] },
];
const MARK = { v: ["확인됨", C.critical, "●"], w: ["추가 확인 필요", C.warning, "▲"], x: ["기각", C.muted, "✕"] };

function fishbone(body, { quality: q }) {
  const top2 = q.pareto.slice(0, 2);
  body.append(h(`
    <p class="q-intro">특성요인도(Fishbone)는 머리에 해결할 문제(결과 특성)를, 뼈에 그 원인 후보를 범주별로 펼쳐 놓는 도구입니다.
      일반적인 브레인스토밍과 달리, 여기 적힌 원인은 모두 이 프로젝트의 데이터와 13개 가설 검증 결과로 <b>확인·기각 여부를 판정</b>했습니다.</p>
    <div class="card"><div class="card-head"><h2>먼저, 어떤 불량이 많은가 — 불량 유형 파레토</h2>
      <span class="muted small">불량 웨이퍼 ${q.pareto.reduce((s, p) => s + p.count, 0)}장</span></div>
      <div id="pareto"></div>
      <div class="takeaway">상위 2개 유형(${top2.map((p) => p.name).join("·")})이 전체 불량의 <b>${top2[1].cum_pct}%</b>를 차지합니다 — 원인 분석은 이 두 유형에 집중합니다.</div></div>
    <div class="card fish"><h2 style="margin-bottom:10px">특성요인도</h2><div id="fish-svg"></div>
      <div class="legend">${Object.values(MARK).map(([t, c, s]) => `<span><b style="color:${c}">${s}</b> ${t}</span>`).join("")}</div></div>
    <div class="card"><h2>원인별 근거</h2><div class="table-wrap"><table class="data plain" id="fish-table"></table></div></div>`));
  pareto(body.querySelector("#pareto"), q.pareto);
  drawFish(body.querySelector("#fish-svg"), q);
  const t = body.querySelector("#fish-table");
  t.innerHTML = `<thead><tr><th>범주</th><th>원인 후보</th><th>판정</th><th>근거</th></tr></thead><tbody>${
    BONES.flatMap((b) => b.causes.map(([k, c, e]) => `<tr><td>${b.cat}</td><td class="wrap">${c}</td>
      <td><b style="color:${MARK[k][1]}">${MARK[k][2]}</b> ${MARK[k][0]}</td><td class="wrap muted">${e}</td></tr>`)).join("")}</tbody>`;
}

function drawFish(container, q) {
  const W = 1200, H = 560, spineY = H / 2, headX = 1030;
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", `0 0 ${W} ${H}`);
  s.setAttribute("role", "img");
  s.setAttribute("aria-label", "웨이퍼 수율 저하 특성요인도");
  const el = (tag, a, txt) => { const n = document.createElementNS(ns, tag); for (const k in a) n.setAttribute(k, a[k]); if (txt) n.textContent = txt; s.appendChild(n); return n; };
  el("line", { x1: 40, y1: spineY, x2: headX, y2: spineY, stroke: C.s1, "stroke-width": 4, "stroke-linecap": "round" });
  el("polygon", { points: `${headX},${spineY - 44} ${W - 20},${spineY - 44} ${W - 20},${spineY + 44} ${headX},${spineY + 44}`, fill: "#1c2a45", stroke: C.s1, "stroke-width": 1.5 });
  const hx = (headX + W - 20) / 2;
  el("text", { x: hx, y: spineY - 14, "text-anchor": "middle", class: "bone-label" }, "웨이퍼 수율 저하");
  el("text", { x: hx, y: spineY + 8, "text-anchor": "middle", fill: C.ink2, "font-size": 12 }, `평균 칩 수율 ${q.overall.baseline_yield}%`);
  el("text", { x: hx, y: spineY + 26, "text-anchor": "middle", fill: C.ink2, "font-size": 12 }, `불량 웨이퍼 ${q.overall.defect_rate_pct}%`);
  const tops = BONES.filter((b) => b.side === "top"), bots = BONES.filter((b) => b.side === "bottom");
  [[tops, -1], [bots, 1]].forEach(([bones, dir]) => {
    bones.forEach((b, i) => {
      const bx = 120 + i * 310, endX = bx + 130, tipY = spineY + dir * 230;
      el("line", { x1: bx, y1: tipY, x2: endX, y2: spineY, stroke: C.axis, "stroke-width": 2.5 });
      el("text", { x: bx - 10, y: tipY + dir * 18 + (dir < 0 ? 4 : 8), class: "bone-label" }, b.cat);
      b.causes.forEach(([k, c], j) => {
        const t = (j + 1) / (b.causes.length + 1);
        const px = bx + (endX - bx) * t, py = tipY + (spineY - tipY) * t;
        el("line", { x1: px, y1: py, x2: px + 22, y2: py, stroke: C.axis, "stroke-width": 1.5 });
        el("text", { x: px + 28, y: py + 4, class: "cause", fill: MARK[k][1] }, MARK[k][2]);
        // 26자 부근의 공백에서 줄바꿈(숫자·단어 중간에서 끊기지 않게)
        const cut = c.length > 26 ? c.lastIndexOf(" ", 26) : -1;
        const words = cut > 0 ? [c.slice(0, cut), c.slice(cut + 1)] : [c];
        words.forEach((w, wi) => el("text", { x: px + 42, y: py + 4 + wi * 14, class: "cause", fill: k === "x" ? C.muted : C.ink2 }, w));
      });
    });
  });
  container.appendChild(s);
}

// ---------------------------------------------------------------- ④ PFMEA
const O_SCALE = [[0.3, 10], [0.15, 8], [0.1, 7], [0.05, 5], [0.02, 3], [0, 2]];
const occ = (rate) => O_SCALE.find(([t]) => rate >= t)[1];
function fmea(body, { quality: q }) {
  const p = q.params;
  const rows = [
    { proc: "식각", mode: "남은 막 두께 과다 (상위 20%)", effect: "가장자리·한 곳에 뭉친 불량 → 칩 불량", S: 7,
      cause: "식각 조건 편차(온도·플라즈마 출력)", rate: p["Thin F2"].defect_rate_high / 100,
      ctrl: "개별 변수 3σ 관리도", D: 7, dNote: "관리도 단독 불량 검출률 7% 수준(⑥)",
      action: "잠정 관리 상한 3,666Å 설정 + 리스크존 자동 경보(구현)", D2: 4 },
    { proc: "계측", mode: "핵심 계측값 누락", effect: "불량 웨이퍼가 정상으로 예측됨(검출 실패)", S: 9,
      cause: "계측 설비 누락 · 데이터 전송 실패", rate: 3 / 1704, rateNote: "3/1,704장",
      ctrl: "없음 — 모델이 중앙값으로 조용히 대치", D: 10, dNote: "예측값이 정상 범위로 나와 알 수 없음",
      action: "Gate 0 규칙: 누락 즉시 심각 경보 + 예측 보류(구현)", D2: 2 },
    { proc: "노광", mode: "회로 선폭 규격 이탈 (25~55nm)", effect: "패턴 불량", S: 6,
      cause: "노광 에너지·초점 편차 (Cp 0.42)", rate: q.capability.Line_CD.out_of_spec_pct / 100,
      ctrl: "규격 검사", D: 5, dNote: "검사로는 잡히나 사후 대응",
      action: "노광 조건 산포 축소, 이탈 Lot 선제 격리", D2: 5 },
    { proc: "산화", mode: "건식 산화 조건", effect: "불량률 9.4% (습식 4.3%)", S: 6,
      cause: "산화 방식별 공정 창 차이", rate: 0.0939,
      ctrl: "산화 온도 단일 관리도", D: 6, dNote: "두 방식 혼합으로 오경보 과다(①)",
      action: "산화 방식별 관리도 분리 운영", D2: 4 },
    { proc: "Lot 운영", mode: "Lot 단위 이상 (Lot 25: 61.1%)", effect: "대량 불량 · 출하 지연", S: 8,
      cause: "특정 Lot의 식각·노광 조건 동시 이탈", rate: 2 / 32, rateNote: "2/32 Lot",
      ctrl: "출하 전 검사", D: 6, dNote: "Lot 완료 후에야 인지",
      action: "Lot p 관리도 실시간 운영 + ML 예측 경보", D2: 3 },
  ].map((r) => ({ ...r, O: occ(r.rate), rpn: r.S * occ(r.rate) * r.D, rpn2: r.S * occ(r.rate) * r.D2 }))
    .sort((a, b) => b.rpn - a.rpn);
  body.append(h(`
    <p class="q-intro">공정 FMEA(PFMEA)는 공정별로 일어날 수 있는 고장을 <b>심각도(S) × 발생도(O) × 검출도(D) = 위험 우선순위(RPN)</b>로
      점수화해 대책 순서를 정합니다. <b>발생도는 실제 데이터의 발생률</b>로 매겼고, 심각도와 검출도는 데이터로 정할 수 없어 <b>가정값</b>입니다.</p>
    <div class="card"><div class="table-wrap"><table class="data plain"><thead><tr>
      <th>공정</th><th>고장 모드</th><th>영향</th><th class="num">S<br><span class="small muted">가정</span></th><th>원인</th>
      <th class="num">O<br><span class="small muted">데이터</span></th><th>현재 관리</th><th class="num">D<br><span class="small muted">가정</span></th>
      <th class="num">RPN</th><th>대책</th><th class="num">대책 후 RPN</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td>${r.proc}</td><td class="wrap"><b>${r.mode}</b></td><td class="wrap">${r.effect}</td>
        <td class="num">${r.S}</td><td class="wrap">${r.cause}</td>
        <td class="num">${r.O}<br><span class="small muted">${r.rateNote || pct(r.rate)}</span></td>
        <td class="wrap">${r.ctrl}<br><span class="small muted">${r.dNote}</span></td><td class="num">${r.D}</td>
        <td class="num"><span class="badge ${r.rpn >= 200 ? "crit" : r.rpn >= 120 ? "warn" : "ok"}">${r.rpn}</span></td>
        <td class="wrap">${r.action}</td>
        <td class="num"><span class="badge ${r.rpn2 >= 200 ? "crit" : r.rpn2 >= 120 ? "warn" : "ok"}">${r.rpn2}</span></td></tr>`).join("")}
    </tbody></table></div>
    <p class="note">발생도 환산: 발생률 30% 이상 10 · 15~30% 8 · 10~15% 7 · 5~10% 5 · 2~5% 3 · 2% 미만 2.
      '대책 후'는 검출도만 개선된다고 본 추정치이며, '(구현)' 표시 대책은 이 대시보드의 경보 기능으로 실제 구현되어 있습니다.</p></div>
    <div class="takeaway"><b>눈여겨볼 점 — 계측 누락</b>은 발생 빈도가 낮아(3장) 발생도는 2에 불과하지만, 검출이 불가능해(D=10)
      RPN이 높게 나옵니다. 드물지만 놓치면 치명적인 고장을 FMEA가 어떻게 드러내는지 보여주는 사례입니다.</div>`));
}

// ---------------------------------------------------------------- ⑤ 8D
function d8(body, { quality: q, wafers, model }) {
  const lot25 = q.p_chart.lots.find((l) => l.lot === 25);
  const ws = wafers.filter((w) => w.lot === 25);
  const med = (k) => q.params[k].median;
  const before = ws.reduce((s, w) => s + model.predict(w.x), 0) / ws.length;
  const after = ws.reduce((s, w) => s + model.predict({ ...w.x, "Thin F2": med("Thin F2"), "Thin F3": med("Thin F3"), "Thin F4": med("Thin F4") }), 0) / ws.length;
  const yieldOf = (t) => (100 * (q.overall.real_die_count - t)) / q.overall.real_die_count;
  const steps = [
    ["D0", "긴급 대응 필요성 판단", `<p>Lot 25 불량률 <b>${pct(lot25.p)}</b> — p 관리도 상한(${pct(lot25.ucl)})의 3배 이상 이탈. 전체 평균(${pct(q.p_chart.p_bar, 2)}) 대비 8배.</p>`],
    ["D1", "팀 구성 (가정)", `<p>품질보증(리더) · 식각 공정 · 노광 공정 · 계측 · 데이터 분석 담당.</p>`],
    ["D2", "문제 기술", `<ul><li><b>무엇이</b>: Lot 25, 54장 중 ${lot25.d}장 불량 — 가장자리 한쪽 뭉침 15, 안쪽 한 곳 뭉침 9, 전체에 흩어짐 7, 긁힌 선 모양 2</li>
      <li><b>얼마나</b>: 웨이퍼당 평균 불량 칩 ${lot25.mean_target}개 (전체 평균 ${q.overall.mean_target}개)</li>
      <li><b>어떻게 알았나</b>: Lot 불량률 p 관리도 이탈</li></ul>`],
    ["D3", "임시 조치 (가정)", `<ul><li>Lot 25 출하 보류 및 전수 재검사</li><li>동일 식각 조건으로 진행 중인 Lot의 남은 막 두께 우선 계측</li></ul>`],
    ["D4", "근본 원인 분석 (데이터 근거)", `<ul>
      <li><b>식각</b>: 2단계 후 남은 막 두께의 Lot 평균이 전체 Lot 대비 <b>+2.36σ</b>, 3단계 +1.80σ — 두께 리스크존 웨이퍼 12장</li>
      <li><b>노광</b>: 회로 선폭 규격 이탈 <b>30/54장(55.6%)</b> — 전체 이탈률 ${q.capability.Line_CD.out_of_spec_pct}%의 2.6배</li>
      <li><b>배제된 원인</b>: 노광 파장(UV 종류)은 Lot 25가 한 종류만 써서 생긴 착시 — 가설 H1에서 10개 모델 모두 유의하지 않음</li></ul>`],
    ["D5", "영구 대책 선정", `<ul><li>식각 남은 막 두께 잠정 관리 상한(3,666Å) 설정</li><li>노광 선폭 산포 축소(Cpk 0.40 → 1.33 목표)</li>
      <li>Lot p 관리도 + ML 불량 예측 경보를 공정 중 실시간 운영</li></ul>`],
    ["D6", "효과 검증 (모델 시뮬레이션)", `<p>Lot 25의 막 두께 3종을 전체 중앙값으로 되돌렸을 때, 이 페이지의 예측 모델로 다시 계산한 결과:</p>
      <div class="stat-row"><div class="stat"><div class="k">대책 전 예측 불량 칩</div><div class="v">${before.toFixed(1)}개</div><div class="d">예측 수율 ${yieldOf(before).toFixed(2)}%</div></div>
      <div class="stat good"><div class="k">대책 후 예측 불량 칩</div><div class="v">${after.toFixed(1)}개</div><div class="d">예측 수율 ${yieldOf(after).toFixed(2)}%</div></div>
      <div class="stat"><div class="k">변화</div><div class="v">${(after - before).toFixed(1)}개</div><div class="d">${(yieldOf(after) - yieldOf(before)).toFixed(2)}%p</div></div></div>
      <p class="note">모델 예측값의 변화이며 인과 효과가 아닙니다. 실제 적용 전 소량 시험 생산으로 확인해야 합니다. (브라우저에서 실시간 계산)</p>`],
    ["D7", "재발 방지", `<ul><li>PFMEA 갱신(④) — 식각·Lot 운영 항목 검출도 개선 반영</li><li>관리도 운영 기준 표준화: 산화 방식별 분리, Lot 부분군 관리</li></ul>`],
    ["D8", "종결 및 공유", `<p>효과 검증 후 종결, 유사 공정 라인에 대책 수평 전개.</p>`],
  ];
  body.append(h(`<p class="q-intro">8D는 불량 발생 시 문제 정의부터 재발 방지까지 8단계로 추적하는 문제 해결 보고서입니다.
    실제 데이터에서 관리 이탈한 <b>Lot 25</b>를 대상으로 작성했습니다. D0·D2·D4·D6은 실제 데이터로 작성했고,
    팀 구성(D1)과 임시 조치(D3)는 상황을 <b>가정해</b> 작성했습니다.</p>
    <div class="d8">${steps.map(([t, n, c]) => `<div class="d8-step"><div class="tag">${t}</div><div><h3>${n}</h3>${c}</div></div>`).join("")}</div>`));
}

// ---------------------------------------------------------------- ⑥ ML vs SPC
function ml(body, { quality: q }) {
  const c = q.compare;
  body.append(h(`
    <p class="q-intro">같은 1,704장의 웨이퍼에서 <b>실제 불량 ${c.n_defect}장</b>을 얼마나 미리 잡아내는지 비교했습니다.
      SPC는 "${c.spc_rule}" 규칙, ML은 학습에 쓰지 않은 Lot에 대한 예측값(교차검증 결과)을 썼습니다 — 학습 데이터로 평가하면 ML이 부당하게 유리해지기 때문입니다.</p>
    <div class="grid-2">
      <div class="card"><h2>같은 오경보율(${pct(c.spc.fpr)})에서 비교</h2>
        <div class="stat-row">
          <div class="stat"><div class="k">SPC 불량 검출률</div><div class="v">${pct(c.spc.recall)}</div><div class="d">${c.spc.tp}/${c.n_defect}장 · 경보 적중률 ${pct(c.spc.precision)}</div></div>
          <div class="stat good"><div class="k">ML 불량 검출률</div><div class="v">${pct(c.ml_same_fpr.recall)}</div><div class="d">${c.ml_same_fpr.tp}/${c.n_defect}장 · 경보 적중률 ${pct(c.ml_same_fpr.precision)}</div></div>
        </div>
        <h3 style="margin-top:14px">오경보율 5%까지 허용하면</h3>
        <div class="stat-row">
          <div class="stat"><div class="k">SPC</div><div class="v">${pct(c.spc_5pct_fpr.recall)}</div><div class="d">${c.spc_5pct_fpr.tp}장 검출</div></div>
          <div class="stat good"><div class="k">ML</div><div class="v">${pct(c.ml_5pct_fpr.recall)}</div><div class="d">${c.ml_5pct_fpr.tp}장 검출</div></div>
        </div>
        <div class="takeaway">같은 수의 오경보를 감수할 때 <b>ML이 불량을 약 ${(c.ml_same_fpr.recall / c.spc.recall).toFixed(1)}배 더 많이</b> 찾았습니다.
          불량은 변수 하나가 한계를 넘어서가 아니라 <b>여러 변수가 동시에 조금씩 치우친 조합</b>에서 주로 생기기 때문입니다.</div>
      </div>
      <div class="card"><h2>전체 기준에서의 판별력 (ROC)</h2><div id="roc"></div>
        <div class="legend"><span><i style="background:${C.s1}"></i>ML — AUC ${c.ml_same_fpr.auc}</span><span><i style="background:${C.s2}"></i>SPC — AUC ${c.spc.auc}</span></div>
        <p class="note">곡선이 왼쪽 위로 붙을수록 적은 오경보로 많은 불량을 잡습니다. 대각선은 무작위 추측 수준입니다.</p></div>
    </div>
    <div class="card"><h2>그래서 SPC를 버려야 할까? — 역할 분담</h2><div class="table-wrap"><table class="data plain">
      <thead><tr><th></th><th>SPC 관리도</th><th>ML 불량 예측</th></tr></thead><tbody>
      <tr><td>강점</td><td class="wrap">어느 변수가 벗어났는지 바로 보이고, 현장 조치가 명확함 · 공정 안정성 자체를 관리</td><td class="wrap">여러 변수의 조합 위험을 잡음 · 불량 검출률이 높음</td></tr>
      <tr><td>약점</td><td class="wrap">변수 하나씩만 봐서 조합 불량을 놓침 · 부분군 설계를 잘못하면 오경보 과다(①의 산화 온도)</td><td class="wrap">왜 위험한지 설명이 필요함(→ 예측 근거로 보완) · 계측 누락 시 조용히 틀림(→ Gate 0로 보완)</td></tr>
      <tr><td>이 프로젝트의 운영안</td><td class="wrap">Lot p 관리도 + 방식별 분리 관리도로 공정 안정성 관리</td><td class="wrap">웨이퍼 단위 불량 예측 경보 + 예측 근거로 조치 변수 제시</td></tr>
      </tbody></table></div>
      <p class="note">한계: 교차검증 예측을 썼지만, 22개 변수 선택 자체는 전체 데이터에서 이뤄져 ML 성능이 소폭 낙관적일 수 있습니다.
        계측 누락 웨이퍼 ${c.missing_metrology_defect.n_missing}장 중 ${c.missing_metrology_defect.n_missing_defect}장이 불량이었으며, 두 방식 모두 이 경우를 스스로 잡지 못합니다.</p></div>`));
  rocChart(body.querySelector("#roc"), [
    { name: "ML", points: c.roc_ml, color: C.s1, marker: [c.ml_same_fpr.fpr, c.ml_same_fpr.recall], labelAt: 0.25, labelDy: -8 },
    { name: "SPC", points: c.roc_spc, color: C.s2, marker: [c.spc.fpr, c.spc.recall], labelAt: 0.6, labelDy: 16 },
  ]);
}

const RENDER = { spc, cpk, fish: fishbone, fmea, d8, ml };
export { pareto };
