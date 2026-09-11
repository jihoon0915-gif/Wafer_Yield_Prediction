// 의존성 없는 SVG 차트 헬퍼. 모든 차트는 viewBox 기반이라 컨테이너 폭에 맞춰 늘어난다.

const NS = "http://www.w3.org/2000/svg";
const C = {
  s1: "#3987e5", s2: "#d95926", good: "#0ca30c", warning: "#fab219", critical: "#d03b3b",
  ink: "#e8edf6", ink2: "#aab4c6", muted: "#7d889c", grid: "#22304a", axis: "#2e3f5f", surface: "#111b2e",
};
export const COLORS = C;

export function svg(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function text(parent, x, y, str, attrs = {}) {
  const t = svg("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function root(container, w, h, label) {
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "chart";
  const s = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label });
  wrap.appendChild(s);
  container.appendChild(wrap);
  return s;
}
export const fmt = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—"
    : Math.abs(v) >= 1e6 ? v.toExponential(2)
    : v.toLocaleString("ko-KR", { maximumFractionDigits: d, minimumFractionDigits: 0 });

function niceTicks(min, max, n = 5) {
  const span = max - min || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const start = Math.ceil(min / step) * step;
  const out = [];
  for (let v = start; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

// ---------- 툴팁 ----------
const tip = () => document.getElementById("tooltip");
export function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.hidden = false;
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const r = t.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > W - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > H - 8) y = evt.clientY - r.height - pad;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export const hideTip = () => { tip().hidden = true; };

// ---------- 발산형 가로 막대 (예측 근거) ----------
export function shapBars(container, items, { top = 10, unit = "개" } = {}) {
  const rows = [...items].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, top);
  const W = 660, rowH = 28, padL = 270, padR = 70, H = rows.length * rowH + 30;
  const s = root(container, W, H, "예측 근거 기여도 막대");
  const maxAbs = Math.max(1e-6, ...rows.map((r) => Math.abs(r.value)));
  const mid = padL + (W - padL - padR) / 2;
  const half = (W - padL - padR) / 2;
  svg("line", { x1: mid, x2: mid, y1: 4, y2: H - 22, stroke: C.axis }, s);
  rows.forEach((r, i) => {
    const y = 8 + i * rowH;
    const len = (Math.abs(r.value) / maxAbs) * half;
    const up = r.value > 0;
    const x = up ? mid : mid - len;
    text(s, padL - 10, y + 15, r.label || r.name, { "text-anchor": "end", class: "lbl" });
    const bar = svg("rect", { x, y: y + 4, width: Math.max(len, 1.5), height: rowH - 10, rx: 3,
      fill: up ? C.critical : C.good, opacity: 0.9 }, s);
    // 값은 오른쪽 고정 열에 — 긴 음수 막대가 변수명 열을 침범해도 겹치지 않게
    text(s, W - 6, y + 15, `${up ? "+" : "−"}${fmt(Math.abs(r.value), 1)}`,
      { "text-anchor": "end", class: "lbl-strong", fill: up ? "#ff8a8a" : "#4cc24c" });
    const hit = svg("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, s);
    hit.addEventListener("pointermove", (e) => showTip(e,
      `<b>${r.label || r.name}</b><br><span class="tt-k">현재 값</span> ${r.raw ?? "—"}<br>` +
      `<span class="tt-k">예측 불량 칩 수</span> ${up ? "+" : "−"}${fmt(Math.abs(r.value), 2)}${unit} ${up ? "증가" : "감소"}`));
    hit.addEventListener("pointerleave", hideTip);
    bar.style.pointerEvents = "none";
  });
  text(s, mid - half, H - 6, "← 불량 감소", { class: "lbl" });
  text(s, mid + half, H - 6, "불량 증가 →", { "text-anchor": "end", class: "lbl" });
}

// ---------- 히스토그램 + 규격선 ----------
export function histogram(container, { values, bins, edges, lines = [], xLabel = "", height = 240, width = 560, label = "분포" }) {
  let counts = bins, e = edges;
  if (!counts) {
    const vs = values.filter((v) => v !== null && !Number.isNaN(v));
    const lo = Math.min(...vs), hi = Math.max(...vs), k = 24;
    e = Array.from({ length: k + 1 }, (_, i) => lo + ((hi - lo) * i) / k);
    counts = new Array(k).fill(0);
    vs.forEach((v) => { counts[Math.min(k - 1, Math.floor(((v - lo) / (hi - lo || 1)) * k))]++; });
  }
  const W = width, H = height, pl = 44, pr = 16, pt = 22, pb = 40;
  const s = root(container, W, H, label);
  const lo = Math.min(e[0], ...lines.map((l) => l.x)), hi = Math.max(e[e.length - 1], ...lines.map((l) => l.x));
  const X = (v) => pl + ((v - lo) / (hi - lo || 1)) * (W - pl - pr);
  const maxC = Math.max(...counts);
  const Y = (c) => H - pb - (c / maxC) * (H - pt - pb);
  niceTicks(0, maxC, 4).forEach((t) => {
    svg("line", { x1: pl, x2: W - pr, y1: Y(t), y2: Y(t), stroke: C.grid }, s);
    text(s, pl - 6, Y(t) + 4, fmt(t, 0), { "text-anchor": "end" });
  });
  counts.forEach((c, i) => {
    const x0 = X(e[i]), x1 = X(e[i + 1]);
    const r = svg("rect", { x: x0 + 1, y: Y(c), width: Math.max(0, x1 - x0 - 2), height: H - pb - Y(c), rx: 2, fill: C.s1, opacity: 0.85 }, s);
    r.addEventListener("pointermove", (ev) => showTip(ev, `<span class="tt-k">구간</span> ${fmt(e[i], 2)} ~ ${fmt(e[i + 1], 2)}<br><b>${c}</b>장`));
    r.addEventListener("pointerleave", hideTip);
  });
  svg("line", { x1: pl, x2: W - pr, y1: H - pb, y2: H - pb, stroke: C.axis }, s);
  niceTicks(lo, hi, 6).forEach((t) => text(s, X(t), H - pb + 16, fmt(t, 1), { "text-anchor": "middle" }));
  if (xLabel) text(s, (W + pl) / 2, H - 6, xLabel, { "text-anchor": "middle", class: "lbl" });
  lines.forEach((l, i) => {
    svg("line", { x1: X(l.x), x2: X(l.x), y1: pt - 6, y2: H - pb, stroke: l.color || C.critical, "stroke-width": 2 }, s);
    text(s, X(l.x) + (l.anchor === "end" ? -5 : 5), pt - 8 + (i % 2) * 12, l.label,
      { "text-anchor": l.anchor || "start", class: "lbl-strong", fill: l.color || C.critical });
  });
}

// ---------- 관리도 (I / MR / p) ----------
export function controlChart(container, { values, center, ucl, lcl, flags = [], marks = [], xs, xLabel = "",
  yLabel = "", height = 260, label = "관리도", tipFn, uclSeries, lclSeries }) {
  const W = 1000, H = height, pl = 58, pr = 70, pt = yLabel ? 28 : 16, pb = 36;
  const s = root(container, W, H, label);
  const vs = values.filter((v) => v !== null && v !== undefined);
  const lim = [...vs, ucl, lcl, ...(uclSeries || []), ...(lclSeries || [])].filter((v) => v !== null && v !== undefined);
  let lo = Math.min(...lim), hi = Math.max(...lim);
  const padV = (hi - lo) * 0.06; lo -= padV; hi += padV;
  const n = values.length;
  const X = (i) => pl + (n <= 1 ? 0 : (i / (n - 1)) * (W - pl - pr));
  const Y = (v) => pt + (1 - (v - lo) / (hi - lo || 1)) * (H - pt - pb);
  niceTicks(lo, hi, 5).forEach((t) => {
    svg("line", { x1: pl, x2: W - pr, y1: Y(t), y2: Y(t), stroke: C.grid }, s);
    text(s, pl - 8, Y(t) + 4, fmt(t, 2), { "text-anchor": "end" });
  });
  if (yLabel) text(s, 12, 12, yLabel, { class: "lbl" });
  const limLine = (series, v, color, name) => {
    if (series) {
      svg("path", { d: series.map((u, i) => `${i ? "L" : "M"}${X(i)},${Y(u)}`).join(""), fill: "none", stroke: color, "stroke-width": 1.5 }, s);
      text(s, W - pr + 6, Y(series[series.length - 1]) + 4, name, { class: "lbl-strong", fill: color });
    } else if (v !== null && v !== undefined) {
      svg("line", { x1: pl, x2: W - pr, y1: Y(v), y2: Y(v), stroke: color, "stroke-width": 1.5 }, s);
      text(s, W - pr + 6, Y(v) + 4, `${name} ${fmt(v, 2)}`, { class: "lbl-strong", fill: color });
    }
  };
  limLine(uclSeries, ucl, C.critical, "UCL");
  limLine(lclSeries, lcl, C.critical, "LCL");
  limLine(null, center, C.ink2, "CL");
  let d = "";
  values.forEach((v, i) => { if (v !== null && v !== undefined) d += `${d && values[i - 1] !== null ? "L" : "M"}${X(i)},${Y(v)}`; });
  svg("path", { d, fill: "none", stroke: C.s1, "stroke-width": n > 300 ? 1 : 1.6, opacity: 0.9 }, s);
  values.forEach((v, i) => {
    if (v === null || v === undefined) return;
    if (flags[i]) svg("circle", { cx: X(i), cy: Y(v), r: 4.5, fill: C.critical, stroke: C.surface, "stroke-width": 2 }, s);
    else if (marks[i]) svg("circle", { cx: X(i), cy: Y(v), r: 3.2, fill: C.warning, stroke: C.surface, "stroke-width": 1.5 }, s);
  });
  svg("line", { x1: pl, x2: W - pr, y1: H - pb, y2: H - pb, stroke: C.axis }, s);
  if (xs) {
    const step = Math.max(1, Math.ceil(n / 16));
    xs.forEach((x, i) => { if (i % step === 0) text(s, X(i), H - pb + 16, x, { "text-anchor": "middle" }); });
  }
  if (xLabel) text(s, W - pr, H - 4, xLabel, { "text-anchor": "end", class: "lbl" });
  // 크로스헤어 + 툴팁
  const cross = svg("line", { y1: pt, y2: H - pb, stroke: C.ink2, "stroke-width": 1, opacity: 0 }, s);
  const hit = svg("rect", { x: pl, y: pt, width: W - pl - pr, height: H - pt - pb, fill: "transparent" }, s);
  hit.addEventListener("pointermove", (e) => {
    const b = s.getBoundingClientRect();
    const px = ((e.clientX - b.left) / b.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - pl) / (W - pl - pr)) * (n - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 0.5);
    showTip(e, tipFn ? tipFn(i) : `#${i + 1}: <b>${fmt(values[i], 2)}</b>`);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
}

// ---------- ROC (단일 축, 2개 계열) ----------
export function rocChart(container, series, { label = "ROC 곡선" } = {}) {
  const W = 460, H = 390, pl = 50, pr = 20, pt = 30, pb = 44;
  const s = root(container, W, H, label);
  const X = (v) => pl + v * (W - pl - pr), Y = (v) => H - pb - v * (H - pt - pb);
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    svg("line", { x1: pl, x2: W - pr, y1: Y(t), y2: Y(t), stroke: C.grid }, s);
    text(s, pl - 8, Y(t) + 4, `${t * 100}%`, { "text-anchor": "end" });
    text(s, X(t), H - pb + 16, `${t * 100}%`, { "text-anchor": "middle" });
  });
  svg("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: C.axis }, s);
  text(s, (W + pl) / 2, H - 6, "오경보율 (정상 웨이퍼를 불량으로 잘못 경보)", { "text-anchor": "middle", class: "lbl" });
  text(s, 12, 14, "불량 검출률 ↑", { class: "lbl" });
  series.forEach((sr) => {
    svg("path", { d: sr.points.map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join(""),
      fill: "none", stroke: sr.color, "stroke-width": 2.2 }, s);
    if (sr.marker) {
      svg("circle", { cx: X(sr.marker[0]), cy: Y(sr.marker[1]), r: 5, fill: sr.color, stroke: C.surface, "stroke-width": 2 }, s);
    }
    const [lx, ly] = sr.points[Math.floor(sr.points.length * sr.labelAt)];
    text(s, X(lx) + 8, Y(ly) + (sr.labelDy || 0), sr.name, { class: "lbl-strong", fill: sr.color });
  });
}

// ---------- 웨이퍼 불량 지도 ----------
export function waferMap(container, mapStr, { title = "" } = {}) {
  const N = 26, cell = 12, pad = 6, W = N * cell + pad * 2, H = W + (title ? 22 : 0);
  const s = root(container, W, H, "웨이퍼 불량 지도");
  const oy = title ? 22 : 0;
  if (title) text(s, W / 2, 15, title, { "text-anchor": "middle", class: "lbl-strong" });
  svg("circle", { cx: W / 2, cy: oy + W / 2, r: W / 2 - 2, fill: "#0d1526", stroke: C.axis }, s);
  if (!mapStr) {
    text(s, W / 2, oy + W / 2, "지도 데이터 없음(원본 절단)", { "text-anchor": "middle", class: "lbl" });
    return;
  }
  for (let i = 0; i < N * N; i++) {
    const v = mapStr[i];
    if (v === "0") continue;
    svg("rect", { x: pad + (i % N) * cell + 1, y: oy + pad + Math.floor(i / N) * cell + 1, width: cell - 2, height: cell - 2, rx: 2,
      fill: v === "2" ? C.critical : "#1f5f45" }, s);
  }
}

// ---------- Pareto (막대 + 누적% 직접 라벨 — 이중 축 없음) ----------
export function pareto(container, items) {
  const W = 1000, H = 300, pl = 44, pr = 12, pt = 36, pb = 52;
  const s = root(container, W, H, "불량 유형 파레토");
  const maxC = Math.max(...items.map((i) => i.count));
  const bw = (W - pl - pr) / items.length;
  const Y = (c) => H - pb - (c / maxC) * (H - pt - pb);
  niceTicks(0, maxC, 4).forEach((t) => {
    svg("line", { x1: pl, x2: W - pr, y1: Y(t), y2: Y(t), stroke: C.grid }, s);
    text(s, pl - 6, Y(t) + 4, fmt(t, 0), { "text-anchor": "end" });
  });
  items.forEach((it, i) => {
    const x = pl + i * bw;
    const vital = it.cum_pct <= 80 || i === 0;
    const r = svg("rect", { x: x + 4, y: Y(it.count), width: bw - 8, height: H - pb - Y(it.count), rx: 3,
      fill: C.s1, opacity: vital ? 0.95 : 0.4 }, s);
    text(s, x + bw / 2, Y(it.count) - 16, `${it.count}장`, { "text-anchor": "middle", class: "lbl-strong" });
    text(s, x + bw / 2, Y(it.count) - 4, `누적 ${it.cum_pct}%`, { "text-anchor": "middle" });
    text(s, x + bw / 2, H - pb + 16, it.name, { "text-anchor": "middle", class: "lbl" });
    text(s, x + bw / 2, H - pb + 30, it.code, { "text-anchor": "middle" });
    r.addEventListener("pointermove", (e) => showTip(e, `<b>${it.name}</b> (${it.code})<br>${it.count}장 · 누적 ${it.cum_pct}%<br><span class="tt-k">평균 불량 칩</span> ${it.mean_target}개`));
    r.addEventListener("pointerleave", hideTip);
  });
  svg("line", { x1: pl, x2: W - pr, y1: H - pb, y2: H - pb, stroke: C.axis }, s);
}
