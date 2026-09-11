// 2.5D(아이소메트릭) 공정 라인. 장비 6대를 컨베이어 위에 배치하고 웨이퍼가 흘러가는 애니메이션을 그린다.
import { svg } from "./charts.js";

const COS = Math.cos(Math.PI / 6), SIN = Math.sin(Math.PI / 6);
const U = 34; // 월드 1단위 = 34px
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const iso = (x, y, z = 0) => [(x - y) * COS * U, (x + y) * SIN * U - z * U];
const pts = (arr) => arr.map((p) => p.join(",")).join(" ");

function anim(parent, tag, attrs) { if (!reduced) svg(tag, attrs, parent); }

function box(g, x, y, w, d, h, shade) {
  const [a, b, c, e] = [[x, y], [x + w, y], [x + w, y + d], [x, y + d]];
  svg("polygon", { points: pts([iso(...a), iso(...b), iso(...b, h), iso(...a, h)]), fill: shade.right, stroke: "#3a4d73", "stroke-width": 0.8 }, g);
  svg("polygon", { points: pts([iso(...e), iso(...c), iso(...c, h), iso(...e, h)]), fill: shade.left, stroke: "#3a4d73", "stroke-width": 0.8 }, g);
  svg("polygon", { points: pts([iso(...b), iso(...c), iso(...c, h), iso(...b, h)]), fill: shade.left, stroke: "#3a4d73", "stroke-width": 0.8 }, g);
  svg("polygon", { points: pts([iso(...a, h), iso(...b, h), iso(...c, h), iso(...e, h)]), fill: shade.top, stroke: "#4a6090", "stroke-width": 0.8 }, g);
}

const STEEL = { top: "#2b3c5e", left: "#1a2541", right: "#223052" };

// 장비별 장식(움직이는 요소) — 각 공정이 하는 일을 한눈에
const DECOR = {
  oxidation(g, x, y) {                 // 확산로: 붉게 달아오른 창 + 열기
    const [wx, wy] = iso(x + 0.15, y + 0.6, 0.9);
    const win = svg("rect", { x: wx - 20, y: wy - 10, width: 26, height: 20, rx: 3, fill: "#ff7b3a", opacity: 0.8,
      transform: `skewY(30) translate(0 ${-wx * 0.577})` }, g);
    anim(win, "animate", { attributeName: "opacity", values: "0.55;0.95;0.55", dur: "2.4s", repeatCount: "indefinite" });
    const [hx, hy] = iso(x + 0.7, y + 0.7, 2.2);
    for (let i = 0; i < 3; i++) {
      const w = svg("path", { d: `M${hx - 12 + i * 12},${hy} q4,-8 0,-16 q-4,-8 0,-16`, fill: "none", stroke: "#ff9a5e", "stroke-width": 1.5, opacity: 0 }, g);
      anim(w, "animate", { attributeName: "opacity", values: "0;0.8;0", dur: "2s", begin: `${i * 0.6}s`, repeatCount: "indefinite" });
      anim(w, "animateTransform", { attributeName: "transform", type: "translate", values: "0 6;0 -6", dur: "2s", begin: `${i * 0.6}s`, repeatCount: "indefinite" });
    }
  },
  coat(g, x, y) {                      // 스핀 코터: 돌아가는 원판
    const [cx, cy] = iso(x + 0.7, y + 0.7, 1.2);
    const disk = svg("g", { transform: `translate(${cx} ${cy}) scale(1 0.55)` }, g);
    svg("circle", { r: 22, fill: "#3f5f9c", stroke: "#7fb2f0", "stroke-width": 1.5 }, disk);
    const arm = svg("g", {}, disk);
    svg("rect", { x: -2, y: -20, width: 4, height: 40, fill: "#cfe1fb", rx: 2 }, arm);
    svg("rect", { x: -20, y: -2, width: 40, height: 4, fill: "#cfe1fb", rx: 2, opacity: 0.6 }, arm);
    anim(arm, "animateTransform", { attributeName: "transform", type: "rotate", from: "0", to: "360", dur: "1.2s", repeatCount: "indefinite" });
  },
  litho(g, x, y) {                     // 노광기: 위에서 내리쬐는 빛
    box(g, x + 0.45, y + 0.45, 0.5, 0.5, 2.6, { top: "#34487a", left: "#1f2c4d", right: "#283a63" });
    const [tx, ty] = iso(x + 0.7, y + 0.7, 2.1);
    const [bx, by] = iso(x + 0.7, y + 0.7, 1.3);
    const beam = svg("polygon", { points: `${tx - 4},${ty} ${tx + 4},${ty} ${bx + 16},${by} ${bx - 16},${by}`, fill: "#b7d3f6", opacity: 0.35 }, g);
    anim(beam, "animate", { attributeName: "opacity", values: "0.1;0.6;0.1", dur: "1.6s", repeatCount: "indefinite" });
  },
  etch(g, x, y) {                      // 식각 챔버: 보랏빛 플라즈마
    const [cx, cy] = iso(x + 0.7, y + 0.7, 1.35);
    const glow = svg("ellipse", { cx, cy, rx: 26, ry: 13, fill: "#a978ff", opacity: 0.5, filter: "url(#blur)" }, g);
    anim(glow, "animate", { attributeName: "opacity", values: "0.25;0.8;0.25", dur: "1.1s", repeatCount: "indefinite" });
    svg("ellipse", { cx, cy, rx: 18, ry: 9, fill: "none", stroke: "#d6c2ff", "stroke-width": 1.2 }, g);
  },
  implant(g, x, y) {                   // 이온주입: 빔라인을 따라 흐르는 이온
    const [ax, ay] = iso(x - 0.2, y + 0.7, 1.6);
    const [bx, by] = iso(x + 1.1, y + 0.7, 1.0);
    svg("line", { x1: ax, y1: ay, x2: bx, y2: by, stroke: "#2e3f5f", "stroke-width": 6, "stroke-linecap": "round" }, g);
    const beam = svg("line", { x1: ax, y1: ay, x2: bx, y2: by, stroke: "#6fe0b0", "stroke-width": 2.5, "stroke-dasharray": "4 8", "stroke-linecap": "round" }, g);
    anim(beam, "animate", { attributeName: "stroke-dashoffset", from: "24", to: "0", dur: "0.6s", repeatCount: "indefinite" });
  },
  inspect(g, x, y) {                   // 검사기: 좌우로 훑는 스캔 라인
    const [cx, cy] = iso(x + 0.7, y + 0.7, 1.25);
    svg("ellipse", { cx, cy, rx: 22, ry: 11, fill: "#16304f", stroke: "#3987e5", "stroke-width": 1 }, g);
    const scan = svg("line", { x1: cx - 18, y1: cy - 8, x2: cx - 18, y2: cy + 8, stroke: "#7fd0ff", "stroke-width": 2 }, g);
    anim(scan, "animateTransform", { attributeName: "transform", type: "translate", values: "0 0;36 0;0 0", dur: "2s", repeatCount: "indefinite" });
  },
};

const LIGHT = { ok: "#0ca30c", warn: "#fab219", crit: "#d03b3b" };

export function renderLine(container, processes, { onSelect }) {
  container.innerHTML = "";
  // 장비를 월드 (1,-1) 방향으로 늘어놓으면 화면에서는 수평 한 줄이 된다(대각선으로 흘러내리지 않음).
  const S = 2.4, n = processes.length, D = 1.45, BW = 0.28;
  const at = (i) => [i * S, -i * S];
  const beltPt = (t, off = 0) => iso(0.7 + D + t + off, 0.7 + D - t + off);
  const t0 = -1.3, t1 = (n - 1) * S + 1.3;
  const labelY = (1.4 + 2 * (D + BW)) * SIN * U + 26;
  const left = Math.min(beltPt(t0, -BW)[0], beltPt(t0, BW)[0]) - 30;
  const right = Math.max(beltPt(t1, -BW)[0], beltPt(t1, BW)[0]) + 30;
  const top = -3.4 * U - 40, bottom = labelY + 30;
  const s = svg("svg", { viewBox: `${left} ${top} ${right - left} ${bottom - top}`, role: "group",
    "aria-label": "공정 라인 — 장비를 선택하면 조건 패널이 열립니다" });
  container.appendChild(s);
  const defs = svg("defs", {}, s);
  const f = svg("filter", { id: "blur", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
  svg("feGaussianBlur", { stdDeviation: 5 }, f);

  // 바닥판 + 컨베이어
  svg("polygon", { points: pts([beltPt(t0, -D - 0.9), beltPt(t1, -D - 0.9), beltPt(t1, BW + 0.5), beltPt(t0, BW + 0.5)]),
    fill: "#0e182b", stroke: "#1a2741" }, s);
  svg("polygon", { points: pts([beltPt(t0, -BW), beltPt(t1, -BW), beltPt(t1, BW), beltPt(t0, BW)]), fill: "#141f36", stroke: "#2e3f5f" }, s);
  const beltLine = svg("line", { x1: beltPt(t0)[0], y1: beltPt(t0)[1], x2: beltPt(t1)[0], y2: beltPt(t1)[1],
    stroke: "#2e3f5f", "stroke-width": 1, "stroke-dasharray": "6 6" }, s);
  anim(beltLine, "animate", { attributeName: "stroke-dashoffset", from: "12", to: "0", dur: "0.8s", repeatCount: "indefinite" });

  // 흘러가는 웨이퍼
  const pathD = `M${beltPt(t0).join(",")} L${beltPt(t1).join(",")}`;
  for (let i = 0; i < 5; i++) {
    const wf = svg("ellipse", { rx: 9, ry: 4.5, fill: "#b9c8e6", stroke: "#e8edf6", "stroke-width": 0.8, opacity: reduced ? 0 : 0.9 }, s);
    anim(wf, "animateMotion", { path: pathD, dur: "14s", begin: `${-i * 2.8}s`, repeatCount: "indefinite" });
  }

  const nodes = {};
  processes.forEach((p, i) => {
    const [x, y] = at(i);
    const g = svg("g", { class: "equip", tabindex: 0, role: "button", "aria-label": `${p.name} — ${p.desc}` }, s);
    const body = svg("g", { class: "eq-body" }, g);
    box(body, x, y, 1.4, 1.4, 1.2, STEEL);
    DECOR[p.id]?.(body, x, y);
    const lx = iso(x + 1.4, y + 1.4, 0)[0], ly = labelY;
    const light = svg("circle", { cx: iso(x + 1.4, y + 0.2, 1.05)[0], cy: iso(x + 1.4, y + 0.2, 1.05)[1], r: 4.5, fill: LIGHT.ok, stroke: "#0b1220", "stroke-width": 1.5 }, g);
    svg("text", { x: lx, y: ly, "text-anchor": "middle", class: "eq-label" }, g).textContent = `${i + 1}. ${p.name}`;
    svg("text", { x: lx, y: ly + 16, "text-anchor": "middle", class: "eq-sub" }, g).textContent =
      p.params.length ? `조절 변수 ${p.params.length}개` : "결과 확인";
    const pick = () => onSelect(p.id);
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    nodes[p.id] = { g, light };
  });

  return {
    setActive(id) { Object.entries(nodes).forEach(([k, v]) => v.g.classList.toggle("active", k === id)); },
    setStatus(id, level) { if (nodes[id]) nodes[id].light.setAttribute("fill", LIGHT[level] || LIGHT.ok); },
  };
}
