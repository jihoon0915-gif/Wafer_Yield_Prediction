// 브라우저용 LightGBM 추론 + TreeSHAP.
// scripts/14_export_web_dashboard.py가 만든 model.json(sklearn 파이프라인을 그대로 옮긴 것)을 읽어
// 대치 → 표준화 → 원핫 → 피처선택 → 트리 150개 합산을 서버 없이 재현한다.

const OXIDE_LSL_NM = 700;
const ZERO_THRESHOLD = 1e-35; // LightGBM kZeroThreshold

export function createModel(spec) {
  const nNum = spec.numeric.length;

  // 사용자는 '산화막 두께'로 다루고, 모델은 700 - 두께(규격 대비 부족분)를 입력으로 받는다.
  function toModelRow(row) {
    const out = { ...row };
    if (row.thickness !== undefined && row.oxid_thickness_spec_gap === undefined) {
      out.oxid_thickness_spec_gap = row.thickness === null ? null : OXIDE_LSL_NM - row.thickness;
    }
    return out;
  }

  function transform(row) {
    const r = toModelRow(row);
    const full = [];
    spec.numeric.forEach((name, i) => {
      let v = r[name];
      if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) v = spec.median[i];
      full.push((Number(v) - spec.mean[i]) / spec.scale[i]);
    });
    spec.categorical.forEach((name, j) => {
      let v = r[name];
      if (v === null || v === undefined || v === "") v = spec.cat_mode[j];
      spec.cat_categories[j].slice(1).forEach((c) => full.push(String(v) === c ? 1 : 0));
    });
    return spec.selected_index.map((i) => full[i]);
  }

  function goesLeft(node, v) {
    const missing =
      Number.isNaN(v) ||
      (node.m === "Zero" && Math.abs(v) <= ZERO_THRESHOLD);
    if (missing && node.m !== "None") return node.d;
    return v <= node.t;
  }

  function treeValue(nodes, x) {
    let i = 0;
    while (nodes[i].f !== undefined) i = goesLeft(nodes[i], x[nodes[i].f]) ? nodes[i].l : nodes[i].r;
    return nodes[i].v;
  }

  // 각 트리의 기댓값(학습 표본 가중 평균 잎 값) — SHAP의 기준값
  function treeExpected(nodes, i = 0) {
    const n = nodes[i];
    if (n.f === undefined) return n.v;
    const cl = nodes[n.l].c, cr = nodes[n.r].c;
    return (cl * treeExpected(nodes, n.l) + cr * treeExpected(nodes, n.r)) / (cl + cr);
  }
  const baseValue = spec.trees.reduce((s, t) => s + treeExpected(t), 0);

  // --- Path-dependent TreeSHAP (Lundberg et al. 2018, shap tree.cpp와 동일한 절차) ---
  function extend(path, depth, zero, one, feat) {
    path[depth] = { f: feat, z: zero, o: one, w: depth === 0 ? 1 : 0 };
    for (let i = depth - 1; i >= 0; i--) {
      path[i + 1].w += (one * path[i].w * (i + 1)) / (depth + 1);
      path[i].w = (zero * path[i].w * (depth - i)) / (depth + 1);
    }
  }

  function unwind(path, depth, idx) {
    const { o: one, z: zero } = path[idx];
    let next = path[depth].w;
    for (let i = depth - 1; i >= 0; i--) {
      if (one !== 0) {
        const tmp = path[i].w;
        path[i].w = (next * (depth + 1)) / ((i + 1) * one);
        next = tmp - (path[i].w * zero * (depth - i)) / (depth + 1);
      } else {
        path[i].w = (path[i].w * (depth + 1)) / (zero * (depth - i));
      }
    }
    for (let i = idx; i < depth; i++) {
      path[i].f = path[i + 1].f;
      path[i].z = path[i + 1].z;
      path[i].o = path[i + 1].o;
    }
  }

  function unwoundSum(path, depth, idx) {
    const { o: one, z: zero } = path[idx];
    let next = path[depth].w;
    let total = 0;
    for (let i = depth - 1; i >= 0; i--) {
      if (one !== 0) {
        const tmp = (next * (depth + 1)) / ((i + 1) * one);
        total += tmp;
        next = path[i].w - (tmp * zero * (depth - i)) / (depth + 1);
      } else {
        total += path[i].w / zero / ((depth - i) / (depth + 1));
      }
    }
    return total;
  }

  function recurse(nodes, x, phi, i, parentPath, depth, zero, one, feat) {
    const path = parentPath.slice(0, depth).map((p) => ({ ...p }));
    extend(path, depth, zero, one, feat);
    const n = nodes[i];
    if (n.f === undefined) {
      for (let k = 1; k <= depth; k++) {
        const w = unwoundSum(path, depth, k);
        phi[path[k].f] += w * (path[k].o - path[k].z) * n.v;
      }
      return;
    }
    const left = goesLeft(n, x[n.f]);
    const hot = left ? n.l : n.r;
    const cold = left ? n.r : n.l;
    let inZero = 1, inOne = 1, d = depth;
    const found = path.findIndex((p, k) => k <= depth && p.f === n.f);
    if (found >= 0) {
      inZero = path[found].z;
      inOne = path[found].o;
      unwind(path, d, found);
      d -= 1;
    }
    recurse(nodes, x, phi, hot, path, d + 1, (nodes[hot].c / n.c) * inZero, inOne, n.f);
    recurse(nodes, x, phi, cold, path, d + 1, (nodes[cold].c / n.c) * inZero, 0, n.f);
  }

  function predictRaw(row) {
    const x = transform(row);
    return spec.trees.reduce((s, t) => s + treeValue(t, x), 0);
  }

  return {
    features: spec.selected_names,
    metrics: spec.metrics,
    baseValue,
    // 결함 다이 수는 음수가 될 수 없음(학습 파이프라인과 동일한 하한)
    predict: (row) => Math.max(0, predictRaw(row)),
    predictRaw,
    explain(row) {
      const x = transform(row);
      const phi = new Array(x.length).fill(0);
      for (const t of spec.trees) recurse(t, x, phi, 0, [], 0, 1, 1, -1);
      return spec.selected_names.map((name, i) => ({ name, value: phi[i] }));
    },
  };
}
