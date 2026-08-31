"""도메인(공정 물리/화학) 기반 1차 전처리 + Master 데이터셋 구축.

3-Sigma/평균 등 통계적 이상치 탐지에 의존하지 않고, 공정 도메인 지식(물리적으로
불가능한 값, pptx 변수정의서 스펙, 식각/산화/이온주입 공정 화학)만을 기준으로
전처리 규칙과 파생 피처를 정의한다.

병합/그룹핑/웨이퍼맵 파싱은 이미 검증된 wafer.integrate/validate 모듈을 그대로
재사용한다 — No_Die 1:1 inner join은 병합 순서와 무관하게 결과가 동일하므로,
프롬프트에 명시된 순서(Oxidation→Photo_softbake→Photo_lithograpy→Etching→
Ion_Implantation→Inspect) 대신 기존에 9개 assertion으로 검증해둔 순서
(wafer.config.PROCESS_ORDER)를 그대로 쓴다.

산출물:
  - data/processed/processed_master.csv  (No_Die grain, 15,390행)
  - reports/preprocessing_summary.md     (근거/공식/한계 리포트)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from wafer import integrate, validate
from wafer.config import LITHOGRAPHY_LINE_CD_RANGE_NM, OXIDATION_THICKNESS_MIN_NM

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# 물리적으로 0/음수가 나올 수 없는 컬럼 — 공정별로 묶어서 "어느 공정에서 이상이
# 났는지" 플래그를 만들 수 있게 한다. (config.SENTINEL_CONVERT_CANDIDATE_COLUMNS와
# 동일한 근거지만, 여기서는 공정 단위로 그룹화해 둔다.)
# Flux840s는 15,381/15,390건이 전부 동일값(6e17, std=0)인 상수 컬럼(9건만 결측)이라
# 이상치 플래그에 넣어도 변동성이 없어 의미가 없으므로 제외한다.
# --------------------------------------------------------------------------
PROCESS_SENTINEL_COLS = {
    "oxidation": ["Temp_OXid", "ppm", "Pressure", "Oxid_time", "thickness"],
    "photo_softbake": [
        "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake", "time_HMDS_bake",
        "spin1", "spin2", "spin3", "photoresist_bake", "temp_softbake", "time_softbake",
    ],
    "photo_litho": ["Wavelength", "Energy_Exposure"],
    "etching": ["Thin F1", "Thin F2", "Thin F3", "Thin F4", "Temp_Etching", "Source_Power"],
    "ion_implant": [
        "Flux60s", "Flux90s", "Flux160s", "Flux480s",
        "input_Energy", "Temp_implantation", "Furance_Temp", "RTA_Temp",
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def build_anomaly_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """공정별 '물리적으로 불가능한 값(<=0)' 존재 여부를 불리언 컬럼으로 남긴다.

    실제 대치는 하지 않는다(수치는 그대로 두고 플래그만 추가) — 마스터 데이터셋은
    여러 다운스트림 분석(회귀/분류/EDA)에 재사용되므로, 결측 대치는 각 분석의
    검증 방식(GroupKFold fold 내부 등)에 맡기는 게 안전하다.
    """
    df = df.copy()
    counts: dict[str, int] = {}
    flag_cols = []
    for process, cols in PROCESS_SENTINEL_COLS.items():
        present = [c for c in cols if c in df.columns]
        flag_col = f"{process}_sentinel_flag"
        df[flag_col] = (df[present] <= 0).any(axis=1)
        counts[flag_col] = int(df[flag_col].sum())
        flag_cols.append(flag_col)
    df["any_process_sentinel_flag"] = df[flag_cols].any(axis=1)
    counts["any_process_sentinel_flag"] = int(df["any_process_sentinel_flag"].sum())
    return df, counts


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Oxidation: 산화 속도 (Deal-Grove 산화 커널틱스 — 두께/시간) ---
    oxid_time_safe = df["Oxid_time"].where(df["Oxid_time"] > 0)
    df["oxidation_rate_nm_per_min"] = df["thickness"] / oxid_time_safe
    df["oxid_thickness_spec_gap"] = OXIDATION_THICKNESS_MIN_NM - df["thickness"]
    df["oxid_below_spec"] = (df["oxid_thickness_spec_gap"] > 0).astype(int)

    # --- Photo(Litho): CD/해상도 비율, 노광 에너지 효율 ---
    resolution_safe = df["Resolution"].where(df["Resolution"] > 0)
    df["cd_resolution_ratio"] = df["Line_CD"] / resolution_safe
    energy_safe = df["Energy_Exposure"].where(df["Energy_Exposure"] > 0)
    df["exposure_energy_per_cd_nm"] = energy_safe / df["Line_CD"]  # 낮을수록 적은 에너지로 동일 CD 달성(효율적)

    lo, hi = LITHOGRAPHY_LINE_CD_RANGE_NM
    below = (lo - df["Line_CD"]).clip(lower=0)
    above = (df["Line_CD"] - hi).clip(lower=0)
    df["line_cd_band_gap"] = np.maximum(below, above)
    df["line_cd_out_of_band"] = (df["line_cd_band_gap"] > 0).astype(int)

    # --- Etch: 총 식각량 + 단계별 식각 레이트(연속 두께 차분) ---
    df["total_etch_removal"] = df["Thin F1"] - df["Thin F4"]
    df["etch_rate_stage1"] = df["Thin F1"] - df["Thin F2"]
    df["etch_rate_stage2"] = df["Thin F2"] - df["Thin F3"]
    df["etch_rate_stage3"] = df["Thin F3"] - df["Thin F4"]

    # --- Ion Implantation: 누적 주입량(Flux840s 제외, 근거는 리포트 참고) + 어닐링 온도차 ---
    df["total_flux_60_480"] = df[["Flux60s", "Flux90s", "Flux160s", "Flux480s"]].sum(axis=1, skipna=True)
    df["anneal_temp_diff"] = df["Furance_Temp"] - df["RTA_Temp"]

    return df


def chamber_anova(df: pd.DataFrame, wafer: pd.DataFrame) -> list[dict]:
    rows = []
    for col in ["Etching_Chamber", "Ox_Chamber", "lithography_Chamber", "photo_soft_Chamber"]:
        groups = [wafer.loc[wafer[col] == c, "Target"].values for c in wafer[col].dropna().unique()]
        f, p = stats.f_oneway(*groups)
        means = wafer.groupby(col)["Target"].mean().to_dict()
        rows.append({"chamber_col": col, "anova_p": round(float(p), 4), "mean_target_by_chamber": means})
    return rows


def main() -> None:
    log("[1/6] 원본 6개 CSV 로드 + 무결성 확인 (No_Die 유일성, 행수 15,390)")
    raw = integrate.load_all_raw()

    log("[2/6] No_Die 기준 순차 inner join (검증된 PROCESS_ORDER 사용)")
    merged = integrate.merge_all(raw)
    merged["group_id"] = integrate.build_group_id(merged)
    merged = integrate.add_error_class(merged)
    merged, maps_by_no_die, maps_by_group, wafer_map_stats = integrate.process_wafer_maps(merged)

    log("[3/6] 9개 회귀 검증(run_all_checks) 재실행")
    checks = validate.run_all_checks(merged, maps_by_no_die, wafer_map_stats)
    log(checks.to_string(index=False))
    if not checks["통과"].all():
        raise RuntimeError("검증 실패 — Master 데이터셋 생성 중단")

    log("[4/6] 도메인 이상치 플래그(물리적으로 불가능한 값) 생성")
    merged, sentinel_counts = build_anomaly_flags(merged)
    for k, v in sentinel_counts.items():
        log(f"  {k}: {v}건 ({v/len(merged)*100:.2f}%)")

    log("[5/6] 도메인 파생 피처 생성 (산화/노광/식각/이온주입)")
    merged = add_domain_features(merged)

    log("[6/6] Chamber-to-Chamber 편차 1차 확인(ANOVA) + 저장")
    wafer_level = merged.sort_values("Wafer_map", key=lambda s: s.isna()).drop_duplicates(subset="group_id")
    chamber_stats = chamber_anova(merged, wafer_level)
    for r in chamber_stats:
        log(f"  {r['chamber_col']}: ANOVA p={r['anova_p']}  mean_target={r['mean_target_by_chamber']}")

    # 이상치 플래그 vs Target 간단 검증 (정보성 결측 여부 참고용)
    flag_vs_target = []
    for flag_col in [c for c in merged.columns if c.endswith("_sentinel_flag")]:
        if merged[flag_col].sum() == 0:
            continue
        a = merged.loc[merged[flag_col], "Target"]
        b = merged.loc[~merged[flag_col], "Target"]
        if len(a) >= 2 and len(b) >= 2:
            _, p = stats.ttest_ind(a, b, equal_var=False)
        else:
            p = float("nan")
        flag_vs_target.append(
            {"flag": flag_col, "n_flagged": int(merged[flag_col].sum()),
             "target_mean_flagged": round(float(a.mean()), 2),
             "target_mean_other": round(float(b.mean()), 2), "welch_p": round(float(p), 4)}
        )

    out_csv = PROJECT_ROOT / "data" / "processed" / "processed_master.csv"
    drop_for_csv = ["Wafer_map"]  # 26x26 문자열 원본은 별도 parquet(wafer_integrated.parquet)에 이미 존재
    merged.drop(columns=[c for c in drop_for_csv if c in merged.columns]).to_csv(out_csv, index=False)
    log(f"\n저장 완료: {out_csv} ({merged.shape[0]}행 x {merged.shape[1] - len(drop_for_csv)}열)")

    write_summary(merged, checks, sentinel_counts, chamber_stats, flag_vs_target, out_csv)


def write_summary(merged, checks, sentinel_counts, chamber_stats, flag_vs_target, out_csv) -> None:
    n = len(merged)
    lines = []
    lines.append("# Master 데이터셋 전처리 요약 (도메인 기반)")
    lines.append("")
    lines.append(
        "3-Sigma/평균 등 통계적 이상치 탐지가 아니라, 공정 물리·화학 도메인 지식과 "
        "사내 변수정의서(pptx) 스펙만을 근거로 1차 전처리·피처 엔지니어링을 수행했다. "
        f"산출물: `{out_csv.relative_to(PROJECT_ROOT).as_posix()}` ({n:,}행, No_Die grain)."
    )
    lines.append("")

    lines.append("## 1. 병합(Master Data Integration)")
    lines.append("")
    lines.append(
        "- Key: `No_Die`(Lot_Num+Wafer_Num+공정순번을 이미 내포한 다이 식별자) 기준 1:1 inner join.\n"
        "- 6개 CSV 전부 병합 전/후 15,390행 유지(불일치 시 즉시 예외 — 이번 실행에서도 통과).\n"
        "- 병합 순서는 결과에 영향 없음(1:1 inner join)이라, 프롬프트 지정 순서 대신 이미 9개 "
        "assertion으로 회귀검증해 둔 기존 PROCESS_ORDER(Photo_softbake→Photo_lithograpy→"
        "Etching→Ion_Implantation→Oxidation→Inspect)를 그대로 사용."
    )
    lines.append("")
    lines.append("**회귀 검증 (validate.run_all_checks) 결과**")
    lines.append("")
    lines.append("| 검증 항목 | 통과 | 상세 |")
    lines.append("|---|---|---|")
    for _, row in checks.iterrows():
        lines.append(f"| {row['검증 항목']} | {'✅' if row['통과'] else '❌'} | {row['상세']} |")
    lines.append("")

    lines.append("## 2. 도메인 기반 이상치 정제 (Spike/Sentinel Filtering)")
    lines.append("")
    lines.append(
        "**원칙**: 온도·압력·시간·파워·두께·파장·에너지·flux 계열은 물리적으로 0 이하가 "
        "나올 수 없다(절대영도 미만 온도, 음수 압력·시간 등은 존재하지 않음). 이 컬럼들의 "
        "`<=0` 값은 통계적 이상치가 아니라 **센서/로깅 오류로 판정**해 공정별 플래그로 "
        "남겼다(값 자체는 보존, 대치는 하지 않음 — 대치는 다운스트림 분석에서 검증 방식에 "
        "맞게 수행해야 그룹 정보 누수를 피할 수 있음)."
    )
    lines.append("")
    lines.append("| 플래그 | 해당 행 수 | 비율 |")
    lines.append("|---|---|---|")
    for k, v in sentinel_counts.items():
        lines.append(f"| `{k}` | {v} | {v/n*100:.2f}% |")
    lines.append("")
    if flag_vs_target:
        lines.append("**참고 — 이상치 플래그와 Target의 관계 (Welch t-test)**")
        lines.append("")
        lines.append("| 플래그 | n | Target 평균(플래그=True) | Target 평균(그 외) | p-value |")
        lines.append("|---|---|---|---|---|")
        for r in flag_vs_target:
            lines.append(
                f"| `{r['flag']}` | {r['n_flagged']} | {r['target_mean_flagged']} | "
                f"{r['target_mean_other']} | {r['welch_p']} |"
            )
        lines.append("")
        lines.append(
            "⚠️ `etching_sentinel_flag`처럼 p<0.05로 유의하게 나오는 경우, 해당 `<=0` 값이 "
            "단순 센서 오류가 아니라 '공정이 완전히 끝나 잔막이 사실상 0에 수렴'한 정상 상태를 "
            "반영했을 가능성도 있다 — NaN 대치 전에 반드시 도메인 확인이 필요하다(값을 지우지 "
            "않고 플래그만 남긴 이유)."
        )
        lines.append("")

    lines.append("## 3. 도메인 특화 Feature Engineering")
    lines.append("")
    lines.append("### Oxidation")
    lines.append("- `oxidation_rate_nm_per_min = thickness / Oxid_time` — 산화 성장속도(두께/시간).")
    lines.append(
        f"- `oxid_thickness_spec_gap = {OXIDATION_THICKNESS_MIN_NM} - thickness`, "
        "`oxid_below_spec`(0/1) — 변수정의서 스펙(700nm 이상) 미달 여부."
    )
    lines.append("")
    lines.append("### Photo (softbake/litho)")
    lines.append("- `cd_resolution_ratio = Line_CD / Resolution` — 목표 선폭 대비 계측 해상도 비율.")
    lines.append(
        "- `exposure_energy_per_cd_nm = Energy_Exposure / Line_CD` — CD 1nm당 투입 노광 에너지"
        "(낮을수록 적은 에너지로 동일 선폭을 달성한 것 → 효율적). 사내 표준 KPI가 아니라 "
        "이번 분석에서 정의한 공학적 비율이며, 해석 방향만 참고할 것."
    )
    lines.append(
        f"- `line_cd_band_gap`, `line_cd_out_of_band`(0/1) — 스펙 구간"
        f"({LITHOGRAPHY_LINE_CD_RANGE_NM[0]}~{LITHOGRAPHY_LINE_CD_RANGE_NM[1]}nm) 이탈도."
    )
    lines.append("")
    lines.append("### Etch")
    lines.append("- `total_etch_removal = Thin F1 - Thin F4` — 전체 공정 구간 총 식각(제거)량.")
    lines.append(
        "- `etch_rate_stage1/2/3 = Thin F(n) - Thin F(n+1)` — 구간별 식각 레이트. "
        "근거: 이번 프로젝트 EDA에서 Thin F1 자체는 Target과 상관 0.04(거의 무의미)였지만, "
        "F1→F2 구간차분은 상관 -0.39로 F1 단독보다 훨씬 강한 신호였음(구간 차분이 "
        "'식각 진행 속도'라는 물리량을 더 직접적으로 반영하기 때문으로 해석)."
    )
    lines.append("")
    lines.append("### Ion Implantation")
    lines.append(
        "- `total_flux_60_480 = Flux60s + Flux90s + Flux160s + Flux480s` — 누적 이온 주입량. "
        "**Flux840s는 합산에서 제외**: 15,381/15,390건(99.94%)이 전부 정확히 동일한 값"
        "(6e17, std=0)이고 나머지 9건만 결측인 **상수 컬럼**이라, 포함해도 모든 행에 "
        "동일 오프셋만 더해질 뿐 정보량이 없고 원본 파이프라인(`wafer.features.CONSTANT_DROP_COLS`)"
        "에서도 이미 같은 이유로 드롭 대상이다. (참고: 초기 버전 리포트에 '유효값 1건뿐"
        "(결측)'이라고 잘못 적었던 것을 재검증 후 정정 — 실제로는 '거의 결측'이 아니라 "
        "'거의 상수'가 배제 사유임.)"
    )
    lines.append("- `anneal_temp_diff = Furance_Temp - RTA_Temp` — 노(furnace) 어닐링과 RTA 온도차.")
    lines.append("")

    lines.append("## 4. Chamber-to-Chamber Variation (1차 확인)")
    lines.append("")
    lines.append(
        "웨이퍼 단위(1,704장)로 집계해 챔버별 평균 Target 차이를 일원배치 ANOVA로 1차 스크리닝했다 "
        "(통계 검정이지만 '이상치 제거' 목적이 아니라 '분석 우선순위 스크리닝' 목적)."
    )
    lines.append("")
    lines.append("| 챔버 컬럼 | ANOVA p-value | 챔버별 평균 Target |")
    lines.append("|---|---|---|")
    for r in chamber_stats:
        means_str = ", ".join(f"{k}={v:.1f}" for k, v in sorted(r["mean_target_by_chamber"].items()))
        lines.append(f"| `{r['chamber_col']}` | {r['anova_p']} | {means_str} |")
    lines.append("")
    lines.append(
        "`photo_soft_Chamber`만 경계선 수준 유의(p≈0.03, 챔버3이 평균적으로 낮음) — 다만 4개 "
        "챔버 컬럼을 동시에 검정했으므로 다중비교 보정(Bonferroni 0.0125) 기준으로는 탈락. "
        "추가 조사 후보로만 취급할 것. Chamber ID는 명목형(순서 없음)이므로 모델링 시 "
        "원-핫 인코딩을 권장하며, 순수 수치형(서열)으로 넣으면 신호가 희석될 수 있다."
    )
    lines.append("")

    lines.append("## 5. 알려진 구조적 한계 (계속 유지)")
    lines.append("")
    lines.append(
        "- **die-행은 사실상 웨이퍼 단위 값의 복제**: 그룹(No_Die가 아닌 Lot_Num+Wafer_Num) "
        "내 9개 행 기준으로 Thin F2/F3/F4/Oxid_time은 그룹 내 분산이 정확히 0%, 그 외 공정 "
        "변수들도 그룹 내 표준편차가 전체의 0.1~1% 수준(사실상 상수). 이 CSV의 '15,390행'은 "
        "물리적으로 독립적인 15,390개 관측치가 아니라 1,704개 웨이퍼가 반복 기록된 것이므로, "
        "모델링 시 유효 표본수를 1,704 기준으로 판단해야 한다.\n"
        "- **UV_type은 다수 Lot에 단일값으로 배정**: 32개 Lot 중 13개는 UV_type을 하나만 "
        "사용(Lot18~22=G, Lot24~27,25=H, Lot29~32=I). UV_type 효과를 분석할 때는 Lot 간 "
        "풀링 비교보다, 같은 Lot 안에서 UV_type이 섞인 19개 Lot만으로 Lot 고정효과를 "
        "제거한 비교가 필요하다."
    )
    lines.append("")

    lines.append("## 6. 파일 스키마")
    lines.append("")
    lines.append(f"- 행: {n:,} (No_Die grain, 6개 원본 CSV 1:1 병합)")
    lines.append(f"- 열: {merged.shape[1] - 1}개 (`Wafer_map` 원본 문자열 제외 — 이미 `data/processed/wafer_integrated.parquet`에 보존됨)")
    lines.append("- 신규 도메인 파생 컬럼: `oxidation_rate_nm_per_min`, `cd_resolution_ratio`, "
                  "`exposure_energy_per_cd_nm`, `total_etch_removal`, `etch_rate_stage1/2/3`, "
                  "`total_flux_60_480`, `anneal_temp_diff`, `*_sentinel_flag`(6개), "
                  "`any_process_sentinel_flag`")

    out_md = PROJECT_ROOT / "reports" / "preprocessing_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log(f"저장 완료: {out_md}")


if __name__ == "__main__":
    main()
