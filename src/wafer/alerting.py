"""이상/불량 탐지 규칙 엔진.

세 가지 경보 규칙 모두 이 프로젝트에서 실제로 검증한 결과에 근거한다.

R1 METROLOGY_MISSING (critical)
    핵심 계측값(Thin F2/F3/F4)이 결측인 채로 들어온 웨이퍼.
    근거: reports/model_interpretation_report.md 3절 — 데이터셋 역대 최다 결함
    웨이퍼(Target=666)가 9개 측정행 전부에서 이 세 값이 NaN이었다. "결함이 심한
    웨이퍼일수록 계측이 누락되는" 구조적 사각지대라, 결측 발생 자체를 경보
    신호로 쓴다(PROCESS_OPTIMIZATION_STRATEGY.md 3절 Gate 0).

R2 ETCH_RISK_ZONE (warning)
    Thin F2/F3/F4가 모두 상위 20% 구간. 근거: scripts/06_simulation_and_alerts.py
    Scenario A — 이 구간 불량률 18.4% vs 하위 20%(스윗스팟) 구간 0.0%.

R3 PREDICTED_TARGET_UCL (2σ warning / 3σ critical)
    예측 결함 다이 수가 웨이퍼 단위 관리상한선을 초과. 근거: SPC 관리도 방식으로
    reports/06_lot_alert_backtest.csv의 Lot 경보규칙과 동일한 μ+kσ 기준을
    웨이퍼 단위로 적용.

임계값은 하드코딩하지 않고 실제 데이터프레임에서 계산한다(AlertThresholds.from_frame).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

CRITICAL = "critical"
WARNING = "warning"

# Gate 0 감시 대상 — SHAP 상위 3개 식각 잔막 두께
METROLOGY_CRITICAL_FEATURES = ("Thin F2", "Thin F3", "Thin F4")


@dataclass(frozen=True)
class Alert:
    rule: str
    severity: str
    title: str
    detail: str
    evidence: str
    metrics: dict[str, Any]
    dedup_key: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "evidence": self.evidence,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AlertThresholds:
    """실제 학습 데이터에서 계산한 경보 임계값."""

    thin_q80: dict[str, float]
    thin_q20: dict[str, float]
    target_mean: float
    target_std: float
    target_ucl_2s: float
    target_ucl_3s: float
    lot_defect_mean: float
    lot_defect_std: float
    lot_defect_ucl_2s: float
    lot_defect_ucl_3s: float

    @classmethod
    def from_frame(cls, df) -> "AlertThresholds":
        target = df["Target"].astype(float)
        t_mu, t_sd = float(target.mean()), float(target.std())

        thin_q80, thin_q20 = {}, {}
        for col in METROLOGY_CRITICAL_FEATURES:
            s = df[col].astype(float)
            thin_q80[col] = float(s.quantile(0.80))
            thin_q20[col] = float(s.quantile(0.20))

        # Lot 단위 불량률 관리도 — 06_lot_alert_backtest.csv와 동일 규칙
        lot_rate = df.groupby("Lot_Num")["error_class"].apply(lambda s: 100.0 * (s != "none").mean())
        l_mu, l_sd = float(lot_rate.mean()), float(lot_rate.std())

        return cls(
            thin_q80=thin_q80,
            thin_q20=thin_q20,
            target_mean=t_mu,
            target_std=t_sd,
            target_ucl_2s=t_mu + 2 * t_sd,
            target_ucl_3s=t_mu + 3 * t_sd,
            lot_defect_mean=l_mu,
            lot_defect_std=l_sd,
            lot_defect_ucl_2s=l_mu + 2 * l_sd,
            lot_defect_ucl_3s=l_mu + 3 * l_sd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrology_watch_features": list(METROLOGY_CRITICAL_FEATURES),
            "etch_risk_zone_q80": {k: round(v, 1) for k, v in self.thin_q80.items()},
            "etch_sweet_spot_q20": {k: round(v, 1) for k, v in self.thin_q20.items()},
            "predicted_target": {
                "mean": round(self.target_mean, 2),
                "std": round(self.target_std, 2),
                "ucl_2sigma": round(self.target_ucl_2s, 2),
                "ucl_3sigma": round(self.target_ucl_3s, 2),
            },
            "lot_defect_rate_pct": {
                "mean": round(self.lot_defect_mean, 2),
                "std": round(self.lot_defect_std, 2),
                "ucl_2sigma": round(self.lot_defect_ucl_2s, 2),
                "ucl_3sigma": round(self.lot_defect_ucl_3s, 2),
            },
        }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def evaluate(
    features: dict[str, Any],
    predicted_target: float,
    thresholds: AlertThresholds,
    *,
    context: str,
    subject: str,
) -> list[Alert]:
    """한 웨이퍼(또는 한 시뮬레이션 설정)에 대해 세 규칙을 모두 평가한다.

    context/subject는 Slack 메시지와 대시보드 알림 이력에서 "어디서 무엇 때문에"를
    구분하기 위한 값(예: context="simulate", subject="슬라이더 설정").
    """
    alerts: list[Alert] = []

    missing = [c for c in METROLOGY_CRITICAL_FEATURES if _is_missing(features.get(c))]
    if missing:
        alerts.append(
            Alert(
                rule="METROLOGY_MISSING",
                severity=CRITICAL,
                title="핵심 계측값 결측 (Gate 0)",
                detail=(
                    f"{subject} — {', '.join(missing)} 값이 비어 있습니다. "
                    "계측 누락 자체가 고결함 웨이퍼의 선행 신호이므로 예측값을 신뢰하지 말고 "
                    "해당 설비의 계측 로그를 먼저 확인해야 합니다."
                ),
                evidence="역대 최다 결함 웨이퍼(Target=666)가 9개 측정행 전부에서 이 값들이 결측이었음",
                metrics={"missing_features": missing, "predicted_target": round(predicted_target, 1)},
                dedup_key=f"METROLOGY_MISSING|{context}|{','.join(missing)}",
            )
        )

    in_risk_zone = [
        c
        for c in METROLOGY_CRITICAL_FEATURES
        if not _is_missing(features.get(c)) and float(features[c]) >= thresholds.thin_q80[c]
    ]
    if len(in_risk_zone) == len(METROLOGY_CRITICAL_FEATURES):
        alerts.append(
            Alert(
                rule="ETCH_RISK_ZONE",
                severity=WARNING,
                title="식각 잔막 리스크존 진입",
                detail=(
                    f"{subject} — Thin F2/F3/F4가 모두 상위 20% 구간입니다 "
                    f"(기준 {', '.join(f'{c}≥{thresholds.thin_q80[c]:.0f}nm' for c in METROLOGY_CRITICAL_FEATURES)}). "
                    "식각 잔막을 하위 20% 구간으로 되돌리는 레시피 보정을 검토하세요."
                ),
                evidence="이 구간 실측 불량률 18.4% vs 하위 20%(스윗스팟) 구간 0.0%",
                metrics={
                    "values": {c: round(float(features[c]), 1) for c in in_risk_zone},
                    "q80_thresholds": {c: round(thresholds.thin_q80[c], 1) for c in in_risk_zone},
                    "predicted_target": round(predicted_target, 1),
                },
                dedup_key=f"ETCH_RISK_ZONE|{context}",
            )
        )

    if predicted_target > thresholds.target_ucl_3s:
        sigma_level, severity = 3, CRITICAL
        ucl = thresholds.target_ucl_3s
    elif predicted_target > thresholds.target_ucl_2s:
        sigma_level, severity = 2, WARNING
        ucl = thresholds.target_ucl_2s
    else:
        sigma_level, severity, ucl = 0, "", 0.0

    if sigma_level:
        alerts.append(
            Alert(
                rule="PREDICTED_TARGET_UCL",
                severity=severity,
                title=f"예측 결함수 관리상한 초과 ({sigma_level}σ)",
                detail=(
                    f"{subject} — 예측 결함 다이 수 {predicted_target:.0f}개가 "
                    f"관리상한선 μ+{sigma_level}σ={ucl:.1f}개를 초과했습니다 "
                    f"(전체 평균 {thresholds.target_mean:.1f}개)."
                ),
                evidence=f"웨이퍼 단위 SPC 관리도 기준(μ={thresholds.target_mean:.1f}, σ={thresholds.target_std:.1f})",
                metrics={
                    "predicted_target": round(predicted_target, 1),
                    "ucl": round(ucl, 1),
                    "sigma_level": sigma_level,
                },
                dedup_key=f"PREDICTED_TARGET_UCL|{context}|{sigma_level}",
            )
        )

    return alerts


class AlertLog:
    """최근 경보 이력 버퍼 + 중복 억제.

    /simulate는 슬라이더를 움직일 때마다 호출되므로, 같은 경보가 초당 여러 번
    발생한다. cooldown_seconds 안에 같은 dedup_key가 다시 오면 억제한다 —
    Slack 도배 방지와 이력 가독성 둘 다를 위한 것.
    """

    def __init__(self, maxlen: int = 100, cooldown_seconds: float = 300.0) -> None:
        self._entries: deque[Alert] = deque(maxlen=maxlen)
        self._last_seen: dict[str, float] = {}
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()

    def record(self, alerts: Iterable[Alert]) -> list[Alert]:
        """쿨다운을 통과한 경보만 이력에 남기고 반환한다(= Slack으로 보낼 것들)."""
        now = time.time()
        fresh: list[Alert] = []
        with self._lock:
            for alert in alerts:
                last = self._last_seen.get(alert.dedup_key)
                if last is not None and now - last < self._cooldown:
                    continue
                self._last_seen[alert.dedup_key] = now
                self._entries.appendleft(alert)
                fresh.append(alert)
        return fresh

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [a.to_dict() for a in list(self._entries)[:limit]]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._last_seen.clear()
