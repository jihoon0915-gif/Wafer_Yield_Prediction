"""Slack Incoming Webhook 전송 계층.

규칙 판정(alerting.py)과 전송을 분리해 둔 이유: 경보 로직은 웹훅 없이도 동작하고
테스트할 수 있어야 하기 때문이다. 웹훅이 설정되지 않은 환경(공개 데모 등)에서는
전송만 조용히 건너뛰고 경보 이력은 그대로 남는다.

설정: 환경변수 SLACK_WEBHOOK_URL (없으면 비활성)
      SLACK_ALERT_MIN_SEVERITY = critical | warning (기본 warning)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import requests

logger = logging.getLogger(__name__)

WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
MIN_SEVERITY_ENV = "SLACK_ALERT_MIN_SEVERITY"

_SEVERITY_RANK = {"warning": 1, "critical": 2}
_SEVERITY_STYLE = {
    "critical": ("🔴", "심각"),
    "warning": ("🟠", "경고"),
}

TIMEOUT_SECONDS = 5.0


def webhook_url() -> str | None:
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    return url or None


def is_enabled() -> bool:
    return webhook_url() is not None


def min_severity() -> str:
    value = os.environ.get(MIN_SEVERITY_ENV, "warning").strip().lower()
    return value if value in _SEVERITY_RANK else "warning"


def config_status() -> dict[str, Any]:
    """웹훅 URL 자체는 절대 노출하지 않고 설정 여부만 알린다."""
    return {
        "enabled": is_enabled(),
        "min_severity": min_severity(),
        "webhook_env_var": WEBHOOK_ENV,
        "hint": None if is_enabled() else f"{WEBHOOK_ENV} 환경변수를 설정하면 Slack 전송이 활성화됩니다.",
    }


def _blocks(alert) -> list[dict[str, Any]]:
    icon, label = _SEVERITY_STYLE.get(alert.severity, ("⚪", alert.severity))
    metric_lines = "\n".join(f"• `{k}`: {v}" for k, v in alert.metrics.items())
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{icon} [{label}] {alert.title}", "emoji": True},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": alert.detail}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*측정값*\n{metric_lines}"}},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"판정 근거: {alert.evidence}"},
                {"type": "mrkdwn", "text": f"규칙 ID: `{alert.rule}`"},
            ],
        },
        {"type": "divider"},
    ]


def send_alert(alert) -> dict[str, Any]:
    """경보 1건 전송. 예외를 밖으로 던지지 않는다 — 알림 실패가 추론 API를
    죽이면 안 되기 때문."""
    url = webhook_url()
    if url is None:
        return {"sent": False, "reason": "webhook_not_configured", "rule": alert.rule}

    if _SEVERITY_RANK.get(alert.severity, 0) < _SEVERITY_RANK[min_severity()]:
        return {"sent": False, "reason": "below_min_severity", "rule": alert.rule}

    icon, label = _SEVERITY_STYLE.get(alert.severity, ("⚪", alert.severity))
    payload = {
        "text": f"{icon} [{label}] {alert.title} — {alert.detail}",  # 알림 미리보기/폴백용
        "blocks": _blocks(alert),
    }

    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return {"sent": True, "rule": alert.rule}
    except requests.RequestException as exc:
        logger.warning("Slack 전송 실패 (rule=%s): %s", alert.rule, exc)
        return {"sent": False, "reason": "request_failed", "rule": alert.rule, "error": str(exc)}


def send_alerts(alerts: Iterable) -> list[dict[str, Any]]:
    return [send_alert(a) for a in alerts]


def send_test_message() -> dict[str, Any]:
    """웹훅 설정 검증용 — 실제 경보와 구분되는 테스트 메시지를 보낸다."""
    url = webhook_url()
    if url is None:
        return {"sent": False, "reason": "webhook_not_configured"}

    payload = {
        "text": "✅ 웨이퍼 수율 모니터링 — Slack 연동 테스트 메시지",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*✅ 웨이퍼 수율 모니터링 연동 테스트*\n"
                        "이 메시지가 보이면 Incoming Webhook 설정이 정상입니다. "
                        "이후 계측 결측(Gate 0)·식각 리스크존·예측 관리상한 초과 시 이 채널로 경보가 전송됩니다."
                    ),
                },
            }
        ],
    }
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return {"sent": True}
    except requests.RequestException as exc:
        logger.warning("Slack 테스트 전송 실패: %s", exc)
        return {"sent": False, "reason": "request_failed", "error": str(exc)}
