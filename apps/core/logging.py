"""
애플리케이션 전역의 로깅 설정을 담당하는 모듈입니다.
"""

from __future__ import annotations

import logging


def configure_logging() -> None:
    """로그 출력 레벨과 포맷을 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

