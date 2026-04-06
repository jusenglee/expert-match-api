"""
애플리케이션 전역의 로깅 설정을 담당하는 모듈입니다.
ContextVar를 사용하여 비동기 요청 간 Trace ID(Request ID)를 공유하고 로그에 출력합니다.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

# 비동기 컨텍스트에서 고유 요청 ID를 저장하기 위한 ContextVar
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# 현재 요청의 로그 메시지들을 캡처하기 위한 ContextVar
captured_logs_ctx: ContextVar[Optional[list[str]]] = ContextVar("captured_logs", default=None)


class RequestIdFilter(logging.Filter):
    """로그 레코드에 현재 컨텍스트의 request_id를 주입하는 필터입니다."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        return True


class ContextLogHandler(logging.Handler):
    """현재 컨텍스트 버퍼(captured_logs_ctx)가 활성화되어 있으면 로그를 해당 버퍼에 담습니다."""

    def emit(self, record: logging.LogRecord) -> None:
        buffer = captured_logs_ctx.get()
        if buffer is not None:
            try:
                msg = self.format(record)
                buffer.append(msg)
            except Exception:
                self.handleError(record)


def configure_logging() -> None:
    """로그 출력 레벨과 포맷을 설정합니다. Trace ID([요청ID])가 포함됩니다."""
    # 공통 포맷터 설정
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [ID:%(request_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Stdout 출력용 핸들러
    stdout_handler = logging.StreamHandler()
    stdout_handler.addFilter(RequestIdFilter())
    stdout_handler.setFormatter(formatter)

    # 2. 컨텍스트 로그 캡처용 핸들러 (UI 노출용)
    capture_handler = ContextLogHandler()
    capture_handler.addFilter(RequestIdFilter())
    capture_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거 후 새 핸들러 등록 (중복 방지)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(capture_handler)

