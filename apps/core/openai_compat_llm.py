"""
OpenAI 호환 API(vLLM, Solar 등)를 LangChain 프레임워크와 연결하는 어댑터 모듈입니다.
스트리밍 응답 처리, 사고 과정(Reasoning) 추출, 트레이싱(Tracing) 메타데이터 관리 등을 담당합니다.
"""

import inspect
import logging
import time
from typing import Any, AsyncIterator, List, Optional

try:
    import langchain
except ImportError:
    langchain = None

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from openai import AsyncOpenAI
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

# 이 래퍼는 vLLM/OpenAI-compat 응답을 LangChain 메시지로 정규화한다.
# 특히 reasoning delta를 content와 분리해 downstream 스트리밍 계층이
# 안전하게 필터링할 수 있도록 `stream_field` 규약을 붙여주는 것이 중요하다.

STREAM_FIELD_KEY = "stream_field"  # "content" | "reasoning"
STREAM_REASONING_KEY = "reasoning_text"  # additional_kwargs에만 저장


def _ensure_langchain_global_compat() -> None:
    """Backfill globals expected by langchain-core 0.3 when langchain 1.x is installed."""
    if langchain is None:
        return
    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "llm_cache"):
        langchain.llm_cache = None


_ensure_langchain_global_compat()


class OpenAICompatStreamError(RuntimeError):
    """
    OpenAI 호환 스트림 응답의 구조가 깨지거나 필드가 비었을 때 발생하는 예외입니다.
    일반적인 타임아웃과 구분하여 Provider 응답 자체의 무효성을 나타냅니다.
    """


class EmptyStreamContentError(OpenAICompatStreamError):
    """
    LLM의 스트리밍 응답이 정상적으로 종료되었으나, 사용자에게 반환할 최종 텍스트(Content)가 
    전혀 생성되지 않았을 때 발생하는 예외입니다. 
    이는 모델의 컨텍스트 한계 초과나 비정상 응답을 탐지하고, 상위 계층에서 폴백(Fallback) 로직을 트리거하는 데 핵심적으로 사용됩니다.
    """


class OpenAICompatChatModel(BaseChatModel):
    """
    LangChain의 BaseChatModel 인터페이스를 준수하면서, vLLM이나 Solar 같은 
    OpenAI 호환 API 서버와 효율적으로 통신할 수 있도록 자체 구현된 커스텀 LLM 어댑터 클래스입니다.

    [주요 기능]
    - 비동기 호출 지원 (스트리밍 및 비스트리밍 방식 모두 최적화)
    - 최신 추론형 LLM의 사고 과정(Reasoning Content, 예: thinking 단계)과 최종 답변을 분리하여 처리
    - 추천 파이프라인 전반의 트러블슈팅을 위한 추적 ID(request_id) 주입 및 메타데이터 로깅
    """

    # 기본 연결 설정
    model_name: str = "/model"
    base_url: str = "http://vllm_solar:8010/v1"
    api_key: str = "EMPTY"
    timeout: float = 120.0

    # 기본 샘플링 설정
    default_temperature: float = 0.2
    default_top_p: float = 0.8

    # Solar/vLLM reasoning 제어(기본: thinking 끄기 + reasoning 출력 포함 안 함)
    # - reasoning_effort: "low" | "medium" | "high" | None
    # - include_reasoning: True | False | None  (vLLM 프로토콜에 존재)
    default_reasoning_effort: Optional[str] = "low"
    default_include_reasoning: Optional[bool] = False

    # stream에서 reasoning delta가 먼저 와도 최종 답변(content)에 섞이지 않게:
    # - True  : reasoning을 "이벤트"로는 내보내되(content=""), reasoning 텍스트는 additional_kwargs에만 담음(추천)
    # - False : reasoning 이벤트 자체를 내보내지 않음(클라이언트는 content 나올 때까지 무응답처럼 보일 수 있음)
    emit_reasoning_events: bool = True

    _client: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    # ---- LangChain required ----
    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """
        LangChain의 동기 generate 계약을 non-stream invoke 경로로 연결한다.
        내부적으로는 `ainvoke_non_stream`을 호출하고, 반환된 text와 trace 정보를 `ChatResult` 형태로 재포장한다.
        """
        raise NotImplementedError("Use ainvoke/astream")

    @property
    def _llm_type(self) -> str:
        """LangChain이 이 모델을 구분할 때 쓸 안정적인 type name을 돌려준다.
        provider 차이와 무관하게 같은 adapter로 인식되게 해 tracing과 serialization을 단순화한다.
        """
        return "openai_compat_chat"

    # ---- internal helpers ----
    def _get_client(self) -> AsyncOpenAI:
        """provider base URL과 API key로 OpenAI 호환 async client를 지연 생성한다.
        같은 모델 인스턴스에서는 한 번 만든 client를 재사용하여 연결 오버헤드를 줄인다.
        """
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def aclose(self) -> None:
        """내부 async client와 매달려 있는 transport 자원을 정리한다.
        서비스 종료나 테스트 마무리 시점에 connection leak을 남기지 않게 하는 종결 후처리다.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _delta_get(delta: Any, key: str) -> Any:
        """stream chunk에서 provider별 delta field를 안전하게 가져온다.
        OpenAI 호환 API라고 해도 chunk shape가 미묘하게 다를 수 있어 이 헬퍼가 필드 접근 차이를 흡수한다.
        """
        if delta is None:
            return None
        if isinstance(delta, dict):
            return delta.get(key)
        return getattr(delta, key, None)

    @staticmethod
    def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
        """Normalize provider tool call objects into plain dicts."""
        normalized_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls or []:
            if isinstance(tool_call, dict):
                function = tool_call.get("function") or {}
                normalized_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type"),
                        "function": {
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                        },
                    }
                )
                continue

            function = getattr(tool_call, "function", None)
            normalized_calls.append(
                {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", None),
                    "function": {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    },
                }
            )
        return normalized_calls

    @staticmethod
    def _resolve_trace_ids(
        kwargs: dict[str, Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """response header와 body에서 추적 가능한 trace/request id를 수집한다.
        빈 응답이나 provider 오류를 triage할 때 같은 호출을 다시 찾을 수 있게 하는 진단 메타데이터다.
        """
        request_id = kwargs.get("request_id")
        conversation_id = kwargs.get("conversation_id")

        config = kwargs.get("config")
        metadata = kwargs.get("metadata")
        if metadata is None and isinstance(config, dict):
            metadata = config.get("metadata")

        if isinstance(metadata, dict):
            request_id = request_id or metadata.get("request_id")
            conversation_id = conversation_id or metadata.get("conversation_id")

        request_id_str = str(request_id).strip() if request_id is not None else ""
        conversation_id_str = (
            str(conversation_id).strip() if conversation_id is not None else ""
        )
        return (request_id_str or None, conversation_id_str or None)

    def _to_openai_messages(self, messages: List[BaseMessage]) -> List[dict[str, str]]:
        """LangChain message 목록을 OpenAI chat completion 메시지 형식으로 변환한다.
        system/human/assistant/tool role을 보존하고, provider가 못 읽는 부가 필드는 제거하여 요청 body를 간결하게 유지한다.
        """
        converted: List[dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                # AIMessage 포함 (그 외 BaseMessage도 안전하게 assistant로 처리)
                role = "assistant"
            converted.append({"role": role, "content": str(m.content)})
        return converted

    def _build_openai_request_kwargs(
        self,
        *,
        request_id: Optional[str],
        stop: Optional[List[str]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """model, messages, timeout, streaming 플래그를 provider 요청 kwargs로 조합한다.
        공통 body와 provider-specific extra body를 분리해 reasoning 옵션이 일반 chat payload를 오염시키지 않게 한다.
        """
        max_tokens_hint = kwargs.get("max_tokens_hint", kwargs.get("max_tokens"))
        request_kwargs: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.default_temperature),
            "top_p": kwargs.get("top_p", self.default_top_p),
        }
        seed = kwargs.get("seed")
        if seed is not None:
            request_kwargs["seed"] = int(seed)
        if max_tokens_hint is not None:
            request_kwargs["max_tokens"] = max_tokens_hint
            request_kwargs["max_completion_tokens"] = max_tokens_hint
        if stop:
            request_kwargs["stop"] = stop
        for key in ("tools", "tool_choice", "response_format", "parallel_tool_calls"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]
        if request_id:
            request_kwargs["extra_headers"] = {"x-request-id": request_id}
        return request_kwargs

    def _build_extra_body(self, kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
        """provider별로 허용되는 추가 request body를 선별해 만든다.
        reasoning effort, parallel tool calls, response format 같은 선택 옵션을 호환 범위 내에서만 싣어 보낸다.
        """
        extra_body = kwargs.get("extra_body")
        body: dict[str, Any] = dict(extra_body) if isinstance(extra_body, dict) else {}

        top_k = kwargs.get("top_k")
        if top_k is not None:
            body.setdefault("top_k", int(top_k))

        # --- reasoning controls ---
        reasoning_effort = kwargs.get("reasoning_effort", self.default_reasoning_effort)
        if reasoning_effort is not None:
            body.setdefault("reasoning_effort", reasoning_effort)

        include_reasoning = kwargs.get(
            "include_reasoning", self.default_include_reasoning
        )
        if include_reasoning is not None:
            body.setdefault("include_reasoning", include_reasoning)

        # --- chat template hints (vLLM examples) ---
        # 일부 서버/모델에서 키가 다를 수 있어 thinking + enable_thinking 둘 다 넣어 호환성 확보
        chat_template_kwargs = body.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}

        disable_thinking = kwargs.get("disable_thinking", True)
        if disable_thinking:
            chat_template_kwargs.setdefault("thinking", False)
            chat_template_kwargs.setdefault("enable_thinking", False)

        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs

        return body or None

    async def _close_stream(self, stream: Any, *, request_id: Optional[str]) -> None:
        """provider stream object가 노출한 aclose/close 후크를 안전하게 호출한다.
        예외 도중에도 stream transport가 남지 않게 정리를 보장하는 후처리 헬퍼다.
        """
        stream_type = type(stream).__name__
        has_aclose = callable(getattr(stream, "aclose", None))
        has_close = callable(getattr(stream, "close", None))
        try:
            if has_aclose:
                await stream.aclose()
            elif has_close:
                res = stream.close()
                if inspect.isawaitable(res):
                    await res
        except Exception as exc:
            logger.warning(
                "[openai_compat_llm] stream close failed: request_id=%s stream_type=%s has_aclose=%s has_close=%s",
                request_id,
                stream_type,
                has_aclose,
                has_close,
                exc_info=exc,
            )
            raise

    # ---- non-stream ----
    async def _agenerate(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> ChatResult:
        """LangChain의 비동기 generate 계약을 non-stream invoke 경로로 연결한다.
        `_generate`와 같은 결과 shape를 맞추되, event loop 안에서 직접 await 가능하게 제공한다.
        """
        client = self._get_client()
        request_id, conversation_id = self._resolve_trace_ids(kwargs)

        request_kwargs = self._build_openai_request_kwargs(
            request_id=request_id,
            stop=kwargs.get("stop"),
            kwargs=kwargs,
        )
        extra_body = self._build_extra_body(kwargs)

        t0 = time.monotonic()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=self._to_openai_messages(messages),
            stream=False,
            extra_body=extra_body,
            **request_kwargs,
        )

        msg = response.choices[0].message if response.choices else None
        content = (getattr(msg, "content", None) if msg is not None else None) or ""
        tool_calls = self._serialize_tool_calls(
            getattr(msg, "tool_calls", None) if msg is not None else None
        )
        # reasoning은 섞지 않고 참고용으로만 (필요시 로그/메트릭으로 사용)
        reasoning = (
            (getattr(msg, "reasoning", None) if msg is not None else None)
            or (getattr(msg, "reasoning_content", None) if msg is not None else None)
            or ""
        )

        dt_ms = (time.monotonic() - t0) * 1000
        usage = (
            response.usage.model_dump() if getattr(response, "usage", None) else None
        )

        logger.info(
            "[openai_compat_llm] non-stream summary: request_id=%s conversation_id=%s model=%s dt_ms=%.1f message_n=%d content_char_n=%d reasoning_char_n=%d tool_call_n=%d request_max_tokens=%s request_top_k=%s usage=%s base_url=%s",
            request_id,
            conversation_id,
            self.model_name,
            dt_ms,
            len(messages),
            len(content),
            len(reasoning),
            len(tool_calls),
            request_kwargs.get("max_tokens"),
            extra_body.get("top_k") if isinstance(extra_body, dict) else None,
            usage,
            self.base_url,
        )
        additional_kwargs: dict[str, Any] = {}
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
        if reasoning:
            additional_kwargs[STREAM_REASONING_KEY] = reasoning
        if usage is not None:
            additional_kwargs["usage"] = usage
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            ]
        )

    async def ainvoke_non_stream(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> AIMessage:
        """streaming을 사용하지 않는 일반 chat completion 호출을 실행한다.
        provider 응답에서 answer text, reasoning text, usage, trace id를 분리해 상위 레이어가 그대로 소비할 수 있는 딕셔너리로 돌려준다.
        """
        result = await self._agenerate(messages, **kwargs)
        if result.generations:
            return result.generations[0].message
        return AIMessage(content="")

    # ---- stream ----
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """OpenAI 호환 streaming 응답을 LangChain `ChatGenerationChunk` 흐름으로 바꾼다.
        reasoning content와 user-visible content를 구분해 누적하고, 빈 콘텐츠 종료는 `EmptyStreamContentError`로 변환해 상위에 알린다.
        """
        client = self._get_client()
        request_id, conversation_id = self._resolve_trace_ids(kwargs)

        request_kwargs = self._build_openai_request_kwargs(
            request_id=request_id,
            stop=stop,
            kwargs=kwargs,
        )
        extra_body = self._build_extra_body(kwargs)

        t0 = time.monotonic()
        stream = await client.chat.completions.create(
            model=self.model_name,
            messages=self._to_openai_messages(messages),
            stream=True,
            extra_body=extra_body,
            **request_kwargs,
        )

        # metrics
        chunk_n = 0
        emitted_any_chunk_n = 0
        emitted_content_chunk_n = 0
        emitted_reasoning_event_n = 0

        content_char_n = 0
        reasoning_char_n = 0

        ttft_any_ms: Optional[float] = None
        ttft_content_ms: Optional[float] = None
        last_finish_reason = None

        closed = False
        primary_exc: Optional[BaseException] = None

        try:
            async for chunk in stream:
                chunk_n += 1
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                choice0 = choices[0]
                last_finish_reason = (
                    getattr(choice0, "finish_reason", None) or last_finish_reason
                )
                delta = getattr(choice0, "delta", None)

                content = self._delta_get(delta, "content")
                reasoning = self._delta_get(delta, "reasoning") or self._delta_get(
                    delta, "reasoning_content"
                )

                # ttft_any: content/reasoning 중 뭐든 처음 도착한 시점
                if ttft_any_ms is None and (content or reasoning):
                    ttft_any_ms = (time.monotonic() - t0) * 1000

                # reasoning-only 이벤트: 최종 content에 섞지 않도록 content=""로 내보내고,
                # reasoning 텍스트는 additional_kwargs에만 실어 보냄.
                if reasoning and not content:
                    reasoning_char_n += len(reasoning)
                    emitted_any_chunk_n += 1
                    if self.emit_reasoning_events:
                        emitted_reasoning_event_n += 1
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",  # 중요: 최종 답변 문자열에 섞이지 않게
                                additional_kwargs={
                                    STREAM_FIELD_KEY: "reasoning",
                                    STREAM_REASONING_KEY: reasoning,
                                },
                            )
                        )
                    continue

                # content 이벤트
                if content:
                    if ttft_content_ms is None:
                        ttft_content_ms = (time.monotonic() - t0) * 1000
                    emitted_any_chunk_n += 1
                    emitted_content_chunk_n += 1
                    content_char_n += len(content)

                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=content,
                            additional_kwargs={STREAM_FIELD_KEY: "content"},
                        )
                    )

            if emitted_content_chunk_n == 0:
                # 계약 위반으로 간주: 스트림은 끝났지만 사용자에게 전달 가능한 content 토큰이 0개.
                # 여기서는 재시도/폴백을 수행하지 않고 EmptyStreamContentError를 상위로 전파한다.
                # 실제 재시도 횟수/폴백 방식(예: non-stream 호출)은 상위 오케스트레이션 레이어에서 결정한다.
                raise EmptyStreamContentError(
                    "No content emitted in stream. "
                    f"model={self.model_name}, base_url={self.base_url}, request_kwargs={request_kwargs}, extra_body={extra_body}"
                )

        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            try:
                await self._close_stream(stream, request_id=request_id)
                closed = True
            except Exception:
                # close 실패는 _close_stream에서 로깅/raise 처리
                if primary_exc is None:
                    raise

            dt_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "[openai_compat_llm] stream summary: request_id=%s conversation_id=%s dt_ms=%.1f ttft_any_ms=%s ttft_content_ms=%s "
                "chunk_n=%d emitted_any_chunk_n=%d emitted_content_chunk_n=%d emitted_reasoning_event_n=%d "
                "content_char_n=%d reasoning_char_n=%d request_max_tokens=%s request_top_k=%s finish_reason=%s closed=%s model=%s base_url=%s",
                request_id,
                conversation_id,
                dt_ms,
                f"{ttft_any_ms:.1f}" if ttft_any_ms is not None else "none",
                f"{ttft_content_ms:.1f}" if ttft_content_ms is not None else "none",
                chunk_n,
                emitted_any_chunk_n,
                emitted_content_chunk_n,
                emitted_reasoning_event_n,
                content_char_n,
                reasoning_char_n,
                request_kwargs.get("max_tokens"),
                extra_body.get("top_k") if isinstance(extra_body, dict) else None,
                last_finish_reason,
                closed,
                self.model_name,
                self.base_url,
            )
