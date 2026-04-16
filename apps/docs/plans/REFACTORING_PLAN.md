# 리팩토링 계획안

**작성일:** 2026-04-09  
**범위:** `apps/` 전체 소스코드 (테스트 제외)  
**제약:** 외부 의존성 서버(LLM, Qdrant, Embedding)는 특정 VPN 환경에서만 호출 가능

---

## 우선순위 산정 기준

```
우선순위 점수 = (영향도 + 리스크) × (6 - 작업량)
```

| 항목 | 영향도(1-5) | 리스크(1-5) | 작업량(1-5) | 점수 |
|------|-----------|-----------|-----------|------|
| final_recommendation_max 미적용 | 4 | 5 | 1 | **45** |
| `_extract_json_object_text` 중복 제거 | 4 | 4 | 1 | **40** |
| VPN 의존 연결 실패 처리 명확화 | 4 | 5 | 2 | **36** |
| service.py silent exception 제거 | 3 | 4 | 2 | **28** |
| 매직 넘버 → constants 모듈 | 3 | 3 | 2 | **24** |
| `_merge_unique_strings` 중복 제거 | 2 | 2 | 1 | **20** |
| 미사용 config 필드 정리 | 2 | 2 | 1 | **20** |
| exception 처리 표준화 | 3 | 3 | 3 | **18** |
| `_judge_single_shortlist` 분리 | 4 | 3 | 3 | **21** |
| `OpenAICompatPlanner.plan` 분리 | 3 | 2 | 3 | **15** |
| `_normalize_judge_payload` 분리 | 3 | 2 | 3 | **15** |

---

## 단계별 실행 계획

### Phase 1 — 즉시 수정 (1~2일, 리스크 최소)

코드 로직 변경 없이 구조만 정리하는 항목들이다. 외부 서버 호출이 없어 VPN 환경과 무관하게 진행 가능.

---

#### 1-A. `final_recommendation_max` 설정 적용 (점수: 45)

**문제:**  
`Settings.final_recommendation_max = 20` 설정이 존재하지만 `service.py`에서 전혀 적용되지 않는다. judge가 40명 숏리스트를 그대로 반환하면 API 응답에 40개가 노출될 수 있다.

**현재 코드 (`service.py` 라인 83~178):**  
`recommend()` 메서드에서 judge 출력을 그대로 response에 담는다. `recommendations` 리스트에 상한이 없다.

**수정 방법:**
```python
# service.py — _build_recommendation_response() 내
recommendations = judge_output.recommended
# 추가
if len(recommendations) > self.settings.final_recommendation_max:
    recommendations = recommendations[:self.settings.final_recommendation_max]
    logger.warning(
        "추천 결과가 최대 허용치를 초과하여 절단: %d → %d",
        len(judge_output.recommended),
        self.settings.final_recommendation_max,
    )
```

**파일:** `apps/recommendation/service.py`  
**VPN 영향:** 없음

---

#### 1-B. `_extract_json_object_text` 중복 제거 (점수: 40)

**문제:**  
`judge.py`(라인 25~93)와 `planner.py`(라인 24~93)에 72줄짜리 완전히 동일한 함수가 2개 존재한다. 향후 한쪽만 수정하면 동작이 달라지는 버그가 발생할 수 있다.

**수정 방법:**
1. `apps/core/json_utils.py` 신규 생성
2. `_extract_json_object_text` 함수를 그대로 이동
3. `judge.py`와 `planner.py`에서 import로 교체

```python
# apps/core/json_utils.py (신규)
from __future__ import annotations
import re
from typing import Any

def extract_json_object_text(content: Any) -> str:
    """LLM 응답에서 첫 번째 JSON 객체를 추출합니다.
    <thinking> 블록 제거 → 마크다운 코드 블록 제거 → 균형 중괄호 매칭 3단계로 처리.
    """
    # ... (현재 코드 그대로 이동)
```

```python
# judge.py 상단
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text

# planner.py 상단
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
```

**파일:** `apps/core/json_utils.py` (신규), `apps/recommendation/judge.py`, `apps/recommendation/planner.py`  
**VPN 영향:** 없음

---

#### 1-C. `_merge_unique_strings` 중복 제거 (점수: 20)

**문제:**  
`judge.py`(라인 219~229)와 `service.py`(라인 261~272)에 동일한 유틸리티 함수가 존재한다.

**수정 방법:**  
`apps/core/json_utils.py` 또는 `apps/core/utils.py`로 이동 후 양쪽에서 import.

```python
# apps/core/utils.py (신규 또는 json_utils.py에 추가)
def merge_unique_strings(base: list[str], additions: list[str]) -> list[str]:
    """두 문자열 리스트를 중복 없이 병합합니다."""
    seen = set(base)
    result = list(base)
    for item in additions:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
```

**파일:** `apps/core/utils.py` (신규), `apps/recommendation/judge.py`, `apps/recommendation/service.py`  
**VPN 영향:** 없음

---

#### 1-D. 미사용 config 필드 정리 (점수: 20)

**문제:**  
다음 설정들이 `Settings`에 정의되어 있으나 코드에서 실제로 사용되지 않는다.

| 필드 | 위치 | 상태 |
|------|------|------|
| `api_prefix` | config.py:46 | 정의만 됨, 라우터에 적용 안 됨 |
| `final_recommendation_min` | config.py:86 | 정의만 됨, 검증 없음 |
| `app_env` | config.py:42 | 읽히지 않음 |

**수정 방법:**
- `api_prefix`: `main.py`의 `APIRouter(prefix=settings.api_prefix)`에 실제로 연결하거나 제거
- `final_recommendation_min`: Phase 1-A에서 `max`를 적용할 때 `min` 검증도 함께 추가
- `app_env`: 로깅 설정이나 startup 분기에 활용하거나 제거

**파일:** `apps/core/config.py`, `apps/api/main.py`  
**VPN 영향:** 없음

---

### Phase 2 — 안정화 수정 (3~5일, 로직 변경 포함)

외부 호출 로직과 연결된 항목들이다. VPN 환경에서 연결 가능한 상태여야 통합 검증이 가능.

---

#### 2-A. VPN 의존 연결 실패 처리 명확화 (점수: 36)

**문제:**  
현재 LLM / Qdrant / Embedding 서버에 접근 불가 시 발생하는 에러가 일반적인 "연결 실패"로 뭉뚱그려진다. VPN 환경에서만 접근 가능하다는 맥락이 에러 메시지에 없어, 운영 중 장애 진단이 어렵다.

`runtime_validation.py`의 체크는 스타트업 시에만 실행된다. 런타임 도중 VPN 연결이 끊기면 일반 500 에러로만 노출된다.

**수정 방법:**

1. 연결 설정에 명시적 timeout 추가:
```python
# openai_compat_llm.py — AsyncOpenAI 생성 시
self._client = AsyncOpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
)
```

2. VPN 맥락이 담긴 연결 오류 클래스 추가:
```python
# apps/core/exceptions.py (신규)
class ExternalDependencyError(RuntimeError):
    """VPN 전용 외부 서버(LLM/Qdrant/Embedding) 연결 실패.
    이 에러는 VPN에 접속된 환경에서만 외부 서버에 도달할 수 있음을 전제합니다.
    """
    def __init__(self, service: str, url: str, cause: Exception):
        super().__init__(
            f"[VPN 의존 서비스 연결 실패] {service} @ {url} — "
            f"VPN 접속 상태를 확인하세요. 원인: {cause}"
        )
        self.service = service
        self.url = url
        self.cause = cause
```

3. `main.py`의 500 핸들러에서 `ExternalDependencyError` 분기 추가:
```python
@app.exception_handler(ExternalDependencyError)
async def external_dep_handler(request, exc):
    logger.error("외부 서버 접근 불가 (VPN 확인 필요): %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})
```

**파일:** `apps/core/exceptions.py` (신규), `apps/core/openai_compat_llm.py`, `apps/api/main.py`  
**VPN 영향:** VPN 연결 상태에서 통합 테스트 필요

---

#### 2-B. service.py silent exception 제거 (점수: 28)

**문제:**  
`service.py`의 `_serialize_query_payload()` 메서드(라인 235~259)에 `except Exception: pass` 구문이 있어 직렬화 실패가 완전히 무시된다.

```python
# 현재 코드 (문제)
try:
    ...
except Exception:
    pass  # ← 에러를 삼키고 빈 dict 반환
```

이 메서드는 trace 데이터 생성용이므로 실패해도 추천 자체는 영향을 받지 않는다. 그러나 버그를 숨긴다.

**수정 방법:**
```python
except Exception as exc:
    logger.warning(
        "trace 직렬화 중 오류 (무시됨): %s",
        exc,
        exc_info=True,
    )
    return {}
```

**파일:** `apps/recommendation/service.py`  
**VPN 영향:** 없음

---

#### 2-C. 매직 넘버 → 상수 모듈 (점수: 24)

**문제:**  
비즈니스 의미가 있는 숫자들이 코드 곳곳에 하드코딩되어 있어, 값 변경 시 어디를 찾아야 하는지 알기 어렵다.

**발견된 주요 매직 넘버:**

| 값 | 파일:라인 | 의미 |
|----|---------|------|
| `80` | judge.py:305 | "높음" 적합도 기준 점수 |
| `50` | judge.py:307 | "중간" 적합도 기준 점수 |
| `150` | judge.py:420,428 | LLM 입력 초록/요약 절단 길이 |
| `5` | judge.py:422,424 | 키워드 최대 포함 수 |
| `3` | judge.py:317 | 추천 이유 최대 수 |
| `400` | judge.py:600 | Map 라운드 max_tokens |
| `2.5` | judge.py:487 | 토큰 추정 나눗수 |

**수정 방법:**
```python
# apps/core/constants.py (신규)

# Judge 적합도 기준
FIT_SCORE_HIGH = 80        # "높음" 이상 기준
FIT_SCORE_MEDIUM = 50      # "중간" 이상 기준

# LLM 입력 직렬화 제한
ABSTRACT_TRUNCATION_LENGTH = 150    # 초록/요약 절단 길이 (chars)
KEYWORD_INCLUDE_LIMIT = 5           # 키워드 최대 포함 수
MAX_REASONS_PER_CANDIDATE = 3       # 추천 이유 최대 수

# Map 라운드 토큰 제어
MAP_PHASE_MAX_TOKENS = 400          # Map 라운드 응답 토큰 상한
TOKEN_ESTIMATION_DIVISOR = 2.5      # chars → tokens 변환 추정치
```

**파일:** `apps/core/constants.py` (신규), `apps/recommendation/judge.py`  
**VPN 영향:** 없음

---

### Phase 3 — 구조 개선 (1~2주, 함수 분리)

큰 함수들을 분리하고 단일 책임 원칙을 강화하는 단계. 로직 버그가 생기지 않도록 신중하게 진행.

---

#### 3-A. `_judge_single_shortlist` 분리 (점수: 21, 136줄)

**현재 구조 (judge.py 583~718):**  
1개 함수가 3가지 책임을 담당하고 있다:
- 입력 직렬화 선택 (Map vs Reduce)
- LLM 호출 및 응답 수신
- 응답 파싱 및 오류 복구

**분리 후 구조:**
```python
async def _judge_single_shortlist(self, ...) -> JudgeOutput:
    """진입점 — 3단계를 순서대로 호출."""
    messages, max_tokens = self._prepare_judge_messages(query, plan, shortlist, context_label)
    raw_payload = await self._invoke_and_parse_json(messages, context_label, ...)
    return self._interpret_judge_payload(raw_payload, shortlist, context_label)

def _prepare_judge_messages(self, ...) -> tuple[list, int | None]:
    """직렬화 방식 선택 + 프롬프트 조립."""
    ...

async def _invoke_and_parse_json(self, messages, ...) -> dict:
    """LLM 호출 + 3단계 JSON 추출."""
    ...

def _interpret_judge_payload(self, raw_payload, shortlist, context_label) -> JudgeOutput:
    """Map/Reduce별 응답 해석 + 정규화 + 폴백."""
    ...
```

**파일:** `apps/recommendation/judge.py`  
**VPN 영향:** VPN 환경에서 map-reduce 통합 검증 필요

---

#### 3-B. `OpenAICompatPlanner.plan` 분리 (점수: 15, 106줄)

**현재 구조 (planner.py 218~323):**  
1개 메서드가 프롬프트 조립 + LLM 호출 + 파싱 + 폴백을 모두 처리.

**분리 후 구조:**
```python
async def plan(self, *, query, filters_override, exclude_orgs, top_k) -> PlannerOutput:
    """진입점 — 3단계를 순서대로 호출."""
    messages = self._build_plan_messages(query, filters_override, exclude_orgs, top_k)
    try:
        raw = await self._invoke_planner(messages)
        return self._parse_planner_output(raw)
    except Exception as exc:
        logger.warning("[Planner] LLM 호출 실패, 휴리스틱으로 폴백: %s", exc)
        return await self._heuristic_fallback.plan(...)

def _build_plan_messages(self, ...) -> list:
    """시스템 프롬프트 + 유저 메시지 조립."""
    ...

async def _invoke_planner(self, messages) -> str:
    """LLM 호출 + JSON 텍스트 반환."""
    ...

def _parse_planner_output(self, json_text: str) -> PlannerOutput:
    """JSON 파싱 + Pydantic 검증."""
    ...
```

**파일:** `apps/recommendation/planner.py`  
**VPN 영향:** VPN 환경에서 LLM 플래너 통합 검증 필요

---

#### 3-C. `_normalize_judge_payload` 분리 (점수: 15, 89줄)

**현재 구조 (judge.py 128~216):**  
단일 함수가 5가지 정규화를 순차적으로 처리.

**분리 후 구조:**
```python
def _normalize_judge_payload(raw: dict, shortlist: list[CandidateCard]) -> dict:
    """정규화 파이프라인 — 순서 보장."""
    raw = _normalize_recommended_list(raw)
    raw = _normalize_recommendation_fields(raw)
    raw = _patch_missing_expert_ids(raw, shortlist)
    raw = _normalize_string_array_fields(raw)
    return raw
```

각 `_normalize_*` 함수는 독립적으로 책임이 명확해진다.

**파일:** `apps/recommendation/judge.py`  
**VPN 영향:** 없음

---

## 신규 파일 목록

리팩토링 완료 후 추가될 파일:

| 파일 | 역할 |
|------|------|
| `apps/core/json_utils.py` | `extract_json_object_text` 공통 함수 |
| `apps/core/utils.py` | `merge_unique_strings` 등 범용 유틸리티 |
| `apps/core/constants.py` | 비즈니스 의미 상수 모음 |
| `apps/core/exceptions.py` | VPN 의존 연결 에러 등 커스텀 예외 |

---

## 수정 대상 파일 전체 목록

| 파일 | Phase | 변경 내용 |
|------|-------|---------|
| `apps/core/config.py` | 1-D | 미사용 필드 제거 / 연결 |
| `apps/core/openai_compat_llm.py` | 2-A | timeout 명시, ExternalDependencyError 래핑 |
| `apps/api/main.py` | 1-D, 2-A | api_prefix 연결, 503 핸들러 추가 |
| `apps/recommendation/service.py` | 1-A, 1-C, 2-B | max 적용, 중복 제거, silent exception 수정 |
| `apps/recommendation/judge.py` | 1-B, 1-C, 2-C, 3-A, 3-C | 중복 제거, 상수 교체, 함수 분리 |
| `apps/recommendation/planner.py` | 1-B, 3-B | 중복 제거, 함수 분리 |

---

## VPN 환경 제약에 따른 검증 전략

외부 서버(LLM, Qdrant, Embedding)는 VPN에서만 호출 가능하므로, 각 Phase별 검증 방법이 다르다.

| Phase | 검증 방법 |
|-------|---------|
| Phase 1 | VPN 불필요. 코드 리뷰 + 로컬 import 검증 |
| Phase 2-A | **VPN 필요.** timeout 동작 및 ExternalDependencyError 발생 확인 |
| Phase 2-B, 2-C | VPN 불필요. 코드 리뷰 |
| Phase 3-A | **VPN 필요.** Map-Reduce 전체 흐름 통합 검증 |
| Phase 3-B | **VPN 필요.** LLM 플래너 응답 파싱 정상 동작 확인 |
| Phase 3-C | VPN 불필요. 정규화 로직 단위 검증 |

---

## 리팩토링 제외 항목 (의도적 유지)

다음 항목들은 리팩토링 대상에서 제외한다.

| 항목 | 제외 이유 |
|------|---------|
| `playground.py` (676줄 HTML) | 운영 UI 변경 리스크 대비 효익이 낮음 |
| `HeuristicPlanner` 도메인 키워드 목록 | 사업 도메인 변경이 필요한 경우 별도 논의 필요 |
| `openai_compat_llm.py` 스트리밍 로직 | 복잡하지만 동작이 검증됨, VPN 없이 테스트 불가 |
| `seed_data.py` / `seed_runner.py` | 데이터 적재 도구 — 구조 변경 시 적재 일관성 리스크 |
| `branch_weights` 미적용 (weighted RRF) | 설계 결정(ADR)에 따라 의도적으로 비활성화됨 |
