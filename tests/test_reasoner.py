"""flat 계약(v2.1) reasoner 테스트.

evidence 참조 id == chunk_id(<doc_type>_<숫자>_c<NNN>). VALID_EVIDENCE_ID_PATTERN은
이 코덱만 허용하고 구설계 "paper:0" 형태는 거부한다. _serialize_candidates는
relevant_evidence(RelevantEvidenceBundle.by_doc_type)와 context_evidence
(CandidateCard.evidence_by_type)를 직렬화한다. LLM은 FakeReasonModel로 대체한다.
"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.domain.models import CandidateCard, ChunkEvidence, PlannerOutput
from apps.recommendation.evidence_selector import (
    RelevantEvidenceBundle,
    RelevantEvidenceItem,
)
from apps.recommendation.reasoner import (
    FIT_HIGH,
    FIT_NORMAL,
    MAX_SELECTED_EVIDENCE_IDS,
    REASON_TOOL_NAME,
    VALID_EVIDENCE_ID_PATTERN,
    _INLINE_EVIDENCE_ID_PATTERN,
    _strip_inline_evidence_ids,
    OpenAICompatReasonGenerator,
    PassThroughReasonGenerator,
    ReasonedCandidate,
    ReasonGenerationOutput,
)


# ---------------------------------------------------------------------------
# fakes / helpers
# ---------------------------------------------------------------------------


class FakeReasonModel:
    """OpenAICompatChatModel.ainvoke_non_stream 대체(네트워크/실모델 미사용)."""

    def __init__(self, content="", *, tool_calls=None, responses=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.responses = list(responses or [])
        self.last_kwargs = None
        self.last_messages = None
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    async def ainvoke_non_stream(self, messages, **kwargs):
        self.last_kwargs = dict(kwargs)
        self.last_messages = messages
        self.call_count += 1
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        if self.responses:
            return self.responses.pop(0)
        additional_kwargs = {"tool_calls": self.tool_calls} if self.tool_calls else {}
        return AIMessage(content=self.content, additional_kwargs=additional_kwargs)


def _make_generator() -> OpenAICompatReasonGenerator:
    return OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )


def _plan(*keywords: str) -> PlannerOutput:
    kws = list(keywords) or ["semiconductor"]
    return PlannerOutput(
        intent_summary="Recommend reviewers",
        retrieval_core=kws,
        core_keywords=kws,
        task_terms=kws,
    )


def _chunk_evidence(
    chunk_id: str,
    doc_type: str,
    *,
    title: str,
    date: str | None = None,
    snippet: str = "",
    doc_attrs: dict | None = None,
) -> ChunkEvidence:
    return ChunkEvidence(
        chunk_id=chunk_id,
        doc_type=doc_type,
        title=title,
        date=date,
        snippet=snippet,
        doc_attrs=doc_attrs or {},
    )


def _candidate(
    expert_id: str,
    name: str,
    score: float,
    *,
    evidence_by_type: dict[str, list[ChunkEvidence]] | None = None,
    risks: list[str] | None = None,
    data_gaps: list[str] | None = None,
) -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id,
        name=name,
        organization="Test Institute",
        degree="박사",
        counts={"article_cnt": 1},
        evidence_by_type=evidence_by_type or {},
        risks=risks or [],
        data_gaps=data_gaps or [],
        rank_score=score,
        shortlist_score=score,
    )


def _relevant_item(item_id: str, doc_type: str, *, title: str, date="2024-01") -> RelevantEvidenceItem:
    return RelevantEvidenceItem(
        item_id=item_id,
        type=doc_type,
        title=title,
        date=date,
        detail="Detail line",
        snippet="snippet body",
        matched_keywords=["medical imaging"],
        match_score=11.5,
    )


def _bundle(expert_id: str, *items: RelevantEvidenceItem) -> RelevantEvidenceBundle:
    by_doc_type: dict[str, list[RelevantEvidenceItem]] = {}
    for item in items:
        by_doc_type.setdefault(item.type, []).append(item)
    return RelevantEvidenceBundle(expert_id=expert_id, by_doc_type=by_doc_type)


# ---------------------------------------------------------------------------
# VALID_EVIDENCE_ID_PATTERN — flat chunk_id 코덱만 허용
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evidence_id",
    [
        "paper_100000045256_c000",
        "patent_42_c001",
        "project_7_c123",
        "assessor_activity_900_c000",
        "specialty_1_c000",
    ],
)
def test_valid_evidence_id_pattern_accepts_flat_chunk_ids(evidence_id):
    assert VALID_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)


@pytest.mark.parametrize(
    "evidence_id",
    [
        "paper:0",  # 구설계 코덱 — 거부
        "PUB_100000045256_c000",  # 대문자 prefix 폐기
        "paper_100000045256",  # chunk index 누락
        "unknown_1_c000",  # 미지원 doc_type
        "paper_1_cXX",  # 숫자 chunk index 아님
        "",
    ],
)
def test_valid_evidence_id_pattern_rejects_non_flat_ids(evidence_id):
    assert VALID_EVIDENCE_ID_PATTERN.fullmatch(evidence_id) is None


# ---------------------------------------------------------------------------
# _strip_inline_evidence_ids — 본문에 누출된 chunk_id 결정론적 제거(백스톱)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "'호텔건물 화재안전성 평가' 논문(paper_100000435395_c000)으로 건축소방을 다룹니다.",
            "'호텔건물 화재안전성 평가' 논문으로 건축소방을 다룹니다.",
        ),
        (
            "specialty_M1013800_c000 전문분야와 project_7_c123 과제를 보유.",
            "전문분야와 과제를 보유.",
        ),
        (
            "근거 [assessor_activity_900_c000] 가 있습니다.",
            "근거 가 있습니다.",
        ),
        (
            "관련 실적 paper_42_c000, 그리고 추가 검토.",
            "관련 실적, 그리고 추가 검토.",
        ),
        (
            "평가（paper_42_c001）결과를 제시.",  # 전각 괄호
            "평가결과를 제시.",
        ),
        (
            "핵심 근거: project_7_c123",  # 문자열 끝
            "핵심 근거:",
        ),
    ],
)
def test_strip_inline_evidence_ids_removes_leaked_ids(raw, expected):
    cleaned, changed = _strip_inline_evidence_ids(raw)
    assert changed is True
    assert cleaned == expected
    assert _INLINE_EVIDENCE_ID_PATTERN.search(cleaned) is None


@pytest.mark.parametrize(
    "text",
    [
        "paper_100000045256 형태는 chunk index가 없습니다.",  # _c<NNN> 없음
        "project_abc 처럼 코덱 미충족은 그대로 둡니다.",  # _c<NNN> 없음
        "스마트 제조 분야 연구를 수행했습니다.",  # 식별자 없음
    ],
)
def test_strip_inline_evidence_ids_keeps_near_miss_text(text):
    cleaned, changed = _strip_inline_evidence_ids(text)
    assert changed is False
    assert cleaned == text


def test_strip_inline_evidence_ids_noop_when_clean():
    text = "건축소방 분야 실증 연구를 수행했습니다."
    cleaned, changed = _strip_inline_evidence_ids(text)
    assert changed is False
    assert cleaned == text


def test_normalize_output_scrubs_leaked_id_from_reason():
    candidate = _candidate("1", "후보", 1.0)
    raw = ReasonGenerationOutput(
        items=[
            ReasonedCandidate(
                expert_id="1",
                fit=FIT_NORMAL,
                recommendation_reason="해당 논문(paper_100000435395_c000)으로 건축소방을 다룹니다.",
                selected_evidence_ids=["paper_100000435395_c000"],
                risks=[],
            )
        ]
    )
    normalized, trace = OpenAICompatReasonGenerator._normalize_output(raw, [candidate])
    reason = normalized.items[0].recommendation_reason
    assert "paper_100000435395_c000" not in reason
    assert reason == "해당 논문으로 건축소방을 다룹니다."
    assert trace["leaked_reason_candidate_ids"] == ["1"]


# ---------------------------------------------------------------------------
# PassThroughReasonGenerator — LLM 미사용 폴백
# ---------------------------------------------------------------------------


def test_pass_through_generator_emits_one_item_per_candidate_with_empty_reason():
    generator = PassThroughReasonGenerator()
    candidates = [
        _candidate("1", "Alpha", 98.0, risks=["limited recent activity"]),
        _candidate("2", "Bravo", 95.0),
    ]

    output = asyncio.run(
        generator.generate(query="Recommend reviewers", plan=_plan(), candidates=candidates)
    )

    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert all(item.recommendation_reason == "" for item in output.items)
    assert all(item.fit == FIT_NORMAL for item in output.items)
    assert all(item.selected_evidence_ids == [] for item in output.items)
    # 후보의 risks는 보존
    assert output.items[0].risks == ["limited recent activity"]
    assert generator.last_trace["mode"] == "pass_through"
    assert generator.last_trace["candidate_count"] == 2
    assert generator.last_trace["returned_ids"] == ["1", "2"]
    assert generator.last_trace["missing_candidate_ids"] == []


# ---------------------------------------------------------------------------
# generate(): JSON fallback / 입력 순서 정규화
# ---------------------------------------------------------------------------


def test_generate_normalizes_output_to_input_order():
    generator = _make_generator()
    generator.model = FakeReasonModel(
        """{
          "items": [
            {"expert_id": "2", "fit": "중간", "recommendation_reason": "Reason for second", "risks": []},
            {"expert_id": "1", "fit": "높음", "recommendation_reason": "Reason for first", "risks": []}
          ],
          "data_gaps": []
        }"""
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("semiconductor"),
            candidates=[_candidate("1", "Alpha", 98.0), _candidate("2", "Bravo", 95.0)],
        )
    )

    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert output.items[0].recommendation_reason == "Reason for first"
    assert output.items[1].recommendation_reason == "Reason for second"
    assert output.items[0].selected_evidence_ids == []
    # tool 호출 실패 → JSON 본문 fallback 파싱
    assert generator.last_trace["mode"] == "json_fallback"
    assert generator.last_trace["returned_ids"] == ["2", "1"]
    assert generator.last_trace["missing_candidate_ids"] == []
    assert generator.last_trace["retry_count"] == 0


def test_generate_fills_missing_candidate_with_empty_reason():
    generator = _make_generator()
    generator.model = FakeReasonModel(
        """{
          "items": [
            {"expert_id": "1", "fit": "보통", "recommendation_reason": "Only first returned", "risks": []}
          ],
          "data_gaps": []
        }"""
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("semiconductor"),
            candidates=[
                _candidate("1", "Alpha", 98.0),
                _candidate("2", "Bravo", 95.0, risks=["data gap"]),
            ],
        )
    )

    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert output.items[1].recommendation_reason == ""
    assert output.items[1].fit == FIT_NORMAL
    # 누락된 후보의 risks는 후보 카드에서 보존
    assert output.items[1].risks == ["data gap"]
    assert generator.last_trace["returned_ids"] == ["1"]
    assert generator.last_trace["missing_candidate_ids"] == ["2"]
    assert generator.last_trace["empty_reason_candidate_ids"] == ["2"]


# ---------------------------------------------------------------------------
# tool_call 경로 + evidence id 검증/정규화
# ---------------------------------------------------------------------------


def test_generate_prefers_tool_call_arguments_and_keeps_valid_evidence_ids():
    generator = _make_generator()
    tool_payload = {
        "items": [
            {
                "expert_id": "1",
                "fit": FIT_HIGH,
                "recommendation_reason": "Structured tool output is available.",
                "selected_evidence_ids": ["paper_100000045256_c000"],
                "risks": [],
            }
        ],
        "data_gaps": [],
    }
    generator.model = FakeReasonModel(
        content='{"items":[],"data_gaps":[]}',
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": REASON_TOOL_NAME,
                    "arguments": json.dumps(tool_payload, ensure_ascii=False),
                },
            }
        ],
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("semiconductor"),
            candidates=[_candidate("1", "Alpha", 98.0)],
        )
    )

    assert output.items[0].recommendation_reason == "Structured tool output is available."
    assert output.items[0].fit == FIT_HIGH
    assert output.items[0].selected_evidence_ids == ["paper_100000045256_c000"]
    assert generator.last_trace["mode"] == "tool_call"
    assert generator.last_trace["invalid_selected_evidence_candidate_ids"] == []


def test_generate_drops_invalid_and_dedup_evidence_ids():
    generator = _make_generator()
    tool_payload = {
        "items": [
            {
                "expert_id": "1",
                "fit": "높음",
                "recommendation_reason": "Has bogus citation.",
                # paper:0 = 구코덱(거부), 중복 제거, 유효 1개만 보존
                "selected_evidence_ids": [
                    "paper:0",
                    "paper_42_c000",
                    "paper_42_c000",
                ],
                "risks": [],
            }
        ],
        "data_gaps": [],
    }
    generator.model = FakeReasonModel(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": REASON_TOOL_NAME,
                    "arguments": json.dumps(tool_payload, ensure_ascii=False),
                },
            }
        ],
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("semiconductor"),
            candidates=[_candidate("1", "Alpha", 98.0)],
        )
    )

    assert output.items[0].selected_evidence_ids == ["paper_42_c000"]
    assert generator.last_trace["invalid_selected_evidence_candidate_ids"] == ["1"]
    assert generator.last_trace["invalid_selected_evidence_ids_by_candidate"]["1"] == ["paper:0"]


def test_generate_coerces_unknown_fit_to_normal():
    generator = _make_generator()
    generator.model = FakeReasonModel(
        """{
          "items": [
            {"expert_id": "1", "fit": "EXCELLENT", "recommendation_reason": "x", "risks": []}
          ],
          "data_gaps": []
        }"""
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("semiconductor"),
            candidates=[_candidate("1", "Alpha", 98.0)],
        )
    )

    assert output.items[0].fit == FIT_NORMAL


# ---------------------------------------------------------------------------
# _serialize_candidates — relevant_evidence / context_evidence
# ---------------------------------------------------------------------------


def test_serialize_candidates_emits_relevant_and_context_evidence():
    generator = _make_generator()
    fake_model = FakeReasonModel(
        """{
          "items": [
            {"expert_id": "1", "fit": "높음", "recommendation_reason": "Relevant imaging evidence.", "risks": []}
          ],
          "data_gaps": []
        }"""
    )
    generator.model = fake_model

    candidate = _candidate(
        "1",
        "Alpha",
        98.0,
        evidence_by_type={
            "paper": [
                _chunk_evidence(
                    "paper_100000045256_c000",
                    "paper",
                    title="Context imaging paper",
                    date="2023-05",
                )
            ]
        },
    )

    asyncio.run(
        generator.generate(
            query="Recommend medical imaging reviewers",
            plan=_plan("medical imaging"),
            candidates=[candidate, _candidate("2", "Bravo", 95.0)],
            relevant_evidence_by_expert_id={
                "1": _bundle(
                    "1",
                    _relevant_item(
                        "paper_100000045256_c000",
                        "paper",
                        title="Medical imaging segmentation",
                    ),
                )
            },
        )
    )

    payload = json.loads(fake_model.calls[0]["messages"][1].content)
    candidate_payload = payload["candidates"][0]

    # core_keywords가 payload로 전달
    assert payload["core_keywords"] == ["medical imaging"]

    # relevant_evidence: bundle.by_doc_type 직렬화
    assert "relevant_evidence" in candidate_payload
    relevant = candidate_payload["relevant_evidence"]
    assert relevant[0]["evidence_id"] == "paper_100000045256_c000"
    assert relevant[0]["type"] == "paper"
    assert relevant[0]["title"] == "Medical imaging segmentation"
    assert relevant[0]["matched_keywords"] == ["medical imaging"]

    # context_evidence: CandidateCard.evidence_by_type 직렬화(제목/날짜 위주)
    assert "context_evidence" in candidate_payload
    context = candidate_payload["context_evidence"]
    assert context[0]["evidence_id"] == "paper_100000045256_c000"
    assert context[0]["type"] == "paper"
    assert context[0]["date"] == "2023-05"

    # 구설계 키 제거 확인
    assert "selected_evidence" not in candidate_payload
    assert "do_not_mention" not in candidate_payload
    assert "relevant_papers" not in candidate_payload
    assert "all_projects" not in candidate_payload

    # 새 카드 형상 직렬화
    assert candidate_payload["expert_id"] == "1"
    assert candidate_payload["doc_types_present"] == ["paper"]
    assert candidate_payload["counts"] == {"article_cnt": 1}


def test_serialize_candidates_uses_empty_bundle_when_no_relevant_evidence():
    profile = OpenAICompatReasonGenerator._serialize_candidates(
        [_candidate("9", "NoEvidence", 50.0)],
        relevant_evidence_by_expert_id=None,
        retrieval_score_traces_by_expert_id=None,
        profile={
            "name": "primary",
            "detail_char_limit": 200,
            "snippet_char_limit": 1000,
            "matched_keywords_limit": 5,
            "matched_filter_limit": 4,
            "relevant_limit": 10,
            "context_limit": 3,
        },
    )

    assert profile[0]["expert_id"] == "9"
    assert profile[0]["relevant_evidence"] == []
    assert profile[0]["context_evidence"] == []


def test_allowed_evidence_ids_enum_restricts_tool_schema():
    generator = _make_generator()
    fake_model = FakeReasonModel(
        content='{"items":[],"data_gaps":[]}',
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": REASON_TOOL_NAME,
                    "arguments": json.dumps(
                        {
                            "items": [
                                {
                                    "expert_id": "1",
                                    "fit": "높음",
                                    "recommendation_reason": "ok",
                                    "selected_evidence_ids": ["paper_42_c000"],
                                    "risks": [],
                                }
                            ],
                            "data_gaps": [],
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    )
    generator.model = fake_model

    asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("medical imaging"),
            candidates=[_candidate("1", "Alpha", 98.0)],
            relevant_evidence_by_expert_id={
                "1": _bundle(
                    "1",
                    _relevant_item("paper_42_c000", "paper", title="A paper"),
                )
            },
        )
    )

    tool_schema = fake_model.calls[0]["kwargs"]["tools"][0]["function"]["parameters"]
    item_props = tool_schema["properties"]["items"]["items"]["properties"]
    # flat 계약: selected_evidence_ids는 도구 스키마에 존재하고 maxItems 제한
    assert "selected_evidence_ids" in item_props
    assert item_props["selected_evidence_ids"]["maxItems"] == MAX_SELECTED_EVIDENCE_IDS
    # allowed evidence ids enum 제약
    assert item_props["selected_evidence_ids"]["items"]["enum"] == ["paper_42_c000"]


# ---------------------------------------------------------------------------
# empty candidates / fallback
# ---------------------------------------------------------------------------


def test_generate_with_no_candidates_returns_empty_output():
    generator = _make_generator()
    generator.model = FakeReasonModel()

    output = asyncio.run(
        generator.generate(query="Recommend reviewers", plan=_plan(), candidates=[])
    )

    assert output.items == []
    assert generator.model.call_count == 0
    assert generator.last_trace["candidate_count"] == 0


def test_generate_falls_back_to_pass_through_when_all_attempts_fail():
    generator = _make_generator()
    # 두 시도 모두 빈 items → returned_ids 없음 → 폴백으로 강등
    generator.model = FakeReasonModel(
        responses=[
            AIMessage(content='{"items":[],"data_gaps":[]}'),
            AIMessage(content='{"items":[],"data_gaps":[]}'),
        ]
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=_plan("semiconductor"),
            candidates=[_candidate("1", "Alpha", 98.0), _candidate("2", "Bravo", 95.0)],
        )
    )

    # 폴백은 모든 후보를 빈 사유로 반환
    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert all(item.recommendation_reason == "" for item in output.items)
    assert generator.model.call_count == 2
    assert generator.last_trace["mode"] == "fallback"
    assert generator.last_trace["retry_count"] == 2


# ---------------------------------------------------------------------------
# pydantic 모델 정규화
# ---------------------------------------------------------------------------


def test_reasoned_candidate_normalizes_whitespace_in_lists():
    item = ReasonedCandidate(
        expert_id="1",
        selected_evidence_ids=["  paper_1_c000  ", "paper_1_c000", "  "],
        risks=["  a  b  ", ""],
    )
    # 공백 압축 + 중복/빈값 제거
    assert item.selected_evidence_ids == ["paper_1_c000"]
    assert item.risks == ["a b"]


def test_reason_generation_output_defaults_empty():
    output = ReasonGenerationOutput()
    assert output.items == []
    assert output.data_gaps == []
