"""flat-contract (v2.1) tests for PassthroughEvidenceSelector.

검색(query_points_groups)이 이미 연구자별 chunk를 하이브리드 RRF 관련도순으로 모으므로, 기본 selector는
lexical 재랭크 없이 CandidateCard.evidence_by_type을 doc_type별로 묶고 family cap만 적용해 그대로 노출한다.
각 항목은 item_id == chunk_id, rerank_source == "passthrough", 점수순(match_score=chunk score) 정렬.
"""
from __future__ import annotations

from apps.domain.models import CandidateCard, ChunkEvidence, PlannerOutput
from apps.recommendation.evidence_selector import (
    PassthroughEvidenceSelector,
    RelevantEvidenceBundle,
)
from apps.search.doc_types import DOC_TYPES, FAMILY_EVIDENCE_CAP


def _plan(*keywords: str) -> PlannerOutput:
    return PlannerOutput(intent_summary="test", core_keywords=list(keywords), retrieval_core=list(keywords))


def _evidence(
    *,
    chunk_id: str,
    doc_type: str,
    title: str | None = None,
    date: str | None = None,
    snippet: str = "",
    doc_attrs: dict | None = None,
    score: float = 0.0,
) -> ChunkEvidence:
    return ChunkEvidence(
        chunk_id=chunk_id, doc_type=doc_type, title=title, date=date,
        snippet=snippet, doc_attrs=doc_attrs or {}, score=score,
    )


def _card(expert_id: str = "1", **evidence_by_type: list[ChunkEvidence]) -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id, name="Alpha", organization="LabX", degree="박사",
        counts={}, evidence_by_type=dict(evidence_by_type),
    )


def test_passthrough_sets_item_id_to_chunk_id_and_source():
    selector = PassthroughEvidenceSelector()
    card = _card(paper=[_evidence(chunk_id="paper_100000045256_c000", doc_type="paper",
                                  title="Medical Imaging", date="2024-01", score=0.9)])
    bundle = selector.select(candidates=[card], plan=_plan("medical imaging"))["1"]

    assert isinstance(bundle, RelevantEvidenceBundle)
    paper = bundle.papers[0]
    assert paper.item_id == "paper_100000045256_c000"  # ADR-0004
    assert paper.type == "paper"
    assert paper.rerank_source == "passthrough"
    assert paper.title == "Medical Imaging"
    assert paper.match_score == 0.9


def test_passthrough_preserves_all_evidence_without_keyword_filter():
    # 키워드와 무관하게 모든 evidence 보존(검색이 이미 관련도 선별).
    selector = PassthroughEvidenceSelector()
    card = _card(paper=[
        _evidence(chunk_id="paper_1_c000", doc_type="paper", title="unrelated cooking", score=0.7),
        _evidence(chunk_id="paper_2_c000", doc_type="paper", title="soft robotics", score=0.6),
    ])
    bundle = selector.select(candidates=[card], plan=_plan("medical imaging"))["1"]
    assert {p.item_id for p in bundle.papers} == {"paper_1_c000", "paper_2_c000"}


def test_passthrough_orders_by_score_desc():
    selector = PassthroughEvidenceSelector()
    card = _card(paper=[
        _evidence(chunk_id="paper_lo_c000", doc_type="paper", title="low", score=0.3),
        _evidence(chunk_id="paper_hi_c000", doc_type="paper", title="high", score=0.95),
        _evidence(chunk_id="paper_mid_c000", doc_type="paper", title="mid", score=0.6),
    ])
    bundle = selector.select(candidates=[card], plan=_plan())["1"]
    assert [p.item_id for p in bundle.papers] == ["paper_hi_c000", "paper_mid_c000", "paper_lo_c000"]


def test_family_cap_limits_achievement_to_ten():
    cap = FAMILY_EVIDENCE_CAP["achievement"]
    assert cap == 10
    selector = PassthroughEvidenceSelector()
    papers = [_evidence(chunk_id=f"paper_{i}_c000", doc_type="paper", title=f"s{i}", score=1.0 - i * 0.01)
              for i in range(cap + 5)]
    bundle = selector.select(candidates=[_card(paper=papers)], plan=_plan())["1"]
    assert len(bundle.papers) == cap
    # 상위 점수 cap개만 유지.
    assert bundle.papers[0].item_id == "paper_0_c000"


def test_family_cap_limits_assessment_and_expertise_to_six():
    assert FAMILY_EVIDENCE_CAP["assessment"] == 6
    assert FAMILY_EVIDENCE_CAP["expertise"] == 6
    selector = PassthroughEvidenceSelector()
    assessor = [_evidence(chunk_id=f"assessor_activity_{i}_c000", doc_type="assessor_activity", title=f"a{i}", score=0.5)
                for i in range(12)]
    specialty = [_evidence(chunk_id=f"specialty_{i}_c000", doc_type="specialty", title=f"s{i}", score=0.5)
                 for i in range(12)]
    bundle = selector.select(candidates=[_card(assessor_activity=assessor, specialty=specialty)], plan=_plan())["1"]
    assert len(bundle.items_of("assessor_activity")) == 6
    assert len(bundle.items_of("specialty")) == 6


def test_custom_family_cap_overrides_default():
    selector = PassthroughEvidenceSelector(
        family_cap={"achievement": 2, "assessment": 6, "expertise": 6, "identity": 1}
    )
    papers = [_evidence(chunk_id=f"paper_{i}_c000", doc_type="paper", title=f"s{i}", score=1.0 - i * 0.01)
              for i in range(8)]
    bundle = selector.select(candidates=[_card(paper=papers)], plan=_plan())["1"]
    assert len(bundle.papers) == 2


def test_all_five_doc_types_are_bucketed():
    selector = PassthroughEvidenceSelector()
    evidence_by_type = {
        doc_type: [_evidence(chunk_id=f"{doc_type}_1_c000", doc_type=doc_type, title="t", score=0.5)]
        for doc_type in DOC_TYPES
    }
    bundle = selector.select(candidates=[_card(**evidence_by_type)], plan=_plan())["1"]
    assert set(bundle.by_doc_type.keys()) == set(DOC_TYPES)
    assert bundle.items_of("assessor_activity")[0].item_id == "assessor_activity_1_c000"
    assert bundle.items_of("specialty")[0].item_id == "specialty_1_c000"
    assert len(bundle.by_item_id()) == len(DOC_TYPES)


def test_detail_is_derived_from_doc_attrs_per_doc_type():
    selector = PassthroughEvidenceSelector()
    card = _card(
        paper=[_evidence(chunk_id="paper_1_c000", doc_type="paper", title="t",
                         doc_attrs={"journal_name": "Nature Imaging"}, score=0.5)],
        project=[_evidence(chunk_id="project_1_c000", doc_type="project", title="t",
                           doc_attrs={"performing_organization": "KAIST"}, score=0.5)],
    )
    bundle = selector.select(candidates=[card], plan=_plan())["1"]
    assert bundle.papers[0].detail == "Nature Imaging"
    assert bundle.projects[0].detail == "KAIST"


def test_title_falls_back_to_chunk_id_when_missing():
    selector = PassthroughEvidenceSelector()
    card = _card(specialty=[_evidence(chunk_id="specialty_42_c000", doc_type="specialty",
                                      title=None, snippet="", score=0.5)])
    item = selector.select(candidates=[card], plan=_plan())["1"].items_of("specialty")[0]
    assert item.title == "specialty_42_c000"
    assert item.item_id == "specialty_42_c000"


def test_trace_records_passthrough_mode_and_empty_candidates():
    selector = PassthroughEvidenceSelector()
    has = _card("1", paper=[_evidence(chunk_id="paper_1_c000", doc_type="paper", title="t", score=0.5)])
    empty = _card("2")  # no evidence
    bundles = selector.select(candidates=[has, empty], plan=_plan())
    assert bundles["2"].all_items() == []
    trace = selector.last_trace
    assert trace["mode"] == "passthrough"
    assert trace["empty_candidate_ids"] == ["2"]
    counts = {c["expert_id"]: c["total"] for c in trace["candidate_evidence_counts"]}
    assert counts == {"1": 1, "2": 0}
