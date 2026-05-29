"""
CrossEncoderEvidenceSelector 동작 검증.

실제 cross-encoder 모델 없이 FakeScorer를 주입하여
관련도 정렬 / floor drop / top-N cap / pre-gate / dedup / lexical fallback을 검증한다.
"""
from __future__ import annotations

from apps.domain.models import (
    CandidateCard,
    IntellectualPropertyEvidence,
    PlannerOutput,
    PublicationEvidence,
    ResearchProjectEvidence,
)
from apps.recommendation.evidence_selector import (
    CrossEncoderEvidenceSelector,
    KeywordEvidenceSelector,
)


class FakeScorer:
    """doc 텍스트에 특정 needle이 포함되면 지정 raw logit을 반환하는 가짜 cross-encoder."""

    model_name = "fake-cross-encoder"

    def __init__(self, score_map=None, default=0.0, raise_exc=False):
        self.score_map = score_map or {}
        self.default = default
        self.raise_exc = raise_exc
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs):
        if self.raise_exc:
            raise RuntimeError("cross-encoder unavailable")
        self.calls.append(list(pairs))
        out = []
        for _query, doc in pairs:
            value = self.default
            for needle, score in self.score_map.items():
                if needle in doc:
                    value = score
                    break
            out.append(value)
        return out

    @property
    def total_pairs(self) -> int:
        return sum(len(call) for call in self.calls)


def _plan(*keywords: str, semantic_query: str = "") -> PlannerOutput:
    return PlannerOutput(
        intent_summary="test intent",
        core_keywords=list(keywords),
        retrieval_core=list(keywords),
        semantic_query=semantic_query,
    )


def _card(*, papers=None, projects=None, patents=None) -> CandidateCard:
    return CandidateCard(
        expert_id="1",
        name="Alpha",
        top_papers=list(papers or []),
        top_projects=list(projects or []),
        top_patents=list(patents or []),
    )


def _paper(title: str, *, year: str, abstract: str = "") -> PublicationEvidence:
    return PublicationEvidence(
        publication_title=title,
        publication_year_month=year,
        abstract=abstract,
    )


def _selector(scorer, **kwargs) -> CrossEncoderEvidenceSelector:
    return CrossEncoderEvidenceSelector(
        scorer=scorer,
        fallback=KeywordEvidenceSelector(reference_year=2026),
        top_n_per_type=kwargs.get("top_n_per_type", 5),
        relevance_floor=kwargs.get("relevance_floor", 0.30),
        pregate_per_type=kwargs.get("pregate_per_type", 20),
        max_pairs_per_request=kwargs.get("max_pairs_per_request", 256),
    )


def test_orders_evidence_by_cross_encoder_score():
    scorer = FakeScorer(score_map={"medical imaging": 4.0, "soft robot": 0.5})
    card = _card(
        papers=[
            _paper("soft robot manipulation", year="2026-01", abstract="hand control"),
            _paper("medical imaging segmentation", year="2022-05", abstract="emergency CT"),
        ]
    )

    bundles = _selector(scorer).select(candidates=[card], plan=_plan("medical imaging"))

    papers = bundles["1"].papers
    assert [p.title for p in papers] == [
        "medical imaging segmentation",
        "soft robot manipulation",
    ]
    assert papers[0].rerank_source == "cross_encoder"
    assert papers[0].match_score > papers[1].match_score


def test_drops_items_below_relevance_floor():
    scorer = FakeScorer(score_map={"relevant": 3.0, "noise": -5.0})
    card = _card(
        papers=[
            _paper("relevant imaging study", year="2024-01"),
            _paper("noise unrelated topic", year="2025-01"),
        ]
    )

    selector = _selector(scorer)
    bundles = selector.select(candidates=[card], plan=_plan("imaging"))

    assert [p.title for p in bundles["1"].papers] == ["relevant imaging study"]
    counts = selector.last_trace["candidate_evidence_counts"][0]
    assert counts["dropped_below_floor"] >= 1


def test_caps_at_top_n_per_type():
    scorer = FakeScorer(default=0.0, score_map={f"paper {i}": 3.0 - i * 0.1 for i in range(8)})
    card = _card(
        papers=[_paper(f"imaging paper {i}", year="2024-01") for i in range(8)]
    )

    bundles = _selector(scorer, top_n_per_type=5).select(
        candidates=[card], plan=_plan("imaging")
    )

    papers = bundles["1"].papers
    assert len(papers) == 5
    # 가장 높은 점수(=paper 0..4)만 유지되어야 한다.
    assert [p.title for p in papers] == [f"imaging paper {i}" for i in range(5)]


def test_pregate_limits_scored_pairs():
    scorer = FakeScorer(default=1.0)
    card = _card(papers=[_paper(f"imaging paper {i}", year="2024-01") for i in range(5)])

    _selector(scorer, pregate_per_type=2).select(
        candidates=[card], plan=_plan("imaging")
    )

    assert scorer.total_pairs == 2


def test_deduplicates_same_title_and_year_before_scoring():
    scorer = FakeScorer(default=2.0)
    card = _card(
        papers=[
            _paper("AI semiconductor platform", year="2025-01", abstract="first"),
            _paper("AI semiconductor platform", year="2025-06", abstract="dup same year"),
        ]
    )

    selector = _selector(scorer)
    bundles = selector.select(candidates=[card], plan=_plan("AI semiconductor"))

    assert len(bundles["1"].papers) == 1
    assert scorer.total_pairs == 1
    assert selector.last_trace["candidate_evidence_counts"][0]["dedup_dropped"] == 1


def test_falls_back_to_lexical_when_scorer_missing():
    card = _card(papers=[_paper("medical imaging study", year="2024-01")])

    selector = _selector(None)
    bundles = selector.select(candidates=[card], plan=_plan("medical imaging"))

    assert selector.last_trace["mode"] == "lexical_fallback"
    assert selector.last_trace["fallback_reason"] == "no_scorer"
    assert [p.title for p in bundles["1"].papers] == ["medical imaging study"]


def test_falls_back_to_lexical_when_scorer_raises():
    scorer = FakeScorer(raise_exc=True)
    card = _card(papers=[_paper("medical imaging study", year="2024-01")])

    selector = _selector(scorer)
    bundles = selector.select(candidates=[card], plan=_plan("medical imaging"))

    assert selector.last_trace["mode"] == "lexical_fallback"
    assert selector.last_trace["fallback_reason"] == "scorer_error"
    assert bundles["1"].papers  # lexical matched "medical imaging"


def test_prefers_semantic_query_over_core_keywords():
    scorer = FakeScorer(default=2.0)
    card = _card(papers=[_paper("imaging paper", year="2024-01")])

    selector = _selector(scorer)
    selector.select(
        candidates=[card],
        plan=_plan("imaging", semantic_query="semantic intent text"),
    )

    assert selector.last_trace["query"] == "semantic intent text"
    # scorer에 전달된 query 측도 semantic_query여야 한다.
    assert scorer.calls[0][0][0] == "semantic intent text"
