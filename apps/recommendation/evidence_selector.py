"""후보 카드의 chunk evidence를 grounding용으로 정리하는 모듈 (v2.1, grouped 검색 기준).

검색(query_points_groups)이 이미 연구자별 top-K chunk를 하이브리드 RRF 관련도순으로 모아주므로,
기본 selector(PassthroughEvidenceSelector)는 **lexical 재랭크 없이** doc_type별로 묶고 family cap만
적용해 그대로 evidence로 노출한다. evidence 참조 id == chunk_id(ADR-0004). 5 doc_type 전부 지원.

CrossEncoderEvidenceSelector는 옵트인(모델 부재/예외 시 passthrough fallback)으로 남겨둔다.
HARD 제약: evidence selector는 grounding 선별·표시만 한다 — 후보(연구자) 순위·탈락·생성에 영향 0.
"""
from __future__ import annotations

import math
from typing import Protocol

from pydantic import BaseModel, Field

from apps.domain.chunk_view import parse_year
from apps.domain.models import CandidateCard, ChunkEvidence, PlannerOutput
from apps.search.doc_types import DOC_TYPE_TO_FAMILY, FAMILY_EVIDENCE_CAP

DEFAULT_TYPE_CAP = 10


class RelevantEvidenceItem(BaseModel):
    item_id: str  # == chunk_id
    type: str  # doc_type
    title: str
    date: str | None = None
    detail: str | None = None
    snippet: str | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    match_score: float = 0.0
    # 선별 출처: "passthrough"(검색 관련도순 그대로) | "cross_encoder". 내부 trace 메타.
    rerank_source: str = "passthrough"


class RelevantEvidenceBundle(BaseModel):
    expert_id: str
    by_doc_type: dict[str, list[RelevantEvidenceItem]] = Field(default_factory=dict)

    def all_items(self) -> list[RelevantEvidenceItem]:
        items: list[RelevantEvidenceItem] = []
        for bucket in self.by_doc_type.values():
            items.extend(bucket)
        return items

    def by_item_id(self) -> dict[str, RelevantEvidenceItem]:
        return {item.item_id: item for item in self.all_items()}

    def items_of(self, doc_type: str) -> list[RelevantEvidenceItem]:
        return self.by_doc_type.get(doc_type, [])

    # 하위호환 편의 프로퍼티(achievement family)
    @property
    def papers(self) -> list[RelevantEvidenceItem]:
        return self.by_doc_type.get("paper", [])

    @property
    def patents(self) -> list[RelevantEvidenceItem]:
        return self.by_doc_type.get("patent", [])

    @property
    def projects(self) -> list[RelevantEvidenceItem]:
        return self.by_doc_type.get("project", [])


class EvidenceSelector(Protocol):
    def select(
        self, *, candidates: list[CandidateCard], plan: PlannerOutput
    ) -> dict[str, RelevantEvidenceBundle]: ...


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _type_cap(doc_type: str, family_cap: dict[str, int]) -> int:
    family = DOC_TYPE_TO_FAMILY.get(doc_type)
    if family and family in family_cap:
        return int(family_cap[family])
    return DEFAULT_TYPE_CAP


def _doc_attrs_text(ev: ChunkEvidence) -> str:
    parts: list[str] = []
    for value in (ev.doc_attrs or {}).values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if v)
    return " ".join(parts)


def _evidence_doc_text(ev: ChunkEvidence) -> str:
    return " ".join(part for part in [ev.title or "", ev.snippet or "", _doc_attrs_text(ev)] if part)


def _detail(ev: ChunkEvidence) -> str | None:
    attrs = ev.doc_attrs or {}
    if ev.doc_type == "paper":
        return attrs.get("journal_name") or attrs.get("indexing_database")
    if ev.doc_type == "project":
        return attrs.get("performing_organization") or attrs.get("managing_agency")
    if ev.doc_type == "patent":
        return attrs.get("application_registration_type") or attrs.get("application_country")
    return None


def _evidence_item(ev: ChunkEvidence, *, source: str) -> RelevantEvidenceItem:
    return RelevantEvidenceItem(
        item_id=ev.chunk_id,
        type=ev.doc_type,
        title=ev.title or (ev.snippet[:60] if ev.snippet else ev.chunk_id),
        date=ev.date,
        detail=_detail(ev),
        snippet=ev.snippet or None,
        match_score=ev.score,
        rerank_source=source,
    )


class PassthroughEvidenceSelector:
    """그룹 chunk(하이브리드 RRF 관련도순)를 doc_type별 묶음 + family cap만 적용해 그대로 노출.

    lexical 재랭크 없음. 각 doc_type 내부는 검색 관련도 점수(match_score=chunk score) 내림차순.
    """

    def __init__(self, *, family_cap: dict[str, int] | None = None) -> None:
        self.family_cap = family_cap or dict(FAMILY_EVIDENCE_CAP)
        self.last_trace: dict[str, object] = {}

    def select(
        self, *, candidates: list[CandidateCard], plan: PlannerOutput
    ) -> dict[str, RelevantEvidenceBundle]:
        _ = plan
        bundles: dict[str, RelevantEvidenceBundle] = {}
        counts: list[dict[str, object]] = []
        empty_ids: list[str] = []

        for candidate in candidates:
            by_doc_type: dict[str, list[RelevantEvidenceItem]] = {}
            for doc_type, evidences in candidate.evidence_by_type.items():
                ranked = sorted(evidences, key=lambda ev: -ev.score)
                items = [
                    _evidence_item(ev, source="passthrough")
                    for ev in ranked[: _type_cap(doc_type, self.family_cap)]
                ]
                if items:
                    by_doc_type[doc_type] = items
            bundle = RelevantEvidenceBundle(expert_id=candidate.expert_id, by_doc_type=by_doc_type)
            bundles[candidate.expert_id] = bundle
            total = len(bundle.all_items())
            if total == 0:
                empty_ids.append(candidate.expert_id)
            counts.append({"expert_id": candidate.expert_id, "total": total,
                           "by_doc_type": {dt: len(v) for dt, v in by_doc_type.items()}})

        self.last_trace = {
            "mode": "passthrough",
            "candidate_evidence_counts": counts,
            "empty_candidate_ids": empty_ids,
        }
        return bundles


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CrossEncoderEvidenceSelector:
    """후보 내부 evidence(chunk)를 cross-encoder 관련도로 재정렬해 doc_type별 top-N만 추리는 옵트인 selector.

    scorer 부재·예외 시 passthrough fallback으로 강등. HARD 제약: grounding 선별만(후보 순위 영향 0).
    """

    def __init__(
        self,
        *,
        scorer: object | None,
        fallback: EvidenceSelector | None = None,
        top_n_per_type: int = 5,
        relevance_floor: float = 0.30,
        pregate_per_type: int = 20,
        max_pairs_per_request: int = 256,
    ) -> None:
        self.scorer = scorer
        self.fallback = fallback or PassthroughEvidenceSelector()
        self.top_n_per_type = top_n_per_type
        self.relevance_floor = relevance_floor
        self.pregate_per_type = pregate_per_type
        self.max_pairs_per_request = max_pairs_per_request
        self.last_trace: dict[str, object] = {}

    def select(
        self, *, candidates: list[CandidateCard], plan: PlannerOutput
    ) -> dict[str, RelevantEvidenceBundle]:
        query = self._resolve_query(plan)
        if self.scorer is None:
            return self._fallback("no_scorer", candidates=candidates, plan=plan, query=query)
        return self._rerank(candidates=candidates, plan=plan, query=query)

    @staticmethod
    def _resolve_query(plan: PlannerOutput) -> str:
        semantic = (getattr(plan, "semantic_query", "") or "").strip()
        if semantic:
            return semantic
        return " ".join(plan.core_keywords or []).strip()

    def _fallback(
        self, reason: str, *, candidates: list[CandidateCard], plan: PlannerOutput, query: str
    ) -> dict[str, RelevantEvidenceBundle]:
        bundles = self.fallback.select(candidates=candidates, plan=plan)
        fallback_trace = getattr(self.fallback, "last_trace", {}) or {}
        self.last_trace = {
            "mode": "passthrough_fallback",
            "fallback_reason": reason,
            "query": query,
            "candidate_evidence_counts": fallback_trace.get("candidate_evidence_counts", []),
        }
        return bundles

    def _rerank(
        self, *, candidates: list[CandidateCard], plan: PlannerOutput, query: str
    ) -> dict[str, RelevantEvidenceBundle]:
        pairs: list[tuple[str, str]] = []
        pair_index: list[tuple[int, str, ChunkEvidence]] = []

        for candidate_index, candidate in enumerate(candidates):
            for doc_type, evidences in candidate.evidence_by_type.items():
                deduped = self._dedup(evidences)
                for ev in deduped[: self.pregate_per_type]:
                    pairs.append((query, _evidence_doc_text(ev)))
                    pair_index.append((candidate_index, doc_type, ev))

        try:
            raw_scores = self._score_pairs(pairs)
        except Exception:  # noqa: BLE001
            return self._fallback("scorer_error", candidates=candidates, plan=plan, query=query)

        scored: dict[int, dict[str, list[tuple[ChunkEvidence, float]]]] = {}
        for (candidate_index, doc_type, ev), raw in zip(pair_index, raw_scores):
            normalized = _sigmoid(float(raw))
            scored.setdefault(candidate_index, {}).setdefault(doc_type, []).append((ev, normalized))

        bundles: dict[str, RelevantEvidenceBundle] = {}
        counts: list[dict[str, object]] = []
        for candidate_index, candidate in enumerate(candidates):
            floor_dropped = 0
            by_doc_type: dict[str, list[RelevantEvidenceItem]] = {}
            for doc_type, entries in scored.get(candidate_index, {}).items():
                kept: list[tuple[ChunkEvidence, float]] = []
                for ev, normalized in entries:
                    if normalized < self.relevance_floor:
                        floor_dropped += 1
                        continue
                    kept.append((ev, normalized))
                kept.sort(key=lambda pair: (-pair[1], -(parse_year(pair[0].date) or 0), pair[0].title or ""))
                items: list[RelevantEvidenceItem] = []
                for ev, normalized in kept[: self.top_n_per_type]:
                    item = _evidence_item(ev, source="cross_encoder")
                    item.match_score = normalized
                    items.append(item)
                if items:
                    by_doc_type[doc_type] = items
            bundle = RelevantEvidenceBundle(expert_id=candidate.expert_id, by_doc_type=by_doc_type)
            bundles[candidate.expert_id] = bundle
            counts.append({"expert_id": candidate.expert_id, "total": len(bundle.all_items()),
                           "dropped_below_floor": floor_dropped})

        self.last_trace = {
            "mode": "cross_encoder",
            "query": query,
            "scorer": getattr(self.scorer, "model_name", None),
            "candidate_evidence_counts": counts,
        }
        return bundles

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        batch = max(1, self.max_pairs_per_request)
        for start in range(0, len(pairs), batch):
            scores.extend(self.scorer.score(pairs[start : start + batch]))  # type: ignore[union-attr]
        return scores

    @staticmethod
    def _dedup(evidences: list[ChunkEvidence]) -> list[ChunkEvidence]:
        seen: set[tuple[str, int | None]] = set()
        out: list[ChunkEvidence] = []
        for ev in evidences:
            key = (_normalize_text(ev.title), parse_year(ev.date))
            if key in seen:
                continue
            seen.add(key)
            out.append(ev)
        return out
