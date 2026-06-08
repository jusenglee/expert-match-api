"""관련도 기반 검색 보조 — concept 해석/태깅, 멀티뷰 융합점수, capped evidence scoring (순수 함수).

설계(사용자 spec):
- 질의를 정규식으로 깎지 않는다. concept 검증은 검색 후 deterministic multi-signal로 한다.
- 멀티뷰 융합은 raw score 합산 금지 — source별 normalized rank score × view_weight.
- researcher 점수 = 단순 합산 ✗, required-concept best + balance + joint + capped support + 품질/감점.
- concept는 planner(LLM) concept_specs > query_exact 합성(retrieval_core) > 없음 순으로 동적 확정한다
  (resolve_concept_plan). 하드코딩 도메인 사전(registry)은 제거됨 — planner 산출이 단일 출처.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from apps.domain.models import ChunkHit, ChunkPayload, ConceptSpec
from apps.search.doc_types import DOC_TYPE_TO_FAMILY
from apps.search.query_builder import GENERIC_SEARCH_TERMS

# query_exact 합성 시 제외할 일반어(역할/불용어). canonical 목록 재사용(중복 정의 금지).
_GENERIC_TERMS: frozenset[str] = frozenset(t.casefold() for t in GENERIC_SEARCH_TERMS)

CONCEPT_VIEW_PREFIX = "concept:"  # 검색 view source 이름: "concept:<id>"

# planner가 term을 과다 생성해 다시 '사전 지옥'이 되지 않도록 cap.
MAX_CONCEPTS = 5
MAX_QUERY_TERMS = 8
MAX_EVIDENCE_TERMS = 12
MAX_WEAK_TERMS = 12


# ---------------------------------------------------------------------------
# Concept Evidence Plan (planner 우선 → registry 보강 → registry 감지 fallback)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ConceptPlan:
    """질의별 concept evidence plan. specs가 단일 출처이고 나머지는 파생(compat)."""

    specs: list[ConceptSpec] = field(default_factory=list)
    source: str = "none"  # "planner" | "registry" | "none"

    @property
    def required(self) -> list[str]:
        return [s.id for s in self.specs if s.role == "required"]

    @property
    def optional(self) -> list[str]:
        return [s.id for s in self.specs if s.role == "optional"]

    @property
    def all_concepts(self) -> list[str]:
        seen: list[str] = []
        for spec in self.specs:
            if spec.id not in seen:
                seen.append(spec.id)
        return seen

    @property
    def concept_queries(self) -> dict[str, str]:
        """concept_id -> 검색 view 텍스트(query_terms 중심; recall용)."""
        out: dict[str, str] = {}
        for spec in self.specs:
            text = " ".join(spec.query_terms).strip() or spec.label or spec.id
            out[spec.id] = text
        return out

    @property
    def focus_query(self) -> str:
        """sparse_focus용 짧은 핵심 명사구(concept label 중심 — SPLADE 과확장 억제)."""
        parts: list[str] = []
        for spec in self.specs:
            token = spec.label or (spec.query_terms[0] if spec.query_terms else spec.id)
            token = " ".join(str(token).split())
            if token and token not in parts:
                parts.append(token)
        return " ".join(parts)

    def spec(self, concept_id: str) -> ConceptSpec | None:
        for spec in self.specs:
            if spec.id == concept_id:
                return spec
        return None


def _normalize_text(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            values.extend(str(item) for item in part if item is not None)
        else:
            values.append(str(part))
    return " ".join(" ".join(values).casefold().split())


def _contains_term(text: str, term: str) -> bool:
    """word-boundary 매칭. 짧은 영숫자(ai 등)는 경계 강제(Al/aluminum 오인 방지). 한글은 부분포함 허용."""
    normalized = term.casefold().strip()
    if not normalized:
        return False
    if normalized.isascii() and normalized.isalnum() and len(normalized) <= 3:
        pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    return normalized in text


def _synthesize_query_exact(plan: Any) -> list[ConceptSpec]:
    """planner·registry 둘 다 실패 시 최후 안전망(source='query_exact').

    planner가 산출한 retrieval_core(없으면 core_keywords) 키워드를 query_terms=evidence_terms로 갖는
    concept으로 합성한다. role='optional'이라 required gate는 걸지 않고(omission 위험 0), score_researcher의
    concept_sum/support 신호만 살려 concept 없는 _score_generic보다 정렬 품질을 높인다. 도메인 하드코딩 0.
    """
    cores = list(getattr(plan, "retrieval_core", None) or getattr(plan, "core_keywords", None) or [])
    specs: list[ConceptSpec] = []
    seen: set[str] = set()
    for raw in cores:
        term = " ".join(str(raw).split())
        key = term.casefold()
        if len(term) < 2 or key in seen or key in _GENERIC_TERMS:
            continue
        seen.add(key)
        specs.append(
            ConceptSpec(
                id=term,
                label=term,
                role="optional",
                query_terms=[term],
                evidence_terms=[term],
                weak_terms=[],
                source="query_exact",
                confidence=0.5,
            )
        )
    return specs


def _apply_caps(specs: list[ConceptSpec]) -> list[ConceptSpec]:
    capped: list[ConceptSpec] = []
    for spec in specs[:MAX_CONCEPTS]:
        capped.append(
            spec.model_copy(
                update={
                    "query_terms": list(spec.query_terms[:MAX_QUERY_TERMS]),
                    "evidence_terms": list(spec.evidence_terms[:MAX_EVIDENCE_TERMS]),
                    "weak_terms": list(spec.weak_terms[:MAX_WEAK_TERMS]),
                }
            )
        )
    return capped


def resolve_concept_plan(plan: Any, raw_query: str) -> ConceptPlan:
    """동적 Concept Evidence Plan 확정 (planner 산출 단일 출처).

    우선순위: planner concept_specs > query_exact 합성(retrieval_core) > 없음.
    하드코딩 도메인 사전(registry)은 제거됨 — concept/evidence는 planner(LLM)가 질의마다 산출한다.
    planner가 빈 출력을 내면 retrieval_core 키워드를 query_exact concept(optional)으로 합성한다.
    raw_query는 호출부 시그니처 호환을 위해 유지(현재 미사용).
    """
    planner_specs = list(getattr(plan, "concept_specs", []) or [])
    if planner_specs:
        specs, source = planner_specs, "planner"
    else:
        specs = _synthesize_query_exact(plan)
        source = "query_exact" if specs else "none"
    return ConceptPlan(specs=_apply_caps(specs), source=source)


# ---------------------------------------------------------------------------
# chunk concept tagging (deterministic multi-signal)
# ---------------------------------------------------------------------------
def _payload_text(payload: ChunkPayload) -> str:
    return _normalize_text(payload.chunk_text, payload.doc_id)


def has_operation_marker(payload: ChunkPayload, markers: frozenset[str]) -> bool:
    """운영성/교육/행정 과제 마커가 chunk 본문에 등장하는가(근거 가치 하향용, 융합점수에 factor 적용).

    markers는 casefold 가정. 공백 제거본과도 대조해 '사업단 운영'↔'사업단운영' 같은 표기차를 흡수한다.
    """
    if not markers:
        return False
    text = _payload_text(payload)
    despaced = text.replace(" ", "")
    return any(marker in despaced for marker in markers)


def chunk_concept_signals(
    payload: ChunkPayload,
    concept_plan: ConceptPlan,
    *,
    view_concept_hits: set[str] | None = None,
) -> dict[str, set[str]]:
    """chunk의 concept별 신호 분류: confirmed / weak_only / view_only.

    - confirmed: evidence_term이 chunk_text/doc_id에 직접 등장 → concept 확정.
    - weak_only: weak_term만 있고 evidence 없음 → 약신호(확정 X).
    - view_only: concept:<id> 검색 view에만 잡힘(evidence/weak 없음) → 약신호(확정 X).
    """
    text = _payload_text(payload)
    hits = view_concept_hits or set()
    confirmed: set[str] = set()
    weak_only: set[str] = set()
    view_only: set[str] = set()
    for spec in concept_plan.specs:
        # evidence 미정의 concept는 query_terms를 확정 근거로 대체(dead concept 방지).
        evidence = spec.evidence_terms or spec.query_terms
        if any(_contains_term(text, term) for term in evidence):
            confirmed.add(spec.id)
        elif any(_contains_term(text, term) for term in spec.weak_terms):
            weak_only.add(spec.id)
        elif spec.id in hits:
            view_only.add(spec.id)
    return {"confirmed": confirmed, "weak_only": weak_only, "view_only": view_only}


def tag_chunk_concepts(
    payload: ChunkPayload,
    *,
    concept_plan: ConceptPlan,
    view_concept_hits: set[str] | None = None,
) -> list[str]:
    """chunk가 '확정'한 concept id 목록(evidence_term 직접 등장만).

    SPLADE concept view-hit과 weak_term은 *약신호*일 뿐 concept 확정 근거가 아니다(거짓 joint 차단).
    예: '지능형 교육시스템'은 반도체 evidence가 없으므로 semiconductor로 확정되지 않는다.
    """
    signals = chunk_concept_signals(payload, concept_plan, view_concept_hits=view_concept_hits)
    confirmed = signals["confirmed"]
    return [c for c in concept_plan.all_concepts if c in confirmed]


def _doc_attrs_text(payload: ChunkPayload) -> str:
    values: list[Any] = []
    for value in (payload.doc_attrs or {}).values():
        if isinstance(value, (str, int, float)):
            values.append(value)
        elif isinstance(value, (list, tuple, set)):
            values.extend(value)
    return _normalize_text(*values)


def chunk_display_only_concepts(payload: ChunkPayload, concept_plan: ConceptPlan) -> list[str]:
    """evidence_term이 doc_attrs(title/keywords 등)엔 있으나 확정 텍스트(chunk_text/doc_id)엔 없는 concept.

    화면 제목은 doc_attrs에서 오므로 '제목엔 보이지만 확정 근거는 아닌'(거짓 confirmed 방지 설계의 부작용)
    concept를 진단/표시용으로 분리 노출한다. 점수/확정 태깅에는 영향 없다(scoring 무변경).
    """
    attrs = _doc_attrs_text(payload)
    if not attrs:
        return []
    body = _payload_text(payload)
    display_only: set[str] = set()
    for spec in concept_plan.specs:
        evidence = spec.evidence_terms or spec.query_terms
        in_attrs = any(_contains_term(attrs, term) for term in evidence)
        in_body = any(_contains_term(body, term) for term in evidence)
        if in_attrs and not in_body:
            display_only.add(spec.id)
    return [c for c in concept_plan.all_concepts if c in display_only]


# ---------------------------------------------------------------------------
# 멀티뷰 융합 점수 (raw score 합산 금지 — normalized rank score)
# ---------------------------------------------------------------------------
def view_rank_score(rank0: int, rrf_k: int) -> float:
    """0-base rank → 1/(rrf_k+rank). dense/SPLADE 스케일 차이를 제거."""
    return 1.0 / (rrf_k + max(0, rank0))


def view_weight_key(source: str) -> str:
    """source 이름 → view_weights 키. concept:<id> 류는 'sparse_concept'로 매핑."""
    return "sparse_concept" if source.startswith(CONCEPT_VIEW_PREFIX) else source


def fuse_chunk_score(
    view_best_rank0: dict[str, int],
    *,
    view_weights: dict[str, float],
    rrf_k: int,
) -> float:
    """chunk가 잡힌 각 view의 (weight × normalized rank score) 합산."""
    total = 0.0
    for source, rank0 in view_best_rank0.items():
        weight = view_weights.get(view_weight_key(source), 0.0)
        total += weight * view_rank_score(rank0, rrf_k)
    return total


# ---------------------------------------------------------------------------
# researcher capped evidence scoring
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ResearcherScore:
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    coverage_type: str = ""  # "joint" | "separate" | "partial"
    matched_concepts: list[str] = field(default_factory=list)
    evidence_by_concept: dict[str, list[ChunkHit]] = field(default_factory=dict)


def _chunk_value(hit: ChunkHit, doc_type_quality: dict[str, float]) -> float:
    return hit.score * doc_type_quality.get(hit.doc_type, 1.0)


# ---------------------------------------------------------------------------
# joint 토큰 변별 — 붙은 단일 합성어(예: '인공지능반도체대학원')가 두 required 개념을 한 토큰에서
# 동시에 confirm해 거짓 joint(×joint weight)를 만드는 것을 차단한다. 개별 concept 확정
# (tag_chunk_concepts)은 그대로 두고, joint는 'required 개념들이 서로 다른 토큰으로 확정될 때'만
# 인정한다. 한국어 정상 단일개념 합성어('차세대지능형반도체' 등)는 영향 없음.
# ---------------------------------------------------------------------------
_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _text_tokens(text: str) -> list[str]:
    """공백/문장부호로 분리한 단어 토큰(한글·영숫자 run). text는 이미 casefold/정규화 가정."""
    return _WORD_TOKEN_RE.findall(text)


def _concept_confirming_tokens(tokens: list[str], spec: ConceptSpec) -> set[str]:
    """evidence_term이 '한 토큰 안에' 등장하는 그 토큰 값들의 집합(값 기준 — 동일 토큰 반복은 1개로)."""
    evidence = spec.evidence_terms or spec.query_terms
    return {tok for tok in tokens if any(_contains_term(tok, term) for term in evidence)}


def _has_distinct_assignment(token_sets: list[set[str]]) -> bool:
    """각 개념에 '서로 다른 토큰'을 1:1 배정할 수 있는가(SDR/Hall) — augmenting path 매칭."""
    owner: dict[str, int] = {}  # token -> 점유 개념 index

    def assign(i: int, seen: set[str]) -> bool:
        for tok in token_sets[i]:
            if tok in seen:
                continue
            seen.add(tok)
            if tok not in owner or assign(owner[tok], seen):
                owner[tok] = i
                return True
        return False

    return all(assign(i, set()) for i in range(len(token_sets)))


def joint_eligible(payload: ChunkPayload, concept_plan: ConceptPlan) -> bool:
    """이 chunk가 required 개념들을 '서로 다른 토큰'으로 확정하는가(붙은 단일 토큰 거짓 joint 차단).

    required<2면 항상 True(joint 개념 자체가 단일). 텍스트에서 토큰을 못 잡거나(직접 주입 등)
    어떤 개념이 토큰 단위로 안 잡히면(멀티워드 evidence 등) 보수적으로 True(강등하지 않음).
    """
    required_specs = [s for s in concept_plan.specs if s.role == "required"]
    if len(required_specs) < 2:
        return True
    tokens = _text_tokens(_payload_text(payload))
    if not tokens:
        return True
    token_sets: list[set[str]] = []
    for spec in required_specs:
        toks = _concept_confirming_tokens(tokens, spec)
        if not toks:
            return True  # 토큰 단위 미검출 → 판단 불가, 보존
        token_sets.append(toks)
    return _has_distinct_assignment(token_sets)


def _confirming_token_values(
    hit: ChunkHit, concept: str, concept_plan: ConceptPlan, *, distinct: bool
) -> set[str]:
    """이 chunk가 concept을 확정하는 '토큰값' 집합.

    distinct=False거나(플래그 off) 텍스트에서 토큰을 못 잡으면(빈 텍스트/멀티워드 evidence) chunk·concept별
    synthetic distinct 토큰을 써서 기존 동작(개념별 독립 best)을 보존한다. distinct=True일 때만 실제
    토큰값을 공유 판정에 쓴다 — '인공지능반도체대학원' 같은 단일 합성 토큰이 두 개념을 동시 충족하는 것을 막는다.
    """
    spec = concept_plan.spec(concept) if distinct else None
    if spec is not None:
        toks = _concept_confirming_tokens(_text_tokens(_payload_text(hit.payload)), spec)
        if toks:
            return toks
    return {f"{hit.chunk_id}#{concept}"}


def _assign_distinct_token_evidence(
    chunks: list[ChunkHit],
    concepts: list[str],
    concept_plan: ConceptPlan,
    doc_type_quality: dict[str, float],
    *,
    distinct: bool,
) -> dict[str, tuple[float, str]]:
    """각 concept에 '서로 다른 토큰값'의 최고가치 증거를 1:1 배정(최대 cardinality 매칭).

    반환: concept -> (chunk_value, chunk_id). 같은 합성 토큰('인공지능반도체대학원')으로만 확정되는 두 개념은
    서로 다른 토큰을 못 잡아 한쪽만 배정된다(chunk가 여러 개여도 토큰값이 같으면 마찬가지) — 단일 chunk든
    중복 chunk든 거짓 다개념 충족/중복 계상을 차단한다. augmenting-path는 value 내림차순 인접으로 결정적.
    """
    units: dict[str, dict[str, tuple[float, str]]] = {}  # concept -> token -> (value, chunk_id)
    for hit in chunks:
        value = _chunk_value(hit, doc_type_quality)
        for concept in hit.concepts:
            if concept not in concepts:
                continue
            slot = units.setdefault(concept, {})
            for tok in _confirming_token_values(hit, concept, concept_plan, distinct=distinct):
                prev = slot.get(tok)
                if prev is None or value > prev[0]:
                    slot[tok] = (value, hit.chunk_id)

    adjacency = {
        c: [tok for tok, _ in sorted(units.get(c, {}).items(), key=lambda kv: (-kv[1][0], kv[0]))]
        for c in concepts
    }
    token_owner: dict[str, str] = {}

    def augment(concept: str, seen: set[str]) -> bool:
        for tok in adjacency.get(concept, []):
            if tok in seen:
                continue
            seen.add(tok)
            if tok not in token_owner or augment(token_owner[tok], seen):
                token_owner[tok] = concept
                return True
        return False

    for c in concepts:
        augment(c, set())
    return {concept: units[concept][tok] for tok, concept in token_owner.items()}


def score_researcher(
    chunks: list[ChunkHit],
    concept_plan: ConceptPlan,
    *,
    weights: dict[str, float],
    doc_type_quality: dict[str, float],
    support_top_k: int,
    weak_evidence_floor: float = 0.0,
    require_distinct_tokens_for_joint: bool = True,
) -> ResearcherScore:
    """관련 chunk(concept 태깅 완료)로 capped evidence score 계산.

    chunk_value = fused_score × doc_type_quality. 개념별 best + balance + joint + capped support.
    개념별 best/balance/concept는 '서로 다른 토큰값'으로 확정된 증거에만 배정(_assign_distinct_token_evidence)
    하여, 단일 합성 토큰('인공지능반도체대학원')이 두 개념을 동시 충족·중복 계상하는 것을 차단한다.
    배정/joint로 쓰인 chunk는 support에서 제외(중복 계상 방지).
    """
    required = concept_plan.required
    scope = list(required) if required else list(concept_plan.all_concepts)
    distinct = require_distinct_tokens_for_joint

    # 표시용 개념별 chunk(value desc) — 점수 배정과 무관하게 보존(evidence_by_concept).
    by_concept: dict[str, list[ChunkHit]] = {}
    for hit in chunks:
        for concept in hit.concepts:
            by_concept.setdefault(concept, []).append(hit)
    for items in by_concept.values():
        items.sort(key=lambda h: -_chunk_value(h, doc_type_quality))

    matched = [c for c in concept_plan.all_concepts if by_concept.get(c)]

    # 개념별 '서로 다른 토큰' 증거 배정: concept -> (chunk_value, chunk_id).
    assigned = _assign_distinct_token_evidence(
        chunks, scope, concept_plan, doc_type_quality, distinct=distinct
    )

    # joint = required 전부를 한 chunk가 동시충족 (붙은 단일 토큰 거짓 joint는 token 변별로 차단)
    joint_chunks = [
        h
        for h in chunks
        if required
        and all(c in h.concepts for c in required)
        and (not distinct or joint_eligible(h.payload, concept_plan))
    ]
    joint_chunks.sort(key=lambda h: -_chunk_value(h, doc_type_quality))
    joint_best = _chunk_value(joint_chunks[0], doc_type_quality) if joint_chunks else 0.0

    # required 충족 여부 (gate): 각 required 개념이 '독립 토큰'으로 배정되어야 충족.
    required_covered = all(c in assigned for c in required) if required else bool(assigned)
    if required_covered:
        coverage_type = "joint" if joint_chunks else "separate"
    else:
        coverage_type = "partial"

    # balance = min(required 개념 배정값). 한쪽 몰빵 억제.
    if required and required_covered:
        balance = min(assigned[c][0] for c in required)
    else:
        balance = 0.0
    # concept = 배정된 개념들의 값 합(서로 다른 토큰 → 중복 계상 없음).
    concept_sum = sum(assigned[c][0] for c in scope if c in assigned)

    # capped support: 배정/joint로 안 쓰인 관련 chunk를 chunk_value 내림차순 top-k, harmonic decay 합.
    used_chunk_ids = {chunk_id for _, chunk_id in assigned.values()}
    if joint_chunks:
        used_chunk_ids.add(joint_chunks[0].chunk_id)
    support_pool = sorted(
        (h for h in chunks if h.concepts and h.chunk_id not in used_chunk_ids),
        key=lambda h: -_chunk_value(h, doc_type_quality),
    )
    support = 0.0
    for index, hit in enumerate(support_pool[: max(0, support_top_k)]):
        support += _chunk_value(hit, doc_type_quality) / (index + 1)

    w = weights
    breakdown = {
        "joint": round(w.get("joint", 0.0) * joint_best, 6),
        "balance": round(w.get("balance", 0.0) * balance, 6),
        "concept": round(w.get("concept", 0.0) * concept_sum, 6),
        "support": round(w.get("support", 0.0) * support, 6),
    }
    score = sum(breakdown.values())

    # weak evidence 감점: 모든 근거가 floor 이하면 강등.
    if weak_evidence_floor > 0.0 and chunks:
        max_val = max(_chunk_value(h, doc_type_quality) for h in chunks)
        if max_val <= weak_evidence_floor:
            penalty = round(score * 0.5, 6)
            breakdown["weak_penalty"] = -penalty
            score -= penalty

    evidence_by_concept: dict[str, list[ChunkHit]] = {c: by_concept.get(c, []) for c in concept_plan.all_concepts if by_concept.get(c)}
    if joint_chunks:
        evidence_by_concept["joint"] = joint_chunks

    return ResearcherScore(
        score=round(score, 6),
        breakdown=breakdown,
        coverage_type=coverage_type,
        matched_concepts=matched,
        evidence_by_concept=evidence_by_concept,
    )
