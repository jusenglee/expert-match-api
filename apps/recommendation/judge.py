from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.timer import Timer
from apps.core.utils import build_deterministic_seed
from apps.core.utils import merge_unique_strings as _merge_unique_strings
from apps.domain.models import (
    CandidateCard,
    EvidenceItem,
    JudgeOutput,
    PlannerOutput,
    RecommendationDecision,
)

logger = logging.getLogger(__name__)

MAP_PHASE_MAX_TOKENS = 3000


def _normalize_string_list(value: Any) -> list[str] | Any:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        normalized_items: list[str] = []
        for item in value:
            if item is None:
                continue
            normalized = item.strip() if isinstance(item, str) else str(item).strip()
            if normalized:
                normalized_items.append(normalized)
        return normalized_items
    return value


def _normalize_rank(value: Any, fallback_rank: int) -> int | Any:
    if value is None:
        return fallback_rank
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback_rank
        try:
            return int(stripped)
        except ValueError:
            return value
    return value


def _normalize_judge_payload(
    payload: Any, shortlist: list[CandidateCard]
) -> tuple[Any, bool]:
    if not isinstance(payload, dict):
        return payload, False

    normalized_payload = dict(payload)
    normalized_applied = False
    name_to_expert_ids: dict[str, list[str]] = {}
    for card in shortlist:
        normalized_name = card.name.strip()
        name_to_expert_ids.setdefault(normalized_name, []).append(card.expert_id)

    for key in ("not_selected_reasons", "data_gaps"):
        normalized_value = _normalize_string_list(normalized_payload.get(key))
        if normalized_value != normalized_payload.get(key):
            normalized_payload[key] = normalized_value
            normalized_applied = True

    recommended = normalized_payload.get("recommended")
    if recommended is None:
        normalized_payload["recommended"] = []
        return normalized_payload, True
    if not isinstance(recommended, list):
        return normalized_payload, normalized_applied

    normalized_recommended: list[Any] = []
    for fallback_rank, item in enumerate(recommended, start=1):
        if not isinstance(item, dict):
            return normalized_payload, normalized_applied

        normalized_item = dict(item)
        rank = _normalize_rank(normalized_item.get("rank"), fallback_rank)
        if rank != normalized_item.get("rank"):
            normalized_item["rank"] = rank
            normalized_applied = True

        for key in ("reasons", "risks"):
            normalized_value = _normalize_string_list(normalized_item.get(key))
            if normalized_value != normalized_item.get(key):
                normalized_item[key] = normalized_value
                normalized_applied = True

        name = normalized_item.get("name")
        if isinstance(name, str):
            stripped_name = name.strip()
            if stripped_name != name:
                normalized_item["name"] = stripped_name
                normalized_applied = True
            name = stripped_name

        expert_id = normalized_item.get("expert_id")
        if isinstance(expert_id, str):
            stripped_expert_id = expert_id.strip()
            if stripped_expert_id != expert_id:
                normalized_item["expert_id"] = stripped_expert_id
                normalized_applied = True
            expert_id = stripped_expert_id

        if not expert_id and isinstance(name, str):
            matches = name_to_expert_ids.get(name, [])
            if len(matches) == 1:
                normalized_item["expert_id"] = matches[0]
                normalized_applied = True
                expert_id = matches[0]

        fit = normalized_item.get("fit")
        if isinstance(fit, str):
            stripped_fit = fit.strip()
            if stripped_fit != fit:
                normalized_item["fit"] = stripped_fit
                normalized_applied = True

        if expert_id:
            card = next(
                (candidate for candidate in shortlist if candidate.expert_id == expert_id),
                None,
            )
            if card:
                if normalized_item.get("rank_score") is None:
                    normalized_item["rank_score"] = card.rank_score
                    normalized_applied = True
                if normalized_item.get("organization") != card.organization:
                    normalized_item["organization"] = card.organization
                    normalized_applied = True

        normalized_recommended.append(normalized_item)

    normalized_payload["recommended"] = normalized_recommended
    return normalized_payload, normalized_applied


class Judge(Protocol):
    async def judge(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput: ...


class HeuristicJudge:
    """Deterministic score-based fallback judge."""

    def __init__(self) -> None:
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _card_sort_key(card: CandidateCard) -> tuple[float, float]:
        return (card.rank_score, card.shortlist_score)

    @staticmethod
    def _build_evidence(card: CandidateCard) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if card.branch_presence_flags.get("art") and card.top_papers:
            paper = card.top_papers[0]
            evidence.append(
                EvidenceItem(
                    type="paper",
                    title=paper.publication_title,
                    date=paper.publication_year_month,
                    detail=paper.journal_name,
                )
            )
        if card.branch_presence_flags.get("pat") and card.top_patents:
            patent = card.top_patents[0]
            evidence.append(
                EvidenceItem(
                    type="patent",
                    title=patent.intellectual_property_title,
                    date=patent.registration_date or patent.application_date,
                    detail=patent.application_registration_type,
                )
            )
        if card.branch_presence_flags.get("pjt") and card.top_projects:
            project = card.top_projects[0]
            evidence.append(
                EvidenceItem(
                    type="project",
                    title=project.display_title,
                    date=project.project_end_date or project.project_start_date,
                    detail=project.managing_agency,
                )
            )
        if card.branch_presence_flags.get("basic"):
            evidence.append(
                EvidenceItem(
                    type="profile",
                    title=f"{card.name} / {card.organization or 'unknown organization'}",
                    detail=f"{card.degree or 'unknown degree'} / {card.major or 'unknown major'}",
                )
            )
        return evidence

    async def judge(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        ranked_cards = sorted(shortlist, key=self._card_sort_key, reverse=True)
        recommendations: list[RecommendationDecision] = []
        global_data_gaps: list[str] = []

        for rank, card in enumerate(ranked_cards[: max(plan.top_k, 1)], start=1):
            evidence = self._build_evidence(card)
            reasons = [
                reason
                for reason in (
                    "Profile data is available."
                    if card.branch_presence_flags.get("basic")
                    else None,
                    "Publication evidence is available."
                    if card.branch_presence_flags.get("art")
                    else None,
                    "Patent evidence is available."
                    if card.branch_presence_flags.get("pat")
                    else None,
                    "Project evidence is available."
                    if card.branch_presence_flags.get("pjt")
                    else None,
                )
                if reason is not None
            ]
            global_data_gaps.extend(card.data_gaps)
            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=card.expert_id,
                    name=card.name,
                    organization=card.organization,
                    fit="중간",
                    reasons=reasons[:3],
                    evidence=evidence[:4],
                    rank_score=card.rank_score,
                )
            )

        self.last_trace = {
            "mode": "deterministic_fallback",
            "candidate_count": len(shortlist),
            "returned_count": len(recommendations),
        }
        return JudgeOutput(
            recommended=recommendations[: plan.top_k],
            not_selected_reasons=[],
            data_gaps=_merge_unique_strings(global_data_gaps),
        )


class OpenAICompatJudge:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = HeuristicJudge()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._judge_semaphore = asyncio.Semaphore(
            max(1, settings.llm_judge_max_concurrency)
        )
        self.last_trace: dict[str, Any] = {}
        self._round_trace: list[dict[str, Any]] = []

    @staticmethod
    def _build_system_prompt(query: str, dumped_shortlist: list[dict[str, Any]]) -> str:
        """최종 판정(Judging)을 위한 고도화된 시스템 프롬프트를 생성합니다.

        적용된 기법:
        - Role Prompting: '수석 평가위원 매칭 시스템' 페르소나
        - Zero-Hallucination: 엄격한 데이터 기반 추천 원칙
        - Explicit Schema: JSON 출력 구조 명시
        - Dented String: textwrap.dedent 적용
        """
        shortlist_json = json.dumps(dumped_shortlist, ensure_ascii=False, indent=2)
        prompt = f"""
            # Role
            당신은 제공된 전문가 이력 데이터를 바탕으로 사용자의 요구사항에 가장 부합하는 전문가를 선별하고 추천하는 '수석 평가위원 매칭 시스템'입니다.
            당신의 목표는 사용자 질문의 핵심 키워드와 전문가의 실제 이력을 엄격하게 교차 검증하여, 가장 적합한 순서대로 추천 결과를 제공하는 것입니다.

            # Instructions
            1. 의도 및 키워드 추출: 아래 [사용자 질문]에서 핵심 기술 및 도메인 키워드를 추출하세요. (예: '반도체', '공정 자동화')
            2. 엄격한 필터링: [후보 전문가 데이터]에 명시된 경력, 프로젝트, 연구 분야가 추출된 키워드와 직접적으로 연관된 전문가만 선별하세요. 
               관련성이 모호하거나 데이터에 없는 내용을 유추하여 추천해서는 절대 안 됩니다.
            3. 관련성 점수 부여 및 정렬 (Ranking): 선별된 전문가들이 사용자 질문과 얼마나 일치하는지 100점 만점 기준으로 점수를 계산하고, 높은 순으로 정렬하세요.
            4. 증거 제시 - 1 : 추천 이유(reasons)와 근거(evidence)는 반드시 제공된 데이터에 기반하여 구체적으로 작성하세요.
            5. 증거 제시 - 2 : 추천 이유(reasons)와 근거(evidence)는 반드시 사용자의 질문 (User Query) 과 연관이 있어야합니다.
            
            # Constraints
            - 출력은 반드시 유효한 단일 JSON 객체여야 합니다.
            - `expert_id`는 숏리스트 입력에서 정확히 복사하십시오.
            - 증거(evidence)의 `type`은 반드시 "paper", "patent", "project", "profile" 중 하나여야 합니다. (약어 금지)
            - 추천할 후보자가 없는 경우 `recommended=[]`를 반환하고, 이유를 `not_selected_reasons`에 설명하십시오.
            - 마크다운이나 코드 펜스를 사용하지 마십시오.

            # Output Schema
            {{
              "recommended": [
                {{
                  "rank": 1,
                  "expert_id": "string",
                  "name": "string",
                  "organization": "string",
                  "fit": "높음|중간|보통",
                  "reasons": ["string"],
                  "evidence": [
                    {{
                      "type": "paper|patent|project|profile",
                      "title": "string",
                      "date": "string",
                      "detail": "string"
                    }}
                  ],
                  "risks": ["string"],
                  "rank_score": 0.0
                }}
              ],
              "not_selected_reasons": ["string"],
              "data_gaps": ["string"]
            }}

            # User Query
            {query}

            # Candidate Expert Data
            {shortlist_json}
            
            
            규칙:
                - 중요: 계획의 `top_k` 필드에 지정된 후보자 수 이하로 반드시 추천해야 합니다. 적합한 후보가 여러 명일 경우 1명만 반환하지 마십시오.
                - 중요: `branch_presence_flags`는 해당 브랜치에 데이터가 존재한다는 것만 나타냅니다. 완벽한 일치를 보장하지는 않습니다. `top_papers`, `top_patents` 등에서 정확한 적합성을 검증하십시오.
                - 중요: 증거(evidence) 항목의 `type`은 반드시 "paper", "patent", "project", "profile" 중 하나여야 합니다. "pjt", "art", "pat"과 같은 브랜치 약어를 사용하지 마십시오. 연구 과제는 반드시 "project"를 사용해야 합니다.
                - `expert_id`는 숏리스트(shortlist) 입력에서 정확히 복사하십시오.
                - 모든 추천에는 `rank`, `expert_id`, `name`, `organization`, `fit`, `reasons`, `evidence`, `risks`가 반드시 포함되어야 합니다.
                - `reasons`, `risks`, `not_selected_reasons`, `data_gaps`는 단일 문자열이 아닌, 반드시 문자열 배열(arrays of strings)이어야 합니다.
                - 숏리스트 입력의 `top_papers`, `top_patents`, `top_projects` 또는 프로필 필드에 이미 존재하는 증거만 사용하십시오.
                - 보수적으로 평가하십시오. 명시되지 않은 자격, 소속, 전문 분야 또는 증거를 임의로 추론하지 마십시오.
                - 숏리스트의 증거가 빈약하거나 모호한 경우, 후보자를 제외하거나 위험(risk)/데이터 공백(data gap)으로 기록하는 것을 우선하십시오.
                - 추천할 후보자가 없는 경우 `recommended=[]`를 반환하고, 그 이유를 `not_selected_reasons` 및/또는 `data_gaps`에 설명하십시오.
                - JSON 객체 외에 마크다운, 코드 블록(code fences) 또는 기타 텍스트를 절대 반환하지 마십시오."
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _build_map_system_prompt() -> str:
        return (
            "당신은 한국 R&D 평가위원 추천 시스템의 고속 선별 필터(fast screening filter)입니다.\n"
            "당신의 역할: 주어진 숏리스트(shortlist)에서 쿼리와 가장 관련성이 높은 후보자를 선택하는 것입니다.\n"
            "오직 JSON 객체만 반환하십시오. 마크다운이나 설명은 포함하지 마십시오.\n"
            "첫 번째 문자는 '{'이어야 하고, 마지막 문자는 '}'이어야 합니다.\n\n"
            "출력 스키마:\n"
            '{"survivors":[{"expert_id":"...","rank":1},{"expert_id":"...","rank":2}]}\n\n'
            "규칙:\n"
            "- plan.top_k에 지정된 수만큼 선택하십시오 (단, 강력한 후보가 많을 경우 더 선택할 수 있습니다).\n"
            "- expert_id는 숏리스트 입력에서 정확히 복사하십시오.\n"
            "- 중요: 입력에 제공된 rank_score는 고도로 최적화된 검색 엔진의 순위를 반영하므로 이를 전적으로 신뢰해야 합니다.\n"
            "- 쿼리와의 관련성을 검증하되, 생존자(survivors)의 순위는 주로 rank_score를 기반으로 할당하십시오 (높은 점수 = 순위 1 등).\n"
            "- 보수적으로 평가하십시오. 숏리스트의 증거가 쿼리를 명확하게 뒷받침하는 경우에만 후보자를 유지하십시오.\n"
            "- 부분적이거나 관련 없는 제목에서 명시되지 않은 전문성을 임의로 추론하지 마십시오.\n"
            "- 오직 JSON 객체만 출력하십시오. 이유, 증거, 설명은 절대 포함하지 마십시오.\n"
        )

    @staticmethod
    def _card_sort_key(card: CandidateCard) -> tuple[float, float]:
        return (card.rank_score, card.shortlist_score)

    @classmethod
    def _compress_cards_by_rank(
        cls, cards: list[CandidateCard], limit: int
    ) -> list[CandidateCard]:
        if limit <= 0:
            return []
        return sorted(cards, key=cls._card_sort_key, reverse=True)[:limit]

    @staticmethod
    def _serialize_shortlist(shortlist: list[CandidateCard]) -> list[dict[str, Any]]:
        dumped_shortlist = [
            card.model_dump(mode="json", exclude_none=True) for card in shortlist
        ]
        for card_dict in dumped_shortlist:
            for paper in card_dict.get("top_papers", []):
                if paper.get("abstract"):
                    paper["abstract"] = paper["abstract"][:80] + "..."
                if paper.get("korean_keywords"):
                    paper["korean_keywords"] = paper["korean_keywords"][:3]
                if paper.get("english_keywords"):
                    paper["english_keywords"] = paper["english_keywords"][:3]
            for project in card_dict.get("top_projects", []):
                if project.get("research_objective_summary"):
                    project["research_objective_summary"] = (
                        project["research_objective_summary"][:80] + "..."
                    )
                if project.get("research_content_summary"):
                    project["research_content_summary"] = (
                        project["research_content_summary"][:80] + "..."
                    )
        return dumped_shortlist

    @staticmethod
    def _serialize_shortlist_for_map(
        shortlist: list[CandidateCard],
    ) -> list[dict[str, Any]]:
        lightweight: list[dict[str, Any]] = []
        for card in shortlist:
            entry: dict[str, Any] = {
                "expert_id": card.expert_id,
                "name": card.name,
                "organization": card.organization,
                "degree": card.degree,
                "major": card.major,
                "rank_score": card.rank_score,
                "shortlist_score": card.shortlist_score,
                "counts": card.counts,
                "branch_coverage": card.branch_coverage,
            }
            if card.top_papers:
                entry["paper_titles"] = [
                    paper.publication_title for paper in card.top_papers
                ]
            if card.top_patents:
                entry["patent_titles"] = [
                    patent.intellectual_property_title for patent in card.top_patents
                ]
            if card.top_projects:
                entry["project_titles"] = [
                    project.display_title for project in card.top_projects
                ]
            lightweight.append(entry)
        return lightweight

    @staticmethod
    def _build_chunks(
        shortlist: list[CandidateCard], batch_size: int
    ) -> list[list[CandidateCard]]:
        return [
            shortlist[index : index + batch_size]
            for index in range(0, len(shortlist), batch_size)
        ]

    @staticmethod
    def _estimate_shortlist_tokens(
        dumped_shortlist: list[dict[str, Any]],
    ) -> tuple[int, list[str]]:
        estimated_tokens = 0
        breakdown: list[str] = []
        for card_dict in dumped_shortlist:
            char_len = len(json.dumps(card_dict, ensure_ascii=False))
            card_tokens = int(char_len / 2.5)
            estimated_tokens += card_tokens
            breakdown.append(f"{card_dict.get('name', 'Unknown')}({card_tokens}t)")
        return estimated_tokens, breakdown

    def _log_shortlist_token_estimate(
        self, dumped_shortlist: list[dict[str, Any]], *, context_label: str
    ) -> int:
        estimated_tokens, breakdown = self._estimate_shortlist_tokens(dumped_shortlist)
        logger.info(
            "Judge token estimate: context=%s total=%d candidates=%d %s",
            context_label,
            estimated_tokens,
            len(dumped_shortlist),
            ", ".join(breakdown[:5]),
        )
        return estimated_tokens

    @staticmethod
    def _reduce_candidate_limit(plan: PlannerOutput) -> int:
        return max(1, plan.top_k + 1)

    @staticmethod
    def _reduce_token_limit() -> int:
        return MAP_PHASE_MAX_TOKENS * 2

    def _evaluate_reduce_gate(
        self, shortlist: list[CandidateCard], plan: PlannerOutput
    ) -> dict[str, Any]:
        dumped_shortlist = self._serialize_shortlist(shortlist)
        token_estimate, _ = self._estimate_shortlist_tokens(dumped_shortlist)
        candidate_limit = self._reduce_candidate_limit(plan)
        token_limit = self._reduce_token_limit()

        reasons: list[str] = []
        if len(shortlist) > candidate_limit:
            reasons.append("candidate_limit")
        if token_estimate > token_limit:
            reasons.append("token_limit")

        return {
            "allowed": not reasons,
            "reason": "_and_".join(reasons) if reasons else "ready",
            "candidate_count": len(shortlist),
            "candidate_limit": candidate_limit,
            "token_estimate": token_estimate,
            "token_limit": token_limit,
        }

    @staticmethod
    def _target_survivor_count(
        current_count: int,
        reduce_candidate_limit: int,
        gate_reason: str,
    ) -> int:
        if current_count <= 1:
            return current_count
        if gate_reason == "token_limit" and current_count <= reduce_candidate_limit:
            return current_count - 1
        target = max(reduce_candidate_limit, current_count // 2)
        if target >= current_count:
            return current_count - 1
        return max(1, target)

    @staticmethod
    def _allocate_chunk_limits(
        chunk_sizes: list[int], target_total: int
    ) -> list[int]:
        if not chunk_sizes:
            return []

        total_size = sum(chunk_sizes)
        capped_target = max(0, min(target_total, total_size))
        if capped_target == 0:
            return [0 for _ in chunk_sizes]

        raw_allocations = [
            (size * capped_target) / total_size for size in chunk_sizes
        ]
        limits = [
            min(size, int(allocation))
            for size, allocation in zip(chunk_sizes, raw_allocations)
        ]
        remainder = capped_target - sum(limits)
        fractions = [
            allocation - int(allocation) for allocation in raw_allocations
        ]
        order = sorted(
            range(len(chunk_sizes)),
            key=lambda index: (fractions[index], chunk_sizes[index], -index),
            reverse=True,
        )
        for index in order:
            if remainder <= 0:
                break
            if limits[index] < chunk_sizes[index]:
                limits[index] += 1
                remainder -= 1
        return limits

    @staticmethod
    def _ordered_cards_from_ids(
        cards: list[CandidateCard], expert_ids: list[str]
    ) -> list[CandidateCard]:
        lookup = {card.expert_id: card for card in cards}
        ordered: list[CandidateCard] = []
        seen: set[str] = set()
        for expert_id in expert_ids:
            if expert_id in seen:
                continue
            card = lookup.get(expert_id)
            if card is None:
                continue
            ordered.append(card)
            seen.add(expert_id)
        return ordered

    @staticmethod
    def _compute_call_seed(
        *,
        query: str,
        plan: PlannerOutput,
        shortlist: list[CandidateCard],
        context_label: str,
        round_index: int,
        batch_index: int,
        batch_count: int,
    ) -> int:
        return build_deterministic_seed(
            "judge",
            context_label,
            round_index,
            batch_index,
            batch_count,
            query,
            plan.model_dump(mode="json"),
            [card.expert_id for card in shortlist],
        )

    async def _invoke_non_stream_with_limit(
        self,
        messages: list[SystemMessage | HumanMessage],
        *,
        context_label: str,
        round_index: int,
        batch_index: int,
        batch_count: int,
        max_tokens_hint: int | None = None,
        seed: int,
    ) -> tuple[Any, float]:
        async with self._judge_semaphore:
            logger.info(
                "LLM judge call: context=%s round=%d batch=%d/%d max_tokens=%s seed=%d",
                context_label,
                round_index,
                batch_index,
                batch_count,
                max_tokens_hint or "unlimited",
                seed,
            )
            invoke_kwargs = build_consistency_invoke_kwargs(
                max_tokens_hint=max_tokens_hint,
                seed=seed,
            )
            with Timer() as timer:
                result = await self.model.ainvoke_non_stream(messages, **invoke_kwargs)
            return result, timer.elapsed_ms

    @staticmethod
    def _parse_map_response(
        raw_payload: dict[str, Any], shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        survivors = raw_payload.get("survivors", [])
        if not isinstance(survivors, list):
            survivors = []

        if not survivors:
            recommended = raw_payload.get("recommended", [])
            if isinstance(recommended, list) and recommended:
                survivors = recommended

        card_lookup = {card.expert_id: card for card in shortlist}
        recommendations: list[RecommendationDecision] = []
        for fallback_rank, entry in enumerate(survivors, start=1):
            if not isinstance(entry, dict):
                continue
            expert_id = entry.get("expert_id", "")
            if isinstance(expert_id, str):
                expert_id = expert_id.strip()
            if not expert_id:
                continue
            card = card_lookup.get(expert_id)
            rank = entry.get("rank", fallback_rank)
            if isinstance(rank, str):
                try:
                    rank = int(rank.strip())
                except ValueError:
                    rank = fallback_rank
            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=expert_id,
                    name=card.name if card else "",
                    organization=card.organization if card else None,
                    fit="보통",
                    reasons=[],
                    evidence=[],
                    risks=[],
                    rank_score=card.rank_score if card else 0.0,
                )
            )
        return JudgeOutput(recommended=recommendations)

    async def _judge_single_shortlist(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        shortlist: list[CandidateCard],
        context_label: str,
        round_index: int,
        batch_index: int,
        batch_count: int,
        selection_limit_override: int | None = None,
    ) -> JudgeOutput:
        is_map_phase = context_label == "map"
        seed = self._compute_call_seed(
            query=query,
            plan=plan,
            shortlist=shortlist,
            context_label=context_label,
            round_index=round_index,
            batch_index=batch_index,
            batch_count=batch_count,
        )

        if is_map_phase:
            dumped_shortlist = self._serialize_shortlist_for_map(shortlist)
            system_prompt = self._build_map_system_prompt()
            max_tokens_hint = MAP_PHASE_MAX_TOKENS
            selection_limit = selection_limit_override
            user_payload = {
                "query": query,
                "plan": plan.model_dump(mode="json"),
                "shortlist": dumped_shortlist,
                "selection_limit": selection_limit,
            }
        else:
            dumped_shortlist = self._serialize_shortlist(shortlist)
            system_prompt = self._build_system_prompt(query, dumped_shortlist)
            max_tokens_hint = None
            selection_limit = None
            user_payload = {
                "query": query,
                "plan": plan.model_dump(mode="json"),
                "shortlist": dumped_shortlist,
            }

        token_estimate = self._log_shortlist_token_estimate(
            dumped_shortlist, context_label=context_label
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
        ]

        try:
            result, elapsed_ms = await self._invoke_non_stream_with_limit(
                messages,
                context_label=context_label,
                round_index=round_index,
                batch_index=batch_index,
                batch_count=batch_count,
                max_tokens_hint=max_tokens_hint,
                seed=seed,
            )
            json_text = _extract_json_object_text(result.content)
            raw_payload = json.loads(json_text)

            if is_map_phase:
                output = self._parse_map_response(raw_payload, shortlist)
            else:
                normalized_payload, _ = _normalize_judge_payload(raw_payload, shortlist)
                output = JudgeOutput.model_validate(normalized_payload)

            self._round_trace.append(
                {
                    "context": context_label,
                    "round": round_index,
                    "batch": batch_index,
                    "batch_count": batch_count,
                    "candidate_count": len(shortlist),
                    "output_count": len(output.recommended),
                    "seed": seed,
                    "max_tokens_hint": max_tokens_hint,
                    "selection_limit": selection_limit,
                    "token_estimate": token_estimate,
                    "status": "ok",
                    "elapsed_ms": elapsed_ms,
                }
            )
            return output
        except Exception as exc:
            logger.warning(
                "Judge fallback activated: context=%s round=%d batch=%d/%d reason=%s",
                context_label,
                round_index,
                batch_index,
                batch_count,
                exc,
            )
            fallback_output = await self.fallback.judge(
                query=query,
                plan=plan,
                shortlist=shortlist,
            )
            self._round_trace.append(
                {
                    "context": context_label,
                    "round": round_index,
                    "batch": batch_index,
                    "batch_count": batch_count,
                    "candidate_count": len(shortlist),
                    "output_count": len(fallback_output.recommended),
                    "seed": seed,
                    "max_tokens_hint": max_tokens_hint,
                    "selection_limit": selection_limit,
                    "token_estimate": token_estimate,
                    "status": "fallback",
                    "reason": str(exc),
                }
            )
            return fallback_output

    async def _judge_batched(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        batch_size = max(1, self.settings.llm_judge_batch_size)
        round_index = 1
        current_shortlist = shortlist
        merged_not_selected_reasons: list[str] = []
        merged_data_gaps: list[str] = []

        while True:
            gate = self._evaluate_reduce_gate(current_shortlist, plan)
            if gate["allowed"]:
                logger.info(
                    "Reduce gate opened: candidates=%d candidate_limit=%d token_estimate=%d token_limit=%d",
                    gate["candidate_count"],
                    gate["candidate_limit"],
                    gate["token_estimate"],
                    gate["token_limit"],
                )
                final_output = await self._judge_single_shortlist(
                    query=query,
                    plan=plan,
                    shortlist=current_shortlist,
                    context_label="reduce",
                    round_index=round_index,
                    batch_index=1,
                    batch_count=1,
                )
                self.last_trace = {
                    "mode": "map_reduce",
                    "rounds": list(self._round_trace),
                    "final_reduce_candidate_count": len(current_shortlist),
                    "final_reduce_candidate_limit": gate["candidate_limit"],
                    "final_reduce_token_estimate": gate["token_estimate"],
                    "final_reduce_token_limit": gate["token_limit"],
                    "final_reduce_gate_reason": gate["reason"],
                }
                return JudgeOutput(
                    recommended=final_output.recommended,
                    not_selected_reasons=_merge_unique_strings(
                        merged_not_selected_reasons,
                        final_output.not_selected_reasons,
                    ),
                    data_gaps=_merge_unique_strings(
                        merged_data_gaps,
                        final_output.data_gaps,
                    ),
                )

            target_survivor_count = self._target_survivor_count(
                len(current_shortlist),
                gate["candidate_limit"],
                gate["reason"],
            )
            chunks = self._build_chunks(current_shortlist, batch_size)
            chunk_limits = self._allocate_chunk_limits(
                [len(chunk) for chunk in chunks], target_survivor_count
            )

            round_jobs = [
                (index, chunk, limit)
                for index, (chunk, limit) in enumerate(zip(chunks, chunk_limits), start=1)
                if limit > 0
            ]
            map_outputs = await asyncio.gather(
                *[
                    self._judge_single_shortlist(
                        query=query,
                        plan=plan,
                        shortlist=chunk,
                        context_label="map",
                        round_index=round_index,
                        batch_index=index,
                        batch_count=len(round_jobs),
                        selection_limit_override=limit,
                    )
                    for index, chunk, limit in round_jobs
                ],
                return_exceptions=False,
            )

            seen_ids: set[str] = set()
            post_llm_cards: list[CandidateCard] = []
            forced_compression_applied = False

            for (batch_index, chunk, limit), output in zip(round_jobs, map_outputs):
                selected_cards = self._ordered_cards_from_ids(
                    chunk,
                    [
                        recommendation.expert_id
                        for recommendation in output.recommended
                        if recommendation.expert_id
                    ],
                )
                merged_not_selected_reasons.extend(output.not_selected_reasons)
                merged_data_gaps.extend(output.data_gaps)

                if not selected_cards or len(selected_cards) > limit:
                    logger.warning(
                        "Forced compression applied: round=%d batch=%d limit=%d output=%d",
                        round_index,
                        batch_index,
                        limit,
                        len(selected_cards),
                    )
                    selected_cards = self._compress_cards_by_rank(chunk, limit)
                    forced_compression_applied = True

                for card in selected_cards:
                    if card.expert_id in seen_ids:
                        continue
                    seen_ids.add(card.expert_id)
                    post_llm_cards.append(card)

            post_llm_count = len(post_llm_cards)
            next_cards = post_llm_cards

            if not next_cards:
                next_cards = self._compress_cards_by_rank(
                    current_shortlist, target_survivor_count
                )
                forced_compression_applied = True

            if len(next_cards) > target_survivor_count:
                next_cards = self._compress_cards_by_rank(
                    next_cards, target_survivor_count
                )
                forced_compression_applied = True

            if len(next_cards) >= len(current_shortlist):
                logger.warning(
                    "Round contraction missed: round=%d input=%d post_llm=%d target=%d",
                    round_index,
                    len(current_shortlist),
                    post_llm_count,
                    target_survivor_count,
                )
                next_cards = self._compress_cards_by_rank(
                    current_shortlist, target_survivor_count
                )
                forced_compression_applied = True

            self._round_trace.append(
                {
                    "context": "map_round_summary",
                    "round": round_index,
                    "input_count": len(current_shortlist),
                    "target_survivor_count": target_survivor_count,
                    "post_llm_count": post_llm_count,
                    "post_compression_count": len(next_cards),
                    "reduce_gate_reason": gate["reason"],
                    "reduce_candidate_limit": gate["candidate_limit"],
                    "reduce_token_estimate": gate["token_estimate"],
                    "reduce_token_limit": gate["token_limit"],
                    "forced_compression_applied": forced_compression_applied,
                }
            )

            current_shortlist = next_cards
            round_index += 1

    async def judge(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        self._round_trace = []

        if not shortlist:
            self.last_trace = {"mode": "empty", "rounds": []}
            return JudgeOutput()

        batch_size = max(1, self.settings.llm_judge_batch_size)
        gate = self._evaluate_reduce_gate(shortlist, plan)

        if not self.settings.use_map_reduce_judging:
            output = await self._judge_single_shortlist(
                query=query,
                plan=plan,
                shortlist=shortlist,
                context_label="single",
                round_index=1,
                batch_index=1,
                batch_count=1,
            )
            self.last_trace = {"mode": "single", "rounds": list(self._round_trace)}
            return output

        if len(shortlist) <= batch_size and gate["allowed"]:
            output = await self._judge_single_shortlist(
                query=query,
                plan=plan,
                shortlist=shortlist,
                context_label="single",
                round_index=1,
                batch_index=1,
                batch_count=1,
            )
            self.last_trace = {
                "mode": "single",
                "rounds": list(self._round_trace),
                "final_reduce_candidate_count": len(shortlist),
                "final_reduce_candidate_limit": gate["candidate_limit"],
                "final_reduce_token_estimate": gate["token_estimate"],
                "final_reduce_token_limit": gate["token_limit"],
                "final_reduce_gate_reason": gate["reason"],
            }
            return output

        return await self._judge_batched(query=query, plan=plan, shortlist=shortlist)
