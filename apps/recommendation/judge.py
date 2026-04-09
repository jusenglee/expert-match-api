from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from apps.core.config import Settings
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.timer import Timer
from apps.domain.models import (
    CandidateCard,
    EvidenceItem,
    JudgeOutput,
    PlannerOutput,
    RecommendationDecision,
)

logger = logging.getLogger(__name__)


def _extract_json_object_text(content: Any) -> str:
    """Extract the first JSON object from an LLM response payload.

    Safely strips <thinking> blocks, markdown code fences, and extraneous text,
    then uses balanced-brace matching to isolate the top-level JSON object.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        text = "".join(parts)
    else:
        text = str(content)

    text = text.strip()

    # 1단계: <thinking>...</thinking> 블록 완전 제거
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    # 2단계: 마크다운 코드 블록 제거
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()
    else:
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```", "", text).strip()

    # 3단계: 균형 잡힌 중괄호 매칭으로 최상위 JSON 객체 추출
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # 균형 매칭 실패 시 폴백
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


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

        fit = normalized_item.get("fit")
        if isinstance(fit, str):
            stripped_fit = fit.strip()
            if stripped_fit != fit:
                normalized_item["fit"] = stripped_fit
                normalized_applied = True

        rank_score = normalized_item.get("rank_score")
        if expert_id:
            card = next((c for c in shortlist if c.expert_id == expert_id), None)
            if card:
                if rank_score is None:
                    normalized_item["rank_score"] = card.rank_score
                    normalized_applied = True

                if normalized_item.get("organization") != card.organization:
                    normalized_item["organization"] = card.organization
                    normalized_applied = True

        normalized_recommended.append(normalized_item)

    normalized_payload["recommended"] = normalized_recommended
    return normalized_payload, normalized_applied


def _merge_unique_strings(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


class Judge(Protocol):
    async def judge(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput: ...


class HeuristicJudge:
    async def judge(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        ranked_cards = sorted(
            shortlist, key=lambda card: card.shortlist_score, reverse=True
        )
        recommendations: list[RecommendationDecision] = []
        global_data_gaps: list[str] = []

        for rank, card in enumerate(ranked_cards[: max(plan.top_k, 3)], start=1):
            reasons: list[str] = []
            evidence: list[EvidenceItem] = []

            if card.branch_presence_flags.get("art"):
                reasons.append("논문실적(데이터) 존재가 확인됩니다.")
                if card.top_papers:
                    evidence.append(
                        EvidenceItem(
                            type="paper",
                            title=card.top_papers[0].publication_title,
                            date=card.top_papers[0].publication_year_month,
                            detail=card.top_papers[0].journal_name,
                        )
                    )

            if card.branch_presence_flags.get("pat"):
                reasons.append("특허실적(데이터) 존재가 확인됩니다.")
                if card.top_patents:
                    evidence.append(
                        EvidenceItem(
                            type="patent",
                            title=card.top_patents[0].intellectual_property_title,
                            date=card.top_patents[0].registration_date
                            or card.top_patents[0].application_date,
                            detail=card.top_patents[0].application_registration_type,
                        )
                    )

            if card.branch_presence_flags.get("pjt"):
                reasons.append("과제수행(데이터) 존재가 확인됩니다.")
                if card.top_projects:
                    evidence.append(
                        EvidenceItem(
                            type="project",
                            title=card.top_projects[0].display_title,
                            date=card.top_projects[0].project_end_date
                            or card.top_projects[0].project_start_date,
                            detail=card.top_projects[0].managing_agency,
                        )
                    )

            if card.branch_presence_flags.get("basic"):
                reasons.append("전문가 프로필(데이터) 존재가 확인됩니다.")
                evidence.append(
                    EvidenceItem(
                        type="profile",
                        title=f"{card.name} / {card.organization or '소속 미상'}",
                        detail=f"{card.degree or '학위 미상'} / {card.major or '전공 미상'}",
                    )
                )

            if card.data_gaps:
                global_data_gaps.extend(card.data_gaps)

            fit = (
                "높음"
                if card.shortlist_score >= 80
                else "중간"
                if card.shortlist_score >= 50
                else "보통"
            )
            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=card.expert_id,
                    name=card.name,
                    organization=card.organization,
                    fit=fit,
                    reasons=reasons[:3],
                    evidence=evidence[:4],
                    rank_score=card.rank_score,
                )
            )

        unique_gaps = list(dict.fromkeys(global_data_gaps))
        not_selected_reasons: list[str] = []
        if len(ranked_cards) > len(recommendations):
            not_selected_reasons.append(
                "상위 추천 대비 근거 다양성이나 최신성이 상대적으로 낮습니다."
            )

        return JudgeOutput(
            recommended=recommendations[:5],
            not_selected_reasons=not_selected_reasons,
            data_gaps=unique_gaps,
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

    @staticmethod
    def _build_system_prompt() -> str:
        """최종 Reduce/Single 라운드용 상세 프롬프트입니다.
        추천 사유, 근거, 리스크 등 완전한 심사 결과를 생성합니다."""
        return """
            You are the senior recommendation judge for a Korean R&D evaluator recommendation system.
            Return exactly one JSON object that matches this schema:
            {
              "recommended": [
                {
                  "rank": 1,
                  "expert_id": "shortlist expert_id",
                  "name": "shortlist name",
                  "organization": "optional organization",
                  "fit": "높음|중간|보통",
                  "reasons": ["reason"],
                  "evidence": [
                    {
                      "type": "paper|patent|project|profile",
                      "title": "exact shortlist evidence title",
                      "date": "optional date",
                      "detail": "optional detail"
                    }
                  ],
                  "risks": ["risk"],
                  "rank_score": 0.0
                }
              ],
              "not_selected_reasons": ["reason"],
              "data_gaps": ["gap"]
            }

            Rules:
            - IMPORTANT: You MUST recommend up to the number of candidates specified in the plan's `top_k` field. Do NOT just return 1 candidate if there are multiple suitable fits.
            - IMPORTANT: `rank_score` is a RELATIVE score within this search result (RRF-based), not an absolute suitability percentage. Do not misuse it as a probability.
            - IMPORTANT: `branch_presence_flags` only indicate that data exists in that branch. It does not guarantee a perfect match. Verify exact suitability from `top_papers`, `top_patents`, etc.
            - Copy `expert_id` exactly from the shortlist input.
            - Every recommendation must include `rank`, `expert_id`, `name`, `organization`, `fit`, `reasons`, `evidence`, and `risks`.
            - `reasons`, `risks`, `not_selected_reasons`, and `data_gaps` must always be arrays of strings, never a single string.
            - Only use evidence that already exists in `top_papers`, `top_patents`, `top_projects`, or the profile fields of the shortlist input.
            - If no candidate should be recommended, return `recommended=[]` and explain why in `not_selected_reasons` and/or `data_gaps`.
            - Do not return markdown, code fences, or any prose outside the JSON object.
        """

    @staticmethod
    def _build_map_system_prompt() -> str:
        """중간 Map 라운드용 경량 프롬프트입니다.
        상세 사유 없이 생존자 expert_id 목록만 빠르게 반환합니다."""
        return (
            "You are a fast screening filter for a Korean R&D evaluator recommendation system.\n"
            "Your job: from the given shortlist, select the candidates most relevant to the query.\n"
            "Return ONLY a JSON object. No markdown, no explanation.\n"
            "First character must be '{', last character must be '}'.\n\n"
            "Output schema:\n"
            '{"survivors":[{"expert_id":"...","rank":1},{"expert_id":"...","rank":2}]}\n\n'
            "Rules:\n"
            "- Select up to the number specified in plan.top_k (or more if many strong candidates exist).\n"
            "- Copy expert_id exactly from the shortlist input.\n"
            "- Rank by relevance to the query (1 = best).\n"
            "- ONLY output the JSON object. No reasons, no evidence, no explanation.\n"
        )

    def _serialize_shortlist(
        self, shortlist: list[CandidateCard]
    ) -> list[dict[str, Any]]:
        """Reduce/Single 라운드용 전체 직렬화 (상세 근거 포함)."""
        dumped_shortlist = [card.model_dump(mode="json") for card in shortlist]
        for card_dict in dumped_shortlist:
            for paper in card_dict.get("top_papers", []):
                if paper.get("abstract"):
                    paper["abstract"] = paper["abstract"][:150] + "..."
                if paper.get("korean_keywords"):
                    paper["korean_keywords"] = paper["korean_keywords"][:5]
                if paper.get("english_keywords"):
                    paper["english_keywords"] = paper["english_keywords"][:5]
            for proj in card_dict.get("top_projects", []):
                if proj.get("research_objective_summary"):
                    proj["research_objective_summary"] = (
                        proj["research_objective_summary"][:150] + "..."
                    )
                if proj.get("research_content_summary"):
                    proj["research_content_summary"] = (
                        proj["research_content_summary"][:150] + "..."
                    )
        return dumped_shortlist

    @staticmethod
    def _serialize_shortlist_for_map(
        shortlist: list[CandidateCard],
    ) -> list[dict[str, Any]]:
        """Map 라운드용 경량 직렬화.
        LLM이 관련성 판단에 필요한 최소 정보만 포함하여 입력 토큰을 절약합니다."""
        lightweight: list[dict[str, Any]] = []
        for card in shortlist:
            entry: dict[str, Any] = {
                "expert_id": card.expert_id,
                "name": card.name,
                "organization": card.organization,
                "degree": card.degree,
                "major": card.major,
                "rank_score": card.rank_score,
                "counts": card.counts,
                "keyword_matched_counts": card.keyword_matched_counts,
                "branch_coverage": card.branch_coverage,
            }
            # 제목만 간결하게 포함 (abstract, 키워드 등 제외)
            if card.top_papers:
                entry["paper_titles"] = [
                    p.publication_title for p in card.top_papers
                ]
            if card.top_patents:
                entry["patent_titles"] = [
                    p.intellectual_property_title for p in card.top_patents
                ]
            if card.top_projects:
                entry["project_titles"] = [
                    p.display_title for p in card.top_projects
                ]
            lightweight.append(entry)
        return lightweight

    @staticmethod
    def _build_chunks(
        shortlist: list[CandidateCard], batch_size: int
    ) -> list[list[CandidateCard]]:
        return [
            shortlist[i : i + batch_size]
            for i in range(0, len(shortlist), batch_size)
        ]

    def _log_shortlist_token_estimate(
        self, dumped_shortlist: list[dict[str, Any]], *, context_label: str
    ) -> None:
        total_estimated_tokens = 0
        token_breakdown = []
        for card_dict in dumped_shortlist:
            char_len = len(json.dumps(card_dict, ensure_ascii=False))
            est_tokens = int(char_len / 2.5)
            total_estimated_tokens += est_tokens
            token_breakdown.append(f"{card_dict.get('name', 'Unknown')}({est_tokens}t)")

        chunked_breakdown = [
            ", ".join(token_breakdown[i : i + 5])
            for i in range(0, len(token_breakdown), 5)
        ]
        logger.info(
            "데이터당 예상 토큰 크기: context=%s 총 추정=%d 토큰 후보 수=%d\n  %s",
            context_label,
            total_estimated_tokens,
            len(dumped_shortlist),
            "\n  ".join(chunked_breakdown) if chunked_breakdown else "(empty)",
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
    ) -> tuple[Any, float]:
        async with self._judge_semaphore:
            logger.info(
                "LLM 호출 슬롯 획득: context=%s round=%d batch=%d/%d max_concurrency=%d max_tokens=%s",
                context_label,
                round_index,
                batch_index,
                batch_count,
                self.settings.llm_judge_max_concurrency,
                max_tokens_hint or "unlimited",
            )
            invoke_kwargs: dict[str, Any] = {}
            if max_tokens_hint is not None:
                invoke_kwargs["max_tokens_hint"] = max_tokens_hint
            with Timer() as t:
                result = await self.model.ainvoke_non_stream(messages, **invoke_kwargs)
            return result, t.elapsed_ms

    @staticmethod
    def _parse_map_response(
        raw_payload: dict[str, Any], shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        """Map 라운드의 경량 응답을 JudgeOutput으로 변환합니다.
        survivors 배열에서 expert_id만 추출하여 최소한의 RecommendationDecision을 생성합니다.
        LLM이 recommended 형식으로 반환한 경우에도 호환 처리합니다."""
        survivors = raw_payload.get("survivors", [])
        if not isinstance(survivors, list):
            survivors = []
        # Fallback: LLM이 기존 recommended 형식으로 응답한 경우 호환 처리
        if not survivors:
            recommended = raw_payload.get("recommended", [])
            if isinstance(recommended, list) and recommended:
                survivors = recommended
                logger.info("Map 응답 호환 처리: 'recommended' 키를 'survivors'로 대체 사용 (항목 수=%d)", len(survivors))

        card_lookup: dict[str, CandidateCard] = {c.expert_id: c for c in shortlist}
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
    ) -> JudgeOutput:
        is_map_phase = context_label == "map"

        # Map 라운드: 경량 직렬화 + 경량 프롬프트 + max_tokens 제한
        if is_map_phase:
            dumped_shortlist = self._serialize_shortlist_for_map(shortlist)
            system_prompt = self._build_map_system_prompt()
            max_tokens_hint = 400
        else:
            dumped_shortlist = self._serialize_shortlist(shortlist)
            system_prompt = self._build_system_prompt()
            max_tokens_hint = None

        self._log_shortlist_token_estimate(
            dumped_shortlist,
            context_label=context_label,
        )

        user_payload = {
            "query": query,
            "plan": plan.model_dump(mode="json"),
            "shortlist": dumped_shortlist,
        }
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
        ]

        normalized_recommendation_count = 0
        try:
            logger.info(
                "LLM 판정 시작: context=%s round=%d batch=%d/%d 후보 수=%d max_tokens=%s",
                context_label,
                round_index,
                batch_index,
                batch_count,
                len(shortlist),
                max_tokens_hint or "unlimited",
            )
            result, elapsed_ms = await self._invoke_non_stream_with_limit(
                messages,
                context_label=context_label,
                round_index=round_index,
                batch_index=batch_index,
                batch_count=batch_count,
                max_tokens_hint=max_tokens_hint,
            )
            json_text = _extract_json_object_text(result.content)
            logger.info(
                "LLM 응답 수신 완료: context=%s round=%d batch=%d/%d 소요시간=%.2fms",
                context_label,
                round_index,
                batch_index,
                batch_count,
                elapsed_ms,
            )
            logger.info(
                "LLM 응답 텍스트(추출됨): context=%s round=%d batch=%d/%d %s",
                context_label,
                round_index,
                batch_index,
                batch_count,
                json_text,
            )

            raw_payload = json.loads(json_text)

            # Map 라운드: 경량 파싱 (survivors 배열만 처리)
            if is_map_phase:
                output = self._parse_map_response(raw_payload, shortlist)
                logger.info(
                    "Map 스크리닝 완료: context=%s round=%d batch=%d/%d 생존자=%d명 소요시간=%.2fms",
                    context_label,
                    round_index,
                    batch_index,
                    batch_count,
                    len(output.recommended),
                    elapsed_ms,
                )
                return output

            # Reduce/Single 라운드: 기존 상세 파싱
            normalized_payload, normalized_applied = _normalize_judge_payload(
                raw_payload, shortlist
            )

            if isinstance(normalized_payload, dict):
                recommended = normalized_payload.get("recommended", [])
                normalized_recommendation_count = (
                    len(recommended) if isinstance(recommended, list) else 0
                )

            if normalized_applied:
                logger.debug(
                    "판정 결과 정규화 적용: context=%s round=%d batch=%d/%d 후보 수=%d -> 최종 추천 수=%d",
                    context_label,
                    round_index,
                    batch_index,
                    batch_count,
                    len(shortlist),
                    normalized_recommendation_count,
                )

            output = JudgeOutput.model_validate(normalized_payload)
            logger.info(
                "최종 판정 성공: context=%s round=%d batch=%d/%d 최종 추천=%d명",
                context_label,
                round_index,
                batch_index,
                batch_count,
                len(output.recommended),
            )
            return output
        except Exception as exc:
            logger.warning(
                "판정 Fallback 활성화: context=%s round=%d batch=%d/%d 사유=%s 후보 수=%d",
                context_label,
                round_index,
                batch_index,
                batch_count,
                exc,
                len(shortlist),
            )
            return await self.fallback.judge(
                query=query, plan=plan, shortlist=shortlist
            )

    async def _judge_batched(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        batch_size = max(1, self.settings.llm_judge_batch_size)
        logger.info(
            "LLM 최종 판정(Judging) 시작: 내부 배치 병렬 심사 (후보 수=%d, batch_size=%d, max_concurrency=%d)",
            len(shortlist),
            batch_size,
            self.settings.llm_judge_max_concurrency,
        )
        if batch_size <= plan.top_k:
            logger.warning(
                "Judge batch_size가 top_k 이하입니다. 축소 단계가 정체될 수 있습니다: batch_size=%d top_k=%d",
                batch_size,
                plan.top_k,
            )

        round_index = 1
        current_shortlist = shortlist
        merged_not_selected_reasons: list[str] = []
        merged_data_gaps: list[str] = []

        while len(current_shortlist) > batch_size:
            chunks = self._build_chunks(current_shortlist, batch_size)
            logger.info(
                "Judge map round 시작: round=%d candidates=%d batches=%d batch_size=%d max_concurrency=%d",
                round_index,
                len(current_shortlist),
                len(chunks),
                batch_size,
                self.settings.llm_judge_max_concurrency,
            )
            map_outputs = await asyncio.gather(
                *[
                    self._judge_single_shortlist(
                        query=query,
                        plan=plan,
                        shortlist=chunk,
                        context_label="map",
                        round_index=round_index,
                        batch_index=index + 1,
                        batch_count=len(chunks),
                    )
                    for index, chunk in enumerate(chunks)
                ],
                return_exceptions=False,
            )

            winner_ids: set[str] = set()
            for output in map_outputs:
                winner_ids.update(rec.expert_id for rec in output.recommended)
                merged_not_selected_reasons.extend(output.not_selected_reasons)
                merged_data_gaps.extend(output.data_gaps)

            if not winner_ids:
                logger.info(
                    "Judge map round 종료: round=%d survivors=0",
                    round_index,
                )
                return JudgeOutput(
                    recommended=[],
                    not_selected_reasons=_merge_unique_strings(
                        merged_not_selected_reasons
                    ),
                    data_gaps=_merge_unique_strings(merged_data_gaps),
                )

            next_shortlist = [
                card for card in current_shortlist if card.expert_id in winner_ids
            ]
            logger.info(
                "Judge map round 종료: round=%d survivors=%d",
                round_index,
                len(next_shortlist),
            )
            if len(next_shortlist) >= len(current_shortlist):
                logger.warning(
                    "Judge map round가 shortlist를 축소하지 못했습니다. 최종 단일 라운드로 전환합니다: round=%d candidates=%d",
                    round_index,
                    len(current_shortlist),
                )
                break

            current_shortlist = next_shortlist
            round_index += 1

        logger.info(
            "Judge final round 시작: round=%d candidates=%d batch_size=%d max_concurrency=%d",
            round_index,
            len(current_shortlist),
            batch_size,
            self.settings.llm_judge_max_concurrency,
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

    async def judge(
        self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]
    ) -> JudgeOutput:
        if not shortlist:
            return JudgeOutput()

        batch_size = max(1, self.settings.llm_judge_batch_size)
        if not self.settings.use_map_reduce_judging:
            logger.info(
                "LLM 최종 판정(Judging) 시작: 단일 심사 (후보 수=%d, internal batching disabled)",
                len(shortlist),
            )
            return await self._judge_single_shortlist(
                query=query,
                plan=plan,
                shortlist=shortlist,
                context_label="single",
                round_index=1,
                batch_index=1,
                batch_count=1,
            )

        if len(shortlist) <= batch_size:
            logger.info(
                "LLM 최종 판정(Judging) 시작: 단일 심사 (후보 수=%d, batch_size=%d)",
                len(shortlist),
                batch_size,
            )
            return await self._judge_single_shortlist(
                query=query,
                plan=plan,
                shortlist=shortlist,
                context_label="single",
                round_index=1,
                batch_index=1,
                batch_count=1,
            )

        return await self._judge_batched(
            query=query,
            plan=plan,
            shortlist=shortlist,
        )
