"""relevance.py 순수 함수 테스트 — concept 해석/태깅, 멀티뷰 융합, capped evidence scoring.

설계 계약(사용자 spec):
- concept 미산출 시 FALLBACK alias로 질의에서 감지(질의 정제엔 미사용).
- 멀티뷰 융합 = raw score 합산 금지 → source별 normalized rank score × view_weight.
- researcher 점수 = required best + balance(한쪽 몰빵 억제) + joint(다개념 동시충족) + capped support.
"""
from __future__ import annotations

from apps.domain.models import ChunkHit, ChunkPayload, ConceptSpec, PlannerOutput
from apps.search.relevance import (
    CONCEPT_VIEW_PREFIX,
    ConceptPlan,
    fuse_chunk_score,
    has_operation_marker,
    resolve_concept_plan,
    score_researcher,
    tag_chunk_concepts,
    view_rank_score,
    view_weight_key,
)

WEIGHTS = {"joint": 1.5, "balance": 1.2, "concept": 0.8, "support": 0.3}
QUALITY = {"specialty": 1.0, "project": 1.0, "paper": 0.9, "patent": 0.8, "assessor_activity": 0.55}
VIEW_WEIGHTS = {"dense_full": 1.0, "sparse_raw": 0.25, "sparse_focus": 0.7, "sparse_concept": 0.5}


def _payload(rid: str, doc_type: str, chunk_id: str, text: str = "", doc_attrs: dict | None = None) -> ChunkPayload:
    return ChunkPayload(researcher_id=rid, researcher_name=rid, doc_type=doc_type,
                        chunk_id=chunk_id, chunk_text=text, doc_attrs=doc_attrs or {})


def _hit(rid: str, doc_type: str, chunk_id: str, concepts: list[str], score: float, text: str = "") -> ChunkHit:
    return ChunkHit(score=score, payload=_payload(rid, doc_type, chunk_id, text), concepts=concepts)


# registry 제거 후 ai+semiconductor 플랜은 planner concept_specs로 직접 구성(테스트 fixture).
_AI_SPEC = ConceptSpec(
    id="ai", label="인공지능", role="required",
    query_terms=["인공지능", "AI", "머신러닝", "딥러닝", "신경망"],
    evidence_terms=["인공지능", "AI", "머신러닝", "딥러닝", "신경망", "뉴로모픽", "LLM"],
    weak_terms=["지능형", "스마트", "자동화"],
)
_SEMI_SPEC = ConceptSpec(
    id="semiconductor", label="반도체", role="required",
    query_terms=["반도체", "시스템반도체", "반도체소자"],
    evidence_terms=["반도체", "시스템반도체", "AI반도체", "반도체소자", "집적회로", "웨이퍼"],
    weak_terms=["지능형", "시스템", "센서", "회로", "소자", "공정"],
)


def _ai_semi_plan() -> ConceptPlan:
    return resolve_concept_plan(
        PlannerOutput(intent_summary="x", concept_specs=[_AI_SPEC, _SEMI_SPEC]), "인공지능 반도체"
    )


# ---------------------------------------------------------------------------
# resolve_concept_plan
# ---------------------------------------------------------------------------
def test_resolve_concept_plan_none_when_no_known_concept():
    # retrieval_core도 없으면(합성 불가) source=none.
    plan = resolve_concept_plan(PlannerOutput(intent_summary="x"), "연구 산업 전문가 경험")
    assert plan.source == "none"
    assert plan.required == []


def test_resolve_concept_plan_query_exact_when_planner_and_registry_miss():
    # planner concept_specs 없음 + registry 밖 도메인 → retrieval_core를 query_exact(optional)로 합성.
    plan = resolve_concept_plan(
        PlannerOutput(intent_summary="x", retrieval_core=["메타물질", "음굴절"]),
        "메타물질 음굴절 광학 소자 전문가 추천",
    )
    assert plan.source == "query_exact"
    assert plan.required == []  # optional이라 required gate 미작동(omission 위험 0)
    assert set(plan.optional) == {"메타물질", "음굴절"}
    # evidence=term이므로 직접 등장 chunk는 확정 → concept 점수 작동
    payload = _payload("M", "paper", "p_c000", "메타물질 음굴절 광학 소자 연구")
    assert set(tag_chunk_concepts(payload, concept_plan=plan)) == {"메타물질", "음굴절"}


def test_resolve_concept_plan_prefers_planner_specs():
    # planner concept_specs가 있으면 최우선 사용(도메인 무관 — registry에 없는 concept도 동작).
    plan = resolve_concept_plan(
        PlannerOutput(
            intent_summary="x",
            concept_specs=[ConceptSpec(
                id="robotics", label="로봇",
                query_terms=["로봇 제어"], evidence_terms=["로봇", "manipulator"], weak_terms=["자동"],
            )],
        ),
        "로봇 제어 연구자",
    )
    assert plan.source == "planner"
    assert plan.required == ["robotics"]
    spec = plan.spec("robotics")
    assert spec is not None and "로봇" in spec.evidence_terms


# ---------------------------------------------------------------------------
# view fusion (normalized rank score, raw score 합산 금지)
# ---------------------------------------------------------------------------
def test_view_rank_score_and_weight_key():
    assert view_rank_score(0, 60) == 1.0 / 60
    assert view_rank_score(2, 60) == 1.0 / 62
    assert view_weight_key("dense_full") == "dense_full"
    assert view_weight_key(f"{CONCEPT_VIEW_PREFIX}semiconductor") == "sparse_concept"


def test_fuse_chunk_score_sums_weighted_rank_scores():
    # dense rank0=0 (w1.0) + concept:ai rank0=0 (w0.5) = 1/60 + 0.5/60
    fused = fuse_chunk_score(
        {"dense_full": 0, f"{CONCEPT_VIEW_PREFIX}ai": 0},
        view_weights=VIEW_WEIGHTS,
        rrf_k=60,
    )
    assert abs(fused - (1.0 / 60 + 0.5 / 60)) < 1e-9


def test_fuse_chunk_score_better_rank_scores_higher():
    top = fuse_chunk_score({"dense_full": 0}, view_weights=VIEW_WEIGHTS, rrf_k=60)
    deep = fuse_chunk_score({"dense_full": 9}, view_weights=VIEW_WEIGHTS, rrf_k=60)
    assert top > deep


# ---------------------------------------------------------------------------
# tag_chunk_concepts (multi-signal: view hit ∪ alias term)
# ---------------------------------------------------------------------------
def test_tag_chunk_concepts_view_hit_alone_does_not_tag_when_alias_exists():
    # SPLADE view 과확장 방지: alias가 있는 concept는 view hit 단독으로 태깅 금지(거짓 joint 차단).
    plan = _ai_semi_plan()
    payload = _payload("M1", "paper", "paper_1_c000", text="데이터 분석 연구")  # alias 없음
    assert tag_chunk_concepts(payload, view_concept_hits={"ai", "semiconductor"}, concept_plan=plan) == []


def test_tag_chunk_concepts_view_hit_never_confirms():
    # 모든 concept에 동일: view-hit only는 약신호이며 concept 확정 근거가 아니다.
    plan = ConceptPlan(
        specs=[ConceptSpec(id="robotics", label="로봇", query_terms=["로봇 제어"], evidence_terms=["로봇"])],
        source="planner",
    )
    payload = _payload("M1", "paper", "paper_1_c000", text="자율 제어 시스템")  # evidence(로봇) 없음
    assert tag_chunk_concepts(payload, view_concept_hits={"robotics"}, concept_plan=plan) == []


def test_tag_chunk_concepts_intelligent_system_not_tagged_semiconductor():
    # '지능형 X 시스템'은 반도체 alias가 없으므로 view hit가 있어도 semiconductor로 태깅하지 않는다.
    plan = _ai_semi_plan()
    payload = _payload("M1", "paper", "paper_1_c000", text="지능형 교육 시스템")
    tagged = tag_chunk_concepts(payload, view_concept_hits={"semiconductor"}, concept_plan=plan)
    assert "semiconductor" not in tagged  # 거짓 joint 차단 (지능형=ai alias라 ai만 가능)


def test_tag_chunk_concepts_via_alias_term_joint():
    plan = _ai_semi_plan()
    payload = _payload("M1", "paper", "paper_1_c000", text="인공지능 반도체 설계 연구")
    tagged = tag_chunk_concepts(payload, view_concept_hits=set(), concept_plan=plan)
    assert tagged == ["ai", "semiconductor"]


def test_tag_chunk_concepts_does_not_confirm_from_doc_attrs_only():
    # concept 확정은 chunk_text/doc_id의 직접 evidence만 사용한다. doc_attrs는 표시/상세 메타일 뿐
    # required gate 확정 근거로 쓰지 않는다.
    plan = _ai_semi_plan()
    payload = _payload(
        "M1", "patent", "patent_1_c000",
        text="지식재산권명: 저전력 신호처리 회로 설계",  # chunk_text엔 반도체 alias 없음
        doc_attrs={"intellectual_property_title": "시스템반도체 집적회로 설계", "keywords": "반도체;집적회로"},
    )
    assert tag_chunk_concepts(payload, view_concept_hits=set(), concept_plan=plan) == []


def test_tag_chunk_concepts_avoids_short_ascii_false_positive():
    plan = _ai_semi_plan()
    # 'ai'가 'aluminum'/'detail' 안에 들어가도 word-boundary로 오인하지 않는다.
    payload = _payload("M1", "patent", "patent_1_c000", text="aluminum detail trail")
    assert tag_chunk_concepts(payload, view_concept_hits=set(), concept_plan=plan) == []


def test_tag_chunk_concepts_user_cases():
    # 사장님이 제시한 5 케이스 (evidence 확정 / weak·view = 약신호).
    plan = _ai_semi_plan()

    def tag(text: str) -> list[str]:
        return tag_chunk_concepts(_payload("M", "paper", "p_c000", text), concept_plan=plan)

    assert tag("지능형 교육시스템") == []                     # 지능형=weak → 확정 X (거짓 joint 차단)
    assert tag("지능형 반도체 설계") == ["semiconductor"]       # 반도체 evidence O, 지능형은 ai weak
    assert tag("인공지능 반도체") == ["ai", "semiconductor"]     # joint
    assert tag("반도체 연구개발 경력") == ["semiconductor"]      # 반도체 evidence O, ai 없음
    assert tag("인공지능 기반 교통 시스템") == ["ai"]            # 인공지능 evidence O, 반도체 없음


def test_chunk_concept_signals_view_only_vs_weak_only():
    from apps.search.relevance import chunk_concept_signals

    plan = _ai_semi_plan()
    # evidence/weak 전혀 없는 텍스트 + semiconductor view hit → view_only (확정 X).
    sig = chunk_concept_signals(
        _payload("M", "paper", "p_c000", "data analysis report"),
        plan, view_concept_hits={"semiconductor"},
    )
    assert sig["confirmed"] == set()
    assert sig["view_only"] == {"semiconductor"}
    # '지능형'은 ai weak → weak_only (확정 X).
    sig2 = chunk_concept_signals(_payload("M", "paper", "p_c001", "지능형 플랫폼"), plan)
    assert "ai" in sig2["weak_only"] and "ai" not in sig2["confirmed"]


# ---------------------------------------------------------------------------
# score_researcher (balance / joint / partial / support cap)
# ---------------------------------------------------------------------------
def _score(chunks):
    plan = _ai_semi_plan()
    return score_researcher(chunks, plan, weights=WEIGHTS, doc_type_quality=QUALITY, support_top_k=3)


def test_score_balanced_outranks_onesided():
    balanced = _score([
        _hit("A", "paper", "paper_a1_c000", ["ai"], 0.5),
        _hit("A", "project", "project_a2_c000", ["semiconductor"], 0.45),
    ])
    onesided = _score([
        _hit("B", "paper", "paper_b1_c000", ["ai"], 0.6),
        _hit("B", "paper", "paper_b2_c000", ["semiconductor"], 0.1),
    ])
    assert balanced.coverage_type == "separate"
    assert onesided.coverage_type == "separate"
    assert balanced.score > onesided.score  # balance term이 한쪽 몰빵 억제


def test_score_joint_chunk_is_strongest():
    joint = _score([_hit("C", "specialty", "specialty_C_c000", ["ai", "semiconductor"], 0.5)])
    separate = _score([
        _hit("A", "paper", "paper_a1_c000", ["ai"], 0.5),
        _hit("A", "project", "project_a2_c000", ["semiconductor"], 0.5),
    ])
    assert joint.coverage_type == "joint"
    assert joint.breakdown["joint"] > 0.0
    assert joint.score > separate.score


def test_score_partial_when_required_missing():
    partial = _score([_hit("D", "paper", "paper_d_c000", ["ai"], 0.9)])
    assert partial.coverage_type == "partial"
    assert partial.matched_concepts == ["ai"]
    assert partial.breakdown["balance"] == 0.0  # 한쪽만 → balance 0


def test_score_support_excludes_best_chunks():
    # 같은 개념 best 1 + 보조 2: support는 best 외 chunk만(중복 계상 방지).
    scored = _score([
        _hit("E", "paper", "paper_e1_c000", ["ai"], 0.6),
        _hit("E", "paper", "paper_e2_c000", ["ai"], 0.3),
        _hit("E", "project", "project_e3_c000", ["semiconductor"], 0.5),
    ])
    assert scored.coverage_type == "separate"
    assert scored.breakdown["support"] > 0.0  # 보조 ai 근거가 support로 계상


def test_score_evidence_by_concept_populated():
    scored = _score([
        _hit("F", "paper", "paper_f1_c000", ["ai"], 0.5),
        _hit("F", "project", "project_f2_c000", ["semiconductor"], 0.4),
    ])
    assert set(scored.evidence_by_concept) >= {"ai", "semiconductor"}


# ---------------------------------------------------------------------------
# joint 토큰 변별 — 붙은 단일 합성어('인공지능반도체대학원')의 거짓 joint 차단
# ---------------------------------------------------------------------------
def _joint_hit(rid: str, chunk_id: str, text: str, score: float = 0.5, doc_type: str = "project") -> ChunkHit:
    # ai+semiconductor를 직접 부여하되, joint 적격성은 payload 텍스트(토큰)에서 재판정된다.
    return ChunkHit(score=score, payload=_payload(rid, doc_type, chunk_id, text),
                    concepts=["ai", "semiconductor"])


def test_glued_program_name_confirms_both_but_not_joint():
    # '인공지능반도체대학원' = 한 토큰에 두 개념 → 개별 confirm은 되나(태깅), 독립 토큰 없어 joint도
    # separate도 아니고 partial(semi 독립근거 없음).
    plan = _ai_semi_plan()
    text = "국문과제명: 인공지능반도체대학원 수행기관: 서울대학교"
    assert set(tag_chunk_concepts(_payload("M", "project", "project_g_c000", text),
                                  concept_plan=plan)) == {"ai", "semiconductor"}
    scored = _score([_joint_hit("M", "project_g_c000", text)])
    assert scored.coverage_type == "partial"
    assert scored.breakdown["joint"] == 0.0


def test_two_glued_chunks_same_compound_still_partial():
    # 관측 #2 김성철: '인공지능반도체대학원' chunk가 2개여도 같은 토큰값 → semi 독립근거 없음 → partial.
    # (chunk_id 기준 중복제거로는 못 막고, 토큰값 기준 배정이라야 차단된다.)
    plan = _ai_semi_plan()
    chunks = [
        ChunkHit(score=0.6, payload=_payload("M", "project", "project_k1_c000", "인공지능반도체대학원"),
                 concepts=["ai", "semiconductor"]),
        ChunkHit(score=0.5, payload=_payload("M", "project", "project_k2_c000", "인공지능반도체대학원"),
                 concepts=["ai", "semiconductor"]),
    ]
    scored = score_researcher(chunks, plan, weights=WEIGHTS, doc_type_quality=QUALITY, support_top_k=3)
    assert scored.coverage_type == "partial"
    assert scored.breakdown["joint"] == 0.0


def test_distinct_tokens_keep_joint():
    # '인공지능 ... 반도체 설계' = 서로 다른 토큰 → joint 유지.
    scored = _score([_joint_hit("M", "project_d_c000", "인공지능 기반 반도체 설계 연구")])
    assert scored.coverage_type == "joint"
    assert scored.breakdown["joint"] > 0.0


def test_glued_token_not_rescued_by_repetition():
    # 붙은 프로그램명이 본문에 2회 등장해도 동일 토큰값 → distinct 아님 → joint/separate 아님(partial).
    scored = _score([_joint_hit("M", "project_r_c000",
                                "인공지능반도체대학원 사업 인공지능반도체대학원 운영")])
    assert scored.coverage_type == "partial"
    assert scored.breakdown["joint"] == 0.0


def test_single_concept_glued_compound_still_joins_with_separate_ai():
    # '차세대지능형반도체'(붙은 단일 semiconductor 합성어) + 별도 '인공지능' 토큰 → distinct → joint.
    plan = _ai_semi_plan()
    text = "인공지능 응용 차세대지능형반도체 기술개발"
    assert set(tag_chunk_concepts(_payload("M", "project", "project_s_c000", text),
                                  concept_plan=plan)) == {"ai", "semiconductor"}
    scored = _score([_joint_hit("M", "project_s_c000", text)])
    assert scored.coverage_type == "joint"


def test_glued_program_outranked_by_real_joint():
    # 붙은 프로그램명(더 높은 raw score)이 실제 distinct-token joint보다 낮게 평가(관측 #2 vs #3 재현).
    glued = _score([_joint_hit("A", "project_a_c000", "인공지능반도체대학원", score=0.6)])
    real = _score([_joint_hit("B", "project_b_c000", "인공지능 프로세서 반도체 설계코드 개발", score=0.5)])
    assert glued.coverage_type == "partial"
    assert real.coverage_type == "joint"
    assert real.score > glued.score


def test_joint_distinct_token_guard_can_be_disabled():
    # 플래그 off면 기존 동작(붙은 단일 토큰도 joint).
    plan = _ai_semi_plan()
    hit = _joint_hit("M", "project_g_c000", "인공지능반도체대학원")
    off = score_researcher([hit], plan, weights=WEIGHTS, doc_type_quality=QUALITY,
                           support_top_k=3, require_distinct_tokens_for_joint=False)
    assert off.coverage_type == "joint"


# ---------------------------------------------------------------------------
# 운영성/교육/행정 과제 마커 (근거 가치 하향)
# ---------------------------------------------------------------------------
def test_has_operation_marker_detects_education_program():
    markers = frozenset({"대학원", "인력양성", "부트캠프"})
    # 붙은 합성어 안에서도 검출('인공지능반도체대학원' → '대학원')
    assert has_operation_marker(_payload("M", "project", "p_c000", "국문과제명: 인공지능반도체대학원"), markers)
    # 표기 공백차 흡수('전문인력 양성' despaced → '인력양성' 매칭)
    assert has_operation_marker(
        _payload("M", "project", "p_c001", "차세대시스템반도체설계 전문인력 양성"), frozenset({"인력양성"})
    )


def test_has_operation_marker_negative_for_design_project():
    markers = frozenset({"대학원", "인력양성", "부트캠프"})
    assert not has_operation_marker(
        _payload("M", "project", "p_c000", "지능형 반도체 설계 핵심기술 개발"), markers
    )
    assert not has_operation_marker(_payload("M", "project", "p_c001", "x"), frozenset())
