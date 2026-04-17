# Golden Tests

## Required Scenarios

### 1. Meta-term removal

Input:

- `AI 반도체 평가위원 추천`

Expected:

- `평가위원`, `추천` only appear in `removed_meta_terms`
- `retrieval_core` contains only domain terms
- raw meta terms are not re-injected into sparse retrieval text

### 2. Dense vs sparse query split

Input:

- planner returns both `retrieval_core` and `semantic_query`

Expected:

- dense retrieval uses `semantic_query`
- sparse retrieval uses `retrieval_core`
- query trace records the dense/sparse base source

### 3. Deterministic must-aspect pruning

Input:

- `retrieval_core = ["의료영상 분석", "AI 기반", "기술"]`
- `must_aspects = ["의료영상 분석", "AI 기반", "기술 개발"]`

Expected:

- `must_aspects` is pruned to the domain-specific phrases only
- generic phrases such as `AI 기반`, `기술 개발` do not survive as hard gates

### 4. Contextual evaluation phrase handling

Input:

- `AI 기반 의료영상 과제를 평가할 수 있는 전문가 추천`

Expected:

- `과제 평가` does not remain in `must_aspects`
- `intent_flags.review_context == true`
- `intent_flags.review_targets == ["과제 평가"]`

### 5. Phrase-based coverage threshold

Input:

- `evidence_aspects = ["medical imaging analysis", "의료영상 분석"]`

Expected:

- coverage threshold is `min(2, 2) = 2`, not token-count based
- threshold uses `evidence_aspects` length when available, not `must_aspects`

### 6. Selector / validator scope alignment

Input:

- selector matches an aspect in evidence `snippet` or `detail`, not in title

Expected:

- validator does not false-fallback on title-only grounds

### 7. Partial batch retry

Input:

- first reasoner batch returns partial output

Expected:

- one compact retry occurs
- unresolved candidates only then fall back to server-generated reasons

### 8. Future-dated projects

Input:

- direct-match project evidence with a future end date

Expected:

- evidence remains eligible
- trace includes `future_selected_evidence_ids`

### 9. Bilingual evidence_aspects matching

Input:

- `query = "AI 기반 의료영상 분석 과제 평가 전문가 추천"`
- researcher with English paper titles/abstracts about medical image analysis

Expected:

- `evidence_aspects` includes both Korean (`의료영상 분석`) and English (`medical image analysis`) terms
- English paper titles/abstracts matched via `evidence_aspects` English terms
- `aspect_source = "evidence_aspects"` in selector trace
- `direct_match_count > 0` for matching researcher

### 10. evidence_aspects fallback to must_aspects

Input:

- LLM-generated `PlannerOutput` with empty `evidence_aspects`

Expected:

- selector falls back to `must_aspects` automatically
- `aspect_source = "must_aspects"` in selector trace
- no error; behavior identical to pre-v0.7.0 pipeline

## Acceptance Criteria

- equal-weight RRF only
- stable / expanded path model remains two-path
- dense/sparse query split is visible in trace
- contextual evaluation phrases are intent flags, not hard lexical gates
- gate and validator use the same phrase/evidence scope assumptions
- `evidence_aspects` is bilingual; `retrieval_core` and `must_aspects` are Korean-only
- selector `aspect_source` trace field is present and correct
