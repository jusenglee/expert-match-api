# WO-0 부속 산출물 · 네이밍 정리 가이드 (researcher_id ↔ expert_id)

> WO-0 §4-D 산출물. **이 WO에서는 코드를 바꾸지 않는다**(비파괴). 내부/외부 식별자 경계를
> 매핑표로 확정하고, 실제 일괄 치환은 **WO-C**가 이 표를 권위로 수행한다.
> 외부 API 필드(`expert_id`)의 변경/alias 여부는 **EXTERNAL_API_CHANGELOG.md / API_SPECIFICATION.md를
> 권위로 WO-C에서 BREAKING 여부와 함께 결정**한다.

## 1. 방침 결정

- v2.0 데이터 1층 키는 **`researcher_id`**(DATA_MODEL §3.1, 샘플 payload)이며, chunk→연구자 집계·dedupe의 기준 키다.
- 따라서 **내부 식별자는 `researcher_id`로 통일**한다. 검색결과·카드·근거 번들 등 파이프라인 내부에서 현재 `expert_id`로 부르는 것을 WO-C에서 `researcher_id`로 정렬한다.
- **외부 응답 필드** `expert_id`(`/recommend`, `/search/candidates` 응답)는 외부 계약이다. 내부를 `researcher_id`로 통일하더라도 응답 필드명을 바꿀지(BREAKING) 또는 응답 직렬화 시점에 `researcher_id→expert_id` alias로 매핑할지는 WO-C에서 결정한다(권위: EXTERNAL_API_CHANGELOG.md).

## 2. 경계 매핑표 (작성 시점 SPLADE 브랜치)

### 저장/적재 측 — 이미 `researcher_id` (변경 없음)
| 심볼 | 위치 |
|---|---|
| `BasicInfo.researcher_id` | `apps/domain/models.py:62` |
| `SeedEvidencePoint.researcher_id` | `apps/domain/models.py:184` |
| payload 인덱스 `basic_info.researcher_id` | `apps/search/schema_registry.py` (인덱스 필드 정의) |
| 샘플 payload L1 `researcher_id` | `apps/docs/전체 샘플 payload (도메인별).txt` |

### 검색결과/API/카드/근거 측 — 현재 `expert_id` (WO-C 치환 검토 대상)
| 심볼 | 위치 | 내부/외부 |
|---|---|---|
| `SearchHit.expert_id` | `apps/domain/models.py:222` | 내부 |
| `GroupedSearchHit.expert_id` | `apps/domain/models.py:235` | 내부 |
| `CandidateCard.expert_id` | `apps/domain/models.py:249` | 내부 |
| `RelevantEvidenceBundle.expert_id` | `apps/recommendation/evidence_selector.py:35` | 내부 |
| `RecommendationDecision.expert_id` | `apps/domain/models.py:272` | **외부(응답)** |
| `bundles[candidate.expert_id]` 등 사용처 | `evidence_selector.py:148/153/157/...`, `service.py`(번들 dict 키), `cards.py`, `reasoner.py`(expert_id 키) | 내부 |

> WO-C는 위 "내부" 항목을 `researcher_id`로 정렬하고, "외부" 항목은 changelog 권위로 별도 처리.

## 3. `service._chunk_candidates` 개명 (WO-0에서 완료)

- `apps/recommendation/service.py`의 `_chunk_candidates` → **`_batch_candidates`** (순수 리네이밍, 동작 불변).
- 정의(:510) + 호출부(:392, :531) 동반 수정 완료. `rg "_chunk_candidates" apps/` → 0건.
- 근거: v2.0에서 "chunk"는 1 Point=1 chunk라는 1급 도메인 용어가 되므로 "후보 배치 분할"을 chunk로 부르면 의미 충돌.

## 4. WO-A / WO-C 연결

- **WO-A**: 소스→chunk 변환 시 `apps/search/doc_types.py`의 `build_chunk_id`/`build_doc_id`로 `chunk_id`/`doc_id` 생성. `researcher_id`는 1층 키로 그대로 사용.
- **WO-C**: 위 "내부" `expert_id`를 `researcher_id`로 정렬, `evidence_selector`의 `item_id`(`paper:N` 등)를 `chunk_id`로 교체, `reasoner.VALID_EVIDENCE_ID_PATTERN`을 `chunk_id` 패턴으로 교체. 모두 `doc_types.py`를 단일 의존점으로 사용.
