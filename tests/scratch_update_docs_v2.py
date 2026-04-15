import os

# 1. Update SERVICE_FLOW.md
sf_path = r"/apps/docs/SERVICE_FLOW.md"
with open(sf_path, "r", encoding="utf-8") as f:
    sf_content = f.read()

sf_replacement_1 = """### 1-3. 검색용 문장을 4가지 관점으로 다시 만드는 단계

같은 질문이라도 보는 관점이 다르면 검색 문장도 달라져야 한다.

그래서 시스템은 한 번의 사용자 질문을 4개의 관점으로 다시 정리한다. **이때 가장 중요한 점은, 사용자의 원문 질문("화재 평가위원을 추천해줘")을 절대 통째로 사용하지 않는다는 것이다.**
대신 시스템은 "평가위원"이나 "추천" 같은 불용어가 삭제된 채 순수 기술 도메인으로만 정제된 LLM의 `core_keywords`(예: "화재 방재")만을 베이스 텍스트로 사용하여, 완전히 오염이 제거된 4개의 관점을 만들어낸다.

- `basic`: 전공, 학위, 소속유형, 직위, 기술분류 관점
- `art`: 논문 제목, 키워드, 초록, 최근 연구성과 관점
- `pat`: 특허명, 출원/등록, 사업화 관련 관점
- `pjt`: 과제명, 연구목표, 연구내용, 전문기관 관점

이렇게 만드는 이유는, 순수 도메인 지식만으로 검색을 시도해야 각 근거의 실제 기술적 강점이 제대로 매칭되기 때문이다."""

sf_content = sf_content.replace(
    '### 1-3. 검색용 문장을 4가지 관점으로 다시 만드는 단계\n\n같은 질문이라도 보는 관점이 다르면 검색 문장도 달라져야 한다.\n\n그래서 시스템은 한 번의 사용자 질문을 4개의 관점으로 다시 정리한다.\n\n- `basic`: 전공, 학위, 소속유형, 직위, 기술분류 관점\n- `art`: 논문 제목, 키워드, 초록, 최근 연구성과 관점\n- `pat`: 특허명, 출원/등록, 사업화 관련 관점\n- `pjt`: 과제명, 연구목표, 연구내용, 전문기관 관점\n\n이렇게 만드는 이유는, 같은 사람이라도 어느 근거에서 강점이 드러나는지가 다르기 때문이다.',
    sf_replacement_1
)

sf_replacement_2 = """- **JSON-only 출력 강제**: "첫 글자는 반드시 `{`이고 마지막 글자는 반드시 `}`" 규칙을 명시한다. 설명, 마크다운, 코드 펜스 출력을 금지한다.
- **메타 불용어 배제 원칙**: "평가위원", "추천해", "전문가", "찾아" 등과 같이 R&D 도메인이 아닌 서술어와 메타 직업군은 임베딩 벡터를 심각하게 오염시키므로, `core_keywords` 및 `branch_query_hints` 추출 시 절대 포함되지 않도록 명시적 배제 규칙을 갖는다."""

sf_content = sf_content.replace(
    '- **JSON-only 출력 강제**: "첫 글자는 반드시 `{`이고 마지막 글자는 반드시 `}`" 규칙을 명시한다. 설명, 마크다운, 코드 펜스 출력을 금지한다.',
    sf_replacement_2
)

with open(sf_path, "w", encoding="utf-8") as f:
    f.write(sf_content)


# 2. Update 지침서_v1.3.md
guide_path = r"/apps/docs/평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.3.md"
with open(guide_path, "r", encoding="utf-8") as f:
    guide_content = f.read()

guide_replacement_1 = """| 출력 원칙 | 추천 + 근거 + 제외 사유 + 데이터 공백 | 운영자 검토 가능성 확보 |
| 임베딩 정제 | 원문 질의 배제 및 순수 도메인 키워드 조합 | "평가위원" 등 불용어로 인한 임베딩 벡터 오염을 원천 차단하여 검색 무결성 보장 |"""

guide_content = guide_content.replace(
    '| 출력 원칙 | 추천 + 근거 + 제외 사유 + 데이터 공백 | 운영자 검토 가능성 확보 |',
    guide_replacement_1
)

guide_replacement_2 = """| Retrieval Orchestrator | dense/sparse 질의 생성, Qdrant prefetch 조립 | 원발성 질의(Raw Query)의 어투를 벗겨내고 순수 추출된 도메인 키워드로만 쿼리 뼈대를 조립하여 검색 노이즈 사전 차단 |"""

guide_content = guide_content.replace(
    '| Retrieval Orchestrator | dense/sparse 질의 생성, Qdrant prefetch 조립 | 검색 파이프라인의 핵심 제어층 |',
    guide_replacement_2
)

with open(guide_path, "w", encoding="utf-8") as f:
    f.write(guide_content)

print("Documentation update complete.")
