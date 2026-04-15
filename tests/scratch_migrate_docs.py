import os

old_file = r"d:\Project\python_project\Ntis_person_API\apps\docs\평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.2.md"
new_file = r"/apps/docs/평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.3.md"

with open(old_file, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Version Update
content = content.replace(
    "문서 버전:** v1.2",
    "문서 버전:** v1.3"
).replace(
    "v1.2 (Map-Reduce 심사, JSON 추출 방어, Planner 프롬프트 개선 반영, 2026-04-09)",
    "v1.3 (LLM 구조적 토큰 다이어트, 기관명 정규화 튜닝, 동적 최신성 가산점 반영, 2026-04-14)"
)

# 2. Section 5.3 Payload and Filter
content = content.replace(
    "- 기관 필터링은 후처리가 아니라 `must_not` + `MatchValue`로 Qdrant 내에서 처리한다.",
    "- 기관 필터링은 파이썬 단에서의 단순 후처리가 아니라, `normalize_org_name` 파이프라인에서 정규화된 키워드(MatchValue)를 통해 Qdrant 내비게이션 단계에서 사전에 철저히 배제되도록 처리한다."
)

# 3. Section 6.3 LLM Role
content = content.replace(
    "- **Map 라운드(예비 심사)**: `max_tokens=3000`, 10명 단위의 배치로 분할, 최소한의 정보(초록 제외) 파싱.",
    "- **Map 라운드(예비 심사)**: `max_tokens=3000`, 20명 단위의 병렬 배치로 분할, 토큰 과부하와 불필요한 추론을 차단하기 위한 컨텍스트 통제 (초록 완전 제외)."
).replace(
    "- **Reduce 라운드(상세 심사)**: 논문/과제 초록 등 전체 컨텍스트 입력. 이유와 증거 생성",
    "- **Reduce 라운드(상세 심사)**: 논문/과제 초록 80자 하드 리밋 및 핵심 키워드 3개 이하로 강력한 모델카드 다이어트를 적용. 또한 LLM의 생성 답변 토큰(Generation Tokens)을 줄여 응답 속도를 비약적으로 높이기 위해 `reasons`, `evidence`를 최대 2개 이하로 제한."
)

with open(new_file, "w", encoding="utf-8") as f:
    f.write(content)

# Delete old file
os.remove(old_file)

print("Migration to v1.3 complete")
