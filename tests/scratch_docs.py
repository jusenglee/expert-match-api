import os
import re

planner_path = r"/apps/recommendation/planner.py"
with open(planner_path, "r", encoding="utf-8") as f:
    planner_content = f.read()

# 1. Modify JSON Schema instruction for core_keywords
planner_content = planner_content.replace(
    '  "core_keywords": ["기술 도메인 핵심어"],\\n\'',
    '  "core_keywords": ["기술 도메인 핵심어 (주의: \'평가위원\', \'전문가\', \'추천\', \'부탁해\' 등 메타 불용어는 철저히 배제할 것)"],\\n\''
)

# 2. Modify Example 3 core_keywords
planner_content = planner_content.replace(
    '"core_keywords":["국가 R&D 성과물","평가위원","연구성과"],\'',
    '"core_keywords":["국가 R&D 성과물","연구성과"],\''
)

with open(planner_path, "w", encoding="utf-8") as f:
    f.write(planner_content)

query_builder_path = r"/apps/search/query_builder.py"
with open(query_builder_path, "r", encoding="utf-8") as f:
    query_builder_content = f.read()

# 3. Modify QueryTextBuilder to use plan.core_keywords instead of query
query_builder_replacement = """
class QueryTextBuilder:
    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, str]:
        hints = plan.branch_query_hints
        # 사용자의 Raw Query(어투, 불용어 포함) 대신 LLM이 정제한 순수 도메인 키워드를 기반 텍스트로 사용
        base_text = " ".join(plan.core_keywords) if plan.core_keywords else query.strip()
        
        return {
            "basic": self._compose(base_text, "전공 학위 소속유형 직위 기술분류 전문가 프로필", hints.get("basic")),
            "art": self._compose(base_text, "논문명 키워드 초록 학술지 최근 연구실적", hints.get("art")),
            "pat": self._compose(base_text, "특허 발명명 출원 등록 사업화 지식재산", hints.get("pat")),
            "pjt": self._compose(base_text, "과제명 연구목표 연구내용 전문기관 연구수행 경험", hints.get("pjt")),
        }
"""

query_builder_content = re.sub(
    r'class QueryTextBuilder:.*?(?=\n    @staticmethod)',
    query_builder_replacement.strip() + '\n\n',
    query_builder_content,
    flags=re.DOTALL
)

with open(query_builder_path, "w", encoding="utf-8") as f:
    f.write(query_builder_content)

print("Modification done.")
