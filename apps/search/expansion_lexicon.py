from __future__ import annotations

from typing import TypedDict


class ExpansionBundle(TypedDict):
    description: str
    art_soft: list[str]
    pat_soft: list[str]
    pjt_soft: list[str]


# 확장 번들 사전 정의 (Expansion Lexicon)
# LLM은 질의와 가장 적합한 번들 ID를 선택합니다.
EXPANSION_LEXICON: dict[str, ExpansionBundle] = {
    "uav": {
        "description": "무인기, 드론, 비행체 제어 관련 기술",
        "art_soft": ["무인기", "UAV", "자율비행"],
        "pat_soft": ["비행체 제어", "드론 시스템", "원격 조종"],
        "pjt_soft": ["현장 실증", "자율 비행", "무인 항공 시스템"],
    },
    "fire_response": {
        "description": "화재 진압, 소방 시스템, 재난 대응 관련 기술",
        "art_soft": ["재난 대응", "소방", "화재 확산 방지"],
        "pat_soft": ["진압 장치", "소방 시스템", "소화 설비"],
        "pjt_soft": ["재난 대응", "현장 적용", "화재 감지 및 진압"],
    },
    "ai_vision": {
        "description": "인공지능 시각 지능, 영상 분석, 객체 검출 기술",
        "art_soft": ["영상 분석", "객체 검출", "딥러닝 시각"],
        "pat_soft": ["이미지 프로세싱", "패턴 인식", "비디오 분석"],
        "pjt_soft": ["실시간 모니터링", "지능형 관제", "AI 기반 분석"],
    },
}


def get_lexicon_summary() -> str:
    """LLM 프롬프트 주입용 사전 요약을 생성합니다."""
    lines = []
    for bundle_id, bundle in EXPANSION_LEXICON.items():
        lines.append(f"- {bundle_id}: {bundle['description']}")
    return "\n".join(lines)


def get_expanded_keywords(bundle_ids: list[str], branch: str) -> list[str]:
    """선택된 번들 ID들로부터 특정 브랜치에 해당하는 확장 키워드들을 수집합니다."""
    expanded = []
    field_map = {
        "art": "art_soft",
        "pat": "pat_soft",
        "pjt": "pjt_soft",
        "basic": None,  # Basic은 확장을 최소화
    }
    
    field_name = field_map.get(branch)
    if not field_name:
        return []

    for bid in bundle_ids:
        bundle = EXPANSION_LEXICON.get(bid)
        if bundle:
            expanded.extend(bundle.get(field_name, [])) # type: ignore
            
    return list(set(expanded)) # 중복 제거
