"""도메인 concept 운영 안전망(정답 사전이 아님).

planner가 질의별 Concept Evidence Plan(`ConceptSpec`)을 산출하면 그것을 최우선으로 쓰고, registry는
  (1) 같은 concept id의 누락 term 보강(enrich),
  (2) planner 미산출 시 fallback concept plan 생성(detect)
에만 사용한다. 사람이 모든 도메인 alias를 무한정 코드에 박는 사전이 아니라, 자주 나오는 ~10개 도메인의
최소 anchor만 관리한다.

각 도메인: query_terms(검색 recall) / evidence_terms(태깅 '확정' strong) / weak_terms(약신호 확정불가).
"""
from __future__ import annotations

from apps.domain.models import ConceptSpec

# id -> (label, query_terms, evidence_terms(strong), weak_terms)
_REGISTRY: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "ai": (
        "인공지능",
        ("인공지능", "AI", "머신러닝", "딥러닝", "신경망"),
        ("인공지능", "AI", "머신러닝", "machine learning", "딥러닝", "deep learning",
         "신경망", "neural network", "뉴로모픽", "neuromorphic", "LLM", "생성형",
         "AI반도체", "인공지능반도체", "지능형반도체"),
        ("지능형", "스마트", "자동화"),
    ),
    "semiconductor": (
        "반도체",
        ("반도체", "시스템반도체", "반도체소자"),
        ("반도체", "시스템반도체", "AI반도체", "인공지능반도체", "지능형반도체", "반도체소자",
         "반도체설계", "반도체공정", "반도체장비", "반도체소재", "집적회로", "웨이퍼", "파운드리",
         "SoC", "CMOS", "MOSFET", "DRAM", "SRAM", "PIM", "NPU", "FPGA", "ASIC", "HBM", "칩"),
        ("지능형", "시스템", "센서", "회로", "소자", "공정", "산업", "개발"),
    ),
    "secondary_battery": (
        "이차전지",
        ("이차전지", "배터리", "리튬이온전지"),
        # "배터리"는 대표어로 evidence 승격(자연어 질의 "전기차 배터리" 감지/확정). 단 "전지"는
        # 연료전지/태양전지 등 오확정 위험으로 weak 유지(substring 매칭).
        ("이차전지", "배터리", "리튬이온전지", "리튬이차전지", "전고체전지", "전고체 배터리",
         "solid-state battery", "양극재", "음극재", "고체전해질", "전해질", "리튬메탈"),
        ("전지", "소재", "에너지"),
    ),
    "bio": (
        "바이오",
        ("바이오", "생명공학", "신약"),
        ("바이오", "생명공학", "biotechnology", "신약", "항체", "유전체", "genomics",
         "세포치료", "면역치료", "단백질", "백신"),
        ("의료", "건강", "치료"),
    ),
    "robot": (
        "로봇",
        ("로봇", "로보틱스", "매니퓰레이터"),
        ("로봇", "로보틱스", "robotics", "매니퓰레이터", "manipulator", "협동로봇",
         "휴머노이드", "로봇제어"),
        ("자동", "제어", "구동"),
    ),
    "autonomous_driving": (
        "자율주행",
        ("자율주행", "ADAS", "무인주행"),
        ("자율주행", "autonomous driving", "ADAS", "무인주행", "라이다", "LiDAR",
         "센서융합", "sensor fusion"),
        ("자동", "주행", "차량", "센서"),
    ),
    "display": (
        "디스플레이",
        ("디스플레이", "OLED", "마이크로LED"),
        ("디스플레이", "display", "OLED", "QLED", "마이크로LED", "micro LED", "LCD",
         "유기발광", "패널"),
        ("화면", "소자", "패널"),
    ),
    "hydrogen": (
        "수소",
        ("수소", "연료전지", "수전해"),
        ("수소", "연료전지", "fuel cell", "수전해", "electrolysis", "그린수소",
         "수소저장", "개질"),
        ("에너지", "친환경", "발전"),
    ),
    "quantum": (
        "양자",
        ("양자", "큐비트", "양자컴퓨팅"),
        ("양자", "quantum", "큐비트", "qubit", "양자컴퓨팅", "quantum computing",
         "양자통신", "양자암호", "양자센서"),
        ("정보", "컴퓨팅", "보안"),
    ),
    "security": (
        "보안",
        ("보안", "사이버보안", "암호"),
        ("보안", "security", "사이버보안", "cybersecurity", "암호", "cryptography",
         "침입탐지", "취약점", "해킹", "악성코드"),
        ("정보", "시스템", "네트워크"),
    ),
}


def all_registry_ids() -> list[str]:
    return list(_REGISTRY)


def registry_spec(concept_id: str) -> ConceptSpec | None:
    """registry 도메인 → 새 ConceptSpec(source=registry). 없으면 None."""
    row = _REGISTRY.get(concept_id)
    if row is None:
        return None
    label, query_terms, evidence_terms, weak_terms = row
    return ConceptSpec(
        id=concept_id,
        label=label,
        role="required",
        query_terms=list(query_terms),
        evidence_terms=list(evidence_terms),
        weak_terms=list(weak_terms),
        source="registry",
        confidence=0.7,
    )
