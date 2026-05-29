"""WO-A: 11 doc_type source→chunk 변환 검증 (권위 샘플 7.1~7.11, 연구자 M1006328 기준).

샘플 payload의 domain_attrs를 입력으로 넣고, chunk_text 직렬화 형식 / event 정규화 / tags /
chunk_id·doc_id가 정확한지 단언한다. (chunk_text_len은 샘플의 표기 숫자가 아니라 실제 본문
길이와 일치하는지 검증 — 샘플 본문은 truncate되어 있어 표기 숫자(432 등)와 다르다.)
"""
from __future__ import annotations

import pytest

from apps.domain.models import ResearcherMeta
from apps.ingest.converters import build_chunk
from apps.ingest.validate_chunks import ROLE_ACTION_STOPWORDS
from apps.search.doc_types import DocType

RESEARCHER_ID = "M1006328"
RESEARCHER_NAME = "홍길동"
DOC_SEQ = "0001"

# 샘플 7.x의 공통 researcher_meta
SAMPLE_META = ResearcherMeta(
    affiliated_organization="주식회사 미소테크",
    highest_degree="박사",
    publication_count=15,
    scie_publication_count=5,
    intellectual_property_count=3,
    research_project_count=5,
    researcher_assessor_count=5,
    expert_assessor_count=2,
)

# 각 샘플: doc_type / 기대 chunk_id / domain_attrs / 기대 chunk_text / event_date / event_year / tags
SAMPLES: list[dict] = [
    {
        "doc_type": DocType.PUBLICATION,
        "chunk_id": "PUB_M1006328_0001_c0",
        "attrs": {
            "title_primary": "대규모 언어 모델의 RAG 최적화 방안",
            "title_secondary": "RAG Optimization Methods for Large Language Models",
            "journal_name": "한국정보과학회 논문지",
            "journal_class": "SCIE",
            "indexing_class": "SCIE",
            "publication_year_month": "2024-05",
            "abstract": "본 연구에서는 벡터 검색 결과의 정확도를 높이기 위해...",
            "keywords": ["RAG", "LLM", "Vector DB"],
        },
        "chunk_text": (
            "논문명: 대규모 언어 모델의 RAG 최적화 방안\n"
            "키워드: RAG, LLM, Vector DB\n"
            "학술지: 한국정보과학회 논문지\n"
            "초록: 본 연구에서는 벡터 검색 결과의 정확도를 높이기 위해..."
        ),
        "event_date": "2024-05-01",
        "event_year": 2024,
        "tags": ["rag", "llm", "vector db"],
    },
    {
        "doc_type": DocType.INTELLECTUAL_PROPERTY,
        "chunk_id": "IP_M1006328_0001_c0",
        "attrs": {
            "ip_type": "특허권",
            "ip_title": "유기전자장치 봉지용 접착 필름",
            "application_registration_type": "등록",
            "application_country": "대한민국",
            "application_number": "1231321",
            "application_date": "2024-05-30",
            "registration_number": "1111000",
            "registration_date": "2025-10-31",
            "ip_summary": "본 발명은 접착 필름, 이를 이용한 유기전자장치 봉지 제품 및 봉지 방법에 관한 것...",
            "ip_foreign_class": "해외미출원(국내)",
        },
        # 요약은 domain_attrs.ip_summary 값을 사용(샘플 chunk_text의 요약과는 truncate가 다름)
        "chunk_text": (
            "지식재산권명: 유기전자장치 봉지용 접착 필름\n"
            "구분: 특허권 등록\n"
            "출원국가: 대한민국\n"
            "요약: 본 발명은 접착 필름, 이를 이용한 유기전자장치 봉지 제품 및 봉지 방법에 관한 것..."
        ),
        "event_date": "2025-10-31",  # registration_date 우선
        "event_year": 2025,
        "tags": [],
    },
    {
        "doc_type": DocType.RESEARCH_PROJECT,
        "chunk_id": "PJT_M1006328_0001_c0",
        "attrs": {
            "project_title_korean": "인공지능 기반의 두경부암 자동 진단 모듈 개발",
            "project_title_english": "AI-based Head and Neck Cancer Diagnosis Module",
            "performing_organization": "주식회사 미소테크",
            "managing_agency": "중소기업기술정보진흥원",
            "research_summary_korean": "본 연구는 alternative splicing event를 활용한 두경부암 진단...",
            "research_summary_english": None,
            "research_period_start": "2019-10-07",
            "research_period_end": "2020-04-06",
        },
        "chunk_text": (
            "국문과제명: 인공지능 기반의 두경부암 자동 진단 모듈 개발\n"
            "수행기관: 주식회사 미소테크\n"
            "전문기관: 중소기업기술정보진흥원\n"
            "국문요약: 본 연구는 alternative splicing event를 활용한 두경부암 진단..."
        ),
        "event_date": "2019-10-07",  # research_period_start
        "event_year": 2019,
        "tags": [],
    },
    {
        "doc_type": DocType.RESEARCHER_ASSESSOR,
        "chunk_id": "RAS_M1006328_0001_c0",
        "attrs": {
            "appointing_organization_type": "정부(공공)",
            "appointment_period_type": "임기제",
            "appointing_organization": "한국연구재단",
            "evaluation_committee_name": "뇌첨단_시장_체내삽입전자약",
            "appointment_period_start": "2025-03-11",
            "appointment_period_end": "2025-03-21",
            "appointment_date": "2025-03-11",
        },
        "chunk_text": (
            "임명기관: 한국연구재단\n"
            "평가위원회명: 뇌첨단_시장_체내삽입전자약\n"
            "임명기관구분: 정부(공공)\n"
            "활동기간구분: 임기제"
        ),
        "event_date": "2025-03-11",  # appointment_date
        "event_year": 2025,
        "tags": [],
    },
    {
        "doc_type": DocType.EXPERT_ASSESSOR,
        "chunk_id": "EAS_M1006328_0001_c0",
        "attrs": {
            "evaluation_agency_name": "한국연구재단",
            "assessment_type_class": "역량평가",
            "assessment_class": "핵심평가위원",
            "assessment_content": "2026년도 기초연구본부 핵심평가위원 명단",
            "valid_period_start": "2025-03-11",
            "valid_period_end": "2025-03-21",
        },
        "chunk_text": (
            "평가전문기관: 한국연구재단\n"
            "활동평가구분: 핵심평가위원\n"
            "활동평가내용: 2026년도 기초연구본부 핵심평가위원 명단\n"
            "유형: 역량평가"
        ),
        "event_date": "2025-03-11",  # valid_period_start
        "event_year": 2025,
        "tags": [],
    },
    {
        "doc_type": DocType.RESEARCHER_TECH,
        "chunk_id": "RTC_M1006328_0001_c0",
        "attrs": {
            "tech_classification_system": "중소기업 기술로드맵",
            "tech_classification": "4차 산업혁명 > 재난/안전 > 식품 위해인자 신속 검출 시스템",
            "keywords": ["식품", "안전", "검출"],
        },
        "chunk_text": (
            "기술분류: 4차 산업혁명 > 재난/안전 > 식품 위해인자 신속 검출 시스템\n"
            "키워드: 식품, 안전, 검출\n"
            "분류체계: 중소기업 기술로드맵"
        ),
        "event_date": None,
        "event_year": None,
        "tags": ["식품", "안전", "검출"],
    },
    {
        "doc_type": DocType.EXPERT_TECH,
        "chunk_id": "ETC_M1006328_0001_c0",
        "attrs": {
            "tech_classification_system": "산업기술분류(24년도)",
            "tech_classification": "전기·전자 > 중전기기 > 자동화제어기기",
            "tech_rank": 1,
        },
        "chunk_text": (
            "기술분류: 전기·전자 > 중전기기 > 자동화제어기기\n"
            "분류체계: 산업기술분류(24년도)\n"
            "기술순위: 1"
        ),
        "event_date": None,
        "event_year": None,
        "tags": [],
    },
    {
        "doc_type": DocType.RESEARCHER_CORE,
        "chunk_id": "RCO_M1006328_0001_c0",
        "attrs": {
            "specialty_names": [
                "AI 데이터 플랫폼",
                "AI Transformation",
                "AI 기반 데이터 관리",
                "AI 기반 데이터 모델링",
                "AI 기반 데이터 분석",
            ],
            "specialty_count": 5,
        },
        "chunk_text": "핵심 전문분야: AI 데이터 플랫폼, AI Transformation, AI 기반 데이터 관리, AI 기반 데이터 모델링, AI 기반 데이터 분석",
        "event_date": None,
        "event_year": None,
        "tags": [
            "ai 데이터 플랫폼",
            "ai transformation",
            "ai 기반 데이터 관리",
            "ai 기반 데이터 모델링",
            "ai 기반 데이터 분석",
        ],
    },
    {
        "doc_type": DocType.RESEARCHER_MAJOR,
        "chunk_id": "RMJ_M1006328_0001_c0",
        "attrs": {
            "specialty_names": [
                "ITS/텔레매틱스",
                "RFID/USN",
                "U-컴퓨팅",
                "가정용 기기 및 전자응용 기기",
                "감성과학",
            ],
            "specialty_count": 5,
        },
        "chunk_text": "전공 전문분야: ITS/텔레매틱스, RFID/USN, U-컴퓨팅, 가정용 기기 및 전자응용 기기, 감성과학",
        "event_date": None,
        "event_year": None,
        "tags": ["its/텔레매틱스", "rfid/usn", "u-컴퓨팅", "가정용 기기 및 전자응용 기기", "감성과학"],
    },
    {
        "doc_type": DocType.EXPERT_SPECIFIC,
        "chunk_id": "ESP_M1006328_0001_c0",
        "attrs": {
            "specific_specialty_name": "기초(의학) 연구자",
            "career_description": "성균관대학교 의과대학 강북삼성병원 소화기내과 부교수",
        },
        "chunk_text": (
            "특정전문분야: 기초(의학) 연구자\n"
            "경력: 성균관대학교 의과대학 강북삼성병원 소화기내과 부교수"
        ),
        "event_date": None,
        "event_year": None,
        "tags": ["기초(의학) 연구자"],
    },
    {
        "doc_type": DocType.PROFILE,
        "chunk_id": "PRF_M1006328_0001_c0",
        "attrs": {
            "researcher_number": "11008395",
            "position_title": "책임연구원",
            "highest_degree": "박사",
            "major_field": "기계공학",
            "publication_count": 15,
            "scie_publication_count": 5,
            "intellectual_property_count": 3,
            "research_project_count": 5,
            "researcher_assessor_count": 5,
            "expert_assessor_count": 2,
        },
        # 이름/소속은 researcher_name + researcher_meta.affiliated_organization에서
        "chunk_text": (
            "이름: 홍길동\n"
            "소속: 주식회사 미소테크\n"
            "직위: 책임연구원\n"
            "학위: 박사\n"
            "전공: 기계공학\n"
            "논문 15편 (SCIE 5편), 지식재산권 3건, 과제 5건"
        ),
        "event_date": None,
        "event_year": None,
        "tags": [],
    },
]


def _build(sample: dict, *, doc_seq: str = DOC_SEQ, chunk_index: int = 0):
    return build_chunk(
        sample["doc_type"],
        researcher_id=RESEARCHER_ID,
        researcher_name=RESEARCHER_NAME,
        doc_seq=doc_seq,
        domain_attrs=sample["attrs"],
        researcher_meta=SAMPLE_META,
        chunk_index=chunk_index,
    )


def build_all_sample_chunks():
    """11 doc_type 샘플 chunk 전부 (invariants 테스트에서 재사용)."""
    return [_build(sample) for sample in SAMPLES]


def test_samples_cover_all_eleven_doc_types():
    assert {s["doc_type"] for s in SAMPLES} == set(DocType)
    assert len(SAMPLES) == 11


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["doc_type"].value)
def test_chunk_text_serialization(sample: dict):
    chunk = _build(sample)
    assert chunk.chunk_text == sample["chunk_text"]
    assert chunk.chunk_text_len == len(sample["chunk_text"])


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["doc_type"].value)
def test_chunk_and_doc_ids(sample: dict):
    chunk = _build(sample)
    assert chunk.chunk_id == sample["chunk_id"]
    assert chunk.doc_id == sample["chunk_id"].rsplit("_c", 1)[0]
    assert chunk.doc_type == sample["doc_type"].value


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["doc_type"].value)
def test_event_normalization(sample: dict):
    chunk = _build(sample)
    assert chunk.event_date == sample["event_date"]
    assert chunk.event_year == sample["event_year"]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["doc_type"].value)
def test_tags_normalization(sample: dict):
    chunk = _build(sample)
    assert chunk.tags == sample["tags"]
    assert all(tag == tag.strip().lower() for tag in chunk.tags)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["doc_type"].value)
def test_meta_injected_identically(sample: dict):
    chunk = _build(sample)
    assert chunk.researcher_meta == SAMPLE_META
    assert chunk.researcher_name == RESEARCHER_NAME


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["doc_type"].value)
def test_chunk_text_has_no_role_action_stopwords(sample: dict):
    chunk = _build(sample)
    for stop in ROLE_ACTION_STOPWORDS:
        assert stop not in chunk.chunk_text


def test_assessor_doc_text_legitimately_contains_evaluator_term():
    """'평가위원'은 assessor 도메인 어휘이므로 chunk_text에 정당하게 포함된다(불용어 아님)."""
    ras = _build(SAMPLES[3])
    eas = _build(SAMPLES[4])
    assert "평가위원회명" in ras.chunk_text
    assert "핵심평가위원" in eas.chunk_text
    # 그러나 요청측 액션어("추천")는 없다
    assert "추천" not in ras.chunk_text and "추천" not in eas.chunk_text
