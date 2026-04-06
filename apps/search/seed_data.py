"""
개발 및 테스트 단계에서 사용할 가상의 전문가(Seed) 데이터를 생성하는 모듈입니다.
실제 외부 데이터 연동 없이도 시스템의 검색 및 추천 로직을 검증할 수 있도록
다양한 프로필(논문 중심, 특허 중심, 과제 중심 등)의 데이터를 제공합니다.
"""

from __future__ import annotations

from copy import deepcopy

from apps.domain.models import BasicInfo, ResearcherProfile, PublicationEvidence, IntellectualPropertyEvidence, ResearchProjectEvidence, ExpertPayload, SeedExpertRecord
from apps.search.text_utils import normalize_org_name


def build_canonical_payload_fixture() -> ExpertPayload:
    """
    가장 표준적이고 완성도가 높은 전문가 데이터(김영찬 박사)의 페이로드를 생성합니다.
    이 데이터는 다른 합성 데이터 생성을 위한 기본 템플릿으로 활용됩니다.
    """
    return ExpertPayload(
        basic_info=BasicInfo(
            researcher_id="11008395",
            researcher_name="김영찬",
            gender="남성",
            affiliated_organization="주식회사 미소테크",
            affiliated_organization_exact=normalize_org_name("주식회사 미소테크"),
            position_title="부장",
        ),
        researcher_profile=ResearcherProfile(
            highest_degree="박사",
            major_field="전기및반도체공학",
            publication_count=4,
            scie_publication_count=1,
            intellectual_property_count=1,
            research_project_count=1,
        ),
        publications=[
            PublicationEvidence(
                publication_title="동시접속 사용자 접근을 고려한 데이터베이스 커넥션 풀 아키텍처",
                journal_name="한국콘텐츠학회논문지",
                journal_index_type="기타(또는 해당없음)",
                publication_year_month="2009-01-01",
            ),
            PublicationEvidence(
                publication_title="논문의 저자 키워드를 이용한 실시간 연구동향 분석시스템 설계 및 구현",
                journal_name="한국전자통신학회 논문지",
                journal_index_type="기타(또는 해당없음)",
                publication_year_month="2018-06-01",
            ),
            PublicationEvidence(
                publication_title="이미지 분할(image segmentation) 관련 연구 동향 파악을 위한 과학계량학 기반 연구개발지형도 분석",
                journal_name="한국산업융합학회논문집",
                journal_index_type="기타(또는 해당없음)",
                publication_year_month="2024-06-01",
            ),
            PublicationEvidence(
                publication_title="Real-Time Fire Classification Models Based on Deep Learning for Building an Intelligent Multi-Sensor System",
                journal_name="Fire",
                journal_index_type="SCIE",
                publication_year_month="2024-09-01",
                korean_keywords=["딥러닝", "화재감지", "멀티센서"],
            ),
        ],
        intellectual_properties=[
            IntellectualPropertyEvidence(
                intellectual_property_title="발명의 명칭",
                application_registration_type="출원",
                application_country="대한민국",
                application_number="1231321",
                application_date="2024-06-29",
            )
        ],
        research_projects=[
            ResearchProjectEvidence(
                project_title_korean="인공지능 기반의 두경부암 자동 진단 모듈 개발",
                performing_organization="주식회사 미소테크",
                managing_agency="중소기업기술정보진흥원",
                project_start_date="2019-10-07",
                project_end_date="2020-04-06",
                reference_year=2019,
                research_content_summary="AI 기반 진단 모듈 개발\n연구목표와 연구내용 요약",
            )
        ]
    )


def build_source_texts(payload: ExpertPayload) -> tuple[str, str, str, str]:
    """
    전문가 페이로드를 바탕으로 4가지 브랜치(기본, 논문, 특허, 과제)별 벡터화용 텍스트를 생성합니다.
    각 텍스트는 해당 브랜치의 Dense/Sparse 벡터 생성을 위한 원천 데이터로 사용됩니다.
    """
    # 1. 기본 정보 텍스트 (이름, 소속, 전공 등)
    basic_parts = [
        f"이름: {payload.basic_info.researcher_name}",
        f"소속기관: {payload.basic_info.affiliated_organization or ''}",
        f"직위: {payload.basic_info.position_title or ''}",
        f"학위: {payload.researcher_profile.highest_degree or ''}",
        f"전공: {payload.researcher_profile.major_field or ''}",
    ]
    # 2. 논문 실적 요약 텍스트
    art_parts = [
        f"{paper.publication_title} {paper.journal_name or ''} {paper.abstract or ''} {', '.join(paper.korean_keywords)} {', '.join(paper.english_keywords)}"
        for paper in payload.publications
    ]
    # 3. 특허 실적 요약 텍스트
    pat_parts = [
        f"{patent.intellectual_property_title} {patent.application_registration_type or ''} {patent.application_country or ''}"
        for patent in payload.intellectual_properties
    ]
    # 4. 연구 과제 요약 텍스트
    pjt_parts = [
        f"{project.project_title_korean or ''} {project.project_title_english or ''} {project.research_content_summary or ''} "
        f"{project.performing_organization or ''} {project.managing_agency or ''}"
        for project in payload.research_projects
    ]
    return (
        "\n".join(part for part in basic_parts if part.strip()),
        "\n".join(art_parts) or "논문 근거 부족",
        "\n".join(pat_parts) or "특허 근거 부족",
        "\n".join(pjt_parts) or "과제 근거 부족",
    )


def record_from_payload(payload: ExpertPayload) -> SeedExpertRecord:
    """단일 전문가 페이로드를 Qdrant에 삽입 가능한 Seed 레코드 형태로 변환합니다."""
    basic_text, art_text, pat_text, pjt_text = build_source_texts(payload)
    return SeedExpertRecord(
        point_id=payload.basic_info.researcher_id,
        payload=payload,
        basic_text=basic_text,
        art_text=art_text,
        pat_text=pat_text,
        pjt_text=pjt_text,
    )


def build_synthetic_records(canonical: ExpertPayload) -> list[SeedExpertRecord]:
    """
    표준 템플릿을 변형하여 다양한 특징을 가진 여러 전문가 레코드를 생성합니다.
    검색 랭킹 점수 차이, 필터링 로직, 브랜치별 가중치 테스트 등을 수행하기 위함입니다.
    """
    records: list[SeedExpertRecord] = [record_from_payload(canonical)]

    # 1. 밸런스형 전문가 (논문, 특허, 과제 고루 분포)
    balanced = deepcopy(canonical)
    balanced.basic_info.researcher_id = "11008396"
    balanced.basic_info.researcher_name = "박균형"
    balanced.basic_info.affiliated_organization = "국립테크대학교"
    balanced.basic_info.affiliated_organization_exact = normalize_org_name("국립테크대학교")
    balanced.researcher_profile.scie_publication_count = 2
    balanced.researcher_profile.publication_count = 5
    balanced.researcher_profile.intellectual_property_count = 2
    balanced.researcher_profile.research_project_count = 3
    balanced.intellectual_properties.append(
        IntellectualPropertyEvidence(
            intellectual_property_title="반도체 불량 예측 시스템",
            application_registration_type="등록",
            application_country="대한민국",
            registration_date="2023-05-01",
        )
    )
    records.append(record_from_payload(balanced))

    # 2. 논문 실적 중심 전문가
    paper_heavy = deepcopy(canonical)
    paper_heavy.basic_info.researcher_id = "11008397"
    paper_heavy.basic_info.researcher_name = "최논문"
    paper_heavy.basic_info.affiliated_organization = "AI반도체연구원"
    paper_heavy.basic_info.affiliated_organization_exact = normalize_org_name("AI반도체연구원")
    paper_heavy.intellectual_properties = []
    paper_heavy.researcher_profile.intellectual_property_count = 0
    paper_heavy.researcher_profile.research_project_count = 1
    paper_heavy.research_projects = paper_heavy.research_projects[:1]
    paper_heavy.publications.extend(
        [
            PublicationEvidence(
                publication_title="AI 반도체 최적화 모델",
                journal_name="IEEE Access",
                journal_index_type="SCIE",
                publication_year_month="2025-02-01",
                korean_keywords=["AI 반도체", "최적화"],
            ),
            PublicationEvidence(
                publication_title="NPU 성능 평가 프레임워크",
                journal_name="Sensors",
                journal_index_type="SCIE",
                publication_year_month="2024-12-01",
                korean_keywords=["NPU", "성능평가"],
            ),
        ]
    )
    paper_heavy.researcher_profile.publication_count = len(paper_heavy.publications)
    paper_heavy.researcher_profile.scie_publication_count = 3
    records.append(record_from_payload(paper_heavy))

    # 3. 특허 실적 중심 전문가
    patent_heavy = deepcopy(canonical)
    patent_heavy.basic_info.researcher_id = "11008398"
    patent_heavy.basic_info.researcher_name = "한특허"
    patent_heavy.basic_info.affiliated_organization = "산업기술연구소"
    patent_heavy.basic_info.affiliated_organization_exact = normalize_org_name("산업기술연구소")
    patent_heavy.publications = patent_heavy.publications[:1]
    patent_heavy.researcher_profile.publication_count = 1
    patent_heavy.researcher_profile.scie_publication_count = 0
    patent_heavy.intellectual_properties = [
        IntellectualPropertyEvidence(
            intellectual_property_title="AI 반도체 냉각 구조",
            application_registration_type="등록",
            application_country="대한민국",
            registration_date="2024-03-10",
        ),
        IntellectualPropertyEvidence(
            intellectual_property_title="엣지 추론 가속 장치",
            application_registration_type="출원",
            application_country="미국",
            application_date="2025-01-20",
        ),
    ]
    patent_heavy.researcher_profile.intellectual_property_count = len(patent_heavy.intellectual_properties)
    records.append(record_from_payload(patent_heavy))

    # 4. 연구 과제 중심 전문가
    project_heavy = deepcopy(canonical)
    project_heavy.basic_info.researcher_id = "11008399"
    project_heavy.basic_info.researcher_name = "윤과제"
    project_heavy.basic_info.affiliated_organization = "스마트제조연합"
    project_heavy.basic_info.affiliated_organization_exact = normalize_org_name("스마트제조연합")
    project_heavy.research_projects.extend(
        [
            ResearchProjectEvidence(
                project_title_korean="차세대 AI 반도체 신뢰성 평가 체계 개발",
                research_content_summary="평가위원 후보 추천과 연계 가능한 반도체 실증 과제",
                performing_organization="스마트제조연합",
                managing_agency="한국산업기술평가관리원",
                project_start_date="2023-01-01",
                project_end_date="2025-12-31",
                reference_year=2025,
            ),
            ResearchProjectEvidence(
                project_title_korean="산업현장 적용형 반도체 AI 품질검증",
                research_content_summary="사업화 연계 성과 중심 프로젝트",
                performing_organization="스마트제조연합",
                managing_agency="중소기업기술정보진흥원",
                project_start_date="2024-01-01",
                project_end_date="2026-01-31",
                reference_year=2026,
            ),
        ]
    )
    project_heavy.researcher_profile.research_project_count = len(project_heavy.research_projects)
    records.append(record_from_payload(project_heavy))

    # 5. 특정 기관(제외 대상) 소속 전문가 테스트용
    excluded_org = deepcopy(canonical)
    excluded_org.basic_info.researcher_id = "11008400"
    excluded_org.basic_info.researcher_name = "오제외"
    excluded_org.basic_info.affiliated_organization = "A기관"
    excluded_org.basic_info.affiliated_organization_exact = normalize_org_name("A기관")
    records.append(record_from_payload(excluded_org))

    # 6. 실적 정보가 매우 부족한 전문가 (낮은 순위 노출 테스트)
    weak_evidence = deepcopy(canonical)
    weak_evidence.basic_info.researcher_id = "11008401"
    weak_evidence.basic_info.researcher_name = "임부족"
    weak_evidence.basic_info.affiliated_organization = "지역혁신센터"
    weak_evidence.basic_info.affiliated_organization_exact = normalize_org_name("지역혁신센터")
    weak_evidence.publications = []
    weak_evidence.researcher_profile.publication_count = 0
    weak_evidence.researcher_profile.scie_publication_count = 0
    weak_evidence.intellectual_properties = []
    weak_evidence.researcher_profile.intellectual_property_count = 0
    weak_evidence.research_projects = weak_evidence.research_projects[:1]
    weak_evidence.researcher_profile.research_project_count = 1
    records.append(record_from_payload(weak_evidence))

    return records
