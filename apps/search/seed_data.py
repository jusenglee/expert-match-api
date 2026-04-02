from __future__ import annotations

from copy import deepcopy

from apps.domain.models import BasicInfo, ResearcherProfile, PublicationEvidence, IntellectualPropertyEvidence, ResearchProjectEvidence, EvaluationActivity, ExpertPayload, SeedExpertRecord
from apps.search.text_utils import normalize_org_name


def build_canonical_payload_fixture() -> ExpertPayload:
    # 개발 seed는 외부 엑셀이나 DB를 직접 읽지 않고,
    # 서비스가 기대하는 최종 payload shape를 코드 안에서 명시적으로 만든다.
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
    basic_parts = [
        f"이름: {payload.basic_info.researcher_name}",
        f"소속기관: {payload.basic_info.affiliated_organization or ''}",
        f"직위: {payload.basic_info.position_title or ''}",
        f"학위: {payload.researcher_profile.highest_degree or ''}",
        f"전공: {payload.researcher_profile.major_field or ''}",
    ]
    art_parts = [
        f"{paper.publication_title} {paper.journal_name or ''} {paper.abstract or ''} {', '.join(paper.korean_keywords)} {', '.join(paper.english_keywords)}"
        for paper in payload.publications
    ]
    pat_parts = [
        f"{patent.intellectual_property_title} {patent.application_registration_type or ''} {patent.application_country or ''}"
        for patent in payload.intellectual_properties
    ]
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
    records: list[SeedExpertRecord] = [record_from_payload(canonical)]

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

    excluded_org = deepcopy(canonical)
    excluded_org.basic_info.researcher_id = "11008400"
    excluded_org.basic_info.researcher_name = "오제외"
    excluded_org.basic_info.affiliated_organization = "A기관"
    excluded_org.basic_info.affiliated_organization_exact = normalize_org_name("A기관")
    records.append(record_from_payload(excluded_org))

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
