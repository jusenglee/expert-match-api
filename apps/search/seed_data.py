from __future__ import annotations

from copy import deepcopy

from apps.domain.models import EvaluationActivity, ExpertPayload, PaperEvidence, PatentEvidence, ProjectEvidence, SeedExpertRecord
from apps.search.text_utils import normalize_org_name


def build_canonical_payload_fixture() -> ExpertPayload:
    # 개발 seed는 외부 엑셀이나 DB를 직접 읽지 않고,
    # 서비스가 기대하는 최종 payload shape를 코드 안에서 명시적으로 만든다.
    return ExpertPayload(
        doc_id="11008395",
        hm_nm="김영찬",
        gndr_slct_nm="남성",
        blng_org_nm="주식회사 미소테크",
        blng_org_nm_exact=normalize_org_name("주식회사 미소테크"),
        position_nm="부장",
        degree_slct_nm="박사",
        major_slct_nm="전기및반도체공학",
        country_nm="대한민국",
        career_years=21,
        technical_classifications=[
            "기타 > 기타 > 기타 > 기타",
            "대응 > 상황관리 > 통합적 의사 결정 지원시스템",
        ],
        article_cnt=4,
        scie_cnt=1,
        patent_cnt=1,
        project_cnt=1,
        evaluation_activity_cnt=1,
        external_activity_cnt=1,
        art=[
            PaperEvidence(
                paper_nm="동시접속 사용자 접근을 고려한 데이터베이스 커넥션 풀 아키텍처",
                jrnl_nm="한국콘텐츠학회논문지",
                sci_slct_nm="기타(또는 해당없음)",
                jrnl_pub_dt="2009-01-01",
            ),
            PaperEvidence(
                paper_nm="논문의 저자 키워드를 이용한 실시간 연구동향 분석시스템 설계 및 구현",
                jrnl_nm="한국전자통신학회 논문지",
                sci_slct_nm="기타(또는 해당없음)",
                jrnl_pub_dt="2018-06-01",
            ),
            PaperEvidence(
                paper_nm="이미지 분할(image segmentation) 관련 연구 동향 파악을 위한 과학계량학 기반 연구개발지형도 분석",
                jrnl_nm="한국산업융합학회논문집",
                sci_slct_nm="기타(또는 해당없음)",
                jrnl_pub_dt="2024-06-01",
            ),
            PaperEvidence(
                paper_nm="Real-Time Fire Classification Models Based on Deep Learning for Building an Intelligent Multi-Sensor System",
                jrnl_nm="Fire",
                sci_slct_nm="SCIE",
                jrnl_pub_dt="2024-09-01",
                kor_kywd="딥러닝, 화재감지, 멀티센서",
            ),
        ],
        pat=[
            PatentEvidence(
                ipr_invention_nm="발명의 명칭",
                ipr_regist_type_nm="출원",
                ipr_regist_nat_nm="대한민국",
                aply_no="1231321",
                aply_dt="2024-06-29",
            )
        ],
        pjt=[
            ProjectEvidence(
                title1="인공지능 기반의 두경부암 자동 진단 모듈 개발",
                pjt_prfrm_org_nm="주식회사 미소테크",
                rsch_mgnt_org_nm="중소기업기술정보진흥원",
                start_dt="2019-10-07",
                end_dt="2020-04-06",
                stan_yr=2019,
                content1="AI 기반 진단 모듈 개발",
                content2="연구목표와 연구내용 요약",
            )
        ],
        evaluation_activities=[
            EvaluationActivity(
                appoint_org_nm="한국연구재단",
                committee_nm="뇌첨단_시장_체내삽입전자약",
                appoint_period="2025-03-11 ~ 2025-03-21",
                appoint_dt="2025-03-11",
            )
        ],
    )


def build_source_texts(payload: ExpertPayload) -> tuple[str, str, str, str]:
    basic_parts = [
        f"이름: {payload.hm_nm}",
        f"소속기관: {payload.blng_org_nm or ''}",
        f"직위: {payload.position_nm or ''}",
        f"학위: {payload.degree_slct_nm or ''}",
        f"전공: {payload.major_slct_nm or ''}",
        f"기술분류: {', '.join(payload.technical_classifications)}",
    ]
    art_parts = [
        f"{paper.paper_nm} {paper.jrnl_nm or ''} {paper.abstract_str or ''} {paper.kor_kywd or ''} {paper.eng_kywd or ''}"
        for paper in payload.art
    ]
    pat_parts = [
        f"{patent.ipr_invention_nm} {patent.ipr_regist_type_nm or ''} {patent.ipr_regist_nat_nm or ''}"
        for patent in payload.pat
    ]
    pjt_parts = [
        f"{project.title1 or ''} {project.title2 or ''} {project.content1 or ''} {project.content2 or ''} "
        f"{project.pjt_prfrm_org_nm or ''} {project.rsch_mgnt_org_nm or ''}"
        for project in payload.pjt
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
        point_id=payload.doc_id,
        payload=payload,
        basic_text=basic_text,
        art_text=art_text,
        pat_text=pat_text,
        pjt_text=pjt_text,
    )


def build_synthetic_records(canonical: ExpertPayload) -> list[SeedExpertRecord]:
    records: list[SeedExpertRecord] = [record_from_payload(canonical)]

    balanced = deepcopy(canonical)
    balanced.doc_id = "11008396"
    balanced.hm_nm = "박균형"
    balanced.blng_org_nm = "국립테크대학교"
    balanced.blng_org_nm_exact = normalize_org_name(balanced.blng_org_nm)
    balanced.scie_cnt = 2
    balanced.article_cnt = 5
    balanced.patent_cnt = 2
    balanced.project_cnt = 3
    balanced.pat.append(
        PatentEvidence(
            ipr_invention_nm="반도체 불량 예측 시스템",
            ipr_regist_type_nm="등록",
            ipr_regist_nat_nm="대한민국",
            regist_dt="2023-05-01",
        )
    )
    records.append(record_from_payload(balanced))

    paper_heavy = deepcopy(canonical)
    paper_heavy.doc_id = "11008397"
    paper_heavy.hm_nm = "최논문"
    paper_heavy.blng_org_nm = "AI반도체연구원"
    paper_heavy.blng_org_nm_exact = normalize_org_name(paper_heavy.blng_org_nm)
    paper_heavy.pat = []
    paper_heavy.patent_cnt = 0
    paper_heavy.project_cnt = 1
    paper_heavy.pjt = paper_heavy.pjt[:1]
    paper_heavy.art.extend(
        [
            PaperEvidence(
                paper_nm="AI 반도체 최적화 모델",
                jrnl_nm="IEEE Access",
                sci_slct_nm="SCIE",
                jrnl_pub_dt="2025-02-01",
                kor_kywd="AI 반도체, 최적화",
            ),
            PaperEvidence(
                paper_nm="NPU 성능 평가 프레임워크",
                jrnl_nm="Sensors",
                sci_slct_nm="SCIE",
                jrnl_pub_dt="2024-12-01",
                kor_kywd="NPU, 성능평가",
            ),
        ]
    )
    paper_heavy.article_cnt = len(paper_heavy.art)
    paper_heavy.scie_cnt = 3
    records.append(record_from_payload(paper_heavy))

    patent_heavy = deepcopy(canonical)
    patent_heavy.doc_id = "11008398"
    patent_heavy.hm_nm = "한특허"
    patent_heavy.blng_org_nm = "산업기술연구소"
    patent_heavy.blng_org_nm_exact = normalize_org_name(patent_heavy.blng_org_nm)
    patent_heavy.art = patent_heavy.art[:1]
    patent_heavy.article_cnt = 1
    patent_heavy.scie_cnt = 0
    patent_heavy.pat = [
        PatentEvidence(
            ipr_invention_nm="AI 반도체 냉각 구조",
            ipr_regist_type_nm="등록",
            ipr_regist_nat_nm="대한민국",
            regist_dt="2024-03-10",
        ),
        PatentEvidence(
            ipr_invention_nm="엣지 추론 가속 장치",
            ipr_regist_type_nm="출원",
            ipr_regist_nat_nm="미국",
            aply_dt="2025-01-20",
        ),
    ]
    patent_heavy.patent_cnt = len(patent_heavy.pat)
    records.append(record_from_payload(patent_heavy))

    project_heavy = deepcopy(canonical)
    project_heavy.doc_id = "11008399"
    project_heavy.hm_nm = "윤과제"
    project_heavy.blng_org_nm = "스마트제조연합"
    project_heavy.blng_org_nm_exact = normalize_org_name(project_heavy.blng_org_nm)
    project_heavy.pjt.extend(
        [
            ProjectEvidence(
                title1="차세대 AI 반도체 신뢰성 평가 체계 개발",
                content1="평가위원 후보 추천과 연계 가능한 반도체 실증 과제",
                pjt_prfrm_org_nm="스마트제조연합",
                rsch_mgnt_org_nm="한국산업기술평가관리원",
                start_dt="2023-01-01",
                end_dt="2025-12-31",
                stan_yr=2025,
            ),
            ProjectEvidence(
                title1="산업현장 적용형 반도체 AI 품질검증",
                content1="사업화 연계 성과 중심 프로젝트",
                pjt_prfrm_org_nm="스마트제조연합",
                rsch_mgnt_org_nm="중소기업기술정보진흥원",
                start_dt="2024-01-01",
                end_dt="2026-01-31",
                stan_yr=2026,
            ),
        ]
    )
    project_heavy.project_cnt = len(project_heavy.pjt)
    records.append(record_from_payload(project_heavy))

    excluded_org = deepcopy(canonical)
    excluded_org.doc_id = "11008400"
    excluded_org.hm_nm = "오제외"
    excluded_org.blng_org_nm = "A기관"
    excluded_org.blng_org_nm_exact = normalize_org_name(excluded_org.blng_org_nm)
    records.append(record_from_payload(excluded_org))

    weak_evidence = deepcopy(canonical)
    weak_evidence.doc_id = "11008401"
    weak_evidence.hm_nm = "임부족"
    weak_evidence.blng_org_nm = "지역혁신센터"
    weak_evidence.blng_org_nm_exact = normalize_org_name(weak_evidence.blng_org_nm)
    weak_evidence.art = []
    weak_evidence.article_cnt = 0
    weak_evidence.scie_cnt = 0
    weak_evidence.pat = []
    weak_evidence.patent_cnt = 0
    weak_evidence.pjt = weak_evidence.pjt[:1]
    weak_evidence.project_cnt = 1
    records.append(record_from_payload(weak_evidence))

    return records
