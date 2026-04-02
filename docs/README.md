# NTIS 평가위원 추천 서비스

Qdrant 하이브리드 검색과 LLM 추천 판단을 결합한 평가위원 추천 API다.

## 현재 상태

- `POST /recommend`: 검색 + 후보 비교 + 최종 추천
- `POST /search/candidates`: 검색 디버그용 후보 조회
- `POST /feedback`: 운영자 피드백 저장
- `GET /health`: 생존 확인
- `GET /health/ready`: 실데이터 연동 readiness 확인
- `ntis-validate-live`: 운영 반입 데이터 계약 검증 CLI

## 문서 길잡이

- [SERVICE_FLOW.md](/D:/Project/python_project/Ntis_person_API/docs/SERVICE_FLOW.md): 비개발자도 읽을 수 있는 서비스 동작 설명 문서
- [CONTRACT.md](/D:/Project/python_project/Ntis_person_API/docs/CONTRACT.md): API, planner, judge, payload 계약 문서
- [RUNBOOK.md](/D:/Project/python_project/Ntis_person_API/docs/RUNBOOK.md): 실데이터 반입 후 운영 점검 절차
- [ENVIRONMENT.md](/D:/Project/python_project/Ntis_person_API/docs/ENVIRONMENT.md): 필수 환경변수와 운영 모드 설정

## 운영 기준

- Qdrant `v1.16.0`
- 컬렉션명 `expert_master`
- `basic/art/pat/pjt` dense + sparse named vector
- branch 내부 `dense + sparse -> RRF`
- 최상위 `4 branch -> RRF`
- branch on/off 없음, 항상 전 branch 검색
- hard filter는 코드가 deterministic 하게 보장
- 운영 경로에서는 heuristic/hash fallback 금지

## 개발과 운영의 차이

- 개발/테스트:
  - fake service, hashing encoder, heuristic planner/judge를 테스트 보조용으로만 사용
- 운영:
  - 실제 OpenAI 호환 LLM backend
  - 실제 OpenAI 호환 embedding backend
  - Qdrant 실데이터
  - readiness/validate-live 통과 후 API 사용

## 빠른 시작

1. `python -m pip install -e .[dev]`
2. Qdrant 실행
3. 환경변수 설정
4. 실데이터 insert
5. `ntis-validate-live`
6. `uvicorn apps.api.main:app --reload`
7. `/health/ready` 확인
8. `/search/candidates`, `/recommend` smoke test

세부 절차는 [RUNBOOK.md](/D:/Project/python_project/Ntis_person_API/docs/RUNBOOK.md)를 참고한다.
