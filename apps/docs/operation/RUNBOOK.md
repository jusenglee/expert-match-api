# 운영 매뉴얼 (RUNBOOK)

이 문서는 전문가 추천 시스템의 설치, 실행, 상태 점검 및 실시간 모니터링을 위한 운영 지침을 담고 있습니다.

## 1. 패키지 설치

최신 Python 3.12 이상의 환경에서 다음 명령어를 실행하여 필요한 패키지를 설치합니다.

```powershell
python -m pip install -e .[dev]
```

## 2. Qdrant 및 데이터 준비

- Qdrant 서버가 구동 중인지, 설정된 URL에 접근 가능한지 확인합니다.
- 기본 컬렉션 이름은 `researcher_recommend_proto`이며, 필요 시 환경 변수 `NTIS_QDRANT_COLLECTION_NAME`으로 덮어쓸 수 있습니다.
- 애플리케이션 시작 시, 설정된 컬렉션의 Sparse Vector 수정자(`IDF`)를 자동으로 확인하고 필요 시 복구를 시도합니다.

## 3. 시스템 준비 상태 점검 (Readiness)

추천 API를 호출하기 전, 다음 순서대로 시스템 상태를 점검하십시오.

1. `ntis-validate-live` 명령 실행 (CLI 도구)
2. `GET /health` 호출 (기본 헬스체크)
3. `GET /health/ready` 호출 (상세 준비 상태 확인)

만약 `/health/ready` 결과가 `503` 에러 또는 `ready: false`를 반환한다면 다음 항목을 점검하십시오:
- Qdrant 컬렉션 존재 여부
- 필수 Named Vector(Dense/Sparse) 존재 여부
- Sparse Vector의 IDF 설정 값
- 필수 Payload 인덱스 생성 여부
- 유효한 샘플 데이터(Point) 존재 여부 및 데이터 구조(`publications[]`, `research_projects[]` 등)

## 4. 서버 실행

다음 명령어를 통해 API 서버를 실행합니다.

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8011 --reload
```

- **LLM 일관성 모드**: Planner와 Reasoner는 환경변수로 노출하지 않은 고정 저변동 샘플링(`temperature=0.0`, `top_p=0.2`, `reasoning_effort=low`, `include_reasoning=false`, `disable_thinking=true`)을 사용합니다. 운영 중 튜닝이 필요하면 코드 변경이 필요합니다.
- **참고**: `NTIS_EMBEDDING_BACKEND=local` 모드 사용 시, 루트의 `multilingual-e5-large-instruct` 폴더 내에 모델 파일들이 온전히 존재해야 합니다.
- **추천 사유 생성**: `/recommend` 요청 시 LLM이 전문가별 추천 사유를 생성하며, 이는 검색 결과의 숏리스트(Top-k)를 대상으로 합니다.

## 5. 브라우저 플레이그라운드 (Playground) 활용

웹 브라우저에서 다음 주소로 접속하여 대화형 테스트를 수행할 수 있습니다.

```text
http://127.0.0.1:8011/
```

**플레이그라운드 주요 기능:**
1. **상태 배지 확인**: 페이지 상단의 배지가 녹색(시스템 정상)인지 확인합니다.
2. **분석 결과 창**: 자연어 질의를 입력하여 실제 추천 결과와 선정 사유를 확인합니다.
3. **실시간 서버 로그 콘솔**: **(신규)** 결과창 하단의 콘솔을 통해 서버 내부에서 발생하는 **Trace ID 기반 한글 로그**를 즉시 모니터링합니다. AI의 사고 과정을 투명하게 확인할 수 있습니다.

## 6. API 테스트 (curl 예시)

**전문가 후보 목록 조회 (`/search/candidates`)**
```powershell
curl -X POST http://127.0.0.1:8011/search/candidates `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 분야의 SCIE 논문 실적이 우수한 전문가 후보를 찾아줘\"}"
```

**최종 전문가 추천 (`/recommend`)**
```powershell
curl -X POST http://127.0.0.1:8011/recommend `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 설계 과제 경험이 있는 전문가를 추천해주고 특정 기관은 제외해줘\", \"exclude_orgs\":[\"A기관\"]}"
```

## 7. 가시성 및 로깅 시스템 (중요)

본 시스템은 모든 요청에 대해 **Trace ID**를 부여하여 추적성을 보장합니다.

### 주요 추적 로그 포인트 (한글 로그)
- **Planner**: 사용자 질의 분석 결과, 하드 필터 추출 정보, LLM 응답 소요시간 및 토큰 수
- **Retriever**: 하이브리드 검색 실행 결과, 검색된 후보 수, 필터 적용 현황
- **Judge Map 라운드**: 배치 분할 정보(라운드 번호, 배치 수, 배치 크기), 후보별 예상 토큰 크기, `max_tokens=3000` 제한 적용, LLM 호출 슬롯 획득(세마포어), 경량 응답 수신 시간, 생존자(survivors) 수
- **Judge Reduce 라운드**: 생존 후보의 전체 직렬화 토큰 추정, `max_tokens=unlimited` 적용, 상세 응답 수신 시간
- **Judge JSON 추출**: 3단계 방어 JSON 추출 결과 (추출된 텍스트 로깅)
- **Judge 정규화**: 문자열/리스트 불일치 자동 교정 횟수
- **Fallback**: LLM 오류 또는 JSON 파싱 실패 시 휴리스틱 모드로의 전환 안내 (Planner/Judge 각각)
- **Data Gap**: 특정 전문가의 데이터 누락(논문 없음 등)에 대한 경고

로그 형식은 다음과 같으며, Playground UI의 콘솔에서 레벨별 색상과 함께 확인할 수 있습니다.
`[시간] [레벨] [ID:TraceID] [모듈명] 메시지`

### 로그 예시 (Map-Reduce 심사)
```
[09:45:54] [INFO] [ID:abc123] [Judge] Map-Reduce 분할 시작: 총 후보=37 배치크기=10 배치수=4
[09:45:54] [INFO] [ID:abc123] [Judge] 데이터당 예상 토큰 크기: context=map 총 추정=1200 토큰 후보 수=10
[09:45:54] [INFO] [ID:abc123] [Judge] LLM 호출 슬롯 획득: context=map round=1 batch=1/4 max_concurrency=10 max_tokens=3000
[09:45:56] [INFO] [ID:abc123] [Judge] LLM 응답 수신 완료: context=map round=1 batch=1/4 소요시간=1823.45ms
[09:45:58] [INFO] [ID:abc123] [Judge] Map 라운드 완료: 생존 후보=15
[09:45:58] [INFO] [ID:abc123] [Judge] LLM 판정 시작: context=reduce round=2 batch=1/1 후보 수=15 max_tokens=unlimited
[09:46:09] [INFO] [ID:abc123] [Judge] LLM 응답 수신 완료: context=reduce round=2 batch=1/1 소요시간=10894.23ms
```
