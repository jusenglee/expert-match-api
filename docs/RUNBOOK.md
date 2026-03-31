# RUNBOOK

## 1. 설치

```powershell
python -m pip install -e .[dev]
```

## 2. Qdrant 준비

- 로컬 Qdrant를 `http://localhost:6333`에 띄운다.
- 기본 컬렉션명은 `expert_master`다.

## 3. 실데이터 insert 후 사전검증

실데이터가 들어간 뒤 아래 순서로 확인한다.

1. `ntis-validate-live`
2. `GET /health`
3. `GET /health/ready`

`ntis-validate-live` 또는 `/health/ready`가 실패하면, 추천 API 호출 전에 아래를 먼저 확인한다.

- 컬렉션 존재 여부
- 8개 named vector 존재 여부
- sparse vector IDF 설정
- payload index 존재 여부
- sample point의 `art[]`, `pat[]`, `pjt[]`
- `blng_org_nm_exact`
- `pjt.start_dt/end_dt/stan_yr`

## 4. 서버 실행

운영 기본:

```powershell
uvicorn apps.api.main:app --reload
```

개발 seed는 운영 경로에서 사용하지 않는다.

## 5. smoke test

### 1) 논문 중심

```powershell
curl -X POST http://127.0.0.1:8000/search/candidates -H "Content-Type: application/json" -d "{\"query\":\"최근 5년 SCIE 논문이 있는 AI 반도체 평가위원 추천\"}"
```

### 2) 특허 중심

```powershell
curl -X POST http://127.0.0.1:8000/search/candidates -H "Content-Type: application/json" -d "{\"query\":\"사업화 경험과 등록 특허가 있는 전문가 추천\"}"
```

### 3) 과제 중심

```powershell
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"query\":\"AI 반도체 과제 수행 경험이 많은 평가위원 추천\"}"
```

### 4) 기관 제외 포함 복합 질의

```powershell
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"query\":\"최근 5년 SCIE 논문과 과제 경험이 있는 AI 반도체 평가위원 추천\",\"exclude_orgs\":[\"A기관\"]}"
```

## 6. 수용 기준

- `/health/ready` 200
- `/search/candidates` 후보 반환
- `/recommend` 추천 반환
- 추천 결과에 evidence 비어 있음 0건
- 기관 제외 질의에서 제외기관 후보 0건
- hard filter 위반 0건

## 7. 운영 로그

기본 로그에는 아래 항목이 포함돼야 한다.

- planner JSON
- branch별 query text
- 적용 hard filter
- 제외기관
- 검색 후보 id 목록
- 최종 추천 evidence 요약
- feedback 저장 결과

