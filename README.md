# 전문가 추천 시스템 프로토타입

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)
![Qdrant](https://img.shields.io/badge/Qdrant-1.10%2B-red)
![OpenAI](https://img.shields.io/badge/OpenAI-1.40%2B-lightgray)

이 프로젝트는 다차원 인물 정보(기본 정보, 논문, 특허, 사업/과제 이력 등)를 바탕으로 요구 조건에 부합하는 전문가를 검색하고 추천하는 API 프로토타입입니다. Qdrant 기반의 하이브리드 탐색(Dense + Sparse)과 대형 언어 모델(LLM)의 제어 및 비교 판단 능력을 함께 결합하여 시스템을 구축했습니다.

## 주요 특징

- 다차원 기반 동시 검색 (Multi-branch Hybrid Search)
  - 인물 정보를 기본 정보(basic), 논문(publications), 특허(intellectual_properties), 과제(research_projects) 4개의 관점으로 분리하여 취급합니다.
  - 각 단위마다 의미론적 탐색(Dense)과 기호/키워드 매칭(Sparse)을 병행하며, RRF(Reciprocal Rank Fusion) 알고리즘을 통해 탐색 결과를 통합합니다.
- LLM 기반 제어 파이프라인
  - Planner: 사용자 질의를 시스템이 처리가능한 검색 엔진 필터(hard_filters) 및 검색 쿼리로 변환합니다.
  - Judge: 1차적으로 도출된 인물 후보군(Candidate Cards)의 관련 요소를 LLM이 대조하여 최종 순위와 선정 사유를 산출합니다.

## 시작 가이드

### 1. 환경 준비 및 패키지 설치
본 시스템은 Python 3.12 이상의 환경이 필요합니다.
```bash
python -m venv .venv
pip install -e .[dev]
```

### 2. 환경 변수 설정
루트 디렉토리에 `.env` 파일을 생성하고 다음 필수 환경 변수를 지정합니다.
```env
APP_ENV=local
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. 서버 실행
개발용 서버 구동 명령어는 다음과 같습니다.
```bash
uvicorn apps.api.main:app --reload
```

## 문서 안내

세부 동작 방식 및 관련 설정은 `docs/` 디렉토리 하위 문서에 작성되어 있습니다.
- [서비스 동작 원리 (SERVICE_FLOW.md)](docs/SERVICE_FLOW.md) : 검색 전처리 로직 및 추천 시스템 흐름
- [운영 관리 (RUNBOOK.md)](docs/RUNBOOK.md) : 데이터 반입 및 시스템 무결성 점검 시나리오
- [환경 변수 (ENVIRONMENT.md)](docs/ENVIRONMENT.md) : 프로젝트 실행 모드 및 부가 설정 항목
- [데이터 명세 (CONTRACT.md)](docs/CONTRACT.md) : 기준 모델 스키마 및 입출력 API 구조

## 기술 스택

- Web Framework: FastAPI
- Vector DB: Qdrant
- LLM Pipeline: OpenAI API, LangChain Core
- Data Validation: Pydantic, Pytest
