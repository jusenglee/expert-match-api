# 골든 테스트 (Golden Tests)

## 시나리오 (Scenarios)

### 1. 논문 중심 질의 (Publication-focused Query)

- 입력: `최근 5년 내 SCIE 논문 실적이 우수한 AI 반도체 심사위원을 추천해줘`
- 기대 결과:
  - `art_sci_slct_nm=SCIE` 필터 적용
  - `art_recent_years=5` 필터 적용
  - 추천 근거(`evidence`)에 최소 하나 이상의 논문 항목 포함

### 2. 특허 중심 질의 (Patent-focused Query)

- 입력: `등록된 특허가 있고 사업화 경험이 있는 전문가를 추천해줘`
- 기대 결과:
  - `patent_cnt_min >= 1` 필터 적용
  - 추천 근거(`evidence`)에 최소 하나 이상의 특허 항목 포함

### 3. 과제 중심 질의 (Project-focused Query)

- 입력: `AI 반도체 과제 수행 경험이 풍부한 심사위원을 추천해줘`
- 기대 결과:
  - `project_cnt_min >= 1` 필터 적용
  - 추천 근거(`evidence`)에 최소 하나 이상의 과제 항목 포함

### 4. 특정 기관 제외 질의 (Excluded Organization Query)

- 입력: `심사위원을 추천해주되 A 기관 소속은 제외해줘`
- 기대 결과:
  - 제외 필터링(`exclude_orgs`) 적용
  - 최종 응답에 제외된 기관 소속 전문가가 포함되지 않음

### 5. 부분 근거 질의 (Partial-evidence Query)

- 입력: `특허와 논문 양쪽 모두에 강점이 있는 전문가를 추천해줘`
- 기대 결과:
  - 데이터 결측치가 있는 경우 `data_gaps` 또는 `risks`를 통해 노출
  - 근거 정보가 전혀 없는 추천 결과는 최종 응답에서 제거됨

### 6. 추천 결과 없음 (Empty Recommendation Result)

- 입력: 하드 필터가 너무 엄격하거나 근거 데이터가 부족한 질의
- 기대 결과:
  - `POST /recommend`가 `200` 상태 코드 반환
  - `recommendations=[]` (빈 리스트)
  - 사유가 `not_selected_reasons` 및/또는 `data_gaps`에 명시됨

### 7. 검색 결과 없음 (Zero-hit Retrieval)

- 입력: 검색 단계에서 후보가 전혀 발견되지 않는 질의
- 기대 결과:
  - `POST /recommend`가 `200` 상태 코드 반환
  - 판정(Judge) 단계를 건너뜀
  - `recommendations=[]`
  - `not_selected_reasons`에 후보 없음 설명 포함

## 수락 기준 (Acceptance Criteria)

- 하드 필터를 위반한 전문가는 최종 응답에서 제외되어야 함
- `searched_branches`는 항상 `basic`, `art`, `pat`, `pjt`를 유지해야 함
- 추천 사유와 근거는 전문가 페이로드 내용과 모순되지 않아야 함
- `POST /search/candidates`와 `POST /recommend` 모두 유효한 응답 구조를 반환해야 함
