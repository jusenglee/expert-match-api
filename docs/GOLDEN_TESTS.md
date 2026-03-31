# Golden Tests

## 시나리오

### 1. 논문 중심 질의

- 입력: `최근 5년 SCIE 논문이 있는 AI 반도체 평가위원 추천`
- 기대:
  - `art_sci_slct_nm=SCIE`
  - `art_recent_years=5`
  - 추천 상위권에 논문 evidence 존재

### 2. 특허 중심 질의

- 입력: `사업화 경험과 등록 특허가 있는 전문가 추천`
- 기대:
  - `patent_cnt_min >= 1`
  - 특허 evidence 존재

### 3. 과제 중심 질의

- 입력: `AI 반도체 과제 수행 경험이 많은 평가위원 추천`
- 기대:
  - `project_cnt_min >= 1`
  - 과제 evidence 존재

### 4. 기관 제외 질의

- 입력: `A기관 제외하고 추천`
- 기대:
  - `blng_org_nm_exact` 기반 exclude 동작
  - 제외기관 후보 미포함

### 5. 근거 부족 질의

- 입력: `특허와 논문이 모두 강한 전문가 추천`
- 기대:
  - 근거 부족 후보는 `data_gaps` 또는 `risks`로 표기

## 수용 기준

- hard filter 위반 0건
- `searched_branches`는 항상 `basic/art/pat/pjt`
- 추천 사유와 evidence가 payload와 충돌하지 않음
- `POST /search/candidates`와 `POST /recommend`가 모두 응답 가능

