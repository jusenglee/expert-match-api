# Golden Tests

## Scenarios

### 1. Publication-focused Query

- Input: `Recommend AI semiconductor reviewers with recent SCIE publications in the last 5 years`
- Expect:
  - `art_sci_slct_nm=SCIE`
  - `art_recent_years=5`
  - recommendation evidence includes at least one paper item

### 2. Patent-focused Query

- Input: `Recommend experts with registered patents and commercialization experience`
- Expect:
  - `patent_cnt_min >= 1`
  - recommendation evidence includes at least one patent item

### 3. Project-focused Query

- Input: `Recommend reviewers with strong AI semiconductor project delivery experience`
- Expect:
  - `project_cnt_min >= 1`
  - recommendation evidence includes at least one project item

### 4. Excluded Organization Query

- Input: `Recommend reviewers but exclude A Organization`
- Expect:
  - exclude filtering is applied
  - excluded organizations do not appear in the final response

### 5. Partial-evidence Query

- Input: `Recommend experts who are strong in both patents and papers`
- Expect:
  - gaps are surfaced through `data_gaps` or `risks`
  - recommendations without evidence are removed from the final response

### 6. Empty Recommendation Result

- Input: a query whose hard filters are too strict or whose evidence coverage is insufficient
- Expect:
  - `POST /recommend` returns `200`
  - `recommendations=[]`
  - the reason is surfaced in `not_selected_reasons` and/or `data_gaps`

### 7. Zero-hit Retrieval

- Input: a query that retrieves zero candidates
- Expect:
  - `POST /recommend` returns `200`
  - the judge stage is skipped
  - `recommendations=[]`
  - `not_selected_reasons` includes a no-candidate explanation

## Acceptance Criteria

- hard-filter violations are excluded from the final response
- `searched_branches` remains `basic`, `art`, `pat`, `pjt`
- recommendation reasons and evidence do not contradict the payload
- both `POST /search/candidates` and `POST /recommend` return valid response shapes
