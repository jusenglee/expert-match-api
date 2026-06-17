"""
[외부 중계 API] POST /openAPI 라이브 HTTP 검증 스크립트.

실행 중인 서버의 /openAPI에 실제 요청을 보내고, 응답이 외부 계약과
필드 단위로 일치하는지 검증한다. 단위 테스트(tests/test_api.py)는 가짜
서비스를 쓰므로, 본 스크립트는 실제 서버·데이터로 계약을 점검하는 보완재다.

검증 대상 계약:
    { "totalCount": int,
      "items": [ { "researcherId": str,
                   "score": float(0~1),
                   "reason": str,
                   "evidences": [ { "type": <6종 enum>,
                                    "title": str,
                                    "date": str | null } ] } ] }
  · 최상위/항목/근거의 키 집합이 정확히 일치(내부 trace 등 누출 없음)
  · score 0~1, evidence.type enum, date 는 문자열 또는 null

종료 코드: 모든 검증 통과=0, 계약 위반=1, 실행 오류(서버 연결 실패 등)=2.

기본적으로 응답 본문 JSON 을 검증 결과와 함께 로그에 출력한다(--quiet 로 생략).

사용 예:
    python -m apps.tools.verify_openapi
    python -m apps.tools.verify_openapi --base-url http://localhost:8000 --query "인공지능 반도체 박사급 전문가 추천"
    OPENAPI_BASE_URL=http://10.0.0.5:8000 python -m apps.tools.verify_openapi --require-results
    python -m apps.tools.verify_openapi --quiet   # 응답 본문 생략, 검증 결과만
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

# 외부 계약상 evidences[*].type 로 허용되는 값(내부 EvidenceItem.type 와 동일).
EVIDENCE_TYPES = frozenset(
    {"paper", "patent", "project", "assessor_activity", "specialty", "profile"}
)
DEFAULT_BASE_URL = "http://203.250.234.159:8013"
DEFAULT_QUERY = (
    "인공지능 반도체 분야에서 최근 국책 과제 수행 경험이 있는 박사급 전문가 추천"
)


class ContractChecker:
    """외부 계약 검증 결과(통과 여부, 이름, 상세)를 순서대로 누적한다."""

    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.results.append((bool(ok), name, detail))
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(ok for ok, _, _ in self.results)

    def report(self) -> str:
        lines = []
        for ok, name, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            suffix = f"  ({detail})" if detail else ""
            lines.append(f"  [{mark}] {name}{suffix}")
        return "\n".join(lines)


def _is_number(value: object) -> bool:
    # JSON 불리언은 파이썬에서 int 의 하위형이므로 명시적으로 배제한다.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_contract(payload: object, *, require_results: bool) -> ContractChecker:
    """응답 payload 를 외부 계약과 대조해 검증 결과를 반환한다."""
    c = ContractChecker()

    if not c.check(isinstance(payload, dict), "응답 본문이 JSON object"):
        return c

    top_keys = set(payload.keys())
    c.check(
        top_keys == {"totalCount", "items"},
        "최상위 키가 정확히 {totalCount, items}",
        f"실제: {sorted(top_keys)}",
    )

    total = payload.get("totalCount")
    items = payload.get("items")

    c.check(
        isinstance(total, int) and not isinstance(total, bool),
        "totalCount 가 정수",
        f"실제: {total!r}",
    )
    is_list = c.check(
        isinstance(items, list),
        "items 가 배열",
        f"실제 타입: {type(items).__name__}",
    )

    if is_list:
        c.check(
            isinstance(total, int) and total == len(items),
            "totalCount == items 길이",
            f"totalCount={total}, len(items)={len(items)}",
        )
        if require_results:
            c.check(
                len(items) > 0,
                "items 가 비어있지 않음 (--require-results)",
                f"len={len(items)}",
            )
        for i, item in enumerate(items):
            _validate_item(c, i, item)

    return c


def _validate_item(c: ContractChecker, i: int, item: object) -> None:
    p = f"items[{i}]"
    if not c.check(isinstance(item, dict), f"{p} 가 object"):
        return

    keys = set(item.keys())
    c.check(
        keys == {"researcherId", "score", "reason", "evidences"},
        f"{p} 키가 정확히 {{researcherId, score, reason, evidences}}",
        f"실제: {sorted(keys)}",
    )

    researcher_id = item.get("researcherId")
    c.check(
        isinstance(researcher_id, str) and researcher_id != "",
        f"{p}.researcherId 가 비어있지 않은 문자열",
        f"실제: {researcher_id!r}",
    )

    score = item.get("score")
    c.check(
        _is_number(score) and 0.0 <= float(score) <= 1.0,
        f"{p}.score 가 0~1 범위 숫자",
        f"실제: {score!r}",
    )

    c.check(
        isinstance(item.get("reason"), str),
        f"{p}.reason 가 문자열",
        f"실제 타입: {type(item.get('reason')).__name__}",
    )

    evidences = item.get("evidences")
    if c.check(
        isinstance(evidences, list),
        f"{p}.evidences 가 배열",
        f"실제 타입: {type(evidences).__name__}",
    ):
        for j, ev in enumerate(evidences):
            _validate_evidence(c, p, j, ev)


def _validate_evidence(c: ContractChecker, parent: str, j: int, ev: object) -> None:
    p = f"{parent}.evidences[{j}]"
    if not c.check(isinstance(ev, dict), f"{p} 가 object"):
        return

    keys = set(ev.keys())
    c.check(
        keys == {"type", "title", "date"},
        f"{p} 키가 정확히 {{type, title, date}}",
        f"실제: {sorted(keys)}",
    )
    c.check(
        ev.get("type") in EVIDENCE_TYPES,
        f"{p}.type 가 허용 enum",
        f"실제: {ev.get('type')!r}",
    )
    c.check(
        isinstance(ev.get("title"), str),
        f"{p}.title 가 문자열",
        f"실제 타입: {type(ev.get('title')).__name__}",
    )
    date = ev.get("date")
    c.check(
        date is None or isinstance(date, str),
        f"{p}.date 가 문자열 또는 null",
        f"실제: {date!r}",
    )


def main() -> int:
    # 윈도우 콘솔(cp949)에서 한글 출력 시 UnicodeEncodeError 방지.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="라이브 /openAPI 외부 계약 검증 스크립트"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAPI_BASE_URL", DEFAULT_BASE_URL),
        help=f"대상 서버 base URL (기본: {DEFAULT_BASE_URL}, 환경변수 OPENAPI_BASE_URL)",
    )
    parser.add_argument(
        "--query", default=DEFAULT_QUERY, help="검증에 사용할 자연어 질의"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="요청 타임아웃(초). LLM 추론으로 느릴 수 있어 넉넉히 둔다",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="items 가 비어있으면 실패 처리(데이터 적재 여부 확인용)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="응답 본문 JSON 출력을 생략하고 검증 결과만 표시(기본은 응답도 출력)",
    )
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/openAPI"
    print(f"POST {url}")
    print(f"  query={args.query!r}")

    try:
        resp = httpx.post(url, json={"query": args.query}, timeout=args.timeout)
    except httpx.HTTPError as exc:
        print(f"[실행 오류] 요청 실패: {exc}", file=sys.stderr)
        return 2

    print(f"  status={resp.status_code}")

    c = ContractChecker()
    status_ok = c.check(resp.status_code == 200, "HTTP 200", f"실제: {resp.status_code}")

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        # JSON 이 아니면 원문 그대로 로그에 남긴다.
        print("--- 응답 본문 (원문) ---")
        print(resp.text)
        c.check(False, "응답이 유효한 JSON", resp.text[:200])
        print("--- 검증 결과 ---")
        print(c.report())
        return 1

    # 응답 본문을 기본으로 로그에 출력한다(--quiet 로 생략 가능).
    if not args.quiet:
        print("--- 응답 본문 ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if status_ok:
        body = validate_contract(payload, require_results=args.require_results)
        c.results.extend(body.results)
    elif isinstance(payload, dict) and "detail" in payload:
        # 200 이 아니면 서버가 준 사유를 그대로 보여준다(503/422 등).
        print(f"  detail={payload.get('detail')!r}")

    print("--- 검증 결과 ---")
    print(c.report())

    total = len(c.results)
    failed = sum(1 for ok, _, _ in c.results if not ok)
    if c.passed:
        item_count = payload.get("totalCount") if isinstance(payload, dict) else "?"
        print(f"\n전체 통과: {total}개 검증 통과 (totalCount={item_count})")
        return 0
    print(f"\n실패: {failed}/{total}개 검증 실패")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
