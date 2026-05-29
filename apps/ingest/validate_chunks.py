"""WO-A: 적재 정합성 불변식 검증 (DATA_MODEL.md §6 7개 항목).

GOLDEN_TESTS.md 시나리오 1·7과 연결. CLI: `python -m apps.ingest.validate_chunks <chunks.jsonl>`.
위반이 있으면 비0 exit + 리포트 출력.

§6-7 불용어 주의: "평가위원"은 assessor doc_type의 정당한 도메인 어휘(평가위원회명/핵심평가위원,
샘플 7.4/7.5)이므로 불용어에 넣지 않는다 — 넣으면 권위 샘플이 위반 처리된다. 요청측 액션/페르소나
표현과 v1.x 누수 구문에 한정한다.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from apps.domain.models import ChunkPayload, ResearcherMeta
from apps.search.doc_types import DOC_TYPES

#: §6-7 — 적재 본문에 들어오면 안 되는 요청(쿼리) 어투/액션·v1.x 누수 구문.
ROLE_ACTION_STOPWORDS: tuple[str, ...] = (
    "추천",
    "추천해",
    "추천해줘",
    "찾아줘",
    "심사평가위원 활동",  # v1.x seed_data 누수 구문
    "심사참여건수",        # v1.x seed_data 누수 구문
)


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    chunk_id: str | None = None


@dataclass
class ValidationReport:
    checked: int = 0
    violations: list[Violation] = field(default_factory=list)
    warnings: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def render(self) -> str:
        lines = [f"checked={self.checked} violations={len(self.violations)} warnings={len(self.warnings)}"]
        for v in self.violations:
            lines.append(f"  [VIOLATION:{v.code}] {v.chunk_id or '-'}: {v.message}")
        for w in self.warnings:
            lines.append(f"  [warn:{w.code}] {w.chunk_id or '-'}: {w.message}")
        if self.ok:
            lines.append("all invariants passed")
        return "\n".join(lines)


def _year_of(event_date: str | None) -> int | None:
    if not event_date:
        return None
    try:
        return int(str(event_date)[:4])
    except ValueError:
        return None


def validate_chunks(
    chunks: list[ChunkPayload],
    *,
    embedding_hashes: dict[str, str] | None = None,
    stopwords: tuple[str, ...] = ROLE_ACTION_STOPWORDS,
) -> ValidationReport:
    report = ValidationReport(checked=len(chunks))
    valid_doc_types = set(DOC_TYPES)

    # §6-1: chunk_id 전역 유일
    seen_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            report.violations.append(
                Violation("dup_chunk_id", "chunk_id가 중복됨(전역 유일 위반)", chunk.chunk_id)
            )
        seen_ids.add(chunk.chunk_id)

    # §6-2: researcher_id별 researcher_meta/researcher_name 동일
    by_researcher: dict[str, list[ChunkPayload]] = {}
    for chunk in chunks:
        by_researcher.setdefault(chunk.researcher_id, []).append(chunk)
    for researcher_id, group in by_researcher.items():
        reference = group[0]
        ref_meta = reference.researcher_meta.model_dump()
        for chunk in group[1:]:
            if chunk.researcher_name != reference.researcher_name:
                report.violations.append(
                    Violation(
                        "meta_name_mismatch",
                        f"researcher_id={researcher_id}의 researcher_name 불일치",
                        chunk.chunk_id,
                    )
                )
            if chunk.researcher_meta.model_dump() != ref_meta:
                report.violations.append(
                    Violation(
                        "meta_mismatch",
                        f"researcher_id={researcher_id}의 researcher_meta 불일치",
                        chunk.chunk_id,
                    )
                )

    for chunk in chunks:
        # §6-3: doc_type enum
        if chunk.doc_type not in valid_doc_types:
            report.violations.append(
                Violation("bad_doc_type", f"정의되지 않은 doc_type: {chunk.doc_type!r}", chunk.chunk_id)
            )

        # §6-4: event_year == year(event_date), 둘 다 null 허용
        has_date = chunk.event_date is not None
        has_year = chunk.event_year is not None
        if has_date != has_year:
            report.violations.append(
                Violation(
                    "event_partial_null",
                    f"event_date/event_year 한쪽만 값 존재(date={chunk.event_date}, year={chunk.event_year})",
                    chunk.chunk_id,
                )
            )
        elif has_date and _year_of(chunk.event_date) != chunk.event_year:
            report.violations.append(
                Violation(
                    "event_year_mismatch",
                    f"event_year({chunk.event_year}) != year(event_date {chunk.event_date})",
                    chunk.chunk_id,
                )
            )

        # §6-5: tags 소문자·trim
        for tag in chunk.tags:
            if tag != tag.strip().lower():
                report.violations.append(
                    Violation("tag_not_normalized", f"tag가 소문자·trim 아님: {tag!r}", chunk.chunk_id)
                )

        # §6-6: dense/sparse 동일 chunk_text (임베딩 해시 제공 시에만)
        if embedding_hashes is not None:
            from apps.ingest.embedder import chunk_text_sha256

            recorded = embedding_hashes.get(chunk.chunk_id)
            if recorded is None:
                report.warnings.append(
                    Violation("no_embedding_hash", "임베딩 입력 해시 없음(검증 생략)", chunk.chunk_id)
                )
            elif recorded != chunk_text_sha256(chunk.chunk_text):
                report.violations.append(
                    Violation(
                        "embedding_input_mismatch",
                        "dense/sparse 입력이 chunk_text와 불일치",
                        chunk.chunk_id,
                    )
                )

        # §6-7: chunk_text에 요청 어투/액션·v1.x 누수 구문 없음
        for stop in stopwords:
            if stop in chunk.chunk_text:
                report.violations.append(
                    Violation("role_action_leak", f"chunk_text에 불용어 포함: {stop!r}", chunk.chunk_id)
                )

    return report


def _load_jsonl(path: str) -> list[ChunkPayload]:
    chunks: list[ChunkPayload] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(ChunkPayload.model_validate(json.loads(line)))
    return chunks


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m apps.ingest.validate_chunks <chunks.jsonl>", file=sys.stderr)
        return 2
    chunks = _load_jsonl(args[0])
    report = validate_chunks(chunks)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
