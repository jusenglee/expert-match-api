"""WO-A: v2.0 chunk 적재 변환·검증 파이프라인 (소스 → chunk payload).

오프라인 변환·검증 패키지. 실제 Qdrant 컬렉션 생성/upsert는 WO-B, 컷오버는 WO-D 소관.
chunk_id 코덱·doc_type/family 상수는 apps.search.doc_types(WO-0)를 단일 출처로 사용한다.
"""
