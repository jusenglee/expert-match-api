"""
Qdrant 컬렉션과 코드 간의 데이터 규약(Contract)이 일치하는지 점검하는 CLI 도구입니다.
실제 DB 서버에 접속하여 필드 존재 여부, 데이터 타입 등을 확인합니다.
"""

from __future__ import annotations

import json

from qdrant_client import QdrantClient

from apps.core.config import get_settings
from apps.core.runtime_validation import validate_runtime_settings
from apps.search.encoders import SpladeSparseEncoder
from apps.search.live_validator import LiveContractValidator
from apps.search.schema_registry import SearchSchemaRegistry
from apps.search.sparse_runtime import (
    prepare_sparse_runtime_environment,
    resolve_sparse_runtime,
)


def main() -> int:
    """
    설정 정보를 로드하고 Qdrant 데이터 상태를 검증합니다.
    검증 결과 보고서를 JSON 형식으로 출력하며, 실패 시 종료 코드 1을 반환합니다.
    """
    settings = get_settings()
    # 운영 환경 설정 준수 여부 확인
    validate_runtime_settings(settings)

    # Qdrant 클라이언트 초기화
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.qdrant_cloud_inference,
    )
    cache_dir = prepare_sparse_runtime_environment(settings)
    sparse_runtime, _ = resolve_sparse_runtime(
        client=client,
        settings=settings,
        cache_dir=cache_dir,
        sparse_encoder_factory=SpladeSparseEncoder,
    )
    
    # 실시간 데이터 규약 검증기 생성 및 실행
    validator = LiveContractValidator(
        client=client,
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        sparse_runtime=sparse_runtime,
    )
    
    report = validator.validate()
    
    # 결과를 한글이 포함된 읽기 좋은 JSON 스냅샷으로 출력
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    
    # 검증 성공 시 0, 실패 시 1 반환
    return 0 if report.ready else 1


if __name__ == "__main__":
    # 시스템 종료 코드를 포함하여 실행
    raise SystemExit(main())
