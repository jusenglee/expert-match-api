import asyncio
from qdrant_client import QdrantClient
from apps.core.config import Settings
from apps.search.filters import QdrantFilterCompiler
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import SearchSchemaRegistry
from apps.search.query_builder import QueryTextBuilder
from apps.api.main import build_dense_encoder

async def main():
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=20.0)
    registry = SearchSchemaRegistry.default()
    dense_encoder = build_dense_encoder(settings)
    query_builder = QueryTextBuilder()
    
    retriever = QdrantHybridRetriever(
        client=client,
        settings=settings,
        registry=registry,
        dense_encoder=dense_encoder,
        query_builder=query_builder,
    )
    
    from apps.domain.models import PlannerOutput
    filter_compiler = QdrantFilterCompiler()
    
    query = "반도체"
    exclude_orgs = ["주식회사 미소테크"]
    print(f"Testing exclude_orgs: {exclude_orgs}")
    
    plan = PlannerOutput(
        intent_summary="test", 
        core_keywords=["test"], 
        exclude_orgs=exclude_orgs
    )
    
    query_filter = filter_compiler.compile(
        hard_filters={},
        exclude_orgs=exclude_orgs,
        include_orgs=[]
    )
    
    result = await retriever.search(
        query=query,
        plan=plan,
        query_filter=query_filter
    )
    
    print(f"Retrieved {len(result.hits)} candidates.")
    
    excluded_found = False
    for candidate in result.hits:
        org = candidate.payload.basic_info.affiliated_organization
        print(f"Name: {candidate.payload.basic_info.researcher_name}, Org: {org}")
        if org == "주식회사 미소테크":
            excluded_found = True
            
    if excluded_found:
        print("[FAIL] Excluded org was found in the results!")
    else:
        print("[SUCCESS] Excluded org was successfully filtered out.")

if __name__ == "__main__":
    asyncio.run(main())
