import asyncio
from qdrant_client import QdrantClient
from apps.core.config import Settings

async def main():
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    # Search for points that have "한국과학기술정보연구원" in "affiliated_organization"
    results = client.scroll(
        collection_name=settings.qdrant_collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "basic_info.affiliated_organization",
                    "match": {"text": "한국과학기술정보연구원"}
                }
            ]
        },
        limit=10,
        with_payload=True
    )
    
    points, _ = results
    print(f"Found {len(points)} points that contain '한국과학기술정보연구원'.")
    for point in points:
        org = point.payload.get("basic_info", {}).get("affiliated_organization")
        org_exact = point.payload.get("basic_info", {}).get("affiliated_organization_exact")
        print(f"ID: {point.id}, Name: {point.payload.get('basic_info', {}).get('researcher_name')}")
        print(f"  Org: {org}")
        print(f"  Org Exact: {org_exact}")

if __name__ == "__main__":
    asyncio.run(main())
