from pinecone import Pinecone
from typing import List, Dict, Any
from settings import Settings

settings = Settings()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def get_pinecone_index():
    global pc_index
    if pc_index is None:
        pc_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        pc_index = pc_client.Index(host=settings.PINECONE_INDEX_HOST)
    return pc_index


async def pinecone_upsert(namespace: str, records: List[Dict[str, Any]]):
    index = get_pinecone_index()
    index.upsert_records(namespace, records)


async def pinecone_query(namespace: str, query_text: str, top_k: int = 5):
    index = get_pinecone_index()
    results = index.search(
        namespace=namespace,
        query={"inputs": {"text": query_text}},
        top_k=top_k,
        fields=["chunk_text", "category"],
    )
    return results


async def upsert_chunks_to_pinecone(space_id: str, doc_id: str, chunks: List[str]):
    index = get_pinecone_index()
    records = []
    for i, ch in enumerate(chunks):
        records.append(
            {
                "_id": f"{doc_id}:{i}",
                "text": ch,
                "space_id": space_id,
                "doc_id": doc_id,
                "chunk_no": i,
            }
        )
    index.upsert_records(space_id, records)


async def query_pinecone(space_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    index = get_pinecone_index()
    print("space id", space_id, query, top_k)
    res = index.search(
        namespace=space_id,
        query={"inputs": {"text": query}, "top_k": top_k},
        fields=["text", "doc_id", "chunk_no"],
    )
    res = res.get("result").get("hits", [])
    matches = []
    print("matches", res)
    for m in res:
        matches.append(
            {
                "id": m["_id"],
                "score": m["_score"],
                "text": m.get("fields", {}).get("text", ""),
                "doc_id": m.get("fields", {}).get("doc_id", ""),
            }
        )
    return matches