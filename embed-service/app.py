import chromadb
import uuid
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

app.mount("/static", StaticFiles(directory="../search-service"), name="static")
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="test-products")

class EmbedRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def serve_home():
    return FileResponse(os.path.join("../search-service", "index.html"))

@app.post("/embed")
async def embed(req: EmbedRequest):
    vectors = model.encode(req.texts).tolist()

    ids = [str(uuid.uuid4()) for _ in req.texts]

    collection.add(
        documents=req.texts,
        embeddings=vectors,
        metadatas=[{"text": text} for text in req.texts],
        ids=ids,
    )

    print(f"Added {len(ids)} items. Total in collection: {collection.count()}")

    return {"embeddings": vectors, "ids": ids}

@app.get("/search")
async def search(query: str, top_k: int = 5):
    """
    Search the collection for the most similar documents to the query.
    Args:
        query (str): The search query.
        top_k (int): The number of most similar documents to return.
    Returns:
        JSON response containing the most similar documents.
    """
    # Step 1: Encode the query into an embedding
    query_vector = model.encode([query]).tolist()

    # Step 2: Perform a search in the collection
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )

    # Step 3: Return the search results
    return {
        "query": query,
        "top_k": top_k,
        "results": results["documents"],
        "scores": results["distances"],  # optional, to return similarity scores
    }

@app.get("/count")
async def count_items():
    return {"count": collection.count()}

@app.get("/get_all")
async def get_all():
    return collection.get()