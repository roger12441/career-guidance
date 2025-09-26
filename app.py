# app.py
from fastapi import FastAPI  # Not Fastapi
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","testpassword"))
# print(driver)
client = chromadb.Client()
# print(client.list/_collections())
# Example output: [{'name': 'Career'}]
with driver.session() as session:
    res = session.run("MATCH (p:Personality)-[:FITS_FOR]->(c:Career) RETURN p.id AS pid, c.id AS cid LIMIT 5")
    for record in res:
        print(record["pid"], record["cid"])



collection = client.get_collection("Career")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/recommend-careers/from-personality/{pid}")
def recommend_from_personality(pid: str, top: int = 5):
    q = """
    MATCH (p:Personality {id:$pid})-[:FITS_FOR]->(c:Career)
    RETURN c.id AS id, c.title AS title, c.description AS desc, c.popularity AS pop
    ORDER BY pop DESC
    LIMIT $top
    """
    with driver.session() as s:
        res = s.run(q, pid=pid, top=top)
        careers = [dict(r) for r in res]
    return {"careers": careers}

@app.get("/semantic-career-search")
def semantic_search(q: str, top: int = 5):
    emb = embed_model.encode(q).tolist()
    results = collection.query(query_embeddings=[emb], n_results=top)
    # results contains ids, documents, metadatas, distances
    out = []
    for i, cid in enumerate(results['ids'][0]):
        out.append({"id": cid, "score": results['distances'][0][i], "title": results['metadatas'][0][i]['title']})
    return {"results": out}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)

