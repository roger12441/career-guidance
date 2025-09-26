# embed_store.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.Client()

# Create (or get) a collection for careers
collection_name = "careers"
try:
    collection = client.create_collection(collection_name)
except Exception:
    # If already exists, just get it
    collection = client.get_collection(collection_name)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your careers CSV
df = pd.read_csv("data/careers.csv")

# Loop through each row and add to ChromaDB
for _, row in df.iterrows():
    emb = embed_model.encode(row['description']).tolist()
    collection.add(
        ids=[str(row['id'])],  # unique ID for each entry
        documents=[row['description']],  # text to embed
        metadatas=[{
            "careers": row['careers'],
            "skills": row['required_skills']
        }]
    )

print(f"✅ Successfully stored {len(df)} careers into ChromaDB collection '{collection_name}'.")
