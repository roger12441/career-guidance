from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector
import os

# ✅ Make sure your Gemini API key is set in the environment
# Example (Windows PowerShell):
#   setx GOOGLE_API_KEY "your-gemini-key"
#
# Example (Linux/macOS):
#   export GOOGLE_API_KEY="your-gemini-key"

google_api_key = os.getenv("AIzaSyCSqcZqt7pUQIzP6VBlVd1pJ_E6Uh3FBfo")

# --- LLM (Gemini) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # fast + cost-effective
    google_api_key=google_api_key
)

# --- Embeddings (Gemini) ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# --- Neo4j Vector Store ---
vector_store = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="caarg343",
    index_name="career_embeddings",
    node_label="Document",
    text_node_property="text",
    embedding_node_property="embedding"
)

# Use vector store as retriever
retriever = vector_store.as_retriever()

# Simple test query
query = "What career paths are good for someone skilled in data analysis and AI?"
docs = retriever.get_relevant_documents(query)

print("Top results from Neo4j:")
for d in docs:
    print("-", d.page_content)
