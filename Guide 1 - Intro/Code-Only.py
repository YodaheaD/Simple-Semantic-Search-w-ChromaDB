from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a Chroma client (in-memory DB)
chroma_client = chromadb.Client()

# Docs already exists, so we just get it
collection = chroma_client.get_collection("docs")
# If the collection did not exist, you would create it like this:
#collection = chroma_client.create_collection("docs")

documents = [
    "The cat sat on the mat.",
    "Dogs are loyal animals.",
    "The stock market is volatile today.",
    "Artificial Intelligence is transforming the world.",
    "I love programming in Python."
]

# Convert documents into vectors (embeddings)
embeddings = model.encode(documents).tolist()

# Store them in the vector database
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc{i}" for i in range(len(documents))]
)

query = "machine learning and AI"
query_embedding = model.encode([query]).tolist()

# Search for the most similar documents
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print(results)

# Optional: Loop results for better formatting
print("\n🔍 Search Results for query:", query)

for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
    print(f"\n📄 File: {doc_id}\n{doc_text[:300]}...")  # show first 300 chars