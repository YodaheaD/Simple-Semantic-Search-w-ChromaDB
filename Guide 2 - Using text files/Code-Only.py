import os
from sentence_transformers import SentenceTransformer
import chromadb

# 1️⃣ Initialize model and Chroma
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("text_files")


# 2️⃣ Read all .txt files from the folder
input_folder = f"C:\\Users\\yodah\\OneDrive\\Desktop\\Code-Work\\Python\\Practice\\Vector-DB-Practice\\inputText"
documents = []
file_ids = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            documents.append(content)
            file_ids.append(filename)  # Use filename as unique ID

print(f"Loaded {len(documents)} text files from '{input_folder}'")

# 3️⃣ Generate embeddings for the documents
embeddings = model.encode(documents).tolist()

# 4️⃣ Add documents to ChromaDB
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=file_ids
)
print("All text files have been embedded and added to the vector database!")
# 5️⃣ Perform a semantic search
query = "What is artificial intelligence?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)


# 6️⃣ Display results
print("\n🔍 Search Results for query:", query)
for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
    print(f"\n📄 File: {doc_id}\n{doc_text[:300]}...")  # show first 300 chars