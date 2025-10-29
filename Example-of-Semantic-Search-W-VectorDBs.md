
## Example of Semantic Search with Vector DBs
---
### Overview

### <ul>What is Semantic Search?
- Semantic Search allows you to find documents based on meaning rather than exact keyword matches. It uses embeddings (numerical representations of text) to capture the semantic content of documents.

### <ul>How is Semantic Search used in ML/AI?
- It’s widely used in applications like chatbots, recommendation systems, and information retrieval to provide more relevant results based on user intent.

### <ul>What is a Vector Database?
- A Vector Database stores these embeddings and enables efficient similarity searches. ChromaDB is one such open-source vector database.

### <ul>Why use a Vector Database for Semantic Search?
- **Scalability** and  **Speed**: Fast retrieval of similar documents.

### <ul>How are vector DBs used in real-world applications?
- **Customer Support**: Quickly find relevant help articles based on user queries.
- **Content Recommendation**: Suggest articles or products based on user interests.
- **Document Retrieval**: Locate relevant documents in large corpora (e.g., legal, medical).

### <ul>What is RAG (Retrieval-Augmented Generation)?
* RAG combines retrieval of relevant documents with generative models (like GPT) to provide accurate and context-aware responses. 
* It enhances the capabilities of language models by grounding their outputs in real-world data.
* In this guide, we’ll focus on the retrieval part using semantic search with a vector database.

<br />
<br />

---

## 🧠 Step-by-Step: Semantic Search with a Vector Database

### **1. Install dependencies**

```bash
pip install chromadb sentence-transformers
```

* `chromadb` → local vector database
* `sentence-transformers` → for creating text embeddings

---

### **2. Import and initialize**

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a Chroma client (in-memory DB)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")
```

---

### **3. Add some example documents**

```python
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
```

---

### **4. Perform a similarity search**

Let’s say a user asks:

> “Tell me something about machine learning.”

```python
query = "machine learning and AI"
query_embedding = model.encode([query]).tolist()

# Search for the most similar documents
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print(results)
```

---

### **5. Output Example**

You’ll get something like:

```python
{
  'ids': [['doc3', 'doc4']],
  'documents': [['Artificial Intelligence is transforming the world.',
                 'I love programming in Python.']],
  'distances': [[0.12, 0.34]]
}
```

💡 This means:
The most relevant document to *“machine learning and AI”* is:

> “Artificial Intelligence is transforming the world.”

---

### **6. What You Just Did**

✅ Converted text → embeddings (vectors)
✅ Stored embeddings in a vector database
✅ Queried by semantic meaning, not by exact keywords

That’s the **core of Retrieval-Augmented Generation (RAG)** used in systems like ChatGPT with external knowledge.

---
