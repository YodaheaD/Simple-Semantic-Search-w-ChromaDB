# Example: Semantic Search with Vector DBs using Text Files

### This guide is similar to the "Example of Semantic Search with Vector DBs" but focuses on using actual text files as input instead of hardcoded strings to mimic a *real-world* ML or AI project.


### Instead of hardcoding documents in an array, you’d load and embed text files dynamically from a folder (`inputText/`).

- Note: Most of the guide is a chatGPT output, use it as a guide.

## What this guide teaches:
* How to read multiple text files from a folder
* Convert their content into embeddings (vectors)
* Store them in a vector database (ChromaDB)
* Perform semantic search queries against those files

---

Use a folder like this:

```
project/
│
├── inputText/
│   ├── doc1.txt
│   ├── doc2.txt
│   ├── doc3.txt
│
└── semantic_search.py ( or in a Jupyter notebook )
```

And have the script:

1. Read all `.txt` files in `inputText/`
2. Store them in Chroma with embeddings
3. Perform a semantic search query

---

## 🧩 Full Example

```python
import os
from sentence_transformers import SentenceTransformer
import chromadb

# 1️⃣ Initialize model and Chroma
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("text_files")

# 2️⃣ Read all .txt files from the folder
input_folder = "inputText"
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

# 3️⃣ Create embeddings
embeddings = model.encode(documents).tolist()

# 4️⃣ Store them in Chroma
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

print("\n🔍 Search Results:")
for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
    print(f"\n📄 File: {doc_id}\n{doc_text[:300]}...")  # show first 300 chars
```

---

## 🧰 How It Works

1. **`os.listdir(input_folder)`** — lists all `.txt` files
2. **`open(file_path, "r")`** — reads their content
3. **`model.encode()`** — converts each text into a vector
4. **`collection.add()`** — stores both text and embeddings
5. **`collection.query()`** — finds most semantically similar text(s)

---

## 🧾 Example Output

If The folder has:

* `ai_intro.txt`
* `cats.txt`
* `stocks.txt`

And you search for:

> “machine learning and intelligence”

You might get:

```
Loaded 3 text files from 'inputText'
All text files have been embedded and added to the vector database!

🔍 Search Results:

📄 File: ai_intro.txt
Artificial intelligence (AI) is the simulation of human intelligence processes by machines...

📄 File: stocks.txt
The stock market is influenced by AI-driven trading algorithms...
```

---

## 💡 Tip

If you plan to re-run this script often, consider:

* Persisting the Chroma database to disk:

  ```python
  chroma_client = chromadb.PersistentClient(path="./chroma_db")
  ```
* So embeddings are not recreated every run.
---

<br />
<br />
<br />
<br />

# Deeper Explanation of Code:

## 🧩 The Output Results Code

```python
print("\n🔍 Search Results for query:", query)
for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
    print(f"\n📄 File: {doc_id}\n{doc_text[:300]}...")  # show first 300 chars
```

---

## 1️⃣ Where `results` Comes From

This line earlier in The code:

```python
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)
```

asks the vector database:

> “Find the 2 most similar documents to this query embedding.”

Chroma returns a **dictionary** that looks like this:

```python
{
  "ids": [["doc1.txt", "doc3.txt"]],
  "documents": [[
      "Artificial Intelligence (AI) refers to ...",
      "The stock market allows companies to raise capital ..."
  ]],
  "distances": [[0.12, 0.34]]
}
```

Each top-level list (`[ ... ]`) represents a *batch* of queries — since you could query multiple embeddings at once.

That’s why you see `[0]` everywhere:
`results["ids"][0]` → the IDs for the **first query**
`results["documents"][0]` → the matching texts for that query

---

## 2️⃣ The `zip()` Function

```python
for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
```

Here’s what happens:

* `zip()` combines the two lists element-by-element:

  ```python
  zip(
    ["doc1.txt", "doc3.txt"],
    ["Artificial Intelligence ...", "The stock market allows ..."]
  )
  ```

  → becomes an iterable like:

  ```python
  [("doc1.txt", "Artificial Intelligence ..."),
   ("doc3.txt", "The stock market allows ...")]
  ```

So on each loop iteration:

* `doc_id` = `"doc1.txt"`
* `doc_text` = `"Artificial Intelligence (AI) refers to ..."`

---

## 3️⃣ The `print()` Formatting

```python
print(f"\n📄 File: {doc_id}\n{doc_text[:300]}...")
```

* `\n` → starts a new line (for nicer formatting)
* `📄 File: {doc_id}` → shows which file was matched
* `{doc_text[:300]}` → prints only the **first 300 characters** of the document to avoid flooding the console
* `...` → added manually to show there’s more text that’s not displayed

So you get neat, readable output:

```
📄 File: doc1.txt
Artificial Intelligence (AI) refers to the simulation ...
```

---

## 4️⃣ Why The Output Looks Like That

The query was:

```
What is artificial intelligence?
```

The vector embedding of that sentence lives near texts that **talk about AI** in semantic space.
When Chroma compared The query embedding against the stored document embeddings, it found:

| Rank | File         | Similarity Reason                                                                           |
| ---- | ------------ | ------------------------------------------------------------------------------------------- |
| 🥇 1 | **doc1.txt** | It directly defines AI — perfect semantic match                                             |
| 🥈 2 | **doc3.txt** | It mentions “artificial intelligence” in the context of finance and trading, still relevant |
| 🥉 3 | **doc2.txt** | Talks about dogs, no semantic overlap, so excluded                                          |

That’s why The output looked like:

```
📄 File: doc1.txt
Artificial Intelligence (AI) refers to the simulation of human intelligence...

📄 File: doc3.txt
The stock market allows companies to raise capital... Artificial intelligence and machine learning...
```

---

## 🧠 Summary

| Code Part                         | Purpose                                                |
| --------------------------------- | ------------------------------------------------------ |
| `results = collection.query(...)` | Retrieves most similar docs                            |
| `results["ids"][0]`               | IDs of matching docs for this query                    |
| `results["documents"][0]`         | Text of those docs                                     |
| `zip()`                           | Pairs each ID with its text                            |
| `[:300]`                          | Truncates output for readability                       |
| Output order                      | Based on similarity to The query (closest → farthest) |

---
