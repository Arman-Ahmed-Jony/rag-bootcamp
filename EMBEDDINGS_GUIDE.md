# Embeddings Guide for RAG Systems

A comprehensive guide to understanding and implementing embeddings in Retrieval-Augmented Generation (RAG) systems.

## Table of Contents

1. [What are Embeddings?](#what-are-embeddings)
2. [Sentence Transformers vs LangChain](#sentence-transformers-vs-langchain)
3. [Embedding Models Comparison](#embedding-models-comparison)
4. [Dual Embedding Architecture](#dual-embedding-architecture)
5. [Implementation Examples](#implementation-examples)
6. [Performance Optimization](#performance-optimization)
7. [Best Practices](#best-practices)

## What are Embeddings?

**Embeddings** are numerical vector representations of text that capture semantic meaning. Similar texts get similar vectors, making them perfect for:

- **Semantic search** (finding similar content)
- **RAG systems** (retrieving relevant documents)
- **Text similarity** (comparing documents)
- **Clustering** (grouping similar texts)

### Key Concepts

```python
# Text → Vector
text = "Python is a programming language"
embedding = model.encode(text)  # [0.1, -0.3, 0.8, ...] (384 dimensions)

# Similar texts have similar vectors
text1 = "Python programming"
text2 = "Python coding"
similarity = cosine_similarity(embedding1, embedding2)  # High similarity
```

## Sentence Transformers vs LangChain

### Sentence Transformers (Direct)

```python
from sentence_transformers import SentenceTransformer

# Direct control
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["text1", "text2"])
```

**Pros:**
- Full control over model
- Slightly faster
- Direct access to all features

**Cons:**
- Manual integration with vector stores
- More code for common operations

### LangChain HuggingFaceEmbeddings

```python
from langchain.embeddings import HuggingFaceEmbeddings

# LangChain integration
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
```

**Pros:**
- Built-in integration with vector stores
- Less code for common operations
- Easy to switch between providers
- Automatic caching

**Cons:**
- Slight overhead
- Less direct control

### Recommendation

**Use LangChain's `HuggingFaceEmbeddings`** for RAG systems - it provides the best balance of functionality and ease of use.

## Embedding Models Comparison

### Popular Models

| Model | Parameters | Dimensions | Speed | Quality | Use Case |
|-------|------------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 22M | 384 | ⚡⚡⚡ | ⭐⭐⭐ | General purpose, fast |
| `all-mpnet-base-v2` | 110M | 768 | ⚡⚡ | ⭐⭐⭐⭐ | High quality |
| `all-MiniLM-L12-v2` | 33M | 384 | ⚡⚡ | ⭐⭐⭐⭐ | Better quality, still fast |
| `paraphrase-multilingual-MiniLM-L12-v2` | 33M | 384 | ⚡⚡ | ⭐⭐⭐⭐ | Multilingual |

### Model Selection Guide

```python
# For RAG systems - start here
model = SentenceTransformer('all-MiniLM-L6-v2')

# For higher quality
model = SentenceTransformer('all-mpnet-base-v2')

# For multilingual content
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# For specific domains
model = SentenceTransformer('allenai-specter')  # Scientific papers
model = SentenceTransformer('microsoft/codebert-base')  # Code
```

### Performance Comparison

```python
import time
from sentence_transformers import SentenceTransformer

def benchmark_models():
    models = {
        'MiniLM-L6': 'all-MiniLM-L6-v2',
        'MPNet': 'all-mpnet-base-v2',
        'Multilingual': 'paraphrase-multilingual-MiniLM-L12-v2'
    }
    
    texts = ["Python is a programming language", "Machine learning uses algorithms"]
    
    for name, model_name in models.items():
        model = SentenceTransformer(model_name)
        
        start = time.time()
        embeddings = model.encode(texts)
        end = time.time()
        
        print(f"{name}: {end-start:.3f}s, Shape: {embeddings.shape}")
```

## Dual Embedding Architecture

**Smart optimization**: Use different models for storage vs. querying to get the best of both worlds.

### Concept

- **Storage/Indexing**: Fast, smaller model (e.g., `all-MiniLM-L6-v2`)
- **Query/Similarity**: Higher quality model (e.g., `all-mpnet-base-v2`)

### Benefits

- ✅ **Storage efficiency**: Smaller embeddings = less memory
- ✅ **Query quality**: Better model = more accurate similarity
- ✅ **Speed**: Fast indexing, quality retrieval

### Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DualEmbeddingRAG:
    def __init__(self):
        # Fast model for storage
        self.storage_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Quality model for queries
        self.query_model = SentenceTransformer('all-mpnet-base-v2')
        
        self.documents = []
        self.storage_embeddings = None
    
    def add_documents(self, texts):
        """Store documents using fast model"""
        self.documents = texts
        self.storage_embeddings = self.storage_model.encode(texts)
    
    def search(self, query, top_k=3):
        """Search using quality model"""
        # Encode query with quality model
        query_embedding = self.query_model.encode([query])
        
        # Encode all documents with quality model for comparison
        doc_embeddings = self.query_model.encode(self.documents)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        
        return results
```

## Implementation Examples

### Basic RAG with LangChain

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

class SimpleRAG:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def load_documents(self, file_paths):
        """Load and process documents"""
        all_docs = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(all_docs)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        return len(texts)
    
    def search(self, query, k=3):
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

# Usage
rag = SimpleRAG()
file_paths = ["../data/pdf_files/document.pdf"]
num_chunks = rag.load_documents(file_paths)
print(f"Loaded {num_chunks} document chunks")

results = rag.search("What is machine learning?", k=3)
for doc in results:
    print(f"Content: {doc.page_content[:200]}...")
```

### Advanced RAG with Dual Embeddings

```python
class AdvancedRAG:
    def __init__(self):
        # Fast model for storage
        self.storage_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Quality model for queries
        self.query_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        self.vectorstore = None
        self.documents = []
    
    def add_documents(self, texts):
        """Store with fast model"""
        self.documents = texts
        self.vectorstore = FAISS.from_texts(texts, self.storage_embeddings)
    
    def search(self, query, top_k=3):
        """Search with quality model"""
        # Get candidates using fast model
        candidates = self.vectorstore.similarity_search(query, k=top_k*2)
        
        # Re-rank with quality model
        candidate_texts = [doc.page_content for doc in candidates]
        query_embedding = self.query_embeddings.embed_query(query)
        candidate_embeddings = self.query_embeddings.embed_documents(candidate_texts)
        
        # Calculate quality similarities
        similarities = []
        for i, embedding in enumerate(candidate_embeddings):
            sim = np.dot(query_embedding, embedding)
            similarities.append((candidates[i], sim))
        
        # Sort by quality similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
```

## Document Chunking and Embedding Sizes

### How Document Size Affects Embedding Storage

**Key Point**: Embedding models don't process entire documents at once. They encode text chunks, and the storage depends on how many chunks your document gets split into.

### Example: 1,000-Word Document

```python
# Document characteristics
document_words = 1000
avg_chars_per_word = 5
total_chars = document_words * avg_chars_per_word  # ~5,000 characters

# Model token limits
miniLM_max_tokens = 512  # ~350-400 words
mpnet_max_tokens = 512   # ~350-400 words

# Chunking calculation
words_per_chunk = 350
chunks_needed = document_words / words_per_chunk  # ~3 chunks
```

### Storage Calculation Per Document

```python
# MiniLM (384 dimensions, 1.5KB per embedding)
miniLM_chunks = 3  # 1,000 words → ~3 chunks
miniLM_storage_per_doc = miniLM_chunks * 1.5  # 4.5 KB per document

# MPNet (768 dimensions, 3KB per embedding)  
mpnet_chunks = 3  # 1,000 words → ~3 chunks
mpnet_storage_per_doc = mpnet_chunks * 3  # 9 KB per document

print(f"1,000-word document storage:")
print(f"MiniLM: {miniLM_storage_per_doc} KB")
print(f"MPNet:  {mpnet_storage_per_doc} KB")
```

### Scaling to Large Document Collections

```python
# For 1 million documents, each 1,000 words
num_documents = 1_000_000
words_per_doc = 1000
chunks_per_doc = 3  # Average

# Total storage calculation
miniLM_total = num_documents * chunks_per_doc * 1.5  # KB
mpnet_total = num_documents * chunks_per_doc * 3     # KB

# Convert to GB
miniLM_gb = miniLM_total / (1024 * 1024)  # 4.5 GB
mpnet_gb = mpnet_total / (1024 * 1024)    # 9 GB

print(f"1M documents (1,000 words each):")
print(f"MiniLM: {miniLM_gb:.1f} GB")
print(f"MPNet:  {mpnet_gb:.1f} GB")
```

### Document Size vs Chunk Count

```python
def calculate_chunks_and_storage(document_words, model_dimensions):
    """Calculate chunks and storage for a document"""
    words_per_chunk = 350  # Typical chunk size
    chunks = max(1, document_words // words_per_chunk)
    
    # Storage per embedding
    bytes_per_dimension = 4  # float32
    storage_per_embedding = model_dimensions * bytes_per_dimension / 1024  # KB
    
    total_storage = chunks * storage_per_embedding
    
    return chunks, total_storage

# Test different document sizes
document_sizes = [500, 1000, 2000, 5000, 10000]  # words

print("Document Size vs Storage Requirements:")
print("Words\tChunks\tMiniLM\tMPNet")
print("-" * 40)

for words in document_sizes:
    chunks_mini, storage_mini = calculate_chunks_and_storage(words, 384)
    chunks_mp, storage_mp = calculate_chunks_and_storage(words, 768)
    
    print(f"{words}\t{chunks_mini}\t{storage_mini:.1f}KB\t{storage_mp:.1f}KB")
```

### Optimal Chunk Size Considerations

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def analyze_chunking_strategy(text, chunk_size=500, chunk_overlap=50):
    """Analyze how different chunk sizes affect storage"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_text(text)
    
    # Calculate storage for different models
    miniLM_storage = len(chunks) * 1.5  # KB
    mpnet_storage = len(chunks) * 3     # KB
    
    return {
        'chunks': len(chunks),
        'miniLM_storage': miniLM_storage,
        'mpnet_storage': mpnet_storage,
        'avg_chunk_size': sum(len(chunk.split()) for chunk in chunks) / len(chunks)
    }

# Example analysis
sample_text = "Your 1000-word document content here..." * 100  # Simulate long text

strategies = [
    {'chunk_size': 200, 'chunk_overlap': 20},
    {'chunk_size': 500, 'chunk_overlap': 50},
    {'chunk_size': 1000, 'chunk_overlap': 100}
]

print("Chunking Strategy Analysis:")
print("Chunk Size\tChunks\tMiniLM\tMPNet\tAvg Words/Chunk")
print("-" * 60)

for strategy in strategies:
    result = analyze_chunking_strategy(sample_text, **strategy)
    print(f"{strategy['chunk_size']}\t\t{result['chunks']}\t{result['miniLM_storage']:.1f}KB\t{result['mpnet_storage']:.1f}KB\t{result['avg_chunk_size']:.0f}")
```

### Key Takeaways

1. **Embedding size is fixed** by the model (384, 768, etc.)
2. **Storage per document** depends on chunk count, not document size
3. **1,000-word document** typically produces 3-4 embeddings
4. **Chunk size affects** both storage and retrieval quality
5. **Balance needed** between chunk size and semantic coherence

## Performance Optimization

### Memory Considerations

```python
# Model sizes (approximate)
model_sizes = {
    'all-MiniLM-L6-v2': '90MB',
    'all-mpnet-base-v2': '420MB',
    'paraphrase-multilingual-MiniLM-L12-v2': '120MB'
}

# Embedding storage per chunk
# 384 dimensions = 1.5KB per embedding (float32)
# 768 dimensions = 3KB per embedding (float32)

# For 1M documents (1,000 words each, ~3 chunks):
# MiniLM: 4.5GB storage
# MPNet: 9GB storage
```

### Batch Processing

```python
# Process large datasets efficiently
embeddings = model.encode(
    texts,
    batch_size=32,  # Process 32 texts at once
    show_progress_bar=True
)
```

### Caching

```python
# Cache embeddings to avoid recomputation
import pickle

def cache_embeddings(texts, model, cache_file="embeddings.pkl"):
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        embeddings = model.encode(texts)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings
```

## Best Practices

### 1. Model Selection

```python
# Start with MiniLM-L6-v2 for most use cases
model = SentenceTransformer('all-MiniLM-L6-v2')

# Upgrade to MPNet if quality is critical
model = SentenceTransformer('all-mpnet-base-v2')

# Use multilingual models for international content
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### 2. Text Preprocessing

```python
def preprocess_text(text):
    """Clean and preprocess text before embedding"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters if needed
    # text = re.sub(r'[^\w\s]', '', text)
    
    # Truncate if too long (most models have limits)
    if len(text) > 512:
        text = text[:512]
    
    return text
```

### 3. Chunking Strategy

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optimize chunk size for your use case
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Adjust based on your content
    chunk_overlap=50,    # Keep some overlap for context
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### 4. Similarity Thresholds

```python
def search_with_threshold(query, documents, model, threshold=0.7):
    """Only return results above similarity threshold"""
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(documents)
    
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Filter by threshold
    results = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            results.append({
                'document': documents[i],
                'similarity': sim
            })
    
    return results
```

### 5. Error Handling

```python
def safe_encode(model, texts, max_retries=3):
    """Safely encode texts with retry logic"""
    for attempt in range(max_retries):
        try:
            return model.encode(texts)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
```

## When to Use Dual Embeddings

### ✅ Good Use Cases

- Large document collections (10K+ documents)
- Real-time search requirements
- Storage/memory constraints
- Quality is important but speed matters

### ❌ Not Recommended When

- Small document collections (< 1K documents)
- Batch processing (not real-time)
- Storage is not a concern
- Maximum quality is required

## Conclusion

Embeddings are the foundation of effective RAG systems. Choose the right model for your use case:

- **Start with**: `all-MiniLM-L6-v2` for most applications
- **Upgrade to**: `all-mpnet-base-v2` for higher quality
- **Use dual embeddings**: For large-scale, real-time systems
- **Consider multilingual**: If your content spans multiple languages

Remember: The best embedding strategy depends on your specific requirements for speed, quality, and resource constraints.

## Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [LangChain Embeddings Guide](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [Hugging Face Model Hub](https://huggingface.co/models?library=sentence-transformers)
- [FAISS Documentation](https://faiss.ai/)
