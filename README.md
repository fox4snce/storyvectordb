# SQLite VectorDB

A dirt-simple, single-file vector database implementation using SQLite. This library provides powerful vector-based search functionality with minimal dependencies - just numpy!

## Why SQLite VectorDB?

- **Ultra-lightweight**: Single Python file, single dependency (numpy)
- **Zero setup**: No servers, no cloud, no API keys - just import and use
- **Surprisingly powerful**: Efficient similarity search, context retrieval, batch operations
- **Fully local**: All data stays on your machine
- **Blazing fast**: Built on SQLite, one of the most battle-tested databases in the world
- **Actually simple**: ~300 lines of clean, documented code you can understand

## Features

- Self-contained implementation in a single Python file
- SQLite-based storage for both vectors and metadata
- Efficient similarity search using cosine similarity
- Context retrieval for surrounding chunks
- Batch operations for better performance
- Progress tracking for long operations
- File origin tracking for easy updates and deletions
- Metadata filtering support
- Database statistics and monitoring
- No external vector database dependencies

## Installation

Just copy `sqlite_vectordb.py` to your project and install numpy:

```bash
pip install numpy
```

Or use the requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from sqlite_vectordb import SQLiteVectorDB

# Initialize the database
db = SQLiteVectorDB("vectors.db")

# Insert a vector with content and metadata
vector = [0.1, 0.2, 0.3, 0.4]  # Your embedding vector
db.insert(
    content="Example text",
    vector=vector,
    metadata={"category": "test"},
    file_id="doc1.txt"
)

# Search for similar vectors
query_vector = [0.15, 0.25, 0.35, 0.45]
results = db.search(query_vector, top_n=5)

# Print results
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Metadata: {result['metadata']}")
    print("---")

# Delete entries by file
db.delete_by_file("doc1.txt")
```

### Batch Operations

```python
# Insert multiple items at once
items = [
    {
        'content': "First document",
        'vector': [0.1, 0.2, 0.3, 0.4],
        'metadata': {'category': 'docs', 'chunk_index': 0},
        'file_id': 'doc1.txt'
    },
    {
        'content': "Second document",
        'vector': [0.2, 0.3, 0.4, 0.5],
        'metadata': {'category': 'docs', 'chunk_index': 1},
        'file_id': 'doc1.txt'
    }
]
ids = db.insert_batch(items)
```

### Get Context Around Results

```python
# Get surrounding chunks for better context
context = db.get_context(chunk_id=42, window_size=2)
for chunk in context:
    print(f"Chunk ID: {chunk['id']}")
    print(f"Content: {chunk['content']}")
    print("---")
```

### Database Statistics

```python
# Get database statistics
stats = db.get_stats()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Unique files: {stats['unique_files']}")
print(f"Chunks per file: {stats['per_file_chunks']}")
```

## Implementation Details

- Vectors are stored as JSON arrays in SQLite
- Cosine similarity is computed using numpy within SQLite functions
- Metadata is stored as JSON for flexibility
- File IDs enable efficient batch operations
- SQLite's transaction support ensures data integrity

## Performance Considerations

- Uses SQLite indexing for fast metadata filtering
- Cosine similarity is computed efficiently using numpy
- JSON storage provides flexibility without unnecessary complexity
- Suitable for small to medium-sized vector collections (millions of vectors)
- No network overhead - everything runs locally

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 