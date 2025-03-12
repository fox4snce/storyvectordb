# SQLite VectorDB

A lightweight, single-file vector database implementation using SQLite as the underlying storage engine. This library provides vector-based search and retrieval functionality without requiring external vector database dependencies.

## Features

- Self-contained implementation in a single Python file
- SQLite-based storage for both vectors and metadata
- Efficient similarity search using cosine similarity
- File origin tracking for easy updates and deletions
- Metadata filtering support
- No external vector database dependencies

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sqlite-vectordb
```

2. Install the required dependencies:
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

### Search with Metadata Filters

```python
# Search with metadata filters
results = db.search(
    query_vector=query_vector,
    top_n=5,
    filters={"category": "test"}
)
```

### List All Entries

```python
# Get all stored vectors
all_entries = db.list_all()
for entry in all_entries:
    print(f"ID: {entry['id']}")
    print(f"Content: {entry['content']}")
    print(f"File ID: {entry['file_id']}")
    print("---")
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
- Suitable for small to medium-sized vector collections

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 