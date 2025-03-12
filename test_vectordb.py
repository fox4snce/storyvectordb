"""
Test file for SQLite VectorDB demonstrating usage with PDF files and different vector sources.

Required packages:
pip install numpy openai markitdown

Usage:
1. Start LM Studio and enable local server (default: http://localhost:5001)
2. Run this script: python test_vectordb.py
"""

import os
import subprocess
from typing import List, Dict
from sqlite_vectordb import SQLiteVectorDB
from openai import OpenAI
import time

# Initialize OpenAI client for LM Studio
client = OpenAI(base_url="http://localhost:5001/v1", api_key="not-needed")

# Constants
FILES_DIR = "files"  # Directory containing PDF files

def get_embedding(text: str) -> List[float]:
    """Get embeddings using LM Studio's local server."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="local").data[0].embedding

def convert_pdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF to markdown using markitdown."""
    output_path = pdf_path.replace('.pdf', '.md')
    subprocess.run(['markitdown', pdf_path, '-o', output_path], check=True)
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    os.remove(output_path)  # Clean up the temporary markdown file
    return content

def test_basic_operations():
    """Test basic database operations with progress tracking."""
    print("\n=== Testing Basic Operations ===")
    
    # Create a test database
    db_path = "test_vectors.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = SQLiteVectorDB(db_path)
    
    # Test batch insertion
    print("Testing batch insertion...")
    test_items = [
        {
            'content': f"Test content {i}",
            'vector': [float(i)] * 10,  # Simple test vectors
            'metadata': {
                'test_id': i,
                'chunk_index': i,  # Add chunk_index for context testing
                'total_chunks': 5
            },
            'file_id': 'test_file'
        }
        for i in range(5)
    ]
    
    ids = db.insert_batch(test_items)
    print(f"Inserted {len(ids)} items in batch")
    
    # Test search with filters
    print("\nTesting search with filters...")
    results = db.search([0.0] * 10, filters={'test_id': 0})
    print(f"Found {len(results)} results with test_id = 0")
    
    # Test context retrieval
    print("\nTesting context retrieval...")
    context = db.get_context(ids[2], window_size=1)
    print(f"Got {len(context)} chunks of context around item {ids[2]}")
    
    # Test database stats
    print("\nDatabase statistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        if key != 'per_file_chunks':
            print(f"{key}: {value}")
    print(f"Chunks per file: {stats['per_file_chunks']}")

def process_pdf_with_progress(db: SQLiteVectorDB, file_path: str):
    """Process a PDF file with progress updates."""
    def progress_callback(current: int, total: int):
        percent = (current / total) * 100
        print(f"\rProcessing {os.path.basename(file_path)}: {percent:.1f}%", end="")
    
    print(f"\nProcessing {os.path.basename(file_path)}...")
    start_time = time.time()
    
    # Convert PDF to markdown first
    content = convert_pdf_to_markdown(file_path)
    
    # Split into chunks
    words = content.split()
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_size = 1000
    
    for word in words:
        word_size = len(word) + 1
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Process chunks with batching
    batch_size = 10
    file_id = os.path.basename(file_path)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        items = []
        
        for j, chunk in enumerate(batch):
            chunk_index = i + j
            vector = get_embedding(chunk)
            items.append({
                'content': chunk,
                'vector': vector,
                'metadata': {
                    'chunk_index': chunk_index,
                    'total_chunks': len(chunks)
                },
                'file_id': file_id
            })
        
        db.insert_batch(items)
        
        if progress_callback:
            progress_callback(i + len(batch), len(chunks))
    
    duration = time.time() - start_time
    print(f"\nCompleted in {duration:.1f} seconds")

def test_pdf_processing():
    """Test PDF processing with the new features."""
    print("\n=== Testing PDF Processing ===")
    
    db_path = "pdf_vectors.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = SQLiteVectorDB(db_path)
    
    # Ensure files directory exists
    if not os.path.exists(FILES_DIR):
        os.makedirs(FILES_DIR)
        print(f"Created {FILES_DIR} directory. Please add PDF files to test.")
        return
    
    # Process each PDF file
    print(f"\nLooking for PDF files in: {os.path.abspath(FILES_DIR)}")
    pdf_files = [f for f in os.listdir(FILES_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found! Please make sure PDF files are in the 'files' directory.")
        return
        
    print(f"Found {len(pdf_files)} PDF files:", pdf_files)
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(FILES_DIR, pdf_file)
            process_pdf_with_progress(db, pdf_path)
        except Exception as e:
            print(f"\nError processing {pdf_file}: {str(e)}")
            continue
    
    # Show database statistics
    print("\nFinal database statistics:")
    stats = db.get_stats()
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Unique files: {stats['unique_files']}")
    print("\nChunks per file:")
    for file_id, count in stats['per_file_chunks'].items():
        print(f"  {file_id}: {count} chunks")
    
    # Test search functionality
    test_queries = [
        "What are the key findings about generalization phenomena?",
        "Explain the relationship between representation learning and similarity learning.",
        "What is the double descent phenomenon?"
    ]
    
    print("\nTesting search functionality...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        # Get embedding for the query
        query_vector = get_embedding(query)
        
        # Search with the new functionality
        results = db.search(query_vector, top_n=3)
        
        if not results:
            print("No results found.")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (similarity: {result['similarity']:.3f}):")
            print(f"From: {result['file_id']}")
            print("-" * 40)
            print(result['content'].strip())
            
            # Get context for this result
            context = db.get_context(result['id'], window_size=1)
            if len(context) > 1:
                print("\nContext:")
                for ctx in context:
                    if ctx['id'] != result['id']:
                        print(f"... {ctx['content'][:100]}...")

if __name__ == "__main__":
    # Verify LM Studio connection
    try:
        test_text = "Testing LM Studio connection."
        test_embedding = get_embedding(test_text)
        print(f"Successfully connected to LM Studio. Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print("Error: Could not connect to LM Studio server. Make sure it's running on localhost:5001")
        print(f"Error details: {str(e)}")
        exit(1)
    
    # Run the tests
    test_basic_operations()
    test_pdf_processing() 