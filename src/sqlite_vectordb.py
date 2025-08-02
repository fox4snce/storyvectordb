"""
SQLite VectorDB - A lightweight vector database implementation using SQLite.

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
import os
from functools import lru_cache

class SQLiteVectorDB:
    """A lightweight vector database implementation using SQLite."""
    
    def __init__(self, db_path: str):
        """Initialize the vector database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with the cosine similarity function registered."""
        conn = sqlite3.connect(self.db_path)
        
        # Performance optimizations
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA cache_size = 100000")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        conn.create_function("cosine_similarity", 2, self._cosine_similarity)
        return conn
    
    def _initialize_db(self):
        """Create the database and required tables if they don't exist."""
        with self._get_connection() as conn:
            # Create the main vectors table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    vector TEXT,  -- JSON array of floats
                    metadata TEXT,  -- JSON object
                    file_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for faster operations
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON vectors(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON vectors(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector ON vectors(vector)")
    
    @staticmethod
    def _cosine_similarity(vec1_json: str, vec2_json: str) -> float:
        """Calculate cosine similarity between two vectors stored as JSON strings."""
        vec1 = np.array(json.loads(vec1_json))
        vec2 = np.array(json.loads(vec2_json))
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return float(dot_product / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
    
    def insert(self, content: str, vector: List[float], metadata: Optional[Dict] = None, file_id: Optional[str] = None) -> int:
        """Insert a new vector with associated content and metadata.
        
        Args:
            content: The original text or content reference
            vector: The embedding vector as a list of floats
            metadata: Optional dictionary of metadata
            file_id: Optional identifier for the source file
            
        Returns:
            The ID of the inserted entry
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vectors (content, vector, metadata, file_id)
                VALUES (?, ?, ?, ?)
                """,
                (
                    content,
                    json.dumps(vector),
                    json.dumps(metadata) if metadata else None,
                    file_id
                )
            )
            return cursor.lastrowid
    
    def insert_batch(self, items: List[Dict[str, Any]]) -> List[int]:
        """Batch insert multiple items at once for better performance.
        
        Args:
            items: List of dicts with keys: content, vector, metadata (optional), file_id (optional)
        Returns:
            List of inserted IDs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            ids = []
            
            # Use a single transaction for all inserts
            conn.execute("BEGIN")
            try:
                for item in items:
                    cursor.execute(
                        """
                        INSERT INTO vectors (content, vector, metadata, file_id)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            item['content'],
                            json.dumps(item['vector']),
                            json.dumps(item.get('metadata')),
                            item.get('file_id')
                        )
                    )
                    ids.append(cursor.lastrowid)
                conn.execute("COMMIT")
            except:
                conn.execute("ROLLBACK")
                raise
            
            return ids
    
    def search(self, query_vector: List[float], top_n: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: The query embedding vector
            top_n: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of dictionaries containing the search results
        """
        query_vector_json = json.dumps(query_vector)
        
        with self._get_connection() as conn:
            where_clause = ""
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            # Use a subquery to ensure we're getting unique results
            query = f"""
                WITH RankedResults AS (
                    SELECT 
                        id,
                        content,
                        vector,
                        metadata,
                        file_id,
                        cosine_similarity(vector, ?) as similarity,
                        ROW_NUMBER() OVER (
                            PARTITION BY file_id 
                            ORDER BY cosine_similarity(vector, ?) DESC
                        ) as rank_in_file
                    FROM vectors
                    {where_clause}
                )
                SELECT 
                    id,
                    content,
                    vector,
                    metadata,
                    file_id,
                    similarity
                FROM RankedResults
                WHERE rank_in_file <= 3  -- Get top 3 from each file
                ORDER BY similarity DESC
                LIMIT ?
            """
            
            # Note: we need to pass query_vector_json twice because it's used twice in the query
            params = [query_vector_json, query_vector_json, *params, top_n]
            cursor = conn.execute(query, params)
            
            results = []
            seen_content = set()  # Track unique content to prevent exact duplicates
            
            for row in cursor:
                content = row[1]
                # Only add if we haven't seen this exact content before
                if content not in seen_content:
                    seen_content.add(content)
                    results.append({
                        'id': row[0],
                        'content': content,
                        'vector': json.loads(row[2]),
                        'metadata': json.loads(row[3]) if row[3] else None,
                        'file_id': row[4],
                        'similarity': row[5]
                    })
            
            return results
    
    def search_fast(self, query_vector: List[float], top_n: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Fast search using thin Python loop instead of SQL ORDER BY."""
        query_vector = np.array(query_vector)
        
        with self._get_connection() as conn:
            # Build WHERE clause for filters
            where_clause = ""
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            # Fetch all vectors and compute similarities in Python
            query = f"""
                SELECT id, content, vector, metadata, file_id
                FROM vectors
                {where_clause}
            """
            
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor:
                vector = np.array(json.loads(row[2]))
                similarity = float(np.dot(query_vector, vector))  # Already unit norm
                
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'vector': vector.tolist(),
                    'metadata': json.loads(row[3]) if row[3] else None,
                    'file_id': row[4],
                    'similarity': similarity
                })
            
            # Sort by similarity and return top_n
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_n]
    
    def get_context(self, chunk_id: int, window_size: int = 1) -> List[Dict]:
        """Get surrounding chunks for better context.
        
        Args:
            chunk_id: The ID of the center chunk
            window_size: Number of chunks to get on each side
        Returns:
            List of chunks ordered by their position
        """
        with self._get_connection() as conn:
            # First get the file_id and chunk_index of our target
            cursor = conn.execute(
                "SELECT file_id, metadata FROM vectors WHERE id = ?",
                (chunk_id,)
            )
            row = cursor.fetchone()
            if not row:
                return []  # Chunk not found
                
            file_id, metadata_json = row
            if not metadata_json:
                return []  # No metadata available
                
            metadata = json.loads(metadata_json)
            chunk_index = metadata.get('chunk_index')
            if chunk_index is None:
                return []  # No chunk index in metadata
            
            # Then get surrounding chunks
            cursor = conn.execute(
                """
                SELECT id, content, metadata
                FROM vectors 
                WHERE file_id = ? 
                AND json_extract(metadata, '$.chunk_index') BETWEEN ? AND ?
                ORDER BY json_extract(metadata, '$.chunk_index')
                """,
                (file_id, chunk_index - window_size, chunk_index + window_size)
            )
            
            return [
                {
                    'id': row[0],
                    'content': row[1],
                    'metadata': json.loads(row[2]) if row[2] else None
                }
                for row in cursor
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the vector database."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_vectors,
                    COUNT(DISTINCT file_id) as unique_files,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM vectors
            """)
            stats = dict(zip(['total_vectors', 'unique_files', 'oldest_entry', 'newest_entry'], 
                           cursor.fetchone()))
            
            # Get per-file stats
            cursor = conn.execute("""
                SELECT file_id, COUNT(*) as chunk_count
                FROM vectors 
                GROUP BY file_id
            """)
            stats['per_file_chunks'] = dict(cursor.fetchall())
            
            return stats
    
    def delete_by_file(self, file_id: str) -> int:
        """Delete all entries associated with a specific file.
        
        Args:
            file_id: The file identifier
            
        Returns:
            Number of entries deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM vectors WHERE file_id = ?", (file_id,))
            return cursor.rowcount
    
    def delete(self, entry_id: int) -> bool:
        """Delete a specific entry by ID.
        
        Args:
            entry_id: The ID of the entry to delete
            
        Returns:
            True if an entry was deleted, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM vectors WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0
    
    def list_all(self) -> List[Dict]:
        """List all stored vectors and their metadata.
        
        Returns:
            List of dictionaries containing all entries
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, content, vector, metadata, file_id FROM vectors")
            
            results = []
            for row in cursor:
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'vector': json.loads(row[2]),
                    'metadata': json.loads(row[3]) if row[3] else None,
                    'file_id': row[4]
                })
            
            return results
    
    def process_file(self, file_path: str, get_embedding_fn: Callable[[str], List[float]], 
                    chunk_size: int = 1000, callback: Optional[Callable[[int, int], None]] = None):
        """Process a file with progress updates.
        
        Args:
            file_path: Path to the file to process
            get_embedding_fn: Function to convert text to embeddings
            chunk_size: Size of text chunks
            callback: Function(current_chunk, total_chunks) for progress updates
        """
        from pathlib import Path
        
        # Get file content (assuming text file - extend this based on file type)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
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
        file_id = Path(file_path).name
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            items = []
            
            for j, chunk in enumerate(batch):
                chunk_index = i + j
                vector = get_embedding_fn(chunk)
                items.append({
                    'content': chunk,
                    'vector': vector,
                    'metadata': {
                        'chunk_index': chunk_index,
                        'total_chunks': len(chunks)
                    },
                    'file_id': file_id
                })
            
            self.insert_batch(items)
            
            if callback:
                callback(i + len(batch), len(chunks)) 