import os
import json
import sqlite3
import numpy as np
import torch
import requests
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

# Check if sentence_transformers is installed, otherwise use a mock
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers not found. Using mock embeddings.")

class RealExternalKnowledgeService:
    """
    External knowledge service with real retrieval capabilities
    Provides vector database storage and web search functionality
    """
    
    def __init__(self, vector_db_path="knowledge.db", embed_dim=768, search_api_key=None):
        self.vector_db_path = vector_db_path
        self.embed_dim = embed_dim
        self.search_api_key = search_api_key
        
        # Initialize encoder if available
        if HAVE_SENTENCE_TRANSFORMERS:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.encoder = None
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with tables for knowledge storage"""
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            domain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_vectors (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES knowledge_entries (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY,
            query_hash TEXT UNIQUE NOT NULL,
            query_text TEXT NOT NULL,
            result_ids TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector embedding"""
        if self.encoder is not None:
            return self.encoder.encode(text)
        else:
            # Mock encoding if sentence_transformers not available
            # Creates a deterministic but unique vector based on text hash
            import hashlib
            text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(text_hash)
            return np.random.randn(self.embed_dim)
    
    def add_knowledge(self, content: str, source: str = None, domain: str = None) -> int:
        """
        Add new knowledge to the vector database
        Returns the ID of the inserted entry
        """
        # Encode content to vector
        vector = self.encode_text(content)
        
        # Store in database
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        try:
            # First insert the text content
            cursor.execute(
                "INSERT INTO knowledge_entries (content, source, domain) VALUES (?, ?, ?)",
                (content, source, domain)
            )
            entry_id = cursor.lastrowid
            
            # Then insert the vector
            cursor.execute(
                "INSERT INTO knowledge_vectors (entry_id, vector) VALUES (?, ?)",
                (entry_id, vector.tobytes())
            )
            
            conn.commit()
            return entry_id
        except Exception as e:
            conn.rollback()
            print(f"Error adding knowledge: {e}")
            return -1
        finally:
            conn.close()
    
    def batch_add_knowledge(self, entries: List[Dict[str, str]]) -> List[int]:
        """
        Add multiple knowledge entries in batch
        Each entry should be a dict with 'content', 'source', and 'domain' keys
        Returns list of entry IDs
        """
        entry_ids = []
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        try:
            for entry in entries:
                content = entry['content']
                source = entry.get('source')
                domain = entry.get('domain')
                
                # Encode content
                vector = self.encode_text(content)
                
                # Insert content
                cursor.execute(
                    "INSERT INTO knowledge_entries (content, source, domain) VALUES (?, ?, ?)",
                    (content, source, domain)
                )
                entry_id = cursor.lastrowid
                entry_ids.append(entry_id)
                
                # Insert vector
                cursor.execute(
                    "INSERT INTO knowledge_vectors (entry_id, vector) VALUES (?, ?)",
                    (entry_id, vector.tobytes())
                )
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error in batch add: {e}")
        finally:
            conn.close()
        
        return entry_ids
    
    def get_information(self, query_embedding, top_k=3, query_text=None, use_web=False) -> torch.Tensor:
        """
        Retrieve information based on query embedding or text
        Returns tensor suitable for model integration
        
        Args:
            query_embedding: Embedding tensor or numpy array
            top_k: Number of results to retrieve
            query_text: Optional text query (used if query_embedding is None or for web search)
            use_web: Whether to augment with web search
        """
        results = []
        
        # Convert query_embedding to numpy if it's a tensor
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # If no embedding but text is provided, encode it
        if query_embedding is None and query_text:
            query_embedding = self.encode_text(query_text)
        
        # Search vector database
        db_results = self._search_vector_db(query_embedding, top_k)
        if db_results:
            results.extend(db_results)
        
        # Augment with web search if enabled and we have query text
        if use_web and query_text and self.search_api_key:
            web_results = self.web_search(query_text)
            if web_results:
                results.extend(web_results[:top_k])
        
        # If we got results, combine them
        if results:
            combined_text = " ".join([r[0] for r in results])
            result_embedding = self.encode_text(combined_text)
            return torch.tensor(result_embedding, dtype=torch.float32)
        else:
            # Return zero vector if no results
            return torch.zeros(self.embed_dim, dtype=torch.float32)
    
    def _search_vector_db(self, query_vector, top_k=3) -> List[Tuple[str, float]]:
        """
        Search vector database for similar content
        Returns list of (content, similarity) tuples
        """
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        try:
            # Get all vectors and calculate similarity
            cursor.execute("""
                SELECT e.id, e.content, v.vector
                FROM knowledge_entries e
                JOIN knowledge_vectors v ON e.id = v.entry_id
            """)
            
            results = cursor.fetchall()
            if not results:
                return []
            
            # Calculate cosine similarities
            similarities = []
            for id, content, vector_bytes in results:
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                # Normalize vectors to calculate cosine similarity
                norm_query = query_vector / np.linalg.norm(query_vector)
                norm_vector = vector / np.linalg.norm(vector)
                
                similarity = np.dot(norm_query, norm_vector)
                similarities.append((id, content, similarity))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[2], reverse=True)
            return [(content, sim) for _, content, sim in similarities[:top_k]]
        
        except Exception as e:
            print(f"Error searching vector DB: {e}")
            return []
        finally:
            conn.close()
    
    def web_search(self, query_text: str) -> List[Tuple[str, float]]:
        """
        Perform web search to augment knowledge
        Returns list of (content, relevance) tuples
        Supports multiple search providers
        """
        if not self.search_api_key:
            print("No search API key provided")
            return []
        
        provider = "serp"  # Default provider
        
        if provider == "serp":
            return self._serp_api_search(query_text)
        elif provider == "bing":
            return self._bing_search(query_text)
        else:
            print(f"Unknown search provider: {provider}")
            return []
    
    def _serp_api_search(self, query_text: str) -> List[Tuple[str, float]]:
        """Search using SerpAPI"""
        try:
            search_params = {
                "q": query_text,
                "api_key": self.search_api_key
            }
            
            response = requests.get(
                "https://serpapi.com/search",
                params=search_params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract organic results
                results = []
                
                # Process organic results
                organic_results = data.get("organic_results", [])
                for i, result in enumerate(organic_results[:3]):
                    snippet = result.get("snippet", "")
                    title = result.get("title", "")
                    
                    # Combine title and snippet
                    content = f"{title}. {snippet}"
                    
                    # Assign decreasing relevance score based on position
                    relevance = 1.0 - (i * 0.1)
                    
                    results.append((content, relevance))
                
                # Also check for knowledge graph if available
                if "knowledge_graph" in data:
                    kg = data["knowledge_graph"]
                    description = kg.get("description", "")
                    if description:
                        results.insert(0, (description, 1.0))  # Add at top with highest relevance
                
                return results
            else:
                print(f"SerpAPI error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in SerpAPI search: {e}")
            return []
    
    def _bing_search(self, query_text: str) -> List[Tuple[str, float]]:
        """Search using Bing API"""
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.search_api_key,
            }
            
            params = {
                'q': query_text,
                'count': 5,
                'offset': 0,
                'mkt': 'en-US',
                'safeSearch': 'Moderate',
            }
            
            response = requests.get(
                'https://api.bing.microsoft.com/v7.0/search',
                headers=headers,
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Process web pages
                web_pages = data.get('webPages', {}).get('value', [])
                for i, page in enumerate(web_pages[:3]):
                    snippet = page.get('snippet', '')
                    name = page.get('name', '')
                    
                    # Combine name and snippet
                    content = f"{name}. {snippet}"
                    
                    # Assign decreasing relevance score based on position
                    relevance = 1.0 - (i * 0.1)
                    
                    results.append((content, relevance))
                
                return results
            else:
                print(f"Bing API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in Bing search: {e}")
            return []
    
    def cache_query_result(self, query_text: str, result_ids: List[int]):
        """Cache query results for faster retrieval"""
        import hashlib
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        try:
            # Store as JSON string
            ids_json = json.dumps(result_ids)
            
            cursor.execute(
                "INSERT OR REPLACE INTO query_cache (query_hash, query_text, result_ids) VALUES (?, ?, ?)",
                (query_hash, query_text, ids_json)
            )
            
            conn.commit()
        except Exception as e:
            print(f"Error caching query: {e}")
        finally:
            conn.close()
    
    def get_cached_results(self, query_text: str, max_age_hours=24) -> Optional[List[int]]:
        """Retrieve cached results if available and not too old"""
        import hashlib
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT result_ids, timestamp FROM query_cache 
                WHERE query_hash = ? 
                """,
                (query_hash,)
            )
            
            result = cursor.fetchone()
            if not result:
                return None
            
            result_ids_json, timestamp_str = result
            
            # Check if cache is still valid
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp
            
            if age.total_seconds() > (max_age_hours * 3600):
                return None  # Cache too old
            
            # Parse JSON string back to list
            return json.loads(result_ids_json)
            
        except Exception as e:
            print(f"Error retrieving cache: {e}")
            return None
        finally:
            conn.close()
    
    def clear_cache(self, older_than_hours=None):
        """Clear the query cache, optionally only removing older entries"""
        conn = sqlite3.connect(self.vector_db_path)
        cursor = conn.cursor()
        
        try:
            if older_than_hours is not None:
                # Calculate cutoff timestamp
                cutoff = datetime.now() - datetime.timedelta(hours=older_than_hours)
                cutoff_str = cutoff.isoformat()
                
                cursor.execute(
                    "DELETE FROM query_cache WHERE timestamp < ?",
                    (cutoff_str,)
                )
            else:
                # Clear all cache
                cursor.execute("DELETE FROM query_cache")
            
            conn.commit()
            print(f"Cleared {cursor.rowcount} cached queries")
        except Exception as e:
            print(f"Error clearing cache: {e}")
        finally:
            conn.close()


# Utility function to set up knowledge database
def setup_knowledge_database(db_path="knowledge.db", sources=None):
    """
    Set up knowledge database with initial data
    
    Args:
        db_path: Path to SQLite database
        sources: List of dictionaries with content, source, and domain
    """
    # Create service
    service = RealExternalKnowledgeService(vector_db_path=db_path)
    
    # If sources provided, add them
    if sources:
        service.batch_add_knowledge(sources)
        print(f"Added {len(sources)} knowledge entries to {db_path}")
    else:
        # Add some default knowledge
        default_knowledge = [
            {
                "content": "Paris is the capital and most populous city of France. It has an estimated population of 2,165,423 residents in 2019 in an area of more than 105 km².",
                "source": "default",
                "domain": "geography"
            },
            {
                "content": "Alexander Graham Bell was a Scottish-born inventor, scientist, and engineer who is credited with inventing and patenting the first practical telephone in 1876.",
                "source": "default",
                "domain": "history"
            },
            {
                "content": "The capital of Japan is Tokyo. It is the most populous metropolitan area in the world, with more than 37.4 million residents.",
                "source": "default",
                "domain": "geography"
            }
        ]
        
        service.batch_add_knowledge(default_knowledge)
        print(f"Added {len(default_knowledge)} default knowledge entries to {db_path}")
    
    return service


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Service Setup')
    parser.add_argument('--db_path', type=str, default='knowledge.db',
                        help='Path to knowledge database')
    parser.add_argument('--search_key', type=str, default=None,
                        help='API key for web search')
    parser.add_argument('--setup', action='store_true',
                        help='Set up database with default knowledge')
    parser.add_argument('--query', type=str, default=None,
                        help='Test query')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_knowledge_database(db_path=args.db_path)
    
    if args.query:
        service = RealExternalKnowledgeService(
            vector_db_path=args.db_path,
            search_api_key=args.search_key
        )
        
        query_embedding = service.encode_text(args.query)
        results = service._search_vector_db(query_embedding, top_k=3)
        
        print(f"Query: {args.query}")
        print("Results:")
        for content, similarity in results:
            print(f"- {content[:100]}... (similarity: {similarity:.4f})")
        
        if args.search_key:
            print("\nWeb search results:")
            web_results = service.web_search(args.query)
            for content, relevance in web_results:
                print(f"- {content[:100]}... (relevance: {relevance:.4f})")
