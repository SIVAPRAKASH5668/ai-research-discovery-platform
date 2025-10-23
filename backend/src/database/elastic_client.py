"""
Elasticsearch Serverless Client - FINAL VERSION WITH RAG SUPPORT
✅ FIXED: vector_search now returns embeddings for relationship calculation
✅ User library support (user_id field)
✅ Multi-tenancy (per-user filtering)
✅ RAG-ready (full_text field for context)
✅ Hybrid search (BM25 + Vector)
✅ Duplicate prevention
"""
import os
import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.helpers import async_bulk
import json
from datetime import datetime


logger = logging.getLogger(__name__)


class ElasticClient:
    """
    🚀 Elasticsearch Serverless Client with RAG Support
    - Vector search (dense_vector 768D)
    - User library (user_id filtering)
    - Full-text storage for RAG
    - Hybrid search (BM25 + KNN)
    """
    
    def __init__(self):
        self.endpoint = os.getenv("ELASTIC_ENDPOINT")
        self.api_key = os.getenv("ELASTIC_API_KEY")
        self.index_name = os.getenv("ELASTIC_INDEX_NAME", "research")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("❌ Missing ELASTIC_ENDPOINT or ELASTIC_API_KEY")
        
        # Initialize clients
        self.client = Elasticsearch(
            self.endpoint,
            api_key=self.api_key,
            verify_certs=True,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
        
        self.async_client = AsyncElasticsearch(
            self.endpoint,
            api_key=self.api_key,
            verify_certs=True,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
        
        # Statistics
        self.stats = {
            "documents_indexed": 0,
            "documents_updated": 0,
            "duplicates_skipped": 0,
            "user_saves": 0,
            "searches_performed": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        logger.info(f"🔍 Elasticsearch initialized - Index: {self.index_name}")


    async def test_connection(self):
        """Test Elasticsearch Serverless connection"""
        try:
            info = await self.async_client.info()
            cluster_name = info.get('cluster_name', 'unknown')
            version = info.get('version', {}).get('number', 'unknown')
            
            logger.info(f"✅ Elasticsearch connected - Cluster: {cluster_name}, Version: {version}")
            
            # Check/create index
            exists = await self.async_client.indices.exists(index=self.index_name)
            if not exists:
                logger.info(f"📋 Creating index: {self.index_name}")
                await self.create_index()
            else:
                logger.info(f"📋 Index exists: {self.index_name}")
                await self._check_and_update_mapping()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            raise


    async def create_index(self):
        """
        Create Elasticsearch Serverless index WITH user_id support
        ✅ NO settings section (shards/replicas managed automatically)
        """
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        # Basic fields
                        "id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "abstract": {"type": "text"},
                        "full_content": {"type": "text"},  # ← For RAG context
                        "authors": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        
                        # ✅ USER LIBRARY SUPPORT
                        "user_id": {"type": "keyword"},  # ← NEW! Filter by user
                        "is_user_saved": {"type": "boolean"},  # ← Quick filter
                        "saved_at": {"type": "date"},  # ← When user saved it
                        "user_notes": {"type": "text"},  # ← User annotations
                        "user_tags": {"type": "keyword"},  # ← User tags
                        
                        # Categorization
                        "domain": {"type": "keyword"},
                        "language": {"type": "keyword"},
                        "publication_date": {"type": "date"},
                        "publication_year": {"type": "integer"},
                        
                        # Source tracking
                        "source": {"type": "keyword"},
                        "api_source": {"type": "keyword"},
                        "citation_count": {"type": "integer"},
                        
                        # Identifiers
                        "doi": {"type": "keyword"},
                        "url": {"type": "keyword"},
                        "pmid": {"type": "keyword"},
                        "arxiv_id": {"type": "keyword"},
                        "core_id": {"type": "keyword"},
                        
                        # Quality metrics
                        "confidence_score": {"type": "float"},
                        "analysis_confidence": {"type": "float"},
                        
                        # ✅ VECTOR FIELD FOR SEMANTIC SEARCH
                        "content_embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # Search helpers
                        "searchable_text": {"type": "text"},
                        "paper_type": {"type": "keyword"},
                        "journal": {"type": "keyword"},
                        
                        # Metadata
                        "search_method": {"type": "keyword"},
                        "ai_agent_used": {"type": "keyword"},
                        "indexed_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
            }
            
            await self.async_client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"✅ Created index with RAG support: {self.index_name}")
            
        except Exception as e:
            if "resource_already_exists_exception" not in str(e):
                logger.error(f"❌ Index creation failed: {e}")
                raise
            else:
                logger.info(f"📋 Index already exists: {self.index_name}")


    async def _check_and_update_mapping(self):
        """Check if vector field and user fields are properly configured"""
        try:
            mapping = await self.async_client.indices.get_mapping(index=self.index_name)
            current_mapping = mapping[self.index_name]["mappings"]["properties"]
            
            # Check vector field
            embedding_field = current_mapping.get("content_embedding", {})
            
            if (embedding_field.get("type") != "dense_vector" or 
                embedding_field.get("dims") != 768):
                
                logger.warning("⚠️ Updating mapping...")
                
                await self.async_client.indices.put_mapping(
                    index=self.index_name,
                    body={
                        "properties": {
                            "content_embedding": {
                                "type": "dense_vector",
                                "dims": 768,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "user_id": {"type": "keyword"},
                            "is_user_saved": {"type": "boolean"},
                            "saved_at": {"type": "date"},
                            "user_notes": {"type": "text"},
                            "user_tags": {"type": "keyword"}
                        }
                    }
                )
                logger.info("✅ Mapping updated with RAG fields")
            else:
                logger.info("✅ Mapping configured correctly")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not check/update mapping: {e}")


    def _generate_paper_id(self, paper: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """
        Generate unique paper ID
        If user_id provided, creates user-specific ID
        """
        # Try paper ID first
        paper_id = paper.get("id", "")
        if paper_id:
            base_id = str(paper_id).replace('/', '_').replace('.', '_')
        # Try DOI
        elif paper.get("doi"):
            base_id = paper["doi"].replace('/', '_').replace('.', '_')
        # Fallback: hash title
        else:
            title = paper.get("title", "untitled")
            base_id = f"hash_{hashlib.md5(title.encode()).hexdigest()}"
        
        # If user-specific, prepend user_id
        if user_id:
            return f"{user_id}_{base_id}"
        
        return base_id


    async def index_paper(self, paper: Dict[str, Any], 
                          embedding: Optional[List[float]] = None,
                          user_id: Optional[str] = None,
                          user_notes: Optional[str] = None,
                          user_tags: Optional[List[str]] = None) -> bool:
        """
        Index single paper with optional user library support
        
        Args:
            paper: Paper data
            embedding: 768D vector
            user_id: If provided, saves to user's library
            user_notes: User annotations
            user_tags: User tags
        """
        try:
            paper_id = self._generate_paper_id(paper, user_id)
            
            # Check if exists (skip for public papers, allow for user saves)
            if not user_id:
                try:
                    exists = await self.async_client.exists(
                        index=self.index_name,
                        id=paper_id
                    )
                    
                    if exists:
                        logger.info(f"⏭️  Skipping duplicate: {paper.get('title', '')[:50]}...")
                        self.stats["duplicates_skipped"] += 1
                        return True
                except Exception as e:
                    logger.debug(f"Exists check failed: {e}")
            
            # Prepare document
            doc = {
                "id": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "full_content": paper.get("full_content", paper.get("abstract", "")),
                "searchable_text": f"{paper.get('title', '')} {paper.get('abstract', '')}",
                "authors": paper.get("authors", []),
                "domain": paper.get("domain", paper.get("research_domain", "General")),
                "language": paper.get("language", "en"),
                "publication_date": paper.get("publication_date", paper.get("published_date")),
                "publication_year": paper.get("publication_year"),
                "source": paper.get("source", "unknown"),
                "api_source": paper.get("api_source", paper.get("source", "unknown")),
                "citation_count": paper.get("citation_count", 0),
                "doi": paper.get("doi", ""),
                "url": paper.get("url", paper.get("source_url", "")),
                "pmid": paper.get("pmid", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "core_id": paper.get("core_id", ""),
                "confidence_score": paper.get("confidence_score", 0.5),
                "analysis_confidence": paper.get("analysis_confidence", 0.8),
                "search_method": paper.get("search_method", "api_search"),
                "ai_agent_used": paper.get("ai_agent_used", "unknown"),
                "paper_type": paper.get("paper_type", "article"),
                "journal": paper.get("journal", ""),
                "indexed_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                
                # ✅ USER LIBRARY FIELDS
                "user_id": user_id,
                "is_user_saved": bool(user_id),
                "saved_at": datetime.now().isoformat() if user_id else None,
                "user_notes": user_notes,
                "user_tags": user_tags or []
            }
            
            # Add embedding if provided
            if embedding and len(embedding) == 768:
                doc["content_embedding"] = embedding
            
            # Index document
            response = await self.async_client.index(
                index=self.index_name,
                id=paper_id,
                document=doc
            )
            
            # Track result
            if response.get('result') == 'created':
                logger.info(f"✅ {'USER SAVE' if user_id else 'NEW'}: {doc['title'][:50]}...")
                if user_id:
                    self.stats["user_saves"] += 1
                else:
                    self.stats["documents_indexed"] += 1
            elif response.get('result') == 'updated':
                logger.info(f"🔄 UPDATED: {doc['title'][:50]}...")
                self.stats["documents_updated"] += 1
            
            self.stats["successful_operations"] += 1
            return True
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(f"❌ Indexing failed for {paper.get('title', 'unknown')}: {e}")
            return False


    async def search_papers(self, query: str, limit: int = 20, 
                            filters: Optional[Dict[str, Any]] = None,
                            user_id: Optional[str] = None,
                            exclude_user_papers: bool = False) -> List[Dict[str, Any]]:
        """
        BM25 text search
        
        Args:
            query: Search query
            limit: Max results
            filters: Additional filters
            user_id: If provided, search only user's papers
            exclude_user_papers: If True, exclude user-saved papers (for public search)
        """
        try:
            search_body = {
                "query": {
                    "bool": {
                        "must": [{
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "abstract^2", "searchable_text", "authors^2", "user_notes"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }]
                    }
                },
                "size": limit,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"citation_count": {"order": "desc", "missing": "_last"}}
                ],
                "_source": {"excludes": ["content_embedding"]}
            }
            
            # Filter by user_id
            if user_id:
                search_body["query"]["bool"]["filter"] = [
                    {"term": {"user_id": user_id}}
                ]
            elif exclude_user_papers:
                search_body["query"]["bool"]["must_not"] = [
                    {"exists": {"field": "user_id"}}
                ]
            
            # Additional filters
            if filters:
                filter_clauses = search_body["query"]["bool"].get("filter", [])
                for key, value in filters.items():
                    if key == "domain":
                        filter_clauses.append({"term": {"domain": value}})
                    elif key == "language":
                        filter_clauses.append({"term": {"language": value}})
                
                if filter_clauses:
                    search_body["query"]["bool"]["filter"] = filter_clauses
            
            response = await self.async_client.search(index=self.index_name, body=search_body)
            
            papers = []
            for hit in response["hits"]["hits"]:
                paper = hit["_source"]
                paper["id"] = hit["_id"]
                paper["relevance_score"] = hit["_score"]
                paper["search_type"] = "text_search"
                papers.append(paper)
            
            self.stats["searches_performed"] += 1
            self.stats["successful_operations"] += 1
            
            logger.info(f"🔍 Text search: found {len(papers)} papers {f'(user: {user_id})' if user_id else ''}")
            return papers
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(f"❌ Text search failed: {e}")
            return []


    async def vector_search(self, query_embedding: List[float], 
                            limit: int = 20,
                            similarity_threshold: float = 0.7,
                            user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Vector search using KNN - NOW RETURNS EMBEDDINGS!
        
        Args:
            query_embedding: 768D vector
            limit: Max results
            similarity_threshold: Min similarity score
            user_id: If provided, search only user's papers
        """
        try:
            if not query_embedding or len(query_embedding) != 768:
                logger.warning("⚠️ Invalid query embedding dimensions")
                return []
            
            search_body = {
                "knn": {
                    "field": "content_embedding",
                    "query_vector": query_embedding,
                    "k": limit,
                    "num_candidates": min(limit * 3, 100)
                },
                "min_score": similarity_threshold,
                # ✅ CRITICAL FIX: Include embedding in results!
                "_source": {
                    "includes": [
                        "id", "title", "abstract", "authors", "year",
                        "publication_year", "language", "source", "doi",
                        "url", "citation_count", 
                        "content_embedding"  # ← MUST INCLUDE FOR RELATIONSHIPS!
                    ]
                }
            }
            
            # Filter by user_id
            if user_id:
                search_body["filter"] = [{"term": {"user_id": user_id}}]
            
            response = await self.async_client.search(index=self.index_name, body=search_body)
            
            papers = []
            for hit in response["hits"]["hits"]:
                paper = hit["_source"]
                paper["id"] = hit["_id"]
                paper["similarity_score"] = hit["_score"]
                paper["search_type"] = "vector_search"
                papers.append(paper)
            
            self.stats["searches_performed"] += 1
            self.stats["successful_operations"] += 1
            
            logger.info(f"🎯 Vector search: found {len(papers)} papers {f'(user: {user_id})' if user_id else ''}")
            return papers
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(f"❌ Vector search failed: {e}")
            import traceback
            traceback.print_exc()
            return []


    async def hybrid_search(self, query: str, query_embedding: List[float], 
                            limit: int = 20,
                            user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search: BM25 + Vector
        
        Args:
            query: Text query
            query_embedding: 768D vector
            limit: Max results
            user_id: If provided, search only user's papers
        """
        try:
            if not query_embedding or len(query_embedding) != 768:
                logger.warning("⚠️ No embedding, using text search only")
                return await self.search_papers(query, limit, user_id=user_id)
            
            # Run both searches in parallel
            text_results = await self.search_papers(query, limit // 2, user_id=user_id)
            vector_results = await self.vector_search(query_embedding, limit // 2, 0.7, user_id=user_id)
            
            # Combine and deduplicate
            combined_papers = []
            seen_ids = set()
            
            for paper in text_results:
                if paper["id"] not in seen_ids:
                    paper["search_type"] = "hybrid_text"
                    combined_papers.append(paper)
                    seen_ids.add(paper["id"])
            
            for paper in vector_results:
                if paper["id"] not in seen_ids:
                    paper["search_type"] = "hybrid_vector"
                    combined_papers.append(paper)
                    seen_ids.add(paper["id"])
            
            # Sort by score
            combined_papers.sort(
                key=lambda x: x.get("similarity_score", x.get("relevance_score", 0)), 
                reverse=True
            )
            
            final_papers = combined_papers[:limit]
            
            self.stats["searches_performed"] += 1
            self.stats["successful_operations"] += 1
            
            logger.info(f"🔀 Hybrid search: {len(final_papers)} papers {f'(user: {user_id})' if user_id else ''}")
            return final_papers
            
        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(f"❌ Hybrid search failed: {e}")
            return await self.search_papers(query, limit, user_id=user_id)


    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get specific paper by ID"""
        try:
            response = await self.async_client.get(
                index=self.index_name,
                id=paper_id,
                _source_excludes=["content_embedding"]
            )
            paper = response["_source"]
            paper["id"] = paper_id
            paper["found"] = True
            return paper
        except Exception as e:
            logger.warning(f"⚠️ Paper not found: {paper_id}")
            return None


    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = await self.async_client.indices.stats(index=self.index_name)
            index_stats = stats["indices"][self.index_name]["total"]
            
            return {
                "index_name": self.index_name,
                "total_documents": index_stats["docs"]["count"],
                "index_size_bytes": index_stats["store"]["size_in_bytes"],
                "index_size_mb": round(index_stats["store"]["size_in_bytes"] / (1024 * 1024), 2),
                "search_operations": self.stats["searches_performed"],
                "documents_indexed": self.stats["documents_indexed"],
                "documents_updated": self.stats["documents_updated"],
                "user_saves": self.stats["user_saves"],
                "duplicates_skipped": self.stats["duplicates_skipped"],
                "success_rate": round(
                    self.stats["successful_operations"] / 
                    max(self.stats["successful_operations"] + self.stats["failed_operations"], 1),
                    3
                ),
                "vector_search_enabled": True,
                "user_library_enabled": True,
                "rag_ready": True,
                "serverless": True
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}


    async def delete_index(self):
        """Delete the entire index"""
        try:
            await self.async_client.indices.delete(index=self.index_name)
            logger.info(f"🗑️ Deleted index: {self.index_name}")
        except Exception as e:
            logger.error(f"❌ Failed to delete index: {e}")


    async def close(self):
        """Close client connections"""
        try:
            await self.async_client.close()
            self.client.close()
            logger.info("🔌 Elasticsearch connections closed")
        except Exception as e:
            logger.error(f"❌ Failed to close connections: {e}")


# ============================================
# HELPER FUNCTIONS
# ============================================


async def test_elasticsearch():
    """Test Elasticsearch Serverless connection"""
    client = ElasticClient()
    
    try:
        print("🔍 Testing Elasticsearch Serverless...")
        await client.test_connection()
        
        stats = await client.get_index_stats()
        print(f"\n📊 Index Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_elasticsearch())
