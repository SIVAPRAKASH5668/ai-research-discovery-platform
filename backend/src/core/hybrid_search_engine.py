# src/core/hybrid_search_engine.py - Elasticsearch hybrid search
import asyncio
import logging
from typing import Dict, Any, List, Optional
import time

from database.elastic_client import ElasticClient
from core.vertex_ai_processor import VertexAIProcessor

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    🔍 Advanced hybrid search engine combining multiple search strategies
    Integrates Elasticsearch text search with Vertex AI semantic search
    """
    
    def __init__(self, elastic_client: ElasticClient, vertex_ai: VertexAIProcessor):
        self.elastic_client = elastic_client
        self.vertex_ai = vertex_ai
        
        # Search configuration
        self.search_strategies = {
            "text_only": self._text_search,
            "vector_only": self._vector_search, 
            "hybrid": self._hybrid_search,
            "adaptive": self._adaptive_search
        }
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "average_response_time": 0.0,
            "strategy_usage": {
                "text_only": 0,
                "vector_only": 0,
                "hybrid": 0,
                "adaptive": 0
            }
        }
        
        logger.info("🔍 Hybrid search engine initialized")

    async def search(self, query: str, language: str = "en", 
                    context: Optional[Dict] = None, limit: int = 20,
                    strategy: str = "adaptive") -> List[Dict[str, Any]]:
        """
        Main search interface with strategy selection
        """
        start_time = time.time()
        
        try:
            if not query or not query.strip():
                logger.warning("⚠️ Empty query provided")
                return []
            
            # Select and execute search strategy
            search_function = self.search_strategies.get(strategy, self._adaptive_search)
            
            results = await search_function(
                query=query.strip(),
                language=language,
                context=context or {},
                limit=limit
            )
            
            # Post-process results
            processed_results = await self._post_process_results(results, query, context)
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_stats(strategy, response_time, True, len(processed_results))
            
            logger.info(f"✅ Search completed: {len(processed_results)} results in {response_time:.2f}s")
            return processed_results
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(strategy, response_time, False, 0)
            logger.error(f"❌ Search failed ({response_time:.2f}s): {e}")
            return []

    async def _adaptive_search(self, query: str, language: str, 
                             context: Dict, limit: int) -> List[Dict[str, Any]]:
        """
        Adaptive search that chooses the best strategy based on query characteristics
        """
        try:
            # Analyze query characteristics
            query_analysis = await self._analyze_query(query, language)
            
            # Choose strategy based on analysis
            if query_analysis["is_semantic"]:
                # Semantic queries benefit from hybrid approach
                logger.debug("📊 Using hybrid search for semantic query")
                return await self._hybrid_search(query, language, context, limit)
            elif query_analysis["is_specific"]:
                # Specific queries work well with text search
                logger.debug("📊 Using text search for specific query")
                return await self._text_search(query, language, context, limit)
            else:
                # Default to hybrid for balanced results
                logger.debug("📊 Using hybrid search as default")
                return await self._hybrid_search(query, language, context, limit)
                
        except Exception as e:
            logger.warning(f"⚠️ Adaptive search failed, falling back to text search: {e}")
            return await self._text_search(query, language, context, limit)

    async def _hybrid_search(self, query: str, language: str, 
                           context: Dict, limit: int) -> List[Dict[str, Any]]:
        """
        Hybrid search combining text and vector approaches
        """
        try:
            # Generate embedding for semantic search
            query_embedding = await self.vertex_ai.generate_embedding(query)
            
            if query_embedding:
                # Use Elasticsearch hybrid search
                results = await self.elastic_client.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    limit=limit
                )
                
                # Enhance with metadata
                for result in results:
                    result["search_strategy"] = "hybrid"
                    result["language"] = language
                    result["query_context"] = context
                
                return results
            else:
                # Fallback to text search if embedding fails
                logger.warning("⚠️ Embedding generation failed, using text search")
                return await self._text_search(query, language, context, limit)
                
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return await self._text_search(query, language, context, limit)

    async def _text_search(self, query: str, language: str, 
                         context: Dict, limit: int) -> List[Dict[str, Any]]:
        """
        Pure text-based search using Elasticsearch
        """
        try:
            # Build filters from context
            filters = self._build_search_filters(language, context)
            
            results = await self.elastic_client.search_papers(
                query=query,
                limit=limit,
                filters=filters
            )
            
            # Enhance with metadata
            for result in results:
                result["search_strategy"] = "text_only"
                result["language"] = language
                result["query_context"] = context
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Text search failed: {e}")
            return []

    async def _vector_search(self, query: str, language: str, 
                           context: Dict, limit: int) -> List[Dict[str, Any]]:
        """
        Pure vector-based semantic search
        """
        try:
            # Generate query embedding
            query_embedding = await self.vertex_ai.generate_embedding(query)
            
            if not query_embedding:
                logger.warning("⚠️ No embedding generated for vector search")
                return []
            
            results = await self.elastic_client.vector_search(
                query_embedding=query_embedding,
                limit=limit,
                similarity_threshold=0.6
            )
            
            # Enhance with metadata
            for result in results:
                result["search_strategy"] = "vector_only"
                result["language"] = language
                result["query_context"] = context
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []

    async def _analyze_query(self, query: str, language: str) -> Dict[str, bool]:
        """
        Analyze query characteristics to determine optimal search strategy
        """
        try:
            query_lower = query.lower()
            
            # Semantic indicators
            semantic_keywords = [
                "similar", "related", "like", "about", "concerning", 
                "research on", "studies about", "papers on"
            ]
            
            # Specific search indicators
            specific_keywords = [
                "author:", "title:", "doi:", "year:",
                "\"", "exact", "specific"
            ]
            
            # Technical term indicators
            technical_indicators = [
                "algorithm", "method", "model", "framework", 
                "approach", "technique", "implementation"
            ]
            
            is_semantic = any(keyword in query_lower for keyword in semantic_keywords)
            is_specific = any(keyword in query_lower for keyword in specific_keywords)
            is_technical = any(keyword in query_lower for keyword in technical_indicators)
            
            return {
                "is_semantic": is_semantic,
                "is_specific": is_specific,
                "is_technical": is_technical,
                "is_long": len(query.split()) > 5,
                "is_multilingual": language != "en"
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Query analysis failed: {e}")
            return {
                "is_semantic": False,
                "is_specific": False,
                "is_technical": False,
                "is_long": False,
                "is_multilingual": False
            }

    def _build_search_filters(self, language: str, context: Dict) -> Dict[str, Any]:
        """
        Build Elasticsearch filters from search context
        """
        filters = {}
        
        # Language filter (optional, allow multilingual by default)
        if context.get("strict_language"):
            filters["language"] = language
        
        # Domain filter
        if context.get("domain"):
            filters["domain"] = context["domain"]
        
        # Date range filter
        if context.get("date_range"):
            filters["date_range"] = context["date_range"]
        
        # Minimum citation filter
        if context.get("min_citations"):
            filters["min_citations"] = context["min_citations"]
        
        return filters

    async def _post_process_results(self, results: List[Dict], query: str, 
                                  context: Optional[Dict]) -> List[Dict[str, Any]]:
        """
        Post-process search results for consistency and enhancement
        """
        try:
            if not results:
                return results
            
            processed = []
            
            for result in results:
                # Ensure required fields exist
                processed_result = {
                    "id": result.get("id", ""),
                    "title": result.get("title", "Unknown Title"),
                    "abstract": result.get("abstract", ""),
                    "authors": result.get("authors", ""),
                    "domain": result.get("domain", "General Research"),
                    "language": result.get("language", "en"),
                    "publication_date": result.get("publication_date"),
                    "source": result.get("source", "elasticsearch"),
                    "citation_count": result.get("citation_count", 0),
                    "doi": result.get("doi", ""),
                    "url": result.get("url", ""),
                    
                    # Search metadata
                    "relevance_score": result.get("relevance_score", result.get("similarity_score", 0.5)),
                    "search_type": result.get("search_type", result.get("search_strategy", "hybrid")),
                    "confidence_score": result.get("confidence_score", 0.8),
                    
                    # Enhanced metadata
                    "search_engine": "elasticsearch",
                    "ai_enhanced": True,
                    "processed_at": time.time()
                }
                
                processed.append(processed_result)
            
            # Sort by relevance score
            processed.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return processed
            
        except Exception as e:
            logger.warning(f"⚠️ Post-processing failed: {e}")
            return results

    def _update_stats(self, strategy: str, response_time: float, 
                     success: bool, result_count: int):
        """
        Update search statistics
        """
        try:
            self.search_stats["total_searches"] += 1
            
            if success:
                self.search_stats["successful_searches"] += 1
            else:
                self.search_stats["failed_searches"] += 1
            
            # Update average response time
            total_time = (self.search_stats["average_response_time"] * 
                         (self.search_stats["total_searches"] - 1) + response_time)
            self.search_stats["average_response_time"] = total_time / self.search_stats["total_searches"]
            
            # Update strategy usage
            if strategy in self.search_stats["strategy_usage"]:
                self.search_stats["strategy_usage"][strategy] += 1
            
        except Exception as e:
            logger.warning(f"⚠️ Stats update failed: {e}")

    async def search_similar_papers(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find papers similar to a given paper using its embedding
        """
        try:
            # Get the reference paper
            reference_paper = await self.elastic_client.get_paper(paper_id)
            if not reference_paper:
                logger.warning(f"⚠️ Reference paper not found: {paper_id}")
                return []
            
            # Use title and abstract for similarity search
            reference_text = f"{reference_paper.get('title', '')} {reference_paper.get('abstract', '')}"
            
            # Generate embedding
            embedding = await self.vertex_ai.generate_embedding(reference_text)
            if not embedding:
                logger.warning("⚠️ Could not generate embedding for reference paper")
                return []
            
            # Perform vector search
            similar_papers = await self.elastic_client.vector_search(
                query_embedding=embedding,
                limit=limit + 1,  # +1 to account for the reference paper itself
                similarity_threshold=0.5
            )
            
            # Filter out the reference paper itself
            filtered_papers = [
                paper for paper in similar_papers 
                if paper.get("id") != paper_id
            ]
            
            return filtered_papers[:limit]
            
        except Exception as e:
            logger.error(f"❌ Similar papers search failed: {e}")
            return []

    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query
        """
        try:
            # Simple implementation - can be enhanced with more sophisticated suggestions
            suggestions = []
            
            # Common research prefixes
            prefixes = [
                "machine learning", "deep learning", "neural networks",
                "natural language processing", "computer vision", "artificial intelligence",
                "data mining", "information retrieval", "pattern recognition"
            ]
            
            partial_lower = partial_query.lower().strip()
            
            for prefix in prefixes:
                if prefix.startswith(partial_lower) and len(partial_lower) > 2:
                    suggestions.append(prefix)
                elif partial_lower in prefix and len(partial_lower) > 3:
                    suggestions.append(prefix)
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.warning(f"⚠️ Search suggestions failed: {e}")
            return []

    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive search statistics
        """
        return {
            **self.search_stats,
            "success_rate": (
                self.search_stats["successful_searches"] / 
                max(self.search_stats["total_searches"], 1)
            ),
            "strategies_available": list(self.search_strategies.keys()),
            "elasticsearch_enabled": True,
            "vertex_ai_enabled": True
        }

    def reset_stats(self):
        """
        Reset search statistics
        """
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "average_response_time": 0.0,
            "strategy_usage": {strategy: 0 for strategy in self.search_strategies.keys()}
        }
        logger.info("📊 Search statistics reset")
