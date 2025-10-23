"""
ðŸ† ELASTIC + GOOGLE CLOUD HACKATHON SUBMISSION
CONVERSATIONAL MULTILINGUAL RESEARCH DISCOVERY WITH VERTEX AI GEMINI + RAG

âœ… FEATURES:
- Conversational AI with Vertex AI Gemini
- Multilingual research discovery
- User library (save papers)
- RAG queries on user's saved papers
- Hybrid search (BM25 + Vector)
- Knowledge graph visualization
- Real-time edge calculation


Hackathon: Elastic + Google Cloud 2025

"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import numpy as np
import json
import re
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Translation support
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("âš ï¸  Translation unavailable. Install: pip install deep-translator")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
try:
    from database.elastic_client import ElasticClient
    from core.vertex_ai_processor import VertexAIProcessor
    from core.hybrid_search_engine import HybridSearchEngine
    from core.graph_builder import EnhancedIntelligentGraphBuilder
    from integrations.research_paper_apis import ResearchPaperAPIsClient
    
    from src.middleware.translation_middleware import auto_translate


    logger.info("âœ… All components imported successfully")
    COMPONENTS_OK = True
except ImportError as e:
    logger.error(f"âŒ Component import failed: {e}")
    COMPONENTS_OK = False

from elasticsearch import Elasticsearch

ELASTIC_ENDPOINT = os.getenv('ELASTIC_ENDPOINT')
ELASTIC_API_KEY = os.getenv('ELASTIC_API_KEY')
ES_INDEX = os.getenv('ELASTIC_INDEX_NAME', 'research')

logger.info("ðŸ”§ Initializing Elasticsearch Client for Cloud Endpoints...")

try:
    es = Elasticsearch(
        ELASTIC_ENDPOINT,
        api_key=ELASTIC_API_KEY,
        verify_certs=True,
        request_timeout=30
    )
    
    es_info = es.info()
    logger.info(f"âœ… Elasticsearch connected: {ELASTIC_ENDPOINT}")
    logger.info(f"âœ… Using index: {ES_INDEX}")
    
except Exception as e:
    logger.error(f"âŒ Elasticsearch initialization failed: {e}")
    es = None
# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ConversationRequest(BaseModel):
    message: str = Field(..., description="User's natural language question")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Previous conversation turns"
    )
    session_id: Optional[str] = Field(default="default", description="Session ID")


class MultilingualResearchRequest(BaseModel):
    query: str = Field(..., description="Research query")
    max_results: Optional[int] = Field(50, ge=10, le=100)
    enable_graph: Optional[bool] = Field(True)
    fetch_from_apis: Optional[bool] = Field(True)
    enable_translation: Optional[bool] = Field(True)


class FindSimilarRequest(BaseModel):
    paper_id: str
    max_results: Optional[int] = Field(10, ge=5, le=20)


class PaperSummarizeRequest(BaseModel):
    paper_id: str


class PaperQuestionRequest(BaseModel):
    paper_id: str
    question: str


# âœ… NEW: User Library Models
class SavePaperRequest(BaseModel):
    user_id: str
    paper_id: str
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class RAGQueryRequest(BaseModel):
    user_id: str
    question: str
    max_papers: int = 10


# ============================================================================
# ðŸ§  VERTEX AI GEMINI INTELLIGENT AGENT
# ============================================================================

async def _intelligent_agent_with_vertex_ai(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    vertex_ai
) -> Dict[str, Any]:
    """
    ðŸ§  Vertex AI Gemini-powered conversational agent
    Understands intent and decides actions
    """
    try:
        system_instruction = """You are an intelligent research assistant powered by Vertex AI Gemini.

**YOUR ROLE**: Understand user intent and provide intelligent responses.

**POSSIBLE INTENTS**:
1. SEARCH - User wants to find research papers
2. EXPLAIN - User wants explanation of concepts
3. ANALYZE - User wants analysis
4. QUESTION - General question

**RESPONSE FORMAT** (JSON):
{
  "intent": "search|explain|analyze|question",
  "query": "extracted search terms (if intent=search)",
  "needs_discovery": true/false,
  "response": "natural language response to user",
  "suggested_actions": ["suggestion 1", "suggestion 2"]
}

**EXAMPLES**:

User: "Find papers on quantum machine learning"
Response:
{
  "intent": "search",
  "query": "quantum machine learning",
  "needs_discovery": true,
  "response": "I'll search for papers on quantum machine learning.",
  "suggested_actions": ["Filter recent papers", "Focus on quantum algorithms"]
}

User: "What is quantum computing?"
Response:
{
  "intent": "explain",
  "needs_discovery": false,
  "response": "Quantum computing uses quantum phenomena like superposition and entanglement to perform computations. Would you like me to find papers on this topic?",
  "suggested_actions": ["Find introductory papers", "Search applications"]
}

**IMPORTANT**: Always provide valid JSON. Extract search terms when user wants papers."""

        # Build conversation context
        contents = []
        
        for turn in conversation_history[-3:]:  # Last 3 turns for context
            if turn.get('user'):
                contents.append({"role": "user", "parts": [{"text": turn['user']}]})
            if turn.get('assistant'):
                contents.append({"role": "model", "parts": [{"text": turn['assistant']}]})
        
        # Add current message
        contents.append({"role": "user", "parts": [{"text": user_message}]})
        
        # API payload
        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        # Call Vertex AI API
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        api_key = os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        
        endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{gemini_model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get('candidates', [])
                
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    
                    if parts:
                        response_text = parts[0].get('text', '')
                        
                        # Extract JSON from response
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            try:
                                intent_data = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                # Fallback parsing
                                intent_data = _fallback_intent_parsing(user_message, response_text)
                        else:
                            intent_data = _fallback_intent_parsing(user_message, response_text)
                        
                        # Ensure required fields
                        intent_data.setdefault('intent', 'search')
                        intent_data.setdefault('needs_discovery', intent_data['intent'] == 'search')
                        intent_data.setdefault('query', user_message)
                        intent_data.setdefault('response', response_text)
                        intent_data.setdefault('suggested_actions', [])
                        
                        logger.info(f"ðŸ§  Agent: {intent_data['intent']} | Discovery: {intent_data['needs_discovery']}")
                        
                        return intent_data
            
            logger.error(f"âŒ Vertex AI API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"âŒ Vertex AI agent failed: {e}")
        
        # Fallback: simple keyword detection
        search_keywords = ['find', 'search', 'papers', 'research', 'studies', 'discover', 'show me', 'look for']
        needs_search = any(keyword in user_message.lower() for keyword in search_keywords)
        
        return {
            "intent": "search" if needs_search else "question",
            "query": user_message,
            "needs_discovery": needs_search,
            "response": f"I'll help you with: {user_message}",
            "suggested_actions": [],
            "error": str(e)
        }


def _fallback_intent_parsing(user_message: str, response_text: str) -> Dict[str, Any]:
    """Fallback intent parsing when JSON extraction fails"""
    search_keywords = ['find', 'search', 'papers', 'research', 'studies', 'discover']
    needs_search = any(keyword in user_message.lower() for keyword in search_keywords)
    
    return {
        "intent": "search" if needs_search else "question",
        "query": user_message,
        "needs_discovery": needs_search,
        "response": response_text,
        "suggested_actions": []
    }


# ============================================================================
# PARALLEL TRANSLATION
# ============================================================================

async def _parallel_translate_query(query: str) -> Dict[str, str]:
    """Translate query to multiple languages in parallel"""
    if not TRANSLATION_AVAILABLE:
        return {'en': query}
    
    try:
        target_languages = {
            'en': 'English',
            'zh-CN': 'Chinese',
            'ja': 'Japanese',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'pt': 'Portuguese'
        }
        
        logger.info(f"ðŸŒ Translating to {len(target_languages)} languages...")
        start = time.time()
        
        async def translate_one(lang_code: str, lang_name: str) -> tuple:
            if lang_code == 'en':
                return (lang_code, query)
            
            try:
                loop = asyncio.get_event_loop()
                translator = GoogleTranslator(source='auto', target=lang_code)
                translated = await loop.run_in_executor(None, translator.translate, query)
                logger.info(f"   âœ… {lang_code}: '{translated}'")
                return (lang_code, translated)
            except Exception as e:
                logger.warning(f"   âš ï¸  {lang_code} failed: {e}")
                return (lang_code, query)
        
        tasks = [translate_one(code, name) for code, name in target_languages.items()]
        results = await asyncio.gather(*tasks)
        
        translations = dict(results)
        elapsed = time.time() - start
        
        logger.info(f"âœ… Translation done in {elapsed:.1f}s")
        return translations
        
    except Exception as e:
        logger.error(f"âŒ Translation failed: {e}")
        return {'en': query}


# ============================================================================
# PARALLEL API FETCHING
# ============================================================================

async def _parallel_fetch_papers(
    translations: Dict[str, str],
    api_client,
    max_results_per_lang: int = 10
) -> List[Dict]:
    """Fetch papers from APIs in parallel"""
    logger.info(f"ðŸ” Fetching papers from APIs...")
    start = time.time()
    
    async def fetch_english(query: str) -> List[Dict]:
        try:
            logger.info(f"   EN: '{query}'")
            papers = await api_client.search_all_apis(query=query, max_results=max_results_per_lang)
            logger.info(f"      âœ… EN: {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"      âš ï¸  EN failed: {e}")
            return []
    
    async def fetch_other_lang(lang_code: str, query: str) -> List[Dict]:
        if lang_code == 'en':
            return []
        
        try:
            logger.info(f"   {lang_code.upper()}: '{query}'")
            papers = await api_client.search_all_apis(query=query, max_results=max_results_per_lang // 2)
            logger.info(f"      âœ… {lang_code.upper()}: {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"      âš ï¸  {lang_code.upper()} failed: {e}")
            return []
    
    # Create tasks
    tasks = [fetch_english(translations.get('en', ''))]
    for lang_code, query in translations.items():
        if lang_code != 'en':
            tasks.append(fetch_other_lang(lang_code, query))
    
    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    all_papers = []
    for result in results:
        if isinstance(result, list):
            all_papers.extend(result)
    
    # Deduplicate by ID
    seen_ids = set()
    unique_papers = []
    for paper in all_papers:
        paper_id = paper.get('id', '')
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper)
    
    elapsed = time.time() - start
    logger.info(f"âœ… Fetch done in {elapsed:.1f}s: {len(unique_papers)} papers")
    
    return unique_papers


# ============================================================================
# PARALLEL INDEXING TO ELASTICSEARCH
# ============================================================================

async def _parallel_index_papers(
    papers: List[Dict],
    vertex_ai,
    elastic_client,
    batch_size: int = 10
) -> int:
    """Index papers to Elasticsearch with embeddings in parallel"""
    logger.info(f"ðŸ’¾ Indexing {len(papers)} papers...")
    start = time.time()
    
    async def index_batch(batch: List[Dict], batch_num: int) -> int:
        logger.info(f"   Batch {batch_num}: {len(batch)} papers")
        indexed = 0
        
        # Generate embeddings in parallel
        embedding_tasks = []
        for paper in batch:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')[:500]}"
            embedding_tasks.append(vertex_ai.generate_embedding(text, task_type="RETRIEVAL_DOCUMENT"))
        
        embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
        
        # Index papers
        for paper, embedding in zip(batch, embeddings):
            if isinstance(embedding, list) and len(embedding) > 0:
                try:
                    success = await elastic_client.index_paper(paper, embedding)
                    if success:
                        indexed += 1
                except Exception as e:
                    logger.warning(f"      Index failed: {e}")
        
        logger.info(f"   âœ… Batch {batch_num}: {indexed}/{len(batch)} indexed")
        return indexed
    
    # Split into batches
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
    logger.info(f"   ðŸ“¦ {len(batches)} batches of {batch_size}")
    
    # Index in parallel
    batch_tasks = [index_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*batch_tasks)
    
    total_indexed = sum(results)
    elapsed = time.time() - start
    
    logger.info(f"âœ… Indexing done in {elapsed:.1f}s: {total_indexed}/{len(papers)} indexed")
    
    return total_indexed


# ============================================================================
# OPTIMIZED ELASTICSEARCH SEARCH
# ============================================================================

async def _true_multilingual_search(
    elastic_client,
    query: str,
    query_vector: Optional[List[float]],
    limit: int
) -> List[Dict]:
    """
    âœ… OPTIMIZED hybrid search using KNN (FAST!)
    Replaces slow script_score with native KNN
    """
    try:
        # Method 1: If we have vector, use native KNN (FAST!)
        if query_vector and len(query_vector) == 768:
            search_body = {
                "size": limit,
                "knn": {  # âœ… Native KNN - FAST!
                    "field": "content_embedding",
                    "query_vector": query_vector,
                    "k": limit,
                    "num_candidates": min(limit * 3, 100),
                    "boost": 1.0
                },
                "query": {  # âœ… Text search runs in parallel
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "abstract^2"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "boost": 0.5
                    }
                },
                "_source": [
                    "id", "title", "abstract", "authors", "year", "publication_year",
                    "language", "source", "doi", "url", "citation_count",
                    "content_embedding"
                ],
                "timeout": "10s"
            }
        else:
            # Method 2: No vector, just text search
            search_body = {
                "size": limit,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "abstract^2"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "_source": [
                    "id", "title", "abstract", "authors", "year", "publication_year",
                    "language", "source", "doi", "url", "citation_count",
                    "content_embedding"
                ],
                "timeout": "10s"
            }
        
        logger.info(f"   ðŸ” Executing {'KNN+text hybrid' if query_vector else 'text-only'} search...")
        
        response = await elastic_client.async_client.search(
            index=elastic_client.index_name,
            body=search_body,
            request_timeout=30  # Increase timeout
        )
        
        papers = []
        for hit in response['hits']['hits']:
            paper = hit['_source']
            paper['id'] = hit['_id']
            paper['search_score'] = hit['_score']
            papers.append(paper)
        
        logger.info(f"âœ… Search returned {len(papers)} papers in {response.get('took', 0)}ms")
        
        return papers
        
    except Exception as e:
        logger.error(f"âŒ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# SIMILARITY EDGE CALCULATION
# ============================================================================

async def _calculate_similarity_edges(
    papers: List[Dict],
    top_n: int = 20,
    min_similarity: float = 0.70
) -> tuple:
    """Calculate similarity edges between top papers"""
    if len(papers) <= 1:
        logger.warning("âš ï¸  Not enough papers for edges")
        return papers, []
    
    logger.info(f"ðŸ”— Calculating edges (threshold: {min_similarity*100}%)...")
    start = time.time()
    
    # Sort by relevance
    sorted_papers = sorted(
        papers,
        key=lambda p: p.get('search_score', 0),
        reverse=True
    )[:top_n]
    
    logger.info(f"   ðŸ“Š Top {len(sorted_papers)} papers")
    
    edges = []
    calculations = 0
    
    for i, paper1 in enumerate(sorted_papers):
        embedding1 = paper1.get('content_embedding')
        
        if not embedding1:
            continue
        
        for j, paper2 in enumerate(sorted_papers[i+1:], start=i+1):
            embedding2 = paper2.get('content_embedding')
            
            if not embedding2:
                continue
            
            calculations += 1
            
            try:
                emb1 = np.array(embedding1)
                emb2 = np.array(embedding2)
                
                similarity = float(np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                ))
                
                if similarity >= min_similarity:
                    edges.append({
                        'source': paper1['id'],
                        'target': paper2['id'],
                        'strength': similarity,
                        'similarity_score': similarity,
                        'label': f"{int(similarity * 100)}%",
                        'color': '#3B82F6',
                        'width': similarity * 5
                    })
            
            except Exception as e:
                logger.warning(f"      âš ï¸  Similarity calculation failed: {e}")
                continue
    
    elapsed = time.time() - start
    
    logger.info(f"âœ… Edges done in {elapsed:.2f}s: {len(edges)} edges from {calculations} comparisons")
    
    return sorted_papers, edges


# ============================================================================
# FASTAPI APP LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    logger.info("=" * 80)
    logger.info("ðŸ¤– CONVERSATIONAL RESEARCH DISCOVERY + RAG")
    logger.info("=" * 80)
    
    # Initialize state
    app.state.elastic = None
    app.state.vertexai = None
    app.state.hybrid_search = None
    app.state.graph_builder = None
    app.state.api_client = None
    
    try:
        if COMPONENTS_OK:
            # Elasticsearch
            elastic = ElasticClient()
            await elastic.test_connection()
            logger.info("   âœ… Elasticsearch: Ready (with RAG support)")
            
            # Vertex AI
            vertex_ai = VertexAIProcessor()
            await vertex_ai.test_connection()
            logger.info("   âœ… Vertex AI: Ready (Gemini + Embeddings)")
            
            # Hybrid Search
            hybrid_search = HybridSearchEngine(elastic, vertex_ai)
            logger.info("   âœ… Hybrid Search: Ready")
            
            # Graph Builder
            graph_builder = EnhancedIntelligentGraphBuilder()
            logger.info("   âœ… Graph Builder: Ready")
            
            # API Client
            api_client = ResearchPaperAPIsClient()
            logger.info("   âœ… Research APIs: Ready")
            
            if TRANSLATION_AVAILABLE:
                logger.info("   âœ… Translation: Ready")
            
            # Set state
            app.state.elastic = elastic
            app.state.vertexai = vertex_ai
            app.state.hybrid_search = hybrid_search
            app.state.graph_builder = graph_builder
            app.state.api_client = api_client
        
        logger.info("=" * 80)
        logger.info("ðŸš€ SYSTEM READY - ALL FEATURES ENABLED")
        logger.info("=" * 80)
        
        yield
        
    except Exception as e:
        logger.error(f"âŒ Initialization failed: {e}")
        yield
    
    finally:
        # Cleanup
        if app.state.elastic:
            await app.state.elastic.close()


# Create FastAPI app
app = FastAPI(
    title="ðŸ¤– Conversational Research Discovery + RAG",
    description="Elastic + Google Cloud - Multilingual research with RAG",
    version="7.0.0-RAG",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Conversational Research Discovery + RAG",
        "status": "running",
        "version": "7.0.0-RAG",
        "features": {
            "conversational_ai": True,
            "multilingual_search": True,
            "user_library": True,
            "rag_queries": True,
            "knowledge_graph": True,
            "vertex_ai_gemini": True,
            "elasticsearch_hybrid": True
        }
    }


@app.get("/api/health")
async def health():
    """System health check"""
    return {
        "status": "healthy",
        "elasticsearch": app.state.elastic is not None,
        "vertex_ai": app.state.vertexai is not None,
        "hybrid_search": app.state.hybrid_search is not None,
        "graph_builder": app.state.graph_builder is not None,
        "api_client": app.state.api_client is not None
    }

"""
==================================================================================
CLOUD STORAGE API ENDPOINTS
Real-time monitoring, save/delete papers, activity tracking
==================================================================================
"""


# ============= MODELS =============

class SavePaperRequest(BaseModel):
    id: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None

# ============= CLOUD STATS ENDPOINT =============
@app.get("/api/cloud/stats")
@auto_translate 
async def get_cloud_stats():
    """Get real-time cloud storage statistics - SERVERLESS COMPATIBLE"""
    try:
        logger.info("ðŸ“Š Fetching cloud statistics")
        
        # Get total papers count
        total_papers = es.count(index=ES_INDEX)['count']
        
        # Get saved papers count
        saved_papers = es.count(
            index=ES_INDEX,
            body={"query": {"exists": {"field": "saved_at"}}}
        )['count']
        
        # Get last sync time
        try:
            last_sync_result = es.search(
                index=ES_INDEX,
                body={
                    "query": {"exists": {"field": "saved_at"}},
                    "sort": [{"saved_at": "desc"}],
                    "size": 1,
                    "_source": ["saved_at"]
                }
            )
            last_sync = last_sync_result['hits']['hits'][0]['_source']['saved_at'] if last_sync_result['hits']['hits'] else None
        except:
            last_sync = None
        
        # Serverless-compatible: Skip cluster health, just mark green if we can query
        stats_data = {
            "totalPapers": total_papers,
            "totalNodes": saved_papers,
            "storageUsed": 0,  # Serverless doesn't expose this
            "lastSync": last_sync,
            "elasticsearchHealth": "green",
            "vertexAIStatus": "online" if hasattr(app.state, 'vertexai') else "offline"
        }
        
        logger.info(f"âœ… Cloud stats: {total_papers} papers, {saved_papers} saved")
        
        return {"success": True, "stats": stats_data}
        
    except Exception as e:
        logger.error(f"âŒ Error fetching cloud stats: {str(e)}")
        return {
            "success": True,
            "stats": {
                "totalPapers": 0,
                "totalNodes": 0,
                "storageUsed": 0,
                "lastSync": None,
                "elasticsearchHealth": "yellow",
                "vertexAIStatus": "offline"
            }
        }

# ============= 2. SAVE PAPER =============

@app.post("/api/cloud/save")
@auto_translate 
async def save_paper_to_cloud(request: SavePaperRequest):
    """Save a paper to cloud storage"""
    try:
        paper_id = request.id
        logger.info(f"ðŸ’¾ Saving paper to cloud: {paper_id}")
        
        paper_data = {
            "id": paper_id,
            "title": request.title,
            "authors": request.authors,
            "abstract": request.abstract,
            "year": request.year,
            "doi": request.doi,
            "url": request.url,
            "pdf_url": request.pdf_url,
            "saved_at": datetime.now().isoformat(),
            "saved_to_cloud": True
        }
        
        # Try to update existing paper, or create new one
        try:
            es.get(index=ES_INDEX, id=paper_id)
            es.update(
                index=ES_INDEX,
                id=paper_id,
                body={"doc": {"saved_at": paper_data["saved_at"], "saved_to_cloud": True}}
            )
            logger.info(f"âœ… Updated existing paper: {paper_id}")
        except:
            es.index(index=ES_INDEX, id=paper_id, document=paper_data)
            logger.info(f"âœ… Created new paper: {paper_id}")
        
        return {
            "success": True,
            "message": "Paper saved to cloud",
            "paper_id": paper_id,
            "saved_at": paper_data["saved_at"]
        }
        
    except Exception as e:
        logger.error(f"âŒ Error saving paper: {str(e)}")
        return {"success": False, "error": str(e)}

# ============= 3. DELETE PAPER =============

@app.delete("/api/cloud/papers/{paper_id}")
@auto_translate 
async def delete_paper_from_cloud(paper_id: str):
    """Delete a paper from cloud storage"""
    try:
        logger.info(f"ðŸ—‘ï¸ Deleting paper from cloud: {paper_id}")
        
        # Remove saved_at flag (keep paper in index)
        es.update(
            index=ES_INDEX,
            id=paper_id,
            body={
                "script": {
                    "source": "ctx._source.remove('saved_at'); ctx._source.remove('saved_to_cloud');"
                }
            }
        )
        
        logger.info(f"âœ… Deleted paper: {paper_id}")
        
        return {
            "success": True,
            "message": "Paper deleted from cloud",
            "paper_id": paper_id
        }
        
    except Exception as e:
        logger.error(f"âŒ Error deleting paper: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/cloud/papers")
async def get_saved_papers(limit: int = 20, offset: int = 0):
    """Get list of all saved papers with full details"""
    try:
        logger.info(f"ðŸ“„ Fetching saved papers (limit={limit}, offset={offset})")
        
        result = es.search(
            index=ES_INDEX,
            body={
                "query": {"exists": {"field": "saved_at"}},
                "sort": [{"saved_at": "desc"}],  # Most recent first
                "from": offset,
                "size": limit
            }
        )
        
        papers = []
        for hit in result['hits']['hits']:
            paper = hit['_source']
            paper['id'] = hit['_id']
            papers.append(paper)
        
        total = result['hits']['total']['value']
        
        logger.info(f"âœ… Retrieved {len(papers)} saved papers (total: {total})")
        
        return {
            "success": True,
            "papers": papers,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"âŒ Error fetching saved papers: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "papers": [],
            "total": 0
        }

# ============= GET ALL PAPERS (NOT JUST SAVED) =============

@app.get("/api/cloud/all-papers")
async def get_all_papers(limit: int = 50, offset: int = 0):
    """Get ALL papers in cloud storage (not just saved ones)"""
    try:
        logger.info(f"ðŸ“„ Fetching all papers (limit={limit}, offset={offset})")
        
        result = es.search(
            index=ES_INDEX,
            body={
                "query": {"match_all": {}},  # â† Get ALL papers
                "sort": [{"indexed_at": {"order": "desc"}}],
                "from": offset,
                "size": limit
            }
        )
        
        papers = []
        for hit in result['hits']['hits']:
            paper = hit['_source']
            paper['id'] = hit['_id']
            # Mark if it's saved or not
            paper['is_saved'] = 'saved_at' in paper
            papers.append(paper)
        
        total = result['hits']['total']['value']
        
        logger.info(f"âœ… Retrieved {len(papers)} papers (total: {total})")
        
        return {
            "success": True,
            "papers": papers,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"âŒ Error fetching all papers: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "papers": [],
            "total": 0
        }

# ============================================================================
# ðŸ’¬ CONVERSATIONAL CHAT ENDPOINT
# ============================================================================

# âœ… Make sure this is imported!
from pydantic import BaseModel

# Your Pydantic model
class ConversationRequest(BaseModel):
    message: str
    conversation_history: list = []
    session_id: str = None

# âœ…âœ…âœ… CRITICAL: Add `http_request: Request` as FIRST parameter!
@app.post("/api/chat")
@auto_translate  # Decorator
async def chat(
    http_request: Request,  # âœ…âœ…âœ… THIS IS CRITICAL - ADD THIS LINE!
    request: ConversationRequest  # Your existing Pydantic model
):
    """
    ðŸ’¬ Conversational research assistant powered by Vertex AI Gemini
    """
    try:
        logger.info("=" * 80)
        logger.info(f"ðŸ’¬ USER: {request.message}")
        logger.info(f"ðŸ” DEBUG: http_request type = {type(http_request)}")
        logger.info(f"ðŸ” DEBUG: request type = {type(request)}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        if not app.state.vertexai or not app.state.elastic:
            return {
                "success": False,
                "error": "System not initialized",
                "agent_response": "I'm currently unavailable. Please try again later."
            }
        
        # Step 1: Vertex AI Gemini Agent
        logger.info("ðŸ§  STEP 1: Vertex AI Gemini Agent Decision...")
        agent_decision = await _intelligent_agent_with_vertex_ai(
            request.message,
            request.conversation_history,
            app.state.vertexai
        )
        
        logger.info(f"   Intent: {agent_decision['intent']}")
        logger.info(f"   Discovery needed: {agent_decision['needs_discovery']}")
        
        # Step 2: If search intent, discover papers
        papers = []
        edges = []
        
        if agent_decision.get('needs_discovery') and agent_decision['intent'] == 'search':
            logger.info("ðŸ” STEP 2: Discovering Papers...")
            
            # Translate query
            translations = await _parallel_translate_query(agent_decision['query'])
            
            # Fetch from APIs
            fetched_papers = await _parallel_fetch_papers(
                translations,
                app.state.api_client,
                max_results_per_lang=10
            )
            
            # Index to Elasticsearch
            indexed_count = await _parallel_index_papers(
                fetched_papers,
                app.state.vertexai,
                app.state.elastic
            )
            
            logger.info(f"   ðŸ“Š Indexed: {indexed_count} papers")
            
            # Search Elasticsearch
            logger.info("ðŸ” STEP 3: Searching Elasticsearch...")
            query_vector = await app.state.vertexai.generate_embedding(
                agent_decision['query'],
                task_type="RETRIEVAL_QUERY"
            )
            
            all_papers = await _true_multilingual_search(
                app.state.elastic,
                agent_decision['query'],
                query_vector,
                limit=50
            )
            
            # Calculate edges
            logger.info("ðŸ”— STEP 4: Calculating Edges...")
            papers, edges = await _calculate_similarity_edges(
                all_papers,
                top_n=20,
                min_similarity=0.70
            )
            
            logger.info(f"   ðŸ“Š Returning {len(papers)} papers with {len(edges)} edges")
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"âœ… COMPLETED in {elapsed:.2f}s")
        logger.info("=" * 80)
        
        return {
            "success": True,
            "agent_response": agent_decision.get('response', ''),
            "intent": agent_decision.get('intent'),
            "papers": papers,
            "edges": edges,
            "total_papers": len(papers),
            "total_edges": len(edges),
            "suggested_actions": agent_decision.get('suggested_actions', []),
            "processing_time": elapsed,
            "session_id": request.session_id
        }
        
    except Exception as e:
        logger.error(f"âŒ Chat failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "agent_response": "I encountered an error. Please try again."
        }


# ============================================================================
# ðŸ“š RESEARCH ENDPOINTS
# ============================================================================

@app.post("/api/research/discover")
@auto_translate 
async def discover_multilingual(request: MultilingualResearchRequest):
    """
    ðŸ“š Multilingual research discovery
    """
    try:
        logger.info(f"ðŸ” DISCOVER: {request.query}")
        start_time = time.time()
        
        if not app.state.elastic or not app.state.vertexai:
            return {"success": False, "error": "System not initialized"}
        
        # Translate
        translations = await _parallel_translate_query(request.query)
        
        # Fetch
        if request.fetch_from_apis:
            fetched_papers = await _parallel_fetch_papers(
                translations,
                app.state.api_client,
                max_results_per_lang=10
            )
            
            # Index
            indexed_count = await _parallel_index_papers(
                fetched_papers,
                app.state.vertexai,
                app.state.elastic
            )
            
            logger.info(f"   Indexed: {indexed_count} papers")
        
        # Search
        query_vector = await app.state.vertexai.generate_embedding(
            request.query,
            task_type="RETRIEVAL_QUERY"
        )
        
        all_papers = await _true_multilingual_search(
            app.state.elastic,
            request.query,
            query_vector,
            limit=request.max_results
        )
        
        # Edges
        papers, edges = await _calculate_similarity_edges(
            all_papers,
            top_n=20,
            min_similarity=0.70
        )
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "papers": papers,
            "edges": edges,
            "total_papers": len(papers),
            "total_edges": len(edges),
            "processing_time": elapsed
        }
        
    except Exception as e:
        logger.error(f"âŒ Discover failed: {e}")
        return {"success": False, "error": str(e)}


from fastapi import Request  # âœ… Make sure this is imported

@app.post("/api/research/find-similar")
@auto_translate 
async def find_similar(
    http_request: Request,  # âœ… ADD THIS LINE!
    request: FindSimilarRequest
):
    """
    ðŸ”— Find similar papers using vector similarity
    âœ… Supports automatic translation to any language
    """
    try:
        logger.info(f"ðŸ” Finding similar papers for {request.paper_id}")
        
        if not app.state.elastic:
            return {"success": False, "error": "System not initialized"}
        
        # âœ… FIX: Explicitly fetch paper WITH embedding
        try:
            paper_response = await app.state.elastic.async_client.get(
                index=app.state.elastic.index_name,
                id=request.paper_id,
                _source_includes=[
                    'id', 'title', 'abstract', 'authors', 'year', 
                    'publication_year', 'language', 'source', 'doi', 
                    'url', 'citation_count', 'content_embedding'
                ]
            )
            
            paper = paper_response["_source"]
            paper['id'] = request.paper_id
            
            logger.info(f"   ðŸ“„ Paper fetched: {paper.get('title', 'Unknown')[:50]}")
            
        except Exception as e:
            logger.error(f"âŒ Paper not found: {request.paper_id} - {e}")
            return {
                "success": False,
                "error": f"Paper not found: {request.paper_id}"
            }
        
        # Check if embedding exists
        embedding = paper.get('content_embedding')
        
        if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
            logger.error(f"âŒ No valid embedding for {request.paper_id}")
            return {
                "success": False,
                "error": "No embedding available for this paper. Try papers with abstracts."
            }
        
        logger.info(f"   âœ… Embedding found: {len(embedding)} dimensions")
        
        # Find similar papers using vector search
        similar_papers = await app.state.elastic.vector_search(
            query_embedding=embedding,
            limit=request.max_results + 1,
            similarity_threshold=0.65
        )
        
        # Remove source paper from results
        similar_papers = [
            p for p in similar_papers 
            if p.get('id') != request.paper_id
        ][:request.max_results]
        
        logger.info(f"âœ… Found {len(similar_papers)} similar papers")
        
        # Build relationships (edges) for frontend
        relationships = []
        source_embedding = np.array(embedding)
        
        for similar_paper in similar_papers:
            sim_embedding = similar_paper.get('content_embedding')
            
            if sim_embedding and len(sim_embedding) > 0:
                sim_emb_array = np.array(sim_embedding)
                similarity = float(
                    np.dot(source_embedding, sim_emb_array) / 
                    (np.linalg.norm(source_embedding) * np.linalg.norm(sim_emb_array))
                )
                
                relationships.append({
                    'source': request.paper_id,
                    'target': similar_paper['id'],
                    'strength': similarity,
                    'similarity_score': similarity,
                    'label': f"{int(similarity * 100)}%",
                    'color': '#3B82F6',
                    'width': similarity * 5
                })
        
        logger.info(f"âœ… Created {len(relationships)} relationships")
        
        # âœ… This return will be automatically translated by @auto_translate!
        return {
            "success": True,
            "source_paper": paper,  # âœ… Will be translated
            "similar_papers": similar_papers,  # âœ… Will be translated
            "relationships": relationships,
            "total_similar": len(similar_papers)
        }
        
    except Exception as e:
        logger.error(f"âŒ Find similar failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/paper/summarize")
async def summarize_paper(
    http_request: Request,  # ✅ Add this to get HTTP headers
    request: PaperSummarizeRequest
):
    """
    📝 Summarize paper with Vertex AI Gemini + Translation Support
    """
    try:
        logger.info(f"📝 Summarizing paper: {request.paper_id}")
        
        start_time = time.time()
        
        # Get user's language from Accept-Language header
        user_language = http_request.headers.get('accept-language', 'en')
        
        # Extract primary language code (e.g., 'zh-CN' -> 'zh')
        if ',' in user_language:
            user_language = user_language.split(',')[0]
        if '-' in user_language:
            user_language = user_language.split('-')[0]
        
        logger.info(f"🌍 User language detected: {user_language}")
        
        if not app.state.vertexai:
            logger.error("❌ Vertex AI not initialized")
            return {"success": False, "error": "System not initialized"}
        
        # Get paper from Elasticsearch
        logger.info(f"📄 Fetching paper from Elasticsearch...")
        
        try:
            paper_doc = es.get(index=ES_INDEX, id=request.paper_id)
            paper = paper_doc['_source']
            
            logger.info(f"✅ Paper retrieved: {paper.get('title', '')[:60]}...")
            
        except Exception as e:
            logger.error(f"❌ Paper not found: {request.paper_id} - {e}")
            return {
                "success": False,
                "error": f"Paper not found: {request.paper_id}"
            }
        
        # Validate content
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        if not abstract or len(abstract.strip()) < 50:
            logger.warning(f"⚠️ Paper {request.paper_id} has insufficient content")
            return {
                "success": False,
                "error": "Paper does not have sufficient content for summarization"
            }
        
        # Generate summary
        logger.info(f"🤖 Generating AI summary in {user_language}...")
        
        # Language mapping for better prompt
        language_names = {
            'zh': 'Chinese (中文)',
            'es': 'Spanish (Español)',
            'fr': 'French (Français)',
            'de': 'German (Deutsch)',
            'ja': 'Japanese (日本語)',
            'ko': 'Korean (한국어)',
            'pt': 'Portuguese (Português)',
            'ru': 'Russian (Русский)',
            'ar': 'Arabic (العربية)',
            'hi': 'Hindi (हिन्दी)',
            'en': 'English'
        }
        
        target_language = language_names.get(user_language, 'English')
        
        # ✅ MULTILINGUAL PROMPT
        prompt = f"""You are a research paper summarization expert with multilingual capabilities.

**Paper Title**: {title}

**Abstract**: {abstract[:1500]}

**Task**: Provide a concise, professional summary in **{target_language}** in 3-4 bullet points covering:
• Main research question/problem
• Methodology approach  
• Key findings/results
• Significance/impact

**IMPORTANT**: Your entire response MUST be in {target_language}. Do not mix languages.

**Summary in {target_language}**:"""
        
        try:
            summary = await app.state.vertexai.generate_text(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            # Validate
            if not summary or len(summary.strip()) < 50:
                logger.error(f"❌ Generated summary too short: {len(summary)} chars")
                raise Exception("Generated summary is too short or empty")
            
            # Check truncation
            if summary.strip().endswith(('*', '•', '-', ':', ',')):
                logger.warning("⚠️ Summary might be truncated, retrying...")
                summary = await app.state.vertexai.generate_text(
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=0.7
                )
            
            logger.info(f"✅ Summary generated in {user_language} ({len(summary)} chars)")
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {
                "success": False,
                "error": f"Failed to generate summary: {str(e)}"
            }
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Summarization completed in {elapsed:.2f}s")
        
        return {
            "success": True,
            "paper_id": request.paper_id,
            "title": title,
            "summary": summary,
            "language": user_language,
            "source": "paper_abstract",
            "generated_by": "Vertex AI Gemini",
            "processing_time": round(elapsed, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Summarize failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }




@app.post("/api/paper/ask")
@auto_translate 
async def ask_paper(request: PaperQuestionRequest):
    """
    â“ Ask question about paper
    """
    try:
        if not app.state.elastic or not app.state.vertexai:
            return {"success": False, "error": "System not initialized"}
        
        # Get paper
        paper = await app.state.elastic.get_paper(request.paper_id)
        
        if not paper:
            return {"success": False, "error": "Paper not found"}
        
        # RAG prompt
        prompt = f"""Answer this question about the research paper:

Question: {request.question}

Paper Details:
Title: {paper.get('title', 'Unknown')}
Authors: {paper.get('authors', 'Unknown')}
Abstract: {paper.get('abstract', 'No abstract')}

Provide a clear, concise answer based only on the paper information above."""

        answer = await app.state.vertexai.generate_text(
            prompt,
            max_tokens=500,
            temperature=0.2
        )
        
        return {
            "success": True,
            "paper_id": request.paper_id,
            "question": request.question,
            "answer": answer,
            "source": "single_paper_rag"
        }
        
    except Exception as e:
        logger.error(f"âŒ Ask failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# ðŸ’¾ USER LIBRARY ENDPOINTS (NEW!)
# ============================================================================

@app.post("/api/library/save")
@auto_translate 
async def save_paper_to_library(request: SavePaperRequest):
    """
    ðŸ’¾ Save paper to user's personal library
    """
    try:
        if not app.state.elastic or not app.state.vertexai:
            return {"success": False, "error": "System not initialized"}
        
        # Get paper from Elasticsearch
        try:
            paper_response = await app.state.elastic.async_client.get(
                index=app.state.elastic.index_name,
                id=request.paper_id
            )
            paper = paper_response["_source"]
        except Exception as e:
            return {"success": False, "error": f"Paper not found: {request.paper_id}"}
        
        # Re-index with user_id
        success = await app.state.elastic.index_paper(
            paper=paper,
            embedding=paper.get("content_embedding"),
            user_id=request.user_id,
            user_notes=request.notes,
            user_tags=request.tags
        )
        
        if success:
            logger.info(f"ðŸ’¾ Saved paper {request.paper_id} for user {request.user_id}")
            return {
                "success": True,
                "message": "Paper saved to your library",
                "paper_id": request.paper_id,
                "user_id": request.user_id
            }
        else:
            return {"success": False, "error": "Failed to save paper"}
        
    except Exception as e:
        logger.error(f"âŒ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/library/{user_id}")
@auto_translate 
async def get_user_library(user_id: str, page: int = 1, size: int = 20):
    """
    ðŸ“š Get user's saved papers
    """
    try:
        if not app.state.elastic:
            return {"success": False, "error": "System not initialized"}
        
        search_body = {
            "query": {
                "term": {"user_id": user_id}
            },
            "sort": [{"saved_at": "desc"}],
            "from": (page - 1) * size,
            "size": size,
            "_source": {"excludes": ["content_embedding"]}
        }
        
        response = await app.state.elastic.async_client.search(
            index=app.state.elastic.index_name,
            body=search_body
        )
        
        papers = []
        for hit in response["hits"]["hits"]:
            paper = hit["_source"]
            paper["id"] = hit["_id"]
            papers.append(paper)
        
        total = response["hits"]["total"]["value"]
        
        return {
            "success": True,
            "papers": papers,
            "total": total,
            "page": page,
            "size": size,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"âŒ Library fetch failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/rag/query")
@auto_translate
async def rag_query_user_library(
    http_request: Request,
    request: RAGQueryRequest
):
    """
    🤖 Multi-Paper RAG: Search papers + Generate answer with Gemini
    ✅ Uses Google Cloud Gemini API (gemini-2.5-flash-lite)
    ✅ Supports automatic translation
    """
    try:
        start_time = time.time()
        
        logger.info(f"🤖 RAG Query: user={request.user_id}, question={request.question}")
        
        if not app.state.elastic:
            return {"success": False, "error": "System not initialized"}
        
        # Step 1: RETRIEVE - Generate query embedding
        query_vector = await app.state.vertexai.generate_embedding(
            text=request.question,
            task_type="RETRIEVAL_QUERY"
        )
        
        # Search ALL papers
        user_papers = await app.state.elastic.hybrid_search(
            query=request.question,
            query_embedding=query_vector,
            limit=request.max_papers
        )
        
        logger.info(f"📚 Retrieved {len(user_papers)} relevant papers")
        
        if not user_papers:
            return {
                "success": True,
                "answer": "No relevant papers found for your question.",
                "papers_used": [],
                "papers_count": 0,
                "source": "no_papers_found"
            }
        
        # Step 2: AUGMENT - Build context from papers
        context_parts = []
        papers_metadata = []
        
        for i, paper in enumerate(user_papers, 1):
            paper_context = f"""
**Paper {i}:**
- **Title:** {paper.get('title', 'Untitled')}
- **Authors:** {paper.get('authors', 'Unknown')}
- **Year:** {paper.get('publication_year', 'Unknown')}
- **Abstract:** {paper.get('abstract', 'No abstract')[:500]}
"""
            context_parts.append(paper_context)
            
            papers_metadata.append({
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper.get("authors", "Unknown"),
                "year": paper.get("publication_year", "Unknown"),
                "abstract": paper.get("abstract", ""),
                "relevance": paper.get("similarity_score", paper.get("relevance_score", 0))
            })
        
        full_context = "\n".join(context_parts)
        
        # Step 3: GENERATE - Create answer with Gemini API
        prompt = f"""You are a research assistant analyzing multiple research papers.

**USER'S QUESTION:**
{request.question}

**RELEVANT PAPERS:**
{full_context}

**INSTRUCTIONS:**
1. Answer the question using information from the papers above
2. Cite specific papers when making claims (use Paper 1, Paper 2, etc.)
3. If papers don't fully answer the question, explain what information is available
4. Keep answer concise but informative (3-4 paragraphs max)
5. Structure response clearly with main findings and supporting evidence

**ANSWER:**"""

        # ✅ USE EXACT GOOGLE CLOUD ENDPOINT (from your curl example)
        api_key = os.getenv("GOOGLE_API_KEY")
        gemini_model = "gemini-2.5-flash-lite"  # ✅ Available model
        
        # ✅ Exact endpoint from your curl command
        endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{gemini_model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            ai_response = await client.post(endpoint, json=payload, headers=headers)
            
            if ai_response.status_code == 200:
                result = ai_response.json()
                candidates = result.get("candidates", [])
                
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    if parts:
                        answer_text = parts[0].get("text", "").strip()
                        
                        elapsed = time.time() - start_time
                        
                        logger.info(f"✅ RAG answer generated in {elapsed:.2f}s")
                        
                        return {
                            "success": True,
                            "answer": answer_text,
                            "papers_used": papers_metadata,
                            "papers_count": len(user_papers),
                            "user_id": request.user_id,
                            "source": "multi_paper_rag_gemini",
                            "generated_by": "google_cloud_gemini",
                            "gemini_model": gemini_model,
                            "processing_time": elapsed
                        }
            
            # Better error handling
            logger.error(f"❌ Gemini API error: {ai_response.status_code} - {ai_response.text}")
            return {
                "success": False, 
                "error": f"AI generation failed: {ai_response.status_code}",
                "details": ai_response.text[:200]
            }
        
    except Exception as e:
        logger.error(f"❌ RAG query failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}



# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

