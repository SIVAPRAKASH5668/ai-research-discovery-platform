# src/core/vertex_ai_processor.py - SAME INTERFACE, OPTIMIZED INTERNALS
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Standardized model response"""
    content: str
    tokens_used: int
    confidence: float
    model_used: str
    processing_time: float

class VertexAIProcessor:
    """
    🧠 Complete Vertex AI processor for Gemini models and embeddings
    Handles all AI processing needs for the research platform
    ✅ SAME INTERFACE - All method names preserved
    🚀 OPTIMIZED - Uses multilingual embeddings internally
    """
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.project_id or not self.api_key:
            raise ValueError("Missing required environment variables: GOOGLE_CLOUD_PROJECT, GOOGLE_API_KEY")
        
        # Model configuration
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.gemini_flash_model = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
        # 🚀 OPTIMIZED: Use multilingual embedding model internally
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-multilingual-embedding-002")
        
        # API endpoints
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models"
        
        # Request configuration
        self.default_timeout = 60
        self.max_retries = 3
        self.request_delay = 0.1
        
        # 🚀 OPTIMIZED: Add caching for performance
        self.embedding_cache = {}
        self.cache_max_size = 1000
        self.cache_ttl = 3600
        
        # Statistics tracking
        self.stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_processing_time": 0.0,
            # 🚀 OPTIMIZED: Add multilingual stats
            "embeddings_generated": 0,
            "cache_hits": 0,
            "multilingual_queries": 0,
            "languages_detected": set()
        }
        
        logger.info(f"🧠 VertexAI Processor initialized - Project: {self.project_id}, Location: {self.location}")

    async def test_connection(self) -> bool:
        """Test Vertex AI connection with embedding generation - SAME SIGNATURE"""
        try:
            start_time = time.time()
            
            # Test embedding generation
            test_embedding = await self.generate_embedding("test connection to vertex ai")
            
            if test_embedding and len(test_embedding) == 768:
                processing_time = time.time() - start_time
                logger.info(f"✅ Vertex AI connection successful ({processing_time:.2f}s)")
                return True
            else:
                raise Exception(f"Invalid embedding response: {len(test_embedding) if test_embedding else 0} dimensions")
                
        except Exception as e:
            logger.error(f"❌ Vertex AI connection test failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Quick health check - SAME SIGNATURE"""
        try:
            embedding = await self.generate_embedding("health check")
            return embedding is not None and len(embedding) == 768
        except:
            return False

    async def generate_text(self, prompt: str, max_tokens: int = 1000, 
                          temperature: float = 0.1, model: str = None) -> str:
        """Generate text using Gemini model - SAME SIGNATURE"""
        start_time = time.time()
        
        try:
            model_name = model or self.gemini_model
            url = f"{self.base_url}/{model_name}:generateContent"
            
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.8,
                    "topK": 40
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            response_data = await self._make_request_with_retry(url, payload)
            
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0]["text"]
                    
                    processing_time = time.time() - start_time
                    self.stats["successful_requests"] += 1
                    self.stats["total_processing_time"] += processing_time
                    
                    estimated_tokens = len(text.split()) * 1.3
                    self.stats["total_tokens_used"] += estimated_tokens
                    
                    logger.debug(f"✅ Generated {len(text)} chars in {processing_time:.2f}s")
                    return text
            
            raise Exception("No valid content in response")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_requests"] += 1
            logger.error(f"❌ Text generation failed ({processing_time:.2f}s): {e}")
            raise

    async def generate_text_stream(self, prompt: str, max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """Generate streaming text response - SAME SIGNATURE"""
        try:
            full_response = await self.generate_text(prompt, max_tokens)
            
            words = full_response.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                yield chunk + " "
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ Streaming text generation failed: {e}")
            yield f"Error: {str(e)}"

    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
        """
        Generate text embedding using Vertex AI - SAME SIGNATURE
        🚀 OPTIMIZED: Uses multilingual model internally with caching
        """
        try:
            if not text or not text.strip():
                return None
            
            start_time = time.time()
            
            # 🚀 OPTIMIZED: Check cache first
            cache_key = self._get_embedding_cache_key(text, task_type)
            cached_result = self._get_cached_embedding(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
            
            # 🚀 OPTIMIZED: Use multilingual embedding model
            url = f"{self.base_url}/{self.embedding_model}:predict"
            
            payload = {
                "instances": [
                    {
                        "content": text.strip()[:8000],
                        "task_type": task_type
                    }
                ]
            }
            
            response_data = await self._make_request_with_retry(url, payload)
            
            if "predictions" in response_data and response_data["predictions"]:
                prediction = response_data["predictions"][0]
                if "embeddings" in prediction and "values" in prediction["embeddings"]:
                    embedding_values = prediction["embeddings"]["values"]
                    
                    if len(embedding_values) == 768:
                        processing_time = time.time() - start_time
                        
                        # 🚀 OPTIMIZED: Cache the result
                        self._cache_embedding(cache_key, embedding_values)
                        
                        # 🚀 OPTIMIZED: Detect language and update stats
                        detected_lang = self._detect_language_simple(text)
                        if detected_lang and detected_lang != 'en':
                            self.stats["languages_detected"].add(detected_lang)
                            self.stats["multilingual_queries"] += 1
                        
                        self.stats["successful_requests"] += 1
                        self.stats["embeddings_generated"] += 1
                        
                        logger.debug(f"✅ Generated embedding: {len(embedding_values)} dimensions")
                        return embedding_values
            
            raise Exception("Invalid embedding response structure")
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"❌ Embedding generation failed: {e}")
            return None

    async def generate_embeddings_batch(self, texts: List[str], 
                                      batch_size: int = 5, 
                                      task_type: str = "RETRIEVAL_DOCUMENT") -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches - SAME SIGNATURE"""
        try:
            if not texts:
                return []
            
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                batch_tasks = [
                    self.generate_embedding(text, task_type) 
                    for text in batch
                ]
                
                batch_embeddings = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for embedding in batch_embeddings:
                    if isinstance(embedding, Exception):
                        embeddings.append(None)
                        logger.warning(f"⚠️ Batch embedding failed: {embedding}")
                    else:
                        embeddings.append(embedding)
                
                if i + batch_size < len(texts):
                    await asyncio.sleep(self.request_delay)
            
            success_count = sum(1 for e in embeddings if e is not None)
            logger.info(f"📊 Batch embeddings: {success_count}/{len(texts)} successful")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            return [None] * len(texts)

    async def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using Gemini - SAME SIGNATURE"""
        try:
            prompt = f"""
            Analyze the sentiment and key aspects of this text:
            
            Text: {text}
            
            Provide analysis in JSON format with:
            - sentiment: positive/negative/neutral
            - confidence: 0.0 to 1.0
            - key_themes: list of main themes
            - emotional_tone: description
            """
            
            response = await self.generate_text(prompt, max_tokens=500, temperature=0.1)
            
            try:
                return json.loads(response)
            except:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "key_themes": ["analysis_failed"],
                    "emotional_tone": "unable_to_determine",
                    "raw_response": response
                }
                
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "key_themes": [],
                "emotional_tone": "error",
                "error": str(e)
            }

    # 🚀 OPTIMIZED: Internal helper methods (not changing public interface)
    def _detect_language_simple(self, text: str) -> Optional[str]:
        """Simple language detection"""
        try:
            text_lower = text.lower()
            
            if any(ord(c) > 0x4e00 and ord(c) < 0x9fff for c in text):
                return 'zh'
            elif any(ord(c) > 0x3040 and ord(c) < 0x309f for c in text):
                return 'ja'
            elif any(ord(c) > 0x0600 and ord(c) < 0x06ff for c in text):
                return 'ar'
            elif any(term in text_lower for term in ['machine learning', 'the', 'and', 'of']):
                return 'en'
            elif any(term in text_lower for term in ['aprendizaje', 'el', 'la', 'de']):
                return 'es'
            elif any(term in text_lower for term in ['apprentissage', 'le', 'la', 'de']):
                return 'fr'
            elif any(term in text_lower for term in ['maschinelles', 'der', 'die', 'das']):
                return 'de'
            
            return 'en'
        except Exception:
            return None

    def _get_embedding_cache_key(self, text: str, task_type: str) -> str:
        """Generate cache key"""
        import hashlib
        key_string = f"{text[:100]}_{task_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get cached embedding"""
        try:
            if cache_key in self.embedding_cache:
                result, timestamp = self.embedding_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return result
                else:
                    del self.embedding_cache[cache_key]
            return None
        except Exception:
            return None

    def _cache_embedding(self, cache_key: str, embedding: List[float]):
        """Cache embedding"""
        try:
            if len(self.embedding_cache) >= self.cache_max_size:
                sorted_items = sorted(
                    self.embedding_cache.items(), 
                    key=lambda x: x[1][1]
                )
                for key, _ in sorted_items[:self.cache_max_size//2]:
                    del self.embedding_cache[key]
            
            self.embedding_cache[cache_key] = (embedding, time.time())
        except Exception as e:
            logger.debug(f"Caching failed: {e}")

    async def _make_request_with_retry(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retry logic - SAME AS BEFORE"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        url_with_key = f"{url}?key={self.api_key}"
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.default_timeout) as client:
                    response = await client.post(url_with_key, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        self.stats["requests_made"] += 1
                        return response.json()
                    elif response.status_code == 429:
                        wait_time = (2 ** attempt) + self.request_delay
                        logger.warning(f"⚠️ Rate limited, retrying in {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = response.text
                        raise Exception(f"HTTP {response.status_code}: {error_text}")
                        
            except httpx.TimeoutException:
                last_exception = Exception(f"Request timeout after {self.default_timeout}s")
                logger.warning(f"⚠️ Request timeout (attempt {attempt + 1})")
            except Exception as e:
                last_exception = e
                logger.warning(f"⚠️ Request failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = (2 ** attempt) + self.request_delay
                await asyncio.sleep(wait_time)
        
        self.stats["requests_made"] += 1
        self.stats["failed_requests"] += 1
        
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Request failed after {self.max_retries} attempts")

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics - SAME SIGNATURE, ENHANCED CONTENT"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["requests_made"], 1)
            ),
            "average_processing_time": (
                self.stats["total_processing_time"] / max(self.stats["successful_requests"], 1)
            ),
            # 🚀 OPTIMIZED: Add multilingual stats
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(self.stats["embeddings_generated"] + self.stats["cache_hits"], 1)
            ),
            "multilingual_coverage": len(self.stats["languages_detected"]),
            "models_available": [self.gemini_model, self.gemini_flash_model, self.embedding_model],
            "embedding_dimensions": 768
        }

    def reset_stats(self):
        """Reset statistics - SAME SIGNATURE"""
        self.stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_processing_time": 0.0,
            "embeddings_generated": 0,
            "cache_hits": 0,
            "multilingual_queries": 0,
            "languages_detected": set()
        }
        logger.info("📊 Statistics reset")
