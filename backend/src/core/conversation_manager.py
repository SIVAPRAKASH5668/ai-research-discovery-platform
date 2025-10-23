# src/core/conversation_manager.py - Conversational research features
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import uuid
import json

from database.elastic_client import ElasticClient
from core.vertex_ai_processor import VertexAIProcessor
from core.hybrid_search_engine import HybridSearchEngine

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    id: str
    session_id: str
    user_message: str
    agent_response: str
    papers_found: List[Dict]
    follow_ups: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

class ConversationManager:
    """
    💬 Advanced conversational research agent manager
    Handles multi-turn conversations about research topics with context awareness
    """
    
    def __init__(self, elastic_client: ElasticClient, vertex_ai: VertexAIProcessor):
        self.elastic_client = elastic_client
        self.vertex_ai = vertex_ai
        self.hybrid_search = HybridSearchEngine(elastic_client, vertex_ai)
        
        # Conversation memory (in production, use Redis or similar)
        self.conversation_memory = {}
        self.research_context = {}
        
        # Conversation configuration
        self.max_context_turns = 10
        self.max_papers_per_response = 10
        self.follow_up_count = 3
        
        # Response templates
        self.response_templates = {
            "greeting": "Hello! I'm here to help you discover and explore research papers. What would you like to research today?",
            "no_results": "I couldn't find papers matching your query. Could you try rephrasing or providing more specific terms?",
            "clarification": "I'd like to help you better. Could you provide more details about what aspect interests you most?",
            "continuation": "Based on our discussion, I found some relevant papers. Would you like me to explore any specific aspect further?"
        }
        
        logger.info("💬 Conversation manager initialized")

    async def process_turn(self, user_message: str, session_id: str, 
                          language: str = "en", context: Dict = None) -> Dict[str, Any]:
        """
        Process a single conversation turn with context awareness
        """
        turn_id = str(uuid.uuid4())
        
        try:
            logger.info(f"💬 Processing turn {turn_id} for session {session_id}")
            
            # Get conversation history and context
            conversation_history = self.conversation_memory.get(session_id, [])
            research_context = self.research_context.get(session_id, {})
            
            # Analyze user intent with conversation context
            intent_analysis = await self._analyze_conversational_intent(
                user_message, conversation_history, research_context
            )
            
            # Process based on intent
            if intent_analysis["intent"] == "greeting":
                return await self._handle_greeting(session_id, user_message)
            elif intent_analysis["intent"] == "clarification_needed":
                return await self._handle_clarification_request(session_id, user_message, intent_analysis)
            elif intent_analysis["intent"] == "research_query":
                return await self._handle_research_query(
                    session_id, user_message, intent_analysis, language, context
                )
            elif intent_analysis["intent"] == "follow_up":
                return await self._handle_follow_up_query(
                    session_id, user_message, intent_analysis, research_context
                )
            else:
                # Default research handling
                return await self._handle_research_query(
                    session_id, user_message, intent_analysis, language, context
                )
                
        except Exception as e:
            logger.error(f"❌ Conversation turn processing failed: {e}")
            return self._create_error_response(session_id, str(e))

    async def _analyze_conversational_intent(self, message: str, 
                                           conversation_history: List[Dict], 
                                           research_context: Dict) -> Dict[str, Any]:
        """
        Analyze user intent considering conversation context
        """
        try:
            # Build context for intent analysis
            context_summary = self._build_context_summary(conversation_history, research_context)
            
            intent_prompt = f"""
            Analyze the user's research intent in this conversation context:
            
            Current message: "{message}"
            
            Conversation context: {context_summary}
            
            Determine the intent and provide analysis:
            1. Intent type: greeting, research_query, follow_up, clarification_needed, or refinement
            2. Research domain (if applicable)
            3. Specific aspects mentioned
            4. Confidence level (0-1)
            5. Suggested search strategy
            
            Response format: JSON with intent, domain, aspects, confidence, strategy
            """
            
            analysis_response = await self.vertex_ai.generate_text(
                intent_prompt, max_tokens=300, temperature=0.1
            )
            
            # Parse response
            try:
                parsed_intent = json.loads(analysis_response)
                return {
                    "intent": parsed_intent.get("intent", "research_query"),
                    "domain": parsed_intent.get("domain", "general"),
                    "aspects": parsed_intent.get("aspects", []),
                    "confidence": parsed_intent.get("confidence", 0.7),
                    "strategy": parsed_intent.get("strategy", "adaptive"),
                    "raw_analysis": analysis_response
                }
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_intent_fallback(message, analysis_response)
                
        except Exception as e:
            logger.warning(f"⚠️ Intent analysis failed: {e}")
            return {
                "intent": "research_query",
                "domain": "general",
                "aspects": [],
                "confidence": 0.5,
                "strategy": "adaptive"
            }

    def _build_context_summary(self, conversation_history: List[Dict], 
                              research_context: Dict) -> str:
        """
        Build a concise summary of conversation context
        """
        try:
            if not conversation_history:
                return "New conversation, no previous context."
            
            # Get recent turns
            recent_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            
            summary_parts = []
            
            # Add recent topics
            if research_context.get("recent_topics"):
                topics = ", ".join(research_context["recent_topics"][:3])
                summary_parts.append(f"Recent topics: {topics}")
            
            # Add recent papers count
            if research_context.get("papers_discovered", 0) > 0:
                summary_parts.append(f"Papers found: {research_context['papers_discovered']}")
            
            # Add last user message
            if recent_turns:
                last_message = recent_turns[-1].get("user_message", "")[:100]
                summary_parts.append(f"Previous query: {last_message}")
            
            return ". ".join(summary_parts) if summary_parts else "Ongoing research conversation."
            
        except Exception as e:
            logger.warning(f"⚠️ Context summary failed: {e}")
            return "Context unavailable."

    async def _handle_greeting(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Handle greeting/introduction messages
        """
        response = self.response_templates["greeting"]
        
        # Provide some popular research suggestions
        follow_ups = [
            "Find papers on machine learning applications",
            "Search for recent AI research",
            "Explore natural language processing papers",
            "Discover computer vision research"
        ]
        
        return {
            "response": response,
            "papers": [],
            "follow_ups": follow_ups,
            "metadata": {
                "turn_type": "greeting",
                "suggestions_provided": True
            }
        }

    async def _handle_research_query(self, session_id: str, user_message: str, 
                                   intent_analysis: Dict, language: str, 
                                   context: Dict) -> Dict[str, Any]:
        """
        Handle research query with conversational context
        """
        try:
            # Perform enhanced search
            papers = await self.hybrid_search.search(
                query=user_message,
                language=language,
                context={
                    **context,
                    "domain": intent_analysis.get("domain"),
                    "strategy": intent_analysis.get("strategy", "adaptive")
                },
                limit=self.max_papers_per_response
            )
            
            if not papers:
                return {
                    "response": self.response_templates["no_results"],
                    "papers": [],
                    "follow_ups": await self._generate_no_results_suggestions(user_message),
                    "metadata": {"turn_type": "no_results", "query": user_message}
                }
            
            # Generate conversational response about the papers
            research_response = await self._generate_research_response(
                user_message, papers, intent_analysis
            )
            
            # Generate contextual follow-ups
            follow_ups = await self._generate_contextual_follow_ups(
                papers, intent_analysis, user_message
            )
            
            # Update research context
            self._update_research_context(session_id, user_message, papers, intent_analysis)
            
            return {
                "response": research_response,
                "papers": papers,
                "follow_ups": follow_ups,
                "metadata": {
                    "turn_type": "research_query",
                    "papers_found": len(papers),
                    "domain": intent_analysis.get("domain"),
                    "search_strategy": intent_analysis.get("strategy")
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Research query handling failed: {e}")
            return self._create_error_response(session_id, str(e))

    async def _generate_research_response(self, query: str, papers: List[Dict], 
                                        intent_analysis: Dict) -> str:
        """
        Generate a conversational response about the research findings
        """
        try:
            # Build context about the papers found
            papers_summary = self._summarize_papers_for_response(papers)
            
            response_prompt = f"""
            Generate a conversational response about research findings:
            
            User asked: "{query}"
            
            Papers found: {len(papers)} papers
            Summary: {papers_summary}
            
            Create a natural, helpful response that:
            1. Acknowledges the user's question
            2. Summarizes what was found
            3. Highlights key insights
            4. Invites further exploration
            
            Keep it conversational and under 200 words.
            """
            
            response = await self.vertex_ai.generate_text(
                response_prompt, max_tokens=250, temperature=0.2
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Response generation failed: {e}")
            return f"I found {len(papers)} papers related to your question about {query}. The research covers various aspects of this topic. Would you like me to explore any specific area in more detail?"

    def _summarize_papers_for_response(self, papers: List[Dict]) -> str:
        """
        Create a brief summary of papers for response generation
        """
        try:
            if not papers:
                return "No papers found"
            
            # Extract key information
            domains = list(set(p.get("domain", "General") for p in papers))
            top_domains = domains[:3]
            
            recent_count = sum(1 for p in papers 
                             if p.get("publication_date", "").startswith(("2023", "2024", "2025")))
            
            summary_parts = []
            summary_parts.append(f"{len(papers)} papers")
            
            if top_domains:
                summary_parts.append(f"domains: {', '.join(top_domains)}")
            
            if recent_count > 0:
                summary_parts.append(f"{recent_count} recent papers")
            
            return ", ".join(summary_parts)
            
        except Exception as e:
            return f"{len(papers)} papers found"

    async def _generate_contextual_follow_ups(self, papers: List[Dict], 
                                            intent_analysis: Dict, 
                                            query: str) -> List[str]:
        """
        Generate contextual follow-up suggestions
        """
        try:
            follow_ups = []
            
            # Based on papers found
            if papers:
                domains = list(set(p.get("domain", "") for p in papers))
                authors = list(set(p.get("authors", "").split(",")[0].strip() for p in papers))
                
                # Domain-based follow-ups
                if len(domains) > 1:
                    follow_ups.append(f"Compare approaches across {domains[0]} and {domains[1]}")
                
                # Author-based follow-ups
                if authors and authors[0]:
                    follow_ups.append(f"Find more papers by {authors[0]}")
                
                # Methodology follow-ups
                follow_ups.append("Explain the methodology from the top paper")
                follow_ups.append("Find recent developments in this area")
                follow_ups.append("What are the main challenges in this field?")
            
            # Intent-based follow-ups
            if intent_analysis.get("domain") and intent_analysis["domain"] != "general":
                follow_ups.append(f"Explore related topics in {intent_analysis['domain']}")
            
            return follow_ups[:self.follow_up_count]
            
        except Exception as e:
            logger.warning(f"⚠️ Follow-up generation failed: {e}")
            return [
                "Tell me more about the key findings",
                "Find similar recent research",
                "What are the practical applications?"
            ]

    async def _generate_no_results_suggestions(self, query: str) -> List[str]:
        """
        Generate suggestions when no results are found
        """
        return [
            f"Try broader terms related to '{query}'",
            "Search for general concepts first",
            "Check spelling and try alternative terms",
            "Browse popular research topics"
        ]

    def _update_research_context(self, session_id: str, query: str, 
                               papers: List[Dict], intent_analysis: Dict):
        """
        Update research context for the session
        """
        try:
            if session_id not in self.research_context:
                self.research_context[session_id] = {}
            
            context = self.research_context[session_id]
            
            # Update recent topics
            current_topics = context.get("recent_topics", [])
            if intent_analysis.get("domain") and intent_analysis["domain"] != "general":
                current_topics.insert(0, intent_analysis["domain"])
            
            context.update({
                "recent_topics": list(dict.fromkeys(current_topics))[:5],  # Remove duplicates, keep 5
                "papers_discovered": context.get("papers_discovered", 0) + len(papers),
                "last_domain": intent_analysis.get("domain", "general"),
                "last_query": query,
                "conversation_turns": context.get("conversation_turns", 0) + 1,
                "updated_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Context update failed: {e}")

    async def _handle_follow_up_query(self, session_id: str, message: str, 
                                    intent_analysis: Dict, research_context: Dict) -> Dict[str, Any]:
        """
        Handle follow-up queries using conversation context
        """
        try:
            # Use previous context to enhance the current query
            enhanced_query = f"{message} {research_context.get('last_domain', '')}"
            
            return await self._handle_research_query(
                session_id, enhanced_query, intent_analysis, "en", research_context
            )
            
        except Exception as e:
            logger.error(f"❌ Follow-up handling failed: {e}")
            return self._create_error_response(session_id, str(e))

    async def _handle_clarification_request(self, session_id: str, message: str, 
                                          intent_analysis: Dict) -> Dict[str, Any]:
        """
        Handle requests for clarification
        """
        response = self.response_templates["clarification"]
        
        follow_ups = [
            "What specific aspect interests you most?",
            "Are you looking for recent research?",
            "Do you need papers from a specific field?",
            "Would you like methodology or application papers?"
        ]
        
        return {
            "response": response,
            "papers": [],
            "follow_ups": follow_ups,
            "metadata": {"turn_type": "clarification", "original_message": message}
        }

    def _parse_intent_fallback(self, message: str, analysis: str) -> Dict[str, Any]:
        """
        Fallback intent parsing when JSON parsing fails
        """
        message_lower = message.lower()
        
        # Simple heuristics
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "start"]):
            intent = "greeting"
        elif any(question in message_lower for question in ["what", "how", "why", "when", "where"]):
            intent = "research_query"
        else:
            intent = "research_query"
        
        return {
            "intent": intent,
            "domain": "general",
            "aspects": [],
            "confidence": 0.6,
            "strategy": "adaptive"
        }

    def _create_error_response(self, session_id: str, error: str) -> Dict[str, Any]:
        """
        Create error response for conversation failures
        """
        return {
            "response": "I apologize, but I encountered an issue processing your request. Could you please try rephrasing your question?",
            "papers": [],
            "follow_ups": [
                "Try a different phrasing",
                "Ask about a specific research topic",
                "Start with a general research area"
            ],
            "metadata": {
                "turn_type": "error",
                "error": error,
                "session_id": session_id
            }
        }

    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session
        """
        return self.conversation_memory.get(session_id, [])

    async def clear_session(self, session_id: str):
        """
        Clear conversation memory for a session
        """
        if session_id in self.conversation_memory:
            del self.conversation_memory[session_id]
        if session_id in self.research_context:
            del self.research_context[session_id]
        
        logger.info(f"🧹 Cleared conversation session: {session_id}")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics
        """
        return {
            "active_sessions": len(self.conversation_memory),
            "total_contexts": len(self.research_context),
            "average_turns_per_session": (
                sum(len(history) for history in self.conversation_memory.values()) / 
                max(len(self.conversation_memory), 1)
            )
        }
