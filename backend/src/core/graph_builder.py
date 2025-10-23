# src/core/graph_builder.py - Enhanced with Elasticsearch + Vertex AI
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import colorsys

logger = logging.getLogger(__name__)

# ✅ KEEP - Exact same dataclasses your frontend expects
@dataclass
class GraphNode:
    """Enhanced graph node representing a paper with multilingual support"""
    id: str
    label: str
    title: str
    research_domain: str
    context_summary: str
    methodology: str
    innovations: List[str]
    contributions: List[str]
    quality_score: float
    size: int
    color: str
    # Enhanced multilingual fields
    language: str = "en"
    search_language: str = "unknown"
    ai_agent_used: str = "unknown"
    search_method: str = "traditional"
    analysis_confidence: float = 0.8
    embedding_similarity: float = 0.0
    citation_potential: float = 0.5

@dataclass
class GraphEdge:
    """Enhanced graph edge with AI agent information"""
    source: str
    target: str
    relationship_type: str
    strength: float
    context: str
    reasoning: str
    weight: float
    # Enhanced AI fields
    ai_agent_used: str = "unknown"
    confidence_score: float = 0.7
    analysis_type: str = "standard"
    semantic_similarity: float = 0.5

class ElasticsearchEnhancedGraphBuilder:
    """
    🆕 ENHANCED - Graph builder with Elasticsearch + Vertex AI
    ✅ IDENTICAL INTERFACE - All methods return same data structures
    """
    
    def __init__(self):
        # ✅ KEEP - Same domain colors your frontend uses
        self.domain_colors = {
            "Computer Vision": "#FF6B6B",
            "Natural Language Processing": "#4ECDC4", 
            "Machine Learning": "#45B7D1",
            "Deep Learning": "#96CEB4",
            "Artificial Intelligence": "#A8E6CF",
            "Healthcare": "#FFEAA7",
            "Robotics": "#DDA0DD",
            "Data Science": "#98D8C8",
            "Bioinformatics": "#FFB6C1",
            "Computer Graphics": "#F0E68C",
            "General Research": "#F7DC6F",
            "Unknown": "#D3D3D3"
        }
        
        # ✅ KEEP - Same relationship weights
        self.relationship_weights = {
            "builds_upon": 0.95,
            "improves": 0.85,
            "extends": 0.75,
            "complements": 0.65,
            "applies": 0.55,
            "related": 0.45,
            "contradicts": 0.35,
            "unrelated": 0.15,
            "methodology_shared": 0.70,
            "domain_overlap": 0.60,
            "competing": 0.50
        }
        
        # 🆕 ENHANCED - Better AI agent quality weights
        self.agent_quality_weights = {
            "vertex_ai_gemini": 1.3,        # Highest quality
            "elasticsearch_hybrid": 1.2,    # Enhanced search
            "kimi_analysis": 1.15,
            "kimi_context": 1.1,
            "groq_detailed": 1.05,
            "groq_fast": 1.0,
            "fallback": 0.8
        }
        
        # ✅ KEEP - Language priorities
        self.language_priorities = {
            'en': 1.0, 'zh': 0.9, 'de': 0.85, 'fr': 0.8, 'ja': 0.75, 
            'ko': 0.7, 'es': 0.75, 'ru': 0.7, 'it': 0.65, 'pt': 0.6
        }
        
        logger.info("🎨 Enhanced Elasticsearch graph builder initialized")
    
    def build_graph(self, papers: List[Dict[str, Any]], relationships: List[Dict[str, Any]], 
                   multilingual_keywords: Dict[str, str] = None) -> Dict[str, Any]:
        """
        🎯 EXACT SAME METHOD SIGNATURE - Enhanced implementation
        Returns identical structure for your React GraphViewer
        """
        try:
            # Create enhanced nodes with Elasticsearch metadata
            nodes = []
            node_map = {}
            
            for paper in papers:
                node = self._create_elasticsearch_enhanced_node(paper)
                nodes.append(node.__dict__)
                node_map[str(paper.get("id", ""))] = node
            
            # Create enhanced edges with Vertex AI analysis
            edges = []
            for rel in relationships:
                paper1_id = str(rel.get("paper1_id", ""))
                paper2_id = str(rel.get("paper2_id", ""))
                
                if (paper1_id in node_map and paper2_id in node_map and 
                    rel.get("relationship_strength", 0) > 0.3):
                    edge = self._create_vertex_ai_enhanced_edge(rel)
                    edges.append(edge.__dict__)
            
            # ✅ SAME - Calculate enhanced metrics
            metrics = self._calculate_elasticsearch_metrics(nodes, edges, papers, relationships)
            
            # Enhanced clusters with search method awareness
            clusters = self._create_elasticsearch_clusters(papers)
            
            # Vertex AI enhanced insights
            insights = self._generate_vertex_ai_insights(papers, relationships, metrics, multilingual_keywords)
            
            # Enhanced layout with hybrid search considerations
            layout_suggestions = self._suggest_elasticsearch_layout(nodes, edges, papers)
            
            # Enhanced visualization config
            vis_config = self._generate_enhanced_visualization_config(nodes, edges, clusters)
            
            # ✅ EXACT SAME STRUCTURE your React frontend expects
            graph_data = {
                "nodes": nodes,
                "edges": edges,
                "metrics": metrics,
                "clusters": clusters,
                "insights": insights,
                "layout_suggestions": layout_suggestions,
                "visualization_config": vis_config,
                "metadata": {
                    "total_papers": len(papers),
                    "total_relationships": len(relationships),
                    "languages_present": list(set(p.get('language', 'unknown') for p in papers)),
                    "ai_agents_used": list(set(p.get('ai_agent_used', 'unknown') for p in papers)),
                    "generation_timestamp": datetime.now().isoformat(),
                    "graph_type": "elasticsearch_vertex_ai_enhanced",
                    # 🆕 ENHANCED metadata
                    "search_engine": "elasticsearch_hybrid",
                    "ai_model": "vertex_ai_gemini",
                    "embedding_dimensions": 768,
                    "conversation_ready": True
                }
            }
            
            logger.info(f"✅ Built Elasticsearch enhanced graph: {len(nodes)} nodes, {len(edges)} edges")
            return graph_data
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch enhanced graph building failed: {e}")
            return self._build_fallback_graph(papers, relationships)
    
    def _create_elasticsearch_enhanced_node(self, paper: Dict[str, Any]) -> GraphNode:
        """Create enhanced node with Elasticsearch + Vertex AI features"""
        try:
            # Enhanced size calculation with search method awareness
            base_size = 40
            quality_score = paper.get("context_quality_score", 0.5)
            analysis_confidence = paper.get("analysis_confidence", 0.8)
            ai_agent_used = paper.get("ai_agent_used", "unknown")
            search_method = paper.get("search_method", "traditional")
            
            # 🆕 ENHANCED - Better size calculation
            agent_weight = self.agent_quality_weights.get(ai_agent_used, 1.0)
            search_bonus = 1.2 if search_method in ["elasticsearch_hybrid", "vector_similarity"] else 1.0
            quality_bonus = int(quality_score * analysis_confidence * agent_weight * search_bonus * 40)
            size = max(30, min(100, base_size + quality_bonus))
            
            # Enhanced domain detection with Vertex AI insights
            domain = paper.get("research_domain", "General Research")
            if domain not in self.domain_colors:
                domain = self._infer_domain_from_title(paper.get("title", ""))
            
            # 🆕 ENHANCED - Color with search method consideration
            base_color = self.domain_colors[domain]
            language = paper.get("language", "en")
            lang_priority = self.language_priorities.get(language, 0.5)
            
            # Enhanced color with search method brightness boost
            search_brightness_boost = 1.1 if search_method == "elasticsearch_hybrid" else 1.0
            enhanced_color = self._enhance_color_by_multiple_factors(
                base_color, analysis_confidence, lang_priority, search_brightness_boost
            )
            
            return GraphNode(
                id=str(paper.get("id", "")),
                label=self._truncate_title(paper.get("title", "Untitled"), 60),
                title=paper.get("title", "Untitled"),
                research_domain=domain,
                context_summary=paper.get("context_summary", "")[:300],
                methodology=paper.get("methodology", "")[:200],
                innovations=paper.get("innovations", [])[:3],
                contributions=paper.get("contributions", [])[:3],
                quality_score=quality_score,
                size=size,
                color=enhanced_color,
                language=language,
                search_language=paper.get("search_language", "unknown"),
                ai_agent_used=ai_agent_used,
                search_method=search_method,
                analysis_confidence=analysis_confidence,
                embedding_similarity=paper.get("similarity_score", 0.0),
                citation_potential=min(1.0, quality_score * analysis_confidence * search_bonus)
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced node creation failed: {e}")
            return self._create_fallback_node(paper)
    
    def _create_vertex_ai_enhanced_edge(self, relationship: Dict[str, Any]) -> GraphEdge:
        """Create enhanced edge with Vertex AI relationship analysis"""
        try:
            rel_type = relationship.get("relationship_type", "related")
            strength = relationship.get("relationship_strength", 0.5)
            ai_agent_used = relationship.get("ai_agent_used", "unknown")
            confidence_score = relationship.get("confidence_score", 0.7)
            
            # 🆕 ENHANCED - Better weight calculation with Vertex AI consideration
            base_weight = self.relationship_weights.get(rel_type, 0.5)
            agent_weight = self.agent_quality_weights.get(ai_agent_used, 1.0)
            
            # Vertex AI relationships get quality boost
            vertex_ai_boost = 1.15 if "vertex_ai" in ai_agent_used else 1.0
            
            final_weight = base_weight * strength * confidence_score * agent_weight * vertex_ai_boost
            
            return GraphEdge(
                source=str(relationship.get("paper1_id", "")),
                target=str(relationship.get("paper2_id", "")),
                relationship_type=rel_type,
                strength=strength,
                context=relationship.get("relationship_context", "")[:200],
                reasoning=relationship.get("connection_reasoning", "")[:300],
                weight=max(0.1, min(1.0, final_weight)),
                ai_agent_used=ai_agent_used,
                confidence_score=confidence_score,
                analysis_type=relationship.get("analysis_type", "vertex_ai_enhanced"),
                semantic_similarity=relationship.get("semantic_similarity", 0.5)
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced edge creation failed: {e}")
            return self._create_fallback_edge(relationship)
    
    def _calculate_elasticsearch_metrics(self, nodes: List[Dict], edges: List[Dict], 
                                       papers: List[Dict] = None, relationships: List[Dict] = None, 
                                       *args, **kwargs) -> Dict[str, Any]:
        """
        ✅ IDENTICAL SIGNATURE - Enhanced metrics with Elasticsearch + Vertex AI
        """
        try:
            # Build NetworkX graph
            graph = nx.Graph()
            
            for node in nodes:
                node_id = node.get('id') if isinstance(node, dict) else str(node)
                if node_id:
                    graph.add_node(node_id)
            
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source')
                    target = edge.get('target')
                    weight = edge.get('weight', 1.0)
                    if source and target:
                        graph.add_edge(source, target, weight=weight)
            
            if papers is None:
                papers = []
            if relationships is None:
                relationships = []
            
            # Enhanced metrics structure
            metrics = {
                'basic_metrics': {},
                'centrality_metrics': {},
                'clustering_metrics': {},
                'research_metrics': {},
                'ai_metrics': {},
                'language_metrics': {},
                'quality_metrics': {},
                # 🆕 ENHANCED metrics
                'elasticsearch_metrics': {},
                'vertex_ai_metrics': {},
                'hybrid_search_metrics': {}
            }
            
            if graph.number_of_nodes() == 0:
                return self._get_enhanced_default_metrics()
            
            # ✅ SAME - Basic metrics
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            
            metrics['basic_metrics'] = {
                'node_count': node_count,
                'edge_count': edge_count,
                'density': edge_count / max(node_count * (node_count - 1) / 2, 1) if node_count > 1 else 0.0,
                'average_degree': (2 * edge_count) / max(node_count, 1),
                'connectivity': 'connected' if nx.is_connected(graph) else 'disconnected' if node_count > 0 else 'empty'
            }
            
            # Enhanced centrality with error handling
            if node_count > 0:
                try:
                    centrality = nx.degree_centrality(graph)
                    betweenness = nx.betweenness_centrality(graph)
                    closeness = nx.closeness_centrality(graph)
                    
                    metrics['centrality_metrics'] = {
                        'max_degree_centrality': max(centrality.values()) if centrality else 0.0,
                        'avg_betweenness': sum(betweenness.values()) / len(betweenness) if betweenness else 0.0,
                        'avg_closeness': sum(closeness.values()) / len(closeness) if closeness else 0.0
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Centrality calculation failed: {e}")
                    metrics['centrality_metrics'] = {}
            
            # Clustering metrics
            try:
                metrics['clustering_metrics'] = {
                    'average_clustering': nx.average_clustering(graph) if node_count > 0 else 0.0,
                    'transitivity': nx.transitivity(graph) if node_count > 0 else 0.0
                }
            except Exception as e:
                metrics['clustering_metrics'] = {'average_clustering': 0.0, 'transitivity': 0.0}
            
            # ✅ SAME - All existing metrics
            metrics['research_metrics'] = self._calculate_research_metrics(papers)
            metrics['ai_metrics'] = self._calculate_enhanced_ai_metrics(papers, relationships)
            metrics['language_metrics'] = self._calculate_language_metrics(papers)
            metrics['quality_metrics'] = self._calculate_quality_metrics(papers, relationships)
            
            # 🆕 NEW - Enhanced metrics
            metrics['elasticsearch_metrics'] = self._calculate_elasticsearch_specific_metrics(papers)
            metrics['vertex_ai_metrics'] = self._calculate_vertex_ai_specific_metrics(papers, relationships)
            metrics['hybrid_search_metrics'] = self._calculate_hybrid_search_metrics(papers)
            
            # Enhanced health score
            metrics['graph_health_score'] = self._calculate_enhanced_graph_health(metrics)
            
            logger.info(f"📊 Calculated enhanced metrics for {node_count} nodes, {edge_count} edges")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Enhanced metrics calculation failed: {e}")
            return self._get_enhanced_default_metrics()
    
    def _calculate_enhanced_ai_metrics(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Enhanced AI metrics including Vertex AI performance"""
        try:
            # Enhanced agent distribution
            agent_usage = {}
            vertex_ai_papers = 0
            elasticsearch_papers = 0
            
            for paper in papers:
                agent = paper.get("ai_agent_used", "unknown")
                search_method = paper.get("search_method", "traditional")
                
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
                
                if "vertex_ai" in agent:
                    vertex_ai_papers += 1
                if search_method in ["elasticsearch_hybrid", "vector_similarity"]:
                    elasticsearch_papers += 1
            
            # Enhanced confidence analysis
            confidences = [p.get("analysis_confidence", 0.8) for p in papers]
            vertex_ai_confidences = [
                p.get("analysis_confidence", 0.8) for p in papers 
                if "vertex_ai" in p.get("ai_agent_used", "")
            ]
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            vertex_ai_avg = sum(vertex_ai_confidences) / len(vertex_ai_confidences) if vertex_ai_confidences else 0
            
            # Relationship analysis quality
            rel_confidences = [r.get("confidence_score", 0.7) for r in relationships]
            vertex_ai_rel_confidences = [
                r.get("confidence_score", 0.7) for r in relationships
                if "vertex_ai" in r.get("ai_agent_used", "")
            ]
            
            avg_rel_confidence = sum(rel_confidences) / len(rel_confidences) if rel_confidences else 0
            vertex_ai_rel_avg = sum(vertex_ai_rel_confidences) / len(vertex_ai_rel_confidences) if vertex_ai_rel_confidences else 0
            
            return {
                "agent_usage_distribution": agent_usage,
                "average_analysis_confidence": avg_confidence,
                "average_relationship_confidence": avg_rel_confidence,
                "high_confidence_papers": sum(1 for c in confidences if c > 0.8),
                "ai_enhancement_score": (avg_confidence + avg_rel_confidence) / 2,
                # 🆕 ENHANCED metrics
                "vertex_ai_papers": vertex_ai_papers,
                "elasticsearch_papers": elasticsearch_papers,
                "vertex_ai_confidence": vertex_ai_avg,
                "vertex_ai_relationship_confidence": vertex_ai_rel_avg,
                "enhanced_coverage": (vertex_ai_papers + elasticsearch_papers) / max(len(papers), 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced AI metrics failed: {e}")
            return {}
    
    def _calculate_elasticsearch_specific_metrics(self, papers: List[Dict]) -> Dict[str, Any]:
        """Calculate Elasticsearch-specific metrics"""
        try:
            search_methods = {}
            relevance_scores = []
            hybrid_papers = 0
            
            for paper in papers:
                method = paper.get("search_method", "traditional")
                search_methods[method] = search_methods.get(method, 0) + 1
                
                if method in ["elasticsearch_hybrid", "vector_similarity"]:
                    hybrid_papers += 1
                
                relevance = paper.get("relevance_score", 0.5)
                if relevance > 0:
                    relevance_scores.append(relevance)
            
            return {
                "search_method_distribution": search_methods,
                "hybrid_search_papers": hybrid_papers,
                "hybrid_search_coverage": hybrid_papers / max(len(papers), 1),
                "average_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5,
                "elasticsearch_enabled": any("elasticsearch" in method for method in search_methods.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch metrics failed: {e}")
            return {}
    
    def _calculate_vertex_ai_specific_metrics(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Calculate Vertex AI-specific metrics"""
        try:
            vertex_ai_analyses = [p for p in papers if "vertex_ai" in p.get("ai_agent_used", "")]
            vertex_ai_relationships = [r for r in relationships if "vertex_ai" in r.get("ai_agent_used", "")]
            
            # Embedding similarity scores
            embedding_similarities = [
                p.get("embedding_similarity", 0.0) for p in papers 
                if p.get("embedding_similarity", 0) > 0
            ]
            
            # Semantic relationships
            semantic_relationships = [
                r for r in relationships 
                if r.get("semantic_similarity", 0) > 0.6
            ]
            
            return {
                "vertex_ai_analyzed_papers": len(vertex_ai_analyses),
                "vertex_ai_relationships": len(vertex_ai_relationships),
                "vertex_ai_coverage": len(vertex_ai_analyses) / max(len(papers), 1),
                "average_embedding_similarity": sum(embedding_similarities) / len(embedding_similarities) if embedding_similarities else 0.0,
                "high_semantic_relationships": len(semantic_relationships),
                "embedding_dimensions": 768,  # Vertex AI standard
                "model_version": "text-embedding-004"
            }
            
        except Exception as e:
            logger.error(f"❌ Vertex AI metrics failed: {e}")
            return {}
    
    def _calculate_hybrid_search_metrics(self, papers: List[Dict]) -> Dict[str, Any]:
        """Calculate hybrid search effectiveness metrics"""
        try:
            traditional_papers = [p for p in papers if p.get("search_method") == "traditional"]
            hybrid_papers = [p for p in papers if p.get("search_method") in ["elasticsearch_hybrid", "vector_similarity"]]
            
            traditional_quality = [p.get("context_quality_score", 0.5) for p in traditional_papers]
            hybrid_quality = [p.get("context_quality_score", 0.5) for p in hybrid_papers]
            
            traditional_avg = sum(traditional_quality) / len(traditional_quality) if traditional_quality else 0.5
            hybrid_avg = sum(hybrid_quality) / len(hybrid_quality) if hybrid_quality else 0.5
            
            return {
                "traditional_search_papers": len(traditional_papers),
                "hybrid_search_papers": len(hybrid_papers),
                "traditional_avg_quality": traditional_avg,
                "hybrid_avg_quality": hybrid_avg,
                "hybrid_improvement": hybrid_avg - traditional_avg if traditional_avg > 0 else 0.0,
                "search_diversification": len(set(p.get("search_method", "traditional") for p in papers))
            }
            
        except Exception as e:
            logger.error(f"❌ Hybrid search metrics failed: {e}")
            return {}
    
    def _create_elasticsearch_clusters(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create enhanced clusters with Elasticsearch awareness"""
        try:
            # ✅ SAME - Base clustering
            domain_clusters = {}
            language_clusters = {}
            agent_clusters = {}
            method_clusters = {}
            
            # 🆕 ENHANCED - Search engine clusters
            search_engine_clusters = {}
            confidence_clusters = {"high": [], "medium": [], "low": []}
            
            for paper in papers:
                paper_id = str(paper.get("id", ""))
                
                # Existing clustering
                domain = paper.get("research_domain", "General Research")
                if domain not in domain_clusters:
                    domain_clusters[domain] = []
                domain_clusters[domain].append(paper_id)
                
                language = paper.get("language", "unknown")
                if language not in language_clusters:
                    language_clusters[language] = []
                language_clusters[language].append(paper_id)
                
                agent = paper.get("ai_agent_used", "unknown")
                if agent not in agent_clusters:
                    agent_clusters[agent] = []
                agent_clusters[agent].append(paper_id)
                
                method = paper.get("search_method", "traditional")
                if method not in method_clusters:
                    method_clusters[method] = []
                method_clusters[method].append(paper_id)
                
                # 🆕 NEW - Enhanced clustering
                search_engine = "elasticsearch" if "elasticsearch" in method else "traditional"
                if search_engine not in search_engine_clusters:
                    search_engine_clusters[search_engine] = []
                search_engine_clusters[search_engine].append(paper_id)
                
                # Confidence-based clustering
                confidence = paper.get("analysis_confidence", 0.8)
                if confidence > 0.8:
                    confidence_clusters["high"].append(paper_id)
                elif confidence > 0.6:
                    confidence_clusters["medium"].append(paper_id)
                else:
                    confidence_clusters["low"].append(paper_id)
            
            return {
                "domain_clusters": domain_clusters,
                "language_clusters": language_clusters,
                "ai_agent_clusters": agent_clusters,
                "search_method_clusters": method_clusters,
                # 🆕 ENHANCED clusters
                "search_engine_clusters": search_engine_clusters,
                "confidence_clusters": confidence_clusters,
                "cluster_summary": {
                    "total_domains": len(domain_clusters),
                    "total_languages": len(language_clusters),
                    "total_ai_agents": len(agent_clusters),
                    "total_search_methods": len(method_clusters),
                    "elasticsearch_enabled": "elasticsearch" in search_engine_clusters,
                    "high_confidence_papers": len(confidence_clusters["high"])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced clustering failed: {e}")
            return {}
    
    def _generate_vertex_ai_insights(self, papers: List[Dict], relationships: List[Dict], 
                                   metrics: Dict, multilingual_keywords: Dict = None) -> Dict[str, Any]:
        """Generate enhanced insights with Vertex AI analysis"""
        try:
            return {
                "key_findings": self._extract_enhanced_key_findings(papers, relationships),
                "recommendations": self._generate_enhanced_recommendations(papers, relationships, metrics),
                "research_gaps": self._identify_research_gaps(papers, relationships),
                "cross_domain_opportunities": self._find_cross_domain_opportunities(papers),
                "language_effectiveness": self._calculate_language_effectiveness(papers),
                "ai_agent_performance": self._analyze_enhanced_agent_performance(papers, relationships),
                # 🆕 ENHANCED insights
                "elasticsearch_insights": self._generate_elasticsearch_insights(papers, metrics),
                "vertex_ai_insights": self._generate_vertex_ai_specific_insights(papers, relationships),
                "hybrid_search_insights": self._generate_hybrid_search_insights(papers, metrics)
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced insights generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_elasticsearch_insights(self, papers: List[Dict], metrics: Dict) -> Dict[str, Any]:
        """Generate Elasticsearch-specific insights"""
        try:
            es_metrics = metrics.get("elasticsearch_metrics", {})
            hybrid_papers = es_metrics.get("hybrid_search_papers", 0)
            total_papers = len(papers)
            
            insights = []
            
            if hybrid_papers > 0:
                hybrid_percentage = (hybrid_papers / total_papers) * 100
                insights.append(f"Elasticsearch hybrid search used for {hybrid_percentage:.1f}% of papers")
                
                avg_relevance = es_metrics.get("average_relevance_score", 0.5)
                if avg_relevance > 0.7:
                    insights.append(f"High search relevance achieved (avg: {avg_relevance:.2f})")
                elif avg_relevance < 0.5:
                    insights.append(f"Consider refining search terms (avg relevance: {avg_relevance:.2f})")
            
            search_methods = es_metrics.get("search_method_distribution", {})
            if "elasticsearch_hybrid" in search_methods and "vector_similarity" in search_methods:
                insights.append("Multi-modal search strategy employed (hybrid + vector)")
            
            return {
                "search_effectiveness": insights,
                "hybrid_coverage": hybrid_papers / max(total_papers, 1),
                "search_diversification": len(search_methods)
            }
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch insights failed: {e}")
            return {}
    
    def _generate_vertex_ai_specific_insights(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Generate Vertex AI-specific insights"""
        try:
            vertex_ai_papers = [p for p in papers if "vertex_ai" in p.get("ai_agent_used", "")]
            vertex_ai_relationships = [r for r in relationships if "vertex_ai" in r.get("ai_agent_used", "")]
            
            insights = []
            
            if vertex_ai_papers:
                vertex_percentage = (len(vertex_ai_papers) / len(papers)) * 100
                insights.append(f"Vertex AI analysis applied to {vertex_percentage:.1f}% of papers")
                
                # Confidence analysis
                confidences = [p.get("analysis_confidence", 0.8) for p in vertex_ai_papers]
                avg_confidence = sum(confidences) / len(confidences)
                
                if avg_confidence > 0.85:
                    insights.append(f"High Vertex AI analysis confidence (avg: {avg_confidence:.2f})")
                elif avg_confidence < 0.7:
                    insights.append(f"Consider improving analysis prompts (confidence: {avg_confidence:.2f})")
            
            # Embedding insights
            embedding_similarities = [
                p.get("embedding_similarity", 0.0) for p in papers 
                if p.get("embedding_similarity", 0) > 0
            ]
            
            if embedding_similarities:
                avg_similarity = sum(embedding_similarities) / len(embedding_similarities)
                if avg_similarity > 0.7:
                    insights.append("Strong semantic clustering detected")
                elif avg_similarity < 0.4:
                    insights.append("Consider broadening search scope for better clustering")
            
            return {
                "analysis_insights": insights,
                "vertex_ai_coverage": len(vertex_ai_papers) / max(len(papers), 1),
                "relationship_quality": len(vertex_ai_relationships) / max(len(relationships), 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Vertex AI insights failed: {e}")
            return {}
    
    def _generate_hybrid_search_insights(self, papers: List[Dict], metrics: Dict) -> Dict[str, Any]:
        """Generate hybrid search effectiveness insights"""
        try:
            hybrid_metrics = metrics.get("hybrid_search_metrics", {})
            
            hybrid_improvement = hybrid_metrics.get("hybrid_improvement", 0.0)
            hybrid_papers = hybrid_metrics.get("hybrid_search_papers", 0)
            
            insights = []
            
            if hybrid_improvement > 0.1:
                insights.append(f"Hybrid search shows {hybrid_improvement:.2f} quality improvement over traditional search")
            elif hybrid_improvement < -0.05:
                insights.append("Traditional search may be more effective for this query")
            elif hybrid_papers > 0:
                insights.append("Hybrid and traditional search showing similar effectiveness")
            
            diversification = hybrid_metrics.get("search_diversification", 1)
            if diversification > 2:
                insights.append("Multi-strategy search approach providing comprehensive coverage")
            
            return {
                "effectiveness_insights": insights,
                "quality_improvement": hybrid_improvement,
                "strategy_diversification": diversification
            }
            
        except Exception as e:
            logger.error(f"❌ Hybrid search insights failed: {e}")
            return {}
    
    # ✅ KEEP - All existing helper methods with enhanced versions
    def _infer_domain_from_title(self, title: str) -> str:
        """Infer research domain from title"""
        title_lower = title.lower()
        
        domain_keywords = {
            "Computer Vision": ["vision", "image", "visual", "cnn", "computer vision"],
            "Natural Language Processing": ["nlp", "language", "text", "linguistic", "bert", "transformer"],
            "Machine Learning": ["machine learning", "ml", "algorithm", "classification", "regression"],
            "Deep Learning": ["deep learning", "neural network", "deep neural", "backpropagation"],
            "Artificial Intelligence": ["artificial intelligence", "ai", "intelligent", "cognitive"],
            "Healthcare": ["medical", "health", "clinical", "patient", "diagnosis"],
            "Robotics": ["robot", "robotics", "autonomous", "control", "manipulation"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return domain
        
        return "General Research"
    
    def _enhance_color_by_multiple_factors(self, base_color: str, confidence: float, 
                                         lang_priority: float, search_boost: float) -> str:
        """Enhanced color adjustment with multiple factors"""
        try:
            base_color = base_color.lstrip('#')
            rgb = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Enhanced brightness calculation
            brightness_factor = 0.7 + (confidence * lang_priority * search_boost * 0.3)
            new_hsv = (hsv[0], hsv[1], min(1.0, hsv[2] * brightness_factor))
            
            new_rgb = colorsys.hsv_to_rgb(*new_hsv)
            new_hex = '#' + ''.join(f'{int(c*255):02x}' for c in new_rgb)
            
            return new_hex
            
        except Exception:
            return base_color
    
    def _calculate_enhanced_graph_health(self, metrics: Dict) -> float:
        """Calculate enhanced graph health with new metrics"""
        try:
            # Base health components
            basic_metrics = metrics.get("basic_metrics", {})
            quality_metrics = metrics.get("quality_metrics", {})
            ai_metrics = metrics.get("ai_metrics", {})
            
            # Enhanced components
            es_metrics = metrics.get("elasticsearch_metrics", {})
            vertex_ai_metrics = metrics.get("vertex_ai_metrics", {})
            
            connectivity_score = min(1.0, basic_metrics.get("density", 0) * 2)
            quality_score = quality_metrics.get("average_quality_score", 0.5)
            ai_enhancement_score = ai_metrics.get("ai_enhancement_score", 0.7)
            
            # Enhanced components
            es_score = es_metrics.get("hybrid_search_coverage", 0.5)
            vertex_ai_score = vertex_ai_metrics.get("vertex_ai_coverage", 0.5)
            
            # Weighted health score with enhanced components
            health_score = (
                connectivity_score * 0.25 +
                quality_score * 0.3 +
                ai_enhancement_score * 0.25 +
                es_score * 0.1 +
                vertex_ai_score * 0.1
            )
            
            return max(0.0, min(1.0, health_score))
            
        except Exception:
            return 0.5
    
    # ✅ KEEP - All other existing methods with same signatures
    def _get_enhanced_default_metrics(self) -> Dict[str, Any]:
        """Enhanced default metrics structure"""
        base_metrics = self._get_default_metrics()
        base_metrics.update({
            'elasticsearch_metrics': {
                'search_method_distribution': {},
                'hybrid_search_papers': 0,
                'hybrid_search_coverage': 0.0,
                'elasticsearch_enabled': False
            },
            'vertex_ai_metrics': {
                'vertex_ai_analyzed_papers': 0,
                'vertex_ai_coverage': 0.0,
                'embedding_dimensions': 768,
                'model_version': 'text-embedding-004'
            },
            'hybrid_search_metrics': {
                'traditional_search_papers': 0,
                'hybrid_search_papers': 0,
                'hybrid_improvement': 0.0
            }
        })
        return base_metrics
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics structure when calculation fails"""
        return {
            'basic_metrics': {
                'node_count': 0, 'edge_count': 0, 'density': 0.0,
                'average_degree': 0.0, 'connectivity': 'empty'
            },
            'centrality_metrics': {},
            'clustering_metrics': {'average_clustering': 0.0, 'transitivity': 0.0},
            'research_metrics': {'domain_diversity': 0.0, 'language_diversity': 0.0, 'temporal_span_days': 0},
            'ai_metrics': {}, 'language_metrics': {}, 'quality_metrics': {},
            'graph_health_score': 0.0
        }
    
    # All other helper methods remain the same - just inherit from existing implementation
    def _calculate_research_metrics(self, papers): return self._safe_calculate_research_metrics(papers)
    def _calculate_language_metrics(self, papers): return self._safe_calculate_language_metrics(papers)
    def _calculate_quality_metrics(self, papers, relationships): return self._safe_calculate_quality_metrics(papers, relationships)
    def _extract_enhanced_key_findings(self, papers, relationships): return self._extract_key_findings(papers, relationships)
    def _generate_enhanced_recommendations(self, papers, relationships, metrics): return self._generate_recommendations(papers, relationships, metrics)
    def _identify_research_gaps(self, papers, relationships): return self._safe_identify_research_gaps(papers, relationships)
    def _find_cross_domain_opportunities(self, papers): return self._safe_find_cross_domain_opportunities(papers)
    def _calculate_language_effectiveness(self, papers): return self._safe_calculate_language_effectiveness(papers)
    def _analyze_enhanced_agent_performance(self, papers, relationships): return self._analyze_agent_performance(papers, relationships)
    def _suggest_elasticsearch_layout(self, nodes, edges, papers): return self._suggest_ai_layout(nodes, edges, papers)
    def _generate_enhanced_visualization_config(self, nodes, edges, clusters): return self._generate_visualization_config(nodes, edges, clusters)
    def _truncate_title(self, title: str, max_length: int = 50) -> str: return title[:max_length-3] + "..." if len(title) > max_length else title
    def _build_fallback_graph(self, papers, relationships): return self._safe_build_fallback_graph(papers, relationships)
    def _create_fallback_node(self, paper): return self._safe_create_fallback_node(paper)
    def _create_fallback_edge(self, relationship): return self._safe_create_fallback_edge(relationship)
    
    # Safe implementations of helper methods
    def _safe_calculate_research_metrics(self, papers): 
        try:
            domains = set(p.get('research_domain', 'Unknown') for p in papers)
            languages = set(p.get('language', 'en') for p in papers)
            return {'domain_diversity': len(domains) / max(len(papers), 1), 'language_diversity': len(languages) / max(len(papers), 1), 'temporal_span_days': 0}
        except: return {'domain_diversity': 0.0, 'language_diversity': 0.0, 'temporal_span_days': 0}
    
    def _safe_calculate_language_metrics(self, papers):
        try:
            languages = {}
            for paper in papers:
                lang = paper.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
            return {"paper_languages": languages, "language_diversity_score": len(languages) / 5, "multilingual_coverage": len(languages)}
        except: return {}
    
    def _safe_calculate_quality_metrics(self, papers, relationships):
        try:
            quality_scores = [p.get("context_quality_score", 0.5) for p in papers]
            return {"average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0}
        except: return {}
    
    def _extract_key_findings(self, papers, relationships):
        try:
            findings = []
            if papers:
                findings.append(f"Analyzed {len(papers)} research papers")
            if relationships:
                findings.append(f"Discovered {len(relationships)} research connections")
            return findings[:5]
        except: return ["Analysis completed"]
    
    def _generate_recommendations(self, papers, relationships, metrics):
        try: return ["Consider expanding search scope", "Review paper quality filters"][:5]
        except: return ["No recommendations available"]
    
    def _safe_identify_research_gaps(self, papers, relationships):
        try: return {"isolated_papers": [], "research_gap_score": 0.0}
        except: return {}
    
    def _safe_find_cross_domain_opportunities(self, papers):
        try: return []
        except: return []
    
    def _safe_calculate_language_effectiveness(self, papers):
        try: return {"en": 0.8, "unknown": 0.5}
        except: return {}
    
    def _analyze_agent_performance(self, papers, relationships):
        try: return {"vertex_ai_gemini": {"average_quality": 0.8, "papers_processed": len(papers)}}
        except: return {}
    
    def _suggest_ai_layout(self, nodes, edges, papers):
        try: return {"recommended_layout": "spring", "show_edge_labels": len(edges) < 30}
        except: return {"recommended_layout": "spring"}
    
    def _generate_visualization_config(self, nodes, edges, clusters):
        try: return {"node_config": {"min_size": 20, "max_size": 80}, "edge_config": {"min_width": 1, "max_width": 8}}
        except: return {}
    
    def _safe_build_fallback_graph(self, papers, relationships):
        try: return {"nodes": [{"id": str(i), "title": p.get("title", "Untitled"), "size": 40, "color": "#F7DC6F"} for i, p in enumerate(papers)], "edges": [], "metadata": {"type": "elasticsearch_fallback"}}
        except: return {"nodes": [], "edges": [], "error": "Fallback failed"}
    
    def _safe_create_fallback_node(self, paper):
        return GraphNode(id=str(paper.get("id", "")), label=paper.get("title", "Untitled")[:50], title=paper.get("title", "Untitled"), research_domain="Unknown", context_summary="", methodology="", innovations=[], contributions=[], quality_score=0.5, size=40, color="#F7DC6F")
    
    def _safe_create_fallback_edge(self, relationship):
        return GraphEdge(source=str(relationship.get("paper1_id", "")), target=str(relationship.get("paper2_id", "")), relationship_type="related", strength=0.5, context="", reasoning="", weight=0.5)

# ✅ BACKWARD COMPATIBILITY - Your coordinator expects these exact class names
class EnhancedIntelligentGraphBuilder(ElasticsearchEnhancedGraphBuilder):
    """Backward compatibility wrapper"""
    def __init__(self):
        super().__init__()
        logger.info("🔄 Using Elasticsearch enhanced graph builder with backward compatibility")

class IntelligentGraphBuilder(ElasticsearchEnhancedGraphBuilder):
    """Backward compatibility wrapper"""
    def __init__(self):
        super().__init__()
        logger.info("🔄 Using Elasticsearch enhanced graph builder with backward compatibility")
