/**
 * ==================================================================================
 * useResearchData Hook - ULTIMATE DEBUG VERSION WITH FULL RAW DATA LOGGING
 * ==================================================================================
 * ✅ Crossref data properly handled
 * ✅ Similar papers preserve original source
 * ✅ All API sources supported (arXiv, CrossRef, PubMed, EuropePMC, Semantic Scholar)
 * ✅ COMPREHENSIVE RAW DATA LOGGING - See everything from backend!
 */

import { useState, useCallback } from 'react'
import { chatAPI, researchAPI } from '../utils/api'

// ============================================================================
// 🔬 MEGA DEBUG LOGGER - Shows ALL raw data from backend
// ============================================================================
const logRawPaperData = (papers, context = 'Backend Response') => {
  if (!papers || papers.length === 0) {
    console.warn('⚠️ No papers to log')
    return
  }
  
  console.log('\n' + '='.repeat(120))
  console.log(`🔬 RAW PAPER DATA FROM ${context.toUpperCase()} - COMPREHENSIVE BREAKDOWN`)
  console.log('='.repeat(120))
  console.log(`Total Papers: ${papers.length}`)
  
  // Group by source
  const papersBySource = {}
  papers.forEach(p => {
    const source = (p.source || p.api_source || 'unknown').toLowerCase()
    if (!papersBySource[source]) {
      papersBySource[source] = []
    }
    papersBySource[source].push(p)
  })
  
  console.log(`Sources Found: ${Object.keys(papersBySource).join(', ')}`)
  
  // Log each source type with FULL sample
  Object.entries(papersBySource).forEach(([source, sourcePapers]) => {
    console.log('\n' + '-'.repeat(120))
    console.log(`📊 SOURCE: ${source.toUpperCase()} (${sourcePapers.length} papers)`)
    console.log('-'.repeat(120))
    
    // Show first paper in FULL DETAIL
    const sample = sourcePapers[0]
    console.log('\n📄 COMPLETE FIRST PAPER RAW DATA:')
    console.log(JSON.stringify(sample, null, 2))
    
    // Field analysis
    console.log('\n📋 FIELD INVENTORY (what fields exist):')
    const fields = Object.keys(sample)
    fields.forEach(field => {
      const value = sample[field]
      const type = Array.isArray(value) ? 'array' : typeof value
      const preview = JSON.stringify(value)?.substring(0, 100) || ''
      console.log(`  • ${field} (${type}): ${preview}${JSON.stringify(value)?.length > 100 ? '...' : ''}`)
    })
    
    // Specific field analysis
    console.log('\n🔎 KEY FIELDS DETAILED ANALYSIS:')
    
    // AUTHORS
    console.log('\n  AUTHORS:')
    console.log(`    Field name: ${sample.authors !== undefined ? 'authors' : sample.author !== undefined ? 'author' : 'MISSING'}`)
    console.log(`    Raw type: ${Array.isArray(sample.authors || sample.author) ? 'array' : typeof (sample.authors || sample.author)}`)
    console.log(`    Raw value:`, sample.authors || sample.author)
    if (Array.isArray(sample.authors)) {
      console.log(`    Array length: ${sample.authors.length}`)
      console.log(`    First author:`, sample.authors[0])
    } else if (Array.isArray(sample.author)) {
      console.log(`    Array length: ${sample.author.length}`)
      console.log(`    First author:`, sample.author[0])
    }
    
    // YEAR/DATE
    console.log('\n  DATE/YEAR FIELDS:')
    console.log(`    year: ${sample.year}`)
    console.log(`    publication_year: ${sample.publication_year}`)
    console.log(`    published_date: ${sample.published_date}`)
    console.log(`    publication_date: ${sample.publication_date}`)
    if (sample['published-print']) {
      console.log(`    published-print:`, sample['published-print'])
      if (sample['published-print']['date-parts']) {
        console.log(`    published-print.date-parts:`, sample['published-print']['date-parts'])
      }
    }
    
    // CITATIONS
    console.log('\n  CITATION FIELDS:')
    console.log(`    citation_count: ${sample.citation_count}`)
    console.log(`    citations: ${sample.citations}`)
    console.log(`    citationCount: ${sample.citationCount}`)
    if (sample['is-referenced-by-count'] !== undefined) {
      console.log(`    is-referenced-by-count: ${sample['is-referenced-by-count']} ← CROSSREF SPECIFIC`)
    }
    
    // TITLE
    console.log('\n  TITLE:')
    console.log(`    Type: ${Array.isArray(sample.title) ? 'array' : typeof sample.title}`)
    if (Array.isArray(sample.title)) {
      console.log(`    Value (array[0]): ${sample.title[0]}`)
      console.log(`    Full array:`, sample.title)
    } else {
      console.log(`    Value: ${sample.title}`)
    }
    
    // ABSTRACT
    console.log('\n  ABSTRACT:')
    console.log(`    Has abstract field: ${sample.abstract !== undefined}`)
    console.log(`    abstract length: ${sample.abstract?.length || 0}`)
    console.log(`    Has HTML tags: ${/<[^>]+>/.test(sample.abstract)}`)
    console.log(`    First 200 chars: ${sample.abstract?.substring(0, 200)}...`)
    if (sample.abstractText) {
      console.log(`    Has abstractText field: ${sample.abstractText?.length || 0} chars`)
    }
    if (sample.summary) {
      console.log(`    Has summary field: ${sample.summary?.length || 0} chars`)
    }
    
    // DOI/URL
    console.log('\n  IDENTIFIERS:')
    console.log(`    doi: ${sample.doi}`)
    console.log(`    DOI: ${sample.DOI}`)
    console.log(`    url: ${sample.url}`)
    console.log(`    source_url: ${sample.source_url}`)
    
    // METADATA
    console.log('\n  METADATA FIELDS:')
    console.log(`    id: ${sample.id}`)
    console.log(`    source: ${sample.source}`)
    console.log(`    api_source: ${sample.api_source}`)
    console.log(`    research_domain: ${sample.research_domain}`)
    console.log(`    language: ${sample.language}`)
    console.log(`    paper_type: ${sample.paper_type}`)
    console.log(`    type: ${sample.type}`)
    console.log(`    domain: ${sample.domain}`)
    
    // TRANSLATION
    if (sample.translated_to || sample.translation_engine) {
      console.log('\n  TRANSLATION INFO:')
      console.log(`    translated_to: ${sample.translated_to}`)
      console.log(`    translation_engine: ${sample.translation_engine}`)
      console.log(`    original_title: ${sample.original_title}`)
      console.log(`    original_abstract length: ${sample.original_abstract?.length || 0}`)
    }
    
    // SIMILARITY
    if (sample.similarity_score) {
      console.log('\n  SIMILARITY:')
      console.log(`    similarity_score: ${sample.similarity_score}`)
    }
  })
  
  console.log('\n' + '='.repeat(120))
  console.log('END OF RAW DATA ANALYSIS')
  console.log('='.repeat(120) + '\n')
}

export const useResearchData = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [nodeDetails, setNodeDetails] = useState(null)
  const [loadingDetails, setLoadingDetails] = useState(false)
  const [statistics, setStatistics] = useState(null)

  // ========================================================================
  // TRANSFORM PAPERS → GRAPH FORMAT (ALL API SOURCES SUPPORTED)
  // ========================================================================
  const transformPapersToGraph = useCallback((papers) => {
    if (!papers || papers.length === 0) {
      console.warn('⚠️ No papers to transform')
      return { 
        nodes: [], 
        links: [],
        _stableVersion: Date.now(),
        _originalTimestamp: Date.now()
      }
    }

    console.log(`\n🔄 Transforming ${papers.length} papers to graph format...`)

    const nodes = papers.map((paper, index) => {
      // ✅ HANDLE AUTHORS (array or string or semicolon-separated or Crossref format)
      let authorsStr = ''
      if (Array.isArray(paper.authors)) {
        authorsStr = paper.authors.join(';')
      } else if (Array.isArray(paper.author)) {
        // Crossref format: [{given: "John", family: "Doe"}]
        authorsStr = paper.author
          .map(a => `${a.given || ''} ${a.family || ''}`.trim())
          .filter(Boolean)
          .join(';')
      } else if (typeof paper.authors === 'string') {
        authorsStr = paper.authors
      }

      // ✅ HANDLE YEAR (try all possible field names from different APIs)
      const year = paper.year || 
                   paper.publication_year || 
                   paper.published_date || 
                   paper.publication_date ||
                   (paper['published-print']?.['date-parts']?.[0]?.[0]) ||  // ✅ CROSSREF FORMAT!
                   'Unknown'

      // ✅ HANDLE CITATION COUNT (try multiple field names)
      const citationCount = paper.citation_count || 
                            paper.citations || 
                            paper.citationCount ||
                            paper['is-referenced-by-count'] ||  // ✅ CROSSREF FORMAT!
                            0

      // ✅ HANDLE ABSTRACT (try multiple fields + clean HTML)
      let abstract = paper.abstract || 
                     paper.original_abstract || 
                     paper.abstractText ||
                     paper.summary ||
                     ''
      
      // ✅ CLEAN HTML TAGS FROM CROSSREF ABSTRACTS
      if (abstract) {
        abstract = abstract.replace(/<[^>]+>/g, '')  // Remove HTML tags
        abstract = abstract.replace(/\s+/g, ' ').trim()  // Normalize whitespace
      }
      
      if (!abstract || abstract.length < 10) {
        abstract = 'No abstract available'
      }

      // ✅ HANDLE URL/DOI (priority order)
      const doi = paper.doi || paper.DOI || ''
      const url = paper.url || 
                  paper.source_url ||
                  (doi ? `https://doi.org/${doi}` : '') ||
                  ''

      // ✅ HANDLE TITLE (may be array in Crossref)
      let title = ''
      if (Array.isArray(paper.title)) {
        title = paper.title[0] || 'Untitled'
      } else {
        title = paper.title || 'Untitled'
      }

      // ✅ PRESERVE ORIGINAL SOURCE (DON'T OVERWRITE WITH 'similar'!)
      const originalSource = (paper.source || 
                              paper.api_source || 
                              paper.research_domain ||
                              'unknown').toLowerCase()

      const node = {
        id: paper.id,
        title: title,
        label: title.substring(0, 60) + '...',
        
        authors: authorsStr,
        published_date: year,
        citation_count: citationCount,
        research_domain: originalSource.toUpperCase(),
        quality_score: paper.quality_score || paper.similarity_score || 0.85,
        
        abstract: abstract,
        doi: doi,
        url: url,
        pdf_url: paper.pdf_url || '',
        
        color: '#3B82F6',  // Blue
        size: 40,
        
        source: originalSource,  // ✅ PRESERVE ORIGINAL SOURCE!
        language: paper.language || 'en',
        year: year,
        
        // ✅ PRESERVE TRANSLATION INFO
        translated_to: paper.translated_to,
        translation_engine: paper.translation_engine,
        original_title: paper.original_title,
        original_abstract: paper.original_abstract,
        
        // ✅ PRESERVE ALL METADATA
        paper_type: paper.paper_type || paper.type || 'research_article',
        domain: paper.domain || 'Academic',
        searchable_text: paper.searchable_text || title,
        
        // ✅ SIMILARITY SCORE (for similar papers)
        similarity_score: paper.similarity_score,
        
        _original: paper  // ✅ KEEP COMPLETE ORIGINAL DATA!
      }

      return node
    })

    console.log(`✅ Created ${nodes.length} nodes`)
    
    // ✅ LOG SOURCE BREAKDOWN
    const sourceCounts = {}
    nodes.forEach(n => {
      sourceCounts[n.source] = (sourceCounts[n.source] || 0) + 1
    })
    console.log('📊 Transformed source breakdown:', sourceCounts)
    
    if (nodes.length > 0) {
      console.log('📝 First transformed node sample:', {
        id: nodes[0].id,
        title: nodes[0].title?.substring(0, 50),
        source: nodes[0].source,
        research_domain: nodes[0].research_domain,
        authors: nodes[0].authors?.substring(0, 30),
        year: nodes[0].published_date,
        citations: nodes[0].citation_count,
        language: nodes[0].language
      })
    }

    return {
      nodes,
      links: [],
      _stableVersion: Date.now(),
      _originalTimestamp: Date.now()
    }
  }, [])

  // ========================================================================
  // SEARCH PAPERS (CHAT API) - WITH COMPREHENSIVE RAW DATA LOGGING
  // ========================================================================
  const searchPapers = useCallback(async (query, conversationHistory = []) => {
    setLoading(true)
    setError(null)

    try {
      console.log('\n🤖 Calling chat API:', query)

      const response = await chatAPI.sendMessage(query, conversationHistory)

      console.log('\n📦 CHAT API RESPONSE:', {
        success: response.success,
        papersCount: response.papers?.length,
        edgesCount: response.edges?.length,
        intent: response.intent
      })

      if (response.success && response.papers && response.papers.length > 0) {
        console.log(`\n✅ Processing ${response.papers.length} papers from backend...`)
        
        // ✅ LOG COMPLETE RAW DATA FROM BACKEND!
        logRawPaperData(response.papers, 'CHAT API')
        
        // Transform papers to graph
        const graphData = transformPapersToGraph(response.papers)
        
        // ✅ ADD EDGES FROM BACKEND
        if (response.edges && response.edges.length > 0) {
          graphData.links = response.edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            strength: edge.strength || edge.similarity_score || 0.7,
            similarity_score: edge.strength || edge.similarity_score || 0.7,
            label: edge.label || `${Math.round((edge.strength || 0.7) * 100)}%`,
            color: edge.color || '#3B82F6',
            width: edge.width || Math.max(1, (edge.strength || 0.7) * 5)
          }))
          
          console.log(`✅ Added ${graphData.links.length} edges from backend`)
        }
        
        setData(graphData)
        
        // ✅ CALCULATE STATISTICS
        const sources = {}
        const languages = {}
        graphData.nodes.forEach(n => {
          sources[n.source] = (sources[n.source] || 0) + 1
          languages[n.language] = (languages[n.language] || 0) + 1
        })
        
        setStatistics({
          totalPapers: graphData.nodes.length,
          languages: Object.keys(languages),
          sources: Object.keys(sources),
          sourceBreakdown: sources,
          languageBreakdown: languages
        })

        console.log('\n✅ FINAL GRAPH DATA SET:', {
          nodes: graphData.nodes.length,
          links: graphData.links.length,
          sources: Object.keys(sources),
          sourceBreakdown: sources,
          languageBreakdown: languages
        })

        return graphData
      } else {
        console.error('❌ Response validation failed')
        throw new Error('No papers found')
      }
    } catch (err) {
      console.error('❌ Search error:', err)
      setError(err)
      throw err
    } finally {
      setLoading(false)
    }
  }, [transformPapersToGraph])

  // ========================================================================
  // FIND SIMILAR PAPERS - ✅ PRESERVE ORIGINAL SOURCE!
  // ========================================================================
  const findSimilarPapers = useCallback(async (paperId, maxResults = 10) => {
    try {
      console.log('\n⚡ Calling find similar API:', paperId, 'max:', maxResults)

      const response = await researchAPI.findSimilar(paperId, maxResults)

      console.log('📦 Similar papers response:', {
        similarPapersCount: response.similarPapers?.length,
        relationshipsCount: response.relationships?.length
      })

      if (response.similarPapers && response.similarPapers.length > 0) {
        const similarPapers = response.similarPapers
        const relationships = response.relationships || []

        console.log(`\n🔗 Processing ${similarPapers.length} similar papers with ${relationships.length} relationships`)

        // ✅ LOG COMPLETE RAW DATA FROM SIMILAR PAPERS API!
        logRawPaperData(similarPapers, 'SIMILAR PAPERS API')

        // Transform to nodes (GREEN + BIGGER!) - ✅ PRESERVE SOURCE!
        const newNodes = similarPapers.map((paper) => {
          let authorsStr = ''
          if (Array.isArray(paper.authors)) {
            authorsStr = paper.authors.join(';')
          } else if (Array.isArray(paper.author)) {
            authorsStr = paper.author
              .map(a => `${a.given || ''} ${a.family || ''}`.trim())
              .filter(Boolean)
              .join(';')
          } else if (typeof paper.authors === 'string') {
            authorsStr = paper.authors
          }

          // ✅ PRESERVE ORIGINAL SOURCE - DON'T OVERWRITE!
          const originalSource = (paper.source || paper.api_source || 'unknown').toLowerCase()

          return {
            id: paper.id,
            title: paper.title || 'Untitled',
            label: (paper.title || 'Untitled').substring(0, 60) + '...',
            
            authors: authorsStr,
            published_date: paper.year || paper.published_date || 'Unknown',
            citation_count: paper.citations || paper.citation_count || 0,
            research_domain: originalSource.toUpperCase(),  // ✅ USE ORIGINAL SOURCE!
            quality_score: paper.similarity_score || 0.8,
            similarity_score: paper.similarity_score || 0.8,
            
            abstract: paper.abstract || '',
            doi: paper.doi || paper.url || '',
            url: paper.url || paper.doi || '',
            pdf_url: paper.pdf_url || '',
            
            color: '#10B981',  // GREEN
            size: 45,  // BIGGER!
            
            source: originalSource,  // ✅ PRESERVE ORIGINAL SOURCE!
            language: paper.language || 'en',
            year: paper.year,
            _original: paper,
            _is_similar: true  // ✅ FLAG as similar without changing source
          }
        })

        // ✅ LOG TRANSFORMED SOURCES
        const transformedSources = {}
        newNodes.forEach(n => {
          transformedSources[n.source] = (transformedSources[n.source] || 0) + 1
        })
        console.log('📊 Similar papers after transformation:', transformedSources)

        // Transform to links WITH EDGE WEIGHTS
        const newLinks = relationships.map((rel) => {
          const edgeWeight = rel.strength || rel.similarity_score || 0.7
          
          return {
            source: rel.source,
            target: rel.target,
            strength: edgeWeight,
            similarity_score: edgeWeight,
            label: `${Math.round(edgeWeight * 100)}%`,
            color: '#10B981',
            width: Math.max(1, edgeWeight * 5)
          }
        })

        console.log(`✅ Created ${newNodes.length} new nodes, ${newLinks.length} new links`)

        // ✅ ADD to existing graph
        setData(prev => {
          if (!prev) {
            return {
              nodes: newNodes,
              links: newLinks,
              _stableVersion: Date.now(),
              _originalTimestamp: Date.now()
            }
          }

          const existingIds = new Set(prev.nodes.map(n => n.id))
          const uniqueNewNodes = newNodes.filter(n => !existingIds.has(n.id))

          const updated = {
            nodes: [...prev.nodes, ...uniqueNewNodes],
            links: [...prev.links, ...newLinks],
            _stableVersion: Date.now(),
            _originalTimestamp: prev._originalTimestamp
          }

          console.log('✅ Updated graph:', {
            totalNodes: updated.nodes.length,
            totalLinks: updated.links.length,
            addedNodes: uniqueNewNodes.length,
            addedLinks: newLinks.length
          })

          return updated
        })

        return { nodes: newNodes, links: newLinks }
      } else {
        console.warn('⚠️ No similar papers found')
        return { nodes: [], links: [] }
      }
    } catch (err) {
      console.error('❌ Find similar error:', err)
      throw err
    }
  }, [])

  // ========================================================================
  // GET NODE DETAILS - WITH COMPLETE RAW DATA LOG
  // ========================================================================
  const getNodeDetails = useCallback((nodeId) => {
    if (!data || !data.nodes) {
      console.warn('⚠️ No graph data available')
      return null
    }

    const node = data.nodes.find(n => n.id === nodeId)
    if (!node) {
      console.warn('⚠️ Node not found:', nodeId)
      return null
    }

    console.log('\n' + '='.repeat(100))
    console.log('📄 GETTING NODE DETAILS - COMPLETE RAW DATA')
    console.log('='.repeat(100))
    console.log('Node ID:', nodeId)
    console.log('Source:', node.source)
    console.log('Research Domain:', node.research_domain)
    console.log('\n🔬 COMPLETE TRANSFORMED NODE OBJECT:')
    console.log(JSON.stringify(node, null, 2))
    console.log('\n🔬 ORIGINAL BACKEND PAPER DATA (_original):')
    console.log(JSON.stringify(node._original, null, 2))
    console.log('='.repeat(100) + '\n')

    setLoadingDetails(false)
    
    // Get connected edges
    const connectedLinks = data.links.filter(
      link => link.source === nodeId || link.source.id === nodeId || 
              link.target === nodeId || link.target.id === nodeId
    )

    const details = {
      ...node,  // ✅ Include ALL node properties
      connected_papers: connectedLinks.length,
      edge_weights: connectedLinks.map(link => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source
        const targetId = typeof link.target === 'object' ? link.target.id : link.target
        
        return {
          connected_to: sourceId === nodeId ? targetId : sourceId,
          similarity: link.strength || link.similarity_score || 0,
          label: link.label || `${Math.round((link.strength || 0) * 100)}%`
        }
      })
    }

    setNodeDetails(details)
    return details
  }, [data])

  // ========================================================================
  // MANUAL GRAPH DATA SETTER (WITH EDGES SUPPORT)
  // ========================================================================
  const setGraphData = useCallback((papers, edges = []) => {
    if (!papers || papers.length === 0) {
      console.warn('⚠️ No papers provided to setGraphData')
      return null
    }

    console.log(`\n📊 Manually setting graph data with ${papers.length} papers and ${edges.length} edges`)

    // ✅ LOG RAW DATA
    logRawPaperData(papers, 'MANUAL SET')

    const graphData = transformPapersToGraph(papers)
    
    // ✅ ADD PRE-CALCULATED EDGES FROM BACKEND
    if (edges && edges.length > 0) {
      graphData.links = edges.map(edge => ({
        source: edge.source,
        target: edge.target,
        strength: edge.strength || edge.similarity_score || 0.7,
        similarity_score: edge.strength || edge.similarity_score || 0.7,
        label: edge.label || `${Math.round((edge.strength || 0.7) * 100)}%`,
        color: edge.color || '#3B82F6',
        width: edge.width || Math.max(1, (edge.strength || 0.7) * 5)
      }))
      
      console.log(`✅ Added ${graphData.links.length} pre-calculated edges`)
    }
    
    setData(graphData)
    
    // ✅ STATISTICS
    const sources = {}
    const languages = {}
    graphData.nodes.forEach(n => {
      sources[n.source] = (sources[n.source] || 0) + 1
      languages[n.language] = (languages[n.language] || 0) + 1
    })
    
    setStatistics({
      totalPapers: graphData.nodes.length,
      languages: Object.keys(languages),
      sources: Object.keys(sources),
      sourceBreakdown: sources,
      languageBreakdown: languages
    })

    console.log('✅ Graph data manually set:', {
      nodes: graphData.nodes.length,
      links: graphData.links.length,
      sourceBreakdown: sources
    })

    return graphData
  }, [transformPapersToGraph])

  // ✅ RETURN ALL FUNCTIONS AND STATE
  return {
    data,
    loading,
    error,
    searchPapers,
    findSimilarPapers,
    getNodeDetails,
    setGraphData,
    nodeDetails,
    loadingDetails,
    statistics
  }
}
