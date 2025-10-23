import axios from 'axios'
import { toast } from 'react-hot-toast'

// ==================================================================================
// API CONFIGURATION
// ==================================================================================
const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  TIMEOUT: 360000, // 6 minutes
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
}

const API_BASE_URL = API_CONFIG.BASE_URL

// ==================================================================================
// GET CURRENT LANGUAGE
// ==================================================================================
const getCurrentLanguage = () => {
  const stored = localStorage.getItem('preferred_language')
  const browser = navigator.language.split('-')[0]
  const finalLang = stored || browser || 'en'
  
  console.log(`🌍 Language: ${finalLang} (stored: ${stored}, browser: ${browser})`)
  
  return finalLang
}

// ==================================================================================
// AXIOS CLIENT
// ==================================================================================
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
})

// ==================================================================================
// REQUEST INTERCEPTOR (FIXED)
// ==================================================================================
apiClient.interceptors.request.use(
  (config) => {
    config.metadata = { startTime: new Date() }
    
    const currentLang = getCurrentLanguage()
    
    // ✅ FIX: Use 'Accept-Language' header (not 'X-User-Language')
    config.headers['Accept-Language'] = currentLang
    config.headers['X-User-Language'] = currentLang  // Keep for backward compatibility
    
    console.log(`\n🚀 API REQUEST: ${config.method?.toUpperCase()} ${config.url}`)
    console.log(`🌍 Language: ${currentLang}`)
    
    return config
  },
  (error) => {
    console.error('❌ Request Error:', error)
    return Promise.reject(error)
  }
)


// ==================================================================================
// RESPONSE INTERCEPTOR
// ==================================================================================
apiClient.interceptors.response.use(
  (response) => {
    const duration = new Date() - response.config.metadata.startTime
    
    console.log(`\n✅ API RESPONSE: ${response.config.url} (${duration}ms)`)
    console.log(`📊 Status: ${response.status}`)
    
    // Check translation info
    if (response.data.papers?.length > 0) {
      const firstPaper = response.data.papers[0]
      console.log(`📄 First paper:`, {
        id: firstPaper.id,
        title: firstPaper.title?.substring(0, 50),
        source: firstPaper.source,
        language: firstPaper.language,
        translated_to: firstPaper.translated_to,
        has_abstract: !!firstPaper.abstract
      })
    }
    
    return response
  },
  async (error) => {
    const duration = new Date() - error.config?.metadata?.startTime
    console.error(`\n❌ API ERROR: ${error.config?.url} (${duration}ms)`)
    console.error(`Status: ${error.response?.status}`)
    console.error(`Message: ${error.response?.data?.error || error.message}`)
    
    // User-friendly error messages
    if (error.response?.status === 401) {
      toast.error('Session expired')
    } else if (error.response?.status === 422) {
      toast.error('Invalid request data')
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again.')
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Request timeout')
    } else if (error.code === 'ERR_NETWORK') {
      toast.error('Network error. Check your connection.')
    }
    
    return Promise.reject(error)
  }
)

// ==================================================================================
// ☁️ CLOUD STORAGE API (FIXED HEADERS)
// ==================================================================================
export const cloudAPI = {
  getStats: async () => {
    try {
      const currentLang = getCurrentLanguage()
      const response = await fetch(`${API_BASE_URL}/api/cloud/stats`, {
        headers: { 
          'Accept-Language': currentLang,
          'X-User-Language': currentLang
        }
      })
      const data = await response.json()
      return data
    } catch (error) {
      console.error('❌ Failed to get stats:', error)
      return {
        success: false,
        stats: {
          totalPapers: 0,
          totalNodes: 0,
          storageUsed: 0,
          lastSync: null,
          elasticsearchHealth: 'unknown',
          vertexAIStatus: 'unknown'
        }
      }
    }
  },

  getSavedPapers: async ({ limit = 20, offset = 0 }) => {
    try {
      const currentLang = getCurrentLanguage()
      const response = await fetch(
        `${API_BASE_URL}/api/cloud/papers?limit=${limit}&offset=${offset}`,
        { 
          headers: { 
            'Accept-Language': currentLang,
            'X-User-Language': currentLang
          }
        }
      )
      const data = await response.json()
      return data
    } catch (error) {
      console.error('❌ Failed to get saved papers:', error)
      return { success: false, papers: [], total: 0 }
    }
  },

  getAllPapers: async ({ limit = 50, offset = 0 }) => {
    try {
      const currentLang = getCurrentLanguage()
      const response = await fetch(
        `${API_BASE_URL}/api/cloud/all-papers?limit=${limit}&offset=${offset}`,
        { 
          headers: { 
            'Accept-Language': currentLang,
            'X-User-Language': currentLang
          }
        }
      )
      const data = await response.json()
      return data
    } catch (error) {
      console.error('❌ Failed to get all papers:', error)
      return { success: false, papers: [], total: 0 }
    }
  },

  savePaper: async (paper) => {
    try {
      const currentLang = getCurrentLanguage()
      const response = await fetch(`${API_BASE_URL}/api/cloud/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept-Language': currentLang,
          'X-User-Language': currentLang
        },
        body: JSON.stringify(paper)
      })
      const data = await response.json()
      
      if (data.success) {
        toast.success('Paper saved to cloud!', { icon: '☁️' })
      } else {
        toast.error('Failed to save paper')
      }
      
      return data
    } catch (error) {
      console.error('❌ Failed to save paper:', error)
      toast.error('Network error while saving')
      return { success: false, error: error.message }
    }
  },

  deletePaper: async (paperId) => {
    try {
      const currentLang = getCurrentLanguage()
      const response = await fetch(`${API_BASE_URL}/api/cloud/papers/${paperId}`, {
        method: 'DELETE',
        headers: { 
          'Accept-Language': currentLang,
          'X-User-Language': currentLang
        }
      })
      const data = await response.json()
      
      if (data.success) {
        toast.success('Paper deleted from cloud', { icon: '🗑️' })
      } else {
        toast.error('Failed to delete paper')
      }
      
      return data
    } catch (error) {
      console.error('❌ Failed to delete paper:', error)
      toast.error('Network error while deleting')
      return { success: false, error: error.message }
    }
  }
}

// ==================================================================================
// RETRY MECHANISM
// ==================================================================================
const retryRequest = async (fn, attempts = API_CONFIG.RETRY_ATTEMPTS) => {
  try {
    return await fn()
  } catch (error) {
    if (attempts > 1 && error.response?.status >= 500) {
      console.warn(`🔄 Retrying... (${API_CONFIG.RETRY_ATTEMPTS - attempts + 1}/${API_CONFIG.RETRY_ATTEMPTS})`)
      await new Promise(resolve => setTimeout(resolve, API_CONFIG.RETRY_DELAY))
      return retryRequest(fn, attempts - 1)
    }
    throw error
  }
}

// ==================================================================================
// 🤖 CONVERSATIONAL AI API
// ==================================================================================
export const chatAPI = {
  sendMessage: async (message, conversationHistory = [], sessionId = null) => {
    if (!message || !message.trim()) {
      throw new Error('Message cannot be empty')
    }

    return retryRequest(async () => {
      const requestData = {
        message: message.trim(),
        conversation_history: conversationHistory,
        session_id: sessionId || `web_${Date.now()}`
      }

      console.log('🤖 Sending chat request:', { message: requestData.message, historyLength: conversationHistory.length })

      const response = await apiClient.post('/api/chat', requestData)

      if (!response.data.success) {
        throw new Error(response.data.agent_response || 'Chat failed')
      }

      const {
        success,
        agent_response,
        intent,
        papers = [],
        edges = [],
        total_papers = 0,
        total_edges = 0,
        suggested_actions = [],
        processing_time,
        session_id: responseSessionId
      } = response.data

      console.log('✅ Chat response received:', {
        intent,
        papersCount: total_papers || papers.length,
        edgesCount: total_edges || edges.length,
        processingTime: processing_time
      })

      return {
        success,
        response: agent_response,
        intent,
        papers: papers || [],
        papersCount: total_papers || papers.length,
        edges: edges || [],
        edgesCount: total_edges || edges.length,
        suggestedActions: suggested_actions || [],
        processingTime: processing_time,
        sessionId: responseSessionId || sessionId
      }
    })
  }
}

// ==================================================================================
// 📚 RESEARCH DISCOVERY API
// ==================================================================================
export const researchAPI = {
  discover: async (query, options = {}) => {
    const requestData = {
      query: query.trim(),
      max_results: options.maxResults || 50,
      enable_graph: options.enableGraph !== false,
      fetch_from_apis: options.fetchFromApis !== false,
      enable_translation: options.enableTranslation !== false
    }

    return retryRequest(async () => {
      console.log('🔍 Sending discover request:', requestData)

      const response = await apiClient.post('/api/research/discover', requestData)
      
      if (!response.data.success) {
        throw new Error(response.data.error || 'Discovery failed')
      }

      return {
        success: true,
        papers: response.data.papers || [],
        edges: response.data.edges || [],
        papersCount: response.data.total_papers || 0,
        edgesCount: response.data.total_edges || 0,
        processingTime: response.data.processing_time
      }
    })
  },

  findSimilar: async (paperId, maxResults = 10) => {
    if (!paperId) {
      throw new Error('Paper ID is required')
    }

    return retryRequest(async () => {
      const requestData = {
        paper_id: paperId,
        max_results: maxResults
      }

      console.log('⚡ Sending find-similar request:', requestData)

      const response = await apiClient.post('/api/research/find-similar', requestData)

      if (!response.data.success) {
        throw new Error(response.data.error || 'Find similar failed')
      }

      return {
        success: true,
        sourcePaper: response.data.source_paper,
        similarPapers: response.data.similar_papers || [],
        relationships: response.data.relationships || []
      }
    })
  }
}

// ==================================================================================
// 🤖 AI ANALYSIS API - WITH RAG QUERY
// ==================================================================================
export const aiAPI = {
  summarizePaper: async (paperId) => {
    if (!paperId) {
      throw new Error('Paper ID is required')
    }

    return retryRequest(async () => {
      const requestData = {
        paper_id: paperId
      }

      console.log('🤖 Sending summarize request:', requestData)

      const response = await apiClient.post('/api/paper/summarize', requestData)

      if (!response.data.success) {
        throw new Error(response.data.error || 'Summarization failed')
      }

      const {
        success,
        paper_id,
        title,
        summary,
        source,
        generated_by,
        processing_time
      } = response.data

      console.log('✅ Summary received:', {
        paperId: paper_id,
        summaryLength: summary?.length || 0,
        generatedBy: generated_by
      })

      return {
        success,
        paperId: paper_id,
        title,
        summary,
        source,
        generatedBy: generated_by,
        processingTime: processing_time
      }
    })
  },

  // ✅ CORRECTED RAG QUERY - Uses /api/rag/query endpoint
  ragQuery: async (userId, question, maxPapers = 10) => {
    if (!question || !question.trim()) {
      throw new Error('Question cannot be empty')
    }

    if (!userId || !userId.trim()) {
      throw new Error('User ID is required')
    }

    return retryRequest(async () => {
      const requestData = {
        user_id: userId,
        question: question.trim(),
        max_papers: maxPapers
      }

      console.log('🔍 Sending RAG query:', requestData)

      // ✅ CORRECTED ENDPOINT: /api/rag/query (not /api/research/rag-query)
      const response = await apiClient.post('/api/rag/query', requestData)

      if (!response.data.success) {
        throw new Error(response.data.error || 'RAG query failed')
      }

      const {
        success,
        answer,
        papers_used = [],
        papers_count = 0,
        user_id,
        source,
        generated_by,
        gemini_model,
        processing_time
      } = response.data

      console.log('✅ RAG query response:', {
        papersCount: papers_count,
        hasAnswer: !!answer,
        answerLength: answer?.length || 0,
        generatedBy: generated_by,
        model: gemini_model,
        processingTime: processing_time
      })

      return {
        success,
        answer: answer || '',
        papersUsed: papers_used || [],
        papersCount: papers_count || 0,
        userId: user_id,
        source: source,
        generatedBy: generated_by,
        geminiModel: gemini_model,
        processingTime: processing_time
      }
    })
  }
}

// ==================================================================================
// 🏥 SYSTEM API
// ==================================================================================
export const systemAPI = {
  checkHealth: async () => {
    try {
      const response = await apiClient.get('/api/health', { timeout: 5000 })
      return {
        status: 'healthy',
        data: response.data
      }
    } catch (error) {
      console.error('❌ System unhealthy:', error.message)
      return {
        status: 'unhealthy',
        error: error.message
      }
    }
  },

  getStats: async () => {
    try {
      const response = await apiClient.get('/api/stats', { timeout: 5000 })
      return response.data
    } catch (error) {
      console.error('❌ Failed to get stats:', error.message)
      throw error
    }
  }
}

// ==================================================================================
// COMBINED EXPORT
// ==================================================================================
export default {
  chat: chatAPI,
  research: researchAPI,
  ai: aiAPI,
  system: systemAPI,
  cloud: cloudAPI,
  client: apiClient
}

export { apiClient, API_BASE_URL, getCurrentLanguage }
