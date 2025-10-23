/**
 * ==================================================================================
 * RESEARCH DISCOVERY - WITH CHAT PANEL
 * ✅ Keep hero content, remove search form
 * ✅ Add floating chat button
 * ==================================================================================
 */

import React, { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster, toast } from 'react-hot-toast'
import { Globe, MessageCircle, X, Send } from 'lucide-react'

import TopBar from './components/layout/TopBar'
import MainContent from './components/layout/MainContent'
import SearchInterface from './components/search/SearchInterface'
import GraphViewer from './components/graph/GraphViewer'
import NodeInspector from './components/panels/NodeInspector'
import ControlPanel from './components/panels/controlPanel'
import StatusBar from './components/layout/StatusBar'
import LoadingScreen from './components/ui/LoadingScreen'
import ErrorBoundary from './components/common/ErrorBoundary'
import CloudDashboard from './components/dashboard/CloudDashboard'
import { useResearchData } from './hooks/useResearchData'
import { useGraphVisualization } from './hooks/useGraphVisualization'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { AppProvider, useAppContext } from './contexts/AppContext'
import api from './utils/api'

import './styles/modern.css'
import './styles/components.css'
import './styles/animations.css'
import './styles/responsive.css'

// ==================================================================================
// CHAT PANEL COMPONENT
// ==================================================================================
const ChatPanel = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [userId] = useState('user123')
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input
    setInput('')
    
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    }])

    setIsLoading(true)

    try {
      const response = await api.ai.ragQuery(userId, userMessage, 5)

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.answer,
        papersUsed: response.papersUsed,
        papersCount: response.papersCount,
        timestamp: new Date()
      }])
    } catch (error) {
      console.error('❌ Chat error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${error.message}`,
        timestamp: new Date()
      }])
    } finally {
      setIsLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <motion.div
      initial={{ x: 400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 400, opacity: 0 }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        width: '420px',
        height: '100vh',
        background: 'rgba(24, 24, 37, 0.95)',
        backdropFilter: 'blur(20px)',
        borderLeft: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 1000,
        boxShadow: '-4px 0 24px rgba(0, 0, 0, 0.3)'
      }}
    >
      {/* Header */}
      <div style={{
        padding: '20px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#fff' }}>
            🤖 Research Assistant
          </h3>
          <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)' }}>
            Powered by Gemini 2.0 Flash
          </p>
        </div>
        <button onClick={onClose} style={{
          background: 'rgba(255, 255, 255, 0.1)',
          border: 'none',
          borderRadius: '8px',
          width: '32px',
          height: '32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s'
        }}>
          <X size={18} color="#fff" />
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {messages.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: '40px 20px',
            color: 'rgba(255, 255, 255, 0.5)'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>💬</div>
            <p>Ask any research question!</p>
            <p style={{ fontSize: '12px', marginTop: '8px', opacity: 0.7 }}>
              I'll search across papers and provide answers.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
          }}>
            <div style={{
              background: msg.role === 'user' 
                ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                : 'rgba(255, 255, 255, 0.05)',
              padding: '12px 16px',
              borderRadius: '12px',
              maxWidth: '85%',
              color: '#fff',
              fontSize: '14px',
              lineHeight: '1.6',
              wordWrap: 'break-word'
            }}>
              {msg.content}
              {msg.papersUsed && msg.papersUsed.length > 0 && (
                <div style={{
                  marginTop: '12px',
                  paddingTop: '12px',
                  borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                  fontSize: '11px',
                  color: 'rgba(255, 255, 255, 0.7)'
                }}>
                  <p style={{ fontWeight: '600', marginBottom: '6px' }}>
                    📚 Referenced {msg.papersCount} papers:
                  </p>
                  <ul style={{ margin: 0, paddingLeft: '16px', listStyle: 'disc' }}>
                    {msg.papersUsed.slice(0, 3).map((paper, i) => (
                      <li key={i} style={{ marginBottom: '4px' }}>
                        {paper.title} ({paper.year})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <span style={{
              fontSize: '10px',
              color: 'rgba(255, 255, 255, 0.4)',
              marginTop: '4px'
            }}>
              {msg.timestamp.toLocaleTimeString()}
            </span>
          </div>
        ))}

        {isLoading && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.05)',
            padding: '12px 16px',
            borderRadius: '12px',
            color: 'rgba(255, 255, 255, 0.7)',
            fontSize: '14px'
          }}>
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{ padding: '20px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSendMessage()
              }
            }}
            placeholder="Ask about research..."
            disabled={isLoading}
            style={{
              flex: 1,
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
              padding: '12px',
              color: '#fff',
              fontSize: '14px',
              resize: 'none',
              minHeight: '48px',
              maxHeight: '120px',
              outline: 'none',
              fontFamily: 'inherit'
            }}
            rows={1}
          />
          <button onClick={handleSendMessage} disabled={isLoading || !input.trim()} style={{
            background: input.trim() && !isLoading
              ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
              : 'rgba(255, 255, 255, 0.1)',
            border: 'none',
            borderRadius: '12px',
            width: '48px',
            height: '48px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: input.trim() && !isLoading ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s'
          }}>
            <Send size={20} color="#fff" />
          </button>
        </div>
      </div>
    </motion.div>
  )
}

// ==================================================================================
// FLOATING CHAT BUTTON
// ==================================================================================
const FloatingChatButton = ({ onClick }) => {
  return (
    <motion.button
      onClick={onClick}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        width: '56px',
        height: '56px',
        borderRadius: '50%',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        border: 'none',
        boxShadow: '0 4px 24px rgba(102, 126, 234, 0.4)',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 999,
        transition: 'all 0.3s ease'
      }}
    >
      <MessageCircle size={24} color="#fff" />
    </motion.button>
  )
}

// ==================================================================================
// LANGUAGE SWITCHER
// ==================================================================================
const LanguageSwitcher = ({ onLanguageChange }) => {
  const [currentLang, setCurrentLang] = useState(
    localStorage.getItem('preferred_language') || 'en'
  )

  const LANGUAGES = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'zh', name: '中文', flag: '🇨🇳' },
    { code: 'es', name: 'Español', flag: '🇪🇸' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'ja', name: '日本語', flag: '🇯🇵' },
    { code: 'ko', name: '한국어', flag: '🇰🇷' },
    { code: 'pt', name: 'Português', flag: '🇧🇷' }
  ]

  const changeLanguage = async (lng) => {
    setCurrentLang(lng)
    localStorage.setItem('preferred_language', lng)
    
    const langName = LANGUAGES.find(l => l.code === lng)?.name || lng
    toast.success(`Language: ${langName}`, { icon: '🌍', duration: 2000 })
    
    if (onLanguageChange) {
      await onLanguageChange(lng)
    }
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      background: 'rgba(30, 30, 46, 0.6)',
      backdropFilter: 'blur(16px)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '8px',
      padding: '6px 10px',
      cursor: 'pointer'
    }}>
      <Globe size={16} style={{ color: '#fff' }} />
      <select 
        value={currentLang} 
        onChange={(e) => changeLanguage(e.target.value)}
        style={{
          background: 'transparent',
          border: 'none',
          color: '#fff',
          fontSize: '13px',
          cursor: 'pointer',
          outline: 'none',
          fontWeight: '500'
        }}
      >
        {LANGUAGES.map(lang => (
          <option key={lang.code} value={lang.code} style={{ background: '#1e1e2e', padding: '8px' }}>
            {lang.flag} {lang.name}
          </option>
        ))}
      </select>
    </div>
  )
}

// ==================================================================================
// MAIN APP COMPONENT
// ==================================================================================
function AppCore() {
  const [currentView, setCurrentView] = useState('empty')
  const [selectedNode, setSelectedNode] = useState(null)
  const [hoveredNode, setHoveredNode] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [graphMode, setGraphMode] = useState('2d')
  const [showControls, setShowControls] = useState(false)
  const [showInspector, setShowInspector] = useState(false)
  const [showCloudDashboard, setShowCloudDashboard] = useState(false)
  const [showChat, setShowChat] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const graphContainerRef = useRef(null)
  const searchInputRef = useRef(null)

  const { addNotification } = useAppContext()

  const {
    data: graphData,
    loading,
    error,
    searchPapers,
    findSimilarPapers,
    getNodeDetails,
    nodeDetails,
    statistics,
    loadingDetails
  } = useResearchData()

  const {
    graphConfig,
    updateGraphConfig,
    resetView,
    focusNode,
    exportGraph,
    getFilteredData
  } = useGraphVisualization(graphData)

  const handleLanguageChange = useCallback(async (newLanguage) => {
    setRefreshTrigger(prev => prev + 1)
    
    if (graphData && graphData.nodes?.length > 0 && searchQuery) {
      toast.loading('Translating...', { id: 'translate' })
      
      try {
        await searchPapers(searchQuery, [])
        toast.success('Translated!', { id: 'translate' })
        addNotification({ type: 'success', message: 'Content translated!' })
      } catch (err) {
        toast.error('Translation failed', { id: 'translate' })
        addNotification({ type: 'error', message: 'Translation failed' })
      }
    }
  }, [graphData, searchQuery, searchPapers, addNotification])

  useKeyboardShortcuts({
    'cmd+k': () => searchInputRef.current?.focus(),
    'cmd+shift+c': () => setShowControls(!showControls),
    'cmd+shift+m': () => setShowChat(!showChat),
    'escape': () => {
      setSelectedNode(null)
      setShowInspector(false)
      setShowCloudDashboard(false)
      setShowChat(false)
    },
    'r': () => resetView()
  })

  const handleSearch = useCallback(async (query) => {
    setCurrentView('loading')
    
    try {
      await searchPapers(query, [])
      setCurrentView('graph')
      setSearchQuery(query)
      toast.success('Papers discovered!', { icon: '🎉' })
      addNotification({ type: 'success', message: 'Papers discovered!' })
    } catch (err) {
      setCurrentView('error')
      toast.error('Search failed: ' + err.message)
      addNotification({ type: 'error', message: 'Search failed' })
    }
  }, [searchPapers, addNotification])

  const handleNodeInteraction = useCallback(async (nodeId, interactionType) => {
    switch (interactionType) {
      case 'hover':
        setHoveredNode(nodeId)
        break
      case 'click':
        if (nodeId) {
          setSelectedNode(nodeId)
          setShowInspector(true)
          getNodeDetails(nodeId)
          focusNode(nodeId)
        } else {
          setSelectedNode(null)
          setShowInspector(false)
        }
        break
      case 'unhover':
        setHoveredNode(null)
        break
      case 'dblclick':
        if (nodeId) {
          toast.loading('Finding similar...', { id: 'similar' })
          try {
            await findSimilarPapers(nodeId, 10)
            toast.success('Similar papers added!', { id: 'similar' })
            addNotification({ type: 'success', message: 'Similar papers added!' })
          } catch (err) {
            toast.error('Failed', { id: 'similar' })
            addNotification({ type: 'error', message: 'Failed' })
          }
        }
        break
    }
  }, [getNodeDetails, focusNode, findSimilarPapers, addNotification])

  useEffect(() => {
    if (loading) setCurrentView('loading')
    else if (error) setCurrentView('error')
    else if (graphData?.nodes?.length > 0) setCurrentView('graph')
    else if (!loading && !graphData) setCurrentView('empty')
  }, [loading, error, graphData])

  const displayData = getFilteredData() || graphData

  return (
    <ErrorBoundary>
      <div className="app-container">
        <TopBar
          onSearch={handleSearch}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          searchInputRef={searchInputRef}
          loading={loading}
          graphMode={graphMode}
          onGraphModeChange={setGraphMode}
          hasData={!!displayData?.nodes?.length}
          onToggleControls={() => setShowControls(!showControls)}
          showControls={showControls}
          onToggleCloud={() => setShowCloudDashboard(!showCloudDashboard)}
          showCloud={showCloudDashboard}
          languageSwitcher={<LanguageSwitcher onLanguageChange={handleLanguageChange} />}
        />

        <MainContent currentView={currentView}>
          <AnimatePresence mode="wait">
            {currentView === 'empty' && (
              <SearchInterface
                key="search"
                onSearch={handleSearch}
                searchInputRef={searchInputRef}
                recentSearches={[]}
                popularTopics={[
                  'Machine Learning',
                  'Computer Vision',
                  'Natural Language Processing',
                  'Healthcare AI',
                  'Climate Science',
                  'Quantum Computing',
                  'Deep Learning',
                  'Robotics',
                  'Blockchain',
                  'Cybersecurity'
                ]}
              />
            )}

            {currentView === 'loading' && (
              <LoadingScreen
                key="loading"
                message="Discovering research connections..."
                progress={loading ? 45 : 0}
              />
            )}

            {currentView === 'graph' && displayData && (
              <GraphViewer
                key="graph"
                ref={graphContainerRef}
                data={displayData}
                mode={graphMode}
                config={graphConfig}
                selectedNode={selectedNode}
                hoveredNode={hoveredNode}
                onNodeInteraction={handleNodeInteraction}
                onConfigChange={updateGraphConfig}
              />
            )}

            {currentView === 'error' && (
              <motion.div
                key="error"
                className="error-state"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <div className="error-content">
                  <div className="error-icon">⚠️</div>
                  <h3>Something went wrong</h3>
                  <p>{error?.message || 'Failed to load research data'}</p>
                  <div className="error-actions">
                    <button onClick={() => setCurrentView('empty')} className="btn-primary">
                      Back to Search
                    </button>
                    <button onClick={() => window.location.reload()} className="btn-secondary">
                      Reload Page
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </MainContent>

        <AnimatePresence>
          {showInspector && selectedNode && (
            <NodeInspector
              key="inspector"
              node={selectedNode}
              nodeDetails={nodeDetails}
              loading={loadingDetails}
              onClose={() => {
                setShowInspector(false)
                setSelectedNode(null)
              }}
              onRelatedNodeClick={(nodeId) => handleNodeInteraction(nodeId, 'click')}
              onFindSimilar={() => handleNodeInteraction(selectedNode, 'dblclick')}
            />
          )}

          {showControls && currentView === 'graph' && (
            <ControlPanel
              key="controls"
              config={graphConfig}
              onConfigChange={updateGraphConfig}
              onResetView={resetView}
              onExport={exportGraph}
              statistics={statistics}
              graphMode={graphMode}
              onGraphModeChange={(mode) => setGraphMode(mode)}
            />
          )}

          {showCloudDashboard && (
            <CloudDashboard
              key="cloud"
              isOpen={showCloudDashboard}
              onClose={() => setShowCloudDashboard(false)}
              refreshTrigger={refreshTrigger}
            />
          )}
        </AnimatePresence>

        {/* Chat Panel */}
        <ChatPanel isOpen={showChat} onClose={() => setShowChat(false)} />

        {/* Floating Chat Button */}
        {!showChat && (
          <FloatingChatButton onClick={() => setShowChat(true)} />
        )}

        <AnimatePresence>
          {currentView === 'graph' && displayData && (
            <StatusBar
              key="status"
              nodeCount={displayData.nodes?.length || 0}
              linkCount={displayData.links?.length || 0}
              selectedNode={selectedNode}
              searchQuery={searchQuery}
              statistics={statistics}
            />
          )}
        </AnimatePresence>

        <Toaster position="bottom-right" toastOptions={{
          duration: 4000,
          style: {
            background: 'rgba(30, 30, 46, 0.95)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#fff',
            borderRadius: '12px',
            padding: '12px 16px',
            fontSize: '14px',
            fontWeight: '500'
          }
        }} />
      </div>
    </ErrorBoundary>
  )
}

function App() {
  return (
    <AppProvider>
      <AppCore />
    </AppProvider>
  )
}

export default App
