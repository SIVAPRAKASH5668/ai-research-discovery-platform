// src/components/chat/ChatInterface.jsx

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, Bot, User, Sparkles, Loader2, 
  Search, X, MessageCircle 
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import { chatAPI } from '../../utils/api'
import '../../styles/ChatInterface.css'

const ChatInterface = ({ onSearch, onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      role: 'assistant',
      content: "👋 Hi! I'm your AI research assistant powered by Vertex AI Gemini. I can help you discover academic papers across multiple languages!\n\nTry asking:\n- \"Find papers about quantum computing\"\n- \"What is machine learning?\"\n- \"Show me Japanese AI research\"",
      timestamp: new Date().toISOString()
    }
  ])
  
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [conversationHistory, setConversationHistory] = useState([])
  const [isTyping, setIsTyping] = useState(false)
  const [sessionId] = useState(`session_${Date.now()}`)
  
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  
  // Auto-scroll
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])
  
  useEffect(() => {
    inputRef.current?.focus()
  }, [])
  
  // ✅ FIXED: Send message WITH edges support
  const sendMessage = useCallback(async (userMessage) => {
    if (!userMessage.trim() || loading) return
    
    const trimmedMessage = userMessage.trim()
    
    // Add user message to UI
    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: trimmedMessage,
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setIsTyping(true)
    
    try {
      console.log('🤖 Sending message:', trimmedMessage)
      console.log('📜 Current conversation history:', conversationHistory)
      
      // ✅ Call chat API ONCE
      const result = await chatAPI.sendMessage(
        trimmedMessage,
        conversationHistory,
        sessionId
      )
      
      console.log('✅ Chat API full response:', result)
      
      setIsTyping(false)
      
      // Add assistant response to UI
      const assistantMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.response,
        intent: result.intent,
        papers: result.papers,
        papersCount: result.papersCount,
        timestamp: new Date().toISOString(),
        metadata: result.metadata
      }
      
      setMessages(prev => [...prev, assistantMsg])
      
      // ✅ Update conversation history
      const newHistoryEntry = {
        user: trimmedMessage,
        assistant: result.response
      }
      
      const updatedHistory = [...conversationHistory, newHistoryEntry]
      setConversationHistory(updatedHistory)
      
      console.log('📜 Updated conversation history:', updatedHistory)
      
      // ✅ CRITICAL FIX: Extract edges from API response
      if (result.papers && result.papers.length > 0) {
        console.log(`✅ Found ${result.papersCount} papers, triggering graph view`)
        
        // ✅ Edges are ALREADY extracted by api.js at top level!
        const edges = result.edges || []
        
        console.log(`🔗 Extracted ${edges.length} edges from result.edges`)
        
        if (edges.length > 0) {
          console.log('🔗 First edge sample:', edges[0])
        } else {
          console.warn('⚠️ No edges found! Check backend response.')
        }
        
        toast.success(`Found ${result.papersCount} papers with ${edges.length} connections!`, {
          duration: 3000,
          icon: '🎉'
        })
        
        if (onSearch) {
          // ✅ Pass papers AND edges to App.jsx
          console.log('📤 Sending to App.jsx:', {
            query: trimmedMessage,
            papersCount: result.papers.length,
            edgesCount: edges.length
          })
          
          onSearch({
            query: trimmedMessage,
            papers: result.papers,
            edges: edges,  // ← CRITICAL!
            conversationHistory: updatedHistory,
            skipBackendCall: true  // ← Prevent duplicate API call
          })
        }
      }
      
    } catch (error) {
      setIsTyping(false)
      console.error('❌ Chat error:', error)
      
      const errorMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `⚠️ Sorry, I encountered an error: ${error.message}\n\nPlease try again or rephrase your question.`,
        isError: true,
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, errorMsg])
      toast.error('Chat error: ' + error.message)
    } finally {
      setLoading(false)
    }
  }, [conversationHistory, sessionId, onSearch, loading])
  
  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !loading) {
      sendMessage(input)
    }
  }
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }
  
  // Suggested prompts
  const suggestedPrompts = [
    "Find papers about quantum computing",
    "What are the latest AI trends?",
    "Show me climate change research",
    "Explain machine learning",
    "Papers on healthcare AI"
  ]
  
  const handleSuggestedPrompt = (prompt) => {
    setInput(prompt)
    inputRef.current?.focus()
  }
  
  return (
    <motion.div 
      className="chat-interface"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
    >
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header-content">
          <div className="chat-header-icon">
            <Bot className="bot-icon" />
            <Sparkles className="sparkle-icon" />
          </div>
          <div className="chat-header-text">
            <h3>AI Research Assistant</h3>
            <p>Powered by Vertex AI Gemini</p>
          </div>
        </div>
        
        {onClose && (
          <button onClick={onClose} className="chat-close-btn">
            <X size={20} />
          </button>
        )}
      </div>
      
      {/* Messages */}
      <div className="chat-messages">
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              className={`message-wrapper ${message.role}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <div className="message-avatar">
                {message.role === 'user' ? <User size={18} /> : <Bot size={18} />}
              </div>
              
              <div className={`message-bubble ${message.isError ? 'error' : ''}`}>
                <div className="message-content">{message.content}</div>
                
                {message.papersCount > 0 && (
                  <div className="message-papers-badge">
                    <Search size={14} />
                    <span>{message.papersCount} papers found</span>
                  </div>
                )}
                
                {message.intent && (
                  <div className="message-intent-badge">
                    {message.intent === 'search' && '🔍'}
                    {message.intent === 'explain' && '💡'}
                    {message.intent === 'analyze' && '📊'}
                    <span>{message.intent}</span>
                  </div>
                )}
                
                <div className="message-timestamp">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {/* Typing Indicator */}
        {isTyping && (
          <motion.div 
            className="message-wrapper assistant"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="message-avatar">
              <Bot size={18} />
            </div>
            <div className="message-bubble typing">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Suggested Prompts */}
      {!loading && messages.length <= 1 && (
        <div className="suggested-prompts">
          <div className="suggested-prompts-label">
            <MessageCircle size={14} />
            <span>Try asking:</span>
          </div>
          <div className="suggested-prompts-list">
            {suggestedPrompts.map((prompt, index) => (
              <button
                key={index}
                className="suggested-prompt-btn"
                onClick={() => handleSuggestedPrompt(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Input */}
      <form onSubmit={handleSubmit} className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about research papers..."
            className="chat-input"
            rows={1}
            disabled={loading}
          />
          
          <button
            type="submit"
            className="chat-send-btn"
            disabled={!input.trim() || loading}
          >
            {loading ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
          </button>
        </div>
        
        <div className="chat-input-hint">
          Press Enter to send • Shift+Enter for new line
        </div>
      </form>
    </motion.div>
  )
}

export default ChatInterface
