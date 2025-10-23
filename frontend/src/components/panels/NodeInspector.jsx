/**
 * ==================================================================================
 * NODE INSPECTOR - COMPLETE WITH FIXED METADATA FOR ALL SOURCES
 * ==================================================================================
 */

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { aiAPI, cloudAPI } from '../../utils/api'  

import {
  X,
  ExternalLink,
  Download,
  Share,
  BookOpen,
  Users,
  Calendar,
  Award,
  TrendingUp,
  Lightbulb,
  AlertCircle,
  HelpCircle,
  ArrowUpRight,
  Sparkles,
  Loader2,
  RefreshCw,
  Cloud,
  Check,
  Link2,
  GitBranch
} from 'lucide-react'
import { toast } from 'react-hot-toast'


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const NodeInspector = ({
  node,
  nodeDetails,
  loading,
  onClose,
  onRelatedNodeClick
}) => {
  const [activeTab, setActiveTab] = useState('overview')
  const [copied, setCopied] = useState(false)
  
  // AI Summary State
  const [aiSummary, setAiSummary] = useState(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [summaryError, setSummaryError] = useState(null)

  // Cloud Save State
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [checkingIfSaved, setCheckingIfSaved] = useState(false)

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BookOpen },
    { id: 'details', label: 'Details', icon: TrendingUp },
    { id: 'connections', label: 'Connections', icon: Users }
  ]

  // ✅ HELPER: Parse authors from various formats
  const parseAuthors = (authorsData) => {
    if (!authorsData) return null
    
    // String format (e.g., "Author1;Author2;Author3")
    if (typeof authorsData === 'string') {
      const authors = authorsData.split(';').filter(a => a.trim())
      if (authors.length === 0) return null
      
      if (authors.length <= 3) {
        return authors.join(', ')
      } else {
        return `${authors.slice(0, 3).join(', ')} et al.`
      }
    }
    
    // Array format (e.g., [{name: "Author1"}, {name: "Author2"}])
    if (Array.isArray(authorsData)) {
      const authorNames = authorsData
        .map(a => a.name || a)
        .filter(n => n && n.trim())
      
      if (authorNames.length === 0) return null
      
      if (authorNames.length <= 3) {
        return authorNames.join(', ')
      } else {
        return `${authorNames.slice(0, 3).join(', ')} et al.`
      }
    }
    
    return null
  }

  // ✅ HELPER: Parse date from various formats
  const parseDate = (dateData) => {
    if (!dateData) return null
    
    // Already formatted string (e.g., "2024", "2024-01", "Jan 2024")
    if (typeof dateData === 'string') {
      return dateData
    }
    
    // Number (year only)
    if (typeof dateData === 'number') {
      return dateData.toString()
    }
    
    // Date object
    if (dateData instanceof Date) {
      return dateData.getFullYear().toString()
    }
    
    return null
  }

  // ✅ HELPER: Parse citation count
  const parseCitations = (citationData) => {
    if (!citationData && citationData !== 0) return null
    
    const count = parseInt(citationData)
    if (isNaN(count)) return null
    
    return count
  }

  // ✅ GET COLOR BASED ON SIMILARITY SCORE
  const getColorFromSimilarity = (similarity) => {
    if (similarity >= 0.8) {
      return {
        color: '#3B82F6',
        bgColor: 'rgba(59, 130, 246, 0.15)',
        borderColor: 'rgba(59, 130, 246, 0.4)',
        gradient: 'linear-gradient(135deg, #3B82F6, #60A5FA)',
        label: 'Very Strong',
        icon: '🔗'
      }
    }
    if (similarity >= 0.6) {
      return {
        color: '#10B981',
        bgColor: 'rgba(16, 185, 129, 0.15)',
        borderColor: 'rgba(16, 185, 129, 0.4)',
        gradient: 'linear-gradient(135deg, #10B981, #34D399)',
        label: 'Strong',
        icon: '✅'
      }
    }
    if (similarity >= 0.4) {
      return {
        color: '#F59E0B',
        bgColor: 'rgba(245, 158, 11, 0.15)',
        borderColor: 'rgba(245, 158, 11, 0.4)',
        gradient: 'linear-gradient(135deg, #F59E0B, #FBBF24)',
        label: 'Moderate',
        icon: '⚡'
      }
    }
    return {
      color: '#EF4444',
      bgColor: 'rgba(239, 68, 68, 0.15)',
      borderColor: 'rgba(239, 68, 68, 0.4)',
      gradient: 'linear-gradient(135deg, #EF4444, #F87171)',
      label: 'Weak',
      icon: '📍'
    }
  }

  useEffect(() => {
    if (nodeDetails?.id) {
      checkIfPaperIsSaved()
    }
  }, [nodeDetails?.id])

  useEffect(() => {
    if (activeTab === 'details' && !aiSummary && !summaryLoading && !summaryError && nodeDetails?.id) {
      console.log('🔄 Details tab opened, generating summary for:', nodeDetails.id)
      generateAISummary()
    }
  }, [activeTab, nodeDetails?.id])

  const checkIfPaperIsSaved = async () => {
    if (!nodeDetails?.id) return
    
    setCheckingIfSaved(true)
    try {
      // ✅ Use existing cloudAPI from utils/api
      const { papers, success } = await cloudAPI.getSavedPapers({ limit: 1000 })
      
      if (success && papers) {
        // Check if current paper is in saved papers
        const isSaved = papers.some(p => p.id === nodeDetails.id)
        setSaved(isSaved)
        
        if (isSaved) {
          console.log('✅ Paper is already saved in cloud')
        }
      } else {
        setSaved(false)
      }
    } catch (error) {
      console.error('❌ Error checking if saved:', error)
      setSaved(false)
    } finally {
      setCheckingIfSaved(false)
    }
  }

  const handleSavePaper = async () => {
    if (!nodeDetails?.id || saving || saved) return
    
    setSaving(true)
    console.log('💾 Saving paper to cloud:', nodeDetails.id)
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/cloud/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          id: nodeDetails.id,
          title: nodeDetails.title,
          authors: nodeDetails.authors,
          abstract: nodeDetails.abstract,
          year: nodeDetails.year || nodeDetails.published_date,
          doi: nodeDetails.doi,
          url: nodeDetails.url,
          pdf_url: nodeDetails.pdf_url,
          research_domain: nodeDetails.research_domain,
          citation_count: nodeDetails.citation_count
        })
      })
      
      const result = await response.json()
      
      if (result.success) {
        setSaved(true)
        toast.success('Paper saved to cloud!', { icon: '☁️', duration: 3000 })
        console.log('✅ Paper saved successfully:', result.paper_id)
        setTimeout(() => checkIfPaperIsSaved(), 3000)
      } else {
        throw new Error(result.error || 'Failed to save paper')
      }
    } catch (error) {
      console.error('❌ Save failed:', error)
      toast.error('Failed to save paper: ' + error.message, { icon: '❌', duration: 4000 })
    } finally {
      setSaving(false)
    }
  }

  const generateAISummary = async () => {
    if (!nodeDetails?.id) {
      console.warn('⚠️ No nodeDetails.id available')
      return
    }

    setSummaryLoading(true)
    setSummaryError(null)
    setAiSummary(null)

    try {
      console.log('🤖 Generating AI summary for:', nodeDetails.id)
      const result = await aiAPI.summarizePaper(nodeDetails.id)

      if (result.success) {
        setAiSummary(result.summary)
        console.log('✅ AI Summary generated:', result.summary.substring(0, 100) + '...')
      } else {
        setSummaryError(result.error || 'Failed to generate summary')
        console.error('❌ Summary generation failed:', result.error)
      }
    } catch (error) {
      console.error('❌ Summary generation error:', error)
      setSummaryError(error.message || 'Network error while generating summary')
    } finally {
      setSummaryLoading(false)
    }
  }

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(nodeDetails?.doi || nodeDetails?.url || '')
      setCopied(true)
      toast.success('Link copied!', { duration: 2000 })
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: nodeDetails?.title || 'Research Paper',
          text: nodeDetails?.context_summary || '',
          url: nodeDetails?.doi || nodeDetails?.url || window.location.href
        })
      } catch (err) {
        console.error('Share failed:', err)
      }
    } else {
      handleCopyLink()
    }
  }

  // ✅ PARSE METADATA WITH FALLBACKS
  const authors = parseAuthors(nodeDetails?.authors)
  const publishedDate = parseDate(nodeDetails?.published_date || nodeDetails?.year)
  const citationCount = parseCitations(nodeDetails?.citation_count || nodeDetails?.citations)

  return (
    <motion.div
      className="node-inspector"
      initial={{ x: -400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: -400, opacity: 0 }}
      transition={{ duration: 0.5, ease: "easeInOut" }}
    >
      {/* Header */}
      <div className="inspector-header">
        <div className="header-title">
          <BookOpen size={20} />
          <h2>Paper Inspector</h2>
        </div>
        <div className="header-actions">
          <motion.button
            className={`action-btn save-btn ${saved ? 'saved' : ''} ${checkingIfSaved ? 'checking' : ''}`}
            onClick={handleSavePaper}
            disabled={saving || saved || checkingIfSaved}
            whileHover={{ scale: saving || saved ? 1 : 1.05 }}
            whileTap={{ scale: saving || saved ? 1 : 0.95 }}
            title={saved ? 'Saved to Cloud' : 'Save to Cloud'}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 12px', fontSize: '14px' }}
          >
            {checkingIfSaved ? (
              <Loader2 size={18} className="spinner" />
            ) : saving ? (
              <>
                <Loader2 size={18} className="spinner" />
                <span>Saving...</span>
              </>
            ) : saved ? (
              <>
                <Check size={18} />
                <span className="saved-text">Saved!</span>
              </>
            ) : (
              <>
                <Cloud size={18} />
                <span>Save</span>
              </>
            )}
          </motion.button>

          <motion.button
            className="action-btn"
            onClick={handleShare}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title="Share"
          >
            <Share size={18} />
          </motion.button>
          
          <motion.button
            className="action-btn"
            onClick={onClose}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <X size={18} />
          </motion.button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="inspector-tabs">
        {tabs.map((tab) => (
          <motion.button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <tab.icon size={16} />
            {tab.label}
          </motion.button>
        ))}
      </div>

      {/* Content */}
      <div className="inspector-content">
        {loading ? (
          <div className="inspector-loading">
            <motion.div
              className="loading-spinner"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            <p>Loading paper details...</p>
          </div>
        ) : nodeDetails ? (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {/* ===== OVERVIEW TAB - FIXED METADATA ===== */}
              {activeTab === 'overview' && (
                <div className="tab-content">
                  <div className="paper-header">
                    <h1 className="paper-title">{nodeDetails.title || 'Untitled Paper'}</h1>
                    
                    {/* ✅ FIXED METADATA DISPLAY */}
                    <div className="paper-meta">
                      {authors && (
                        <div className="meta-item">
                          <Users size={16} />
                          <span>{authors}</span>
                        </div>
                      )}
                      
                      {publishedDate && (
                        <div className="meta-item">
                          <Calendar size={16} />
                          <span>{publishedDate}</span>
                        </div>
                      )}
                      
                      {citationCount !== null && (
                        <div className="meta-item">
                          <Award size={16} />
                          <span>{citationCount} citation{citationCount !== 1 ? 's' : ''}</span>
                        </div>
                      )}

                      {/* ✅ SHOW SOURCE */}
                      {nodeDetails.source && (
                        <div className="meta-item">
                          <ExternalLink size={16} />
                          <span className="source-badge">{nodeDetails.source.toUpperCase()}</span>
                        </div>
                      )}
                    </div>

                    {/* ✅ SHOW WHEN NO METADATA AVAILABLE */}
                    {!authors && !publishedDate && !citationCount && (
                      <div className="paper-meta">
                        <div className="meta-item" style={{ color: '#94A3B8', fontStyle: 'italic' }}>
                          <AlertCircle size={16} />
                          <span>Metadata not available</span>
                        </div>
                      </div>
                    )}

                    <div className="paper-badges">
                      <span className="domain-badge">{nodeDetails.research_domain || nodeDetails.source?.toUpperCase() || 'RESEARCH'}</span>
                      {nodeDetails.quality_score && (
                        <span className="quality-badge">
                          {Math.round(nodeDetails.quality_score * 100)}% Quality
                        </span>
                      )}
                      {nodeDetails.similarity_score && (
                        <span className="similarity-badge">
                          {Math.round(nodeDetails.similarity_score * 100)}% Similarity
                        </span>
                      )}
                    </div>
                  </div>

                  {nodeDetails.abstract && (
                    <div className="content-section">
                      <h3>Abstract</h3>
                      <p className="abstract-text">{nodeDetails.abstract}</p>
                    </div>
                  )}

                  {nodeDetails.context_summary && (
                    <div className="content-section">
                      <h3>Context Summary</h3>
                      <div className="ai-summary">
                        <p>{nodeDetails.context_summary}</p>
                      </div>
                    </div>
                  )}

                  <div className="action-links">
                    {nodeDetails.doi && (
                      <motion.a
                        href={nodeDetails.doi.startsWith('http') ? nodeDetails.doi : `https://doi.org/${nodeDetails.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="action-link primary"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <ExternalLink size={16} />
                        View Paper
                      </motion.a>
                    )}
                    
                    {nodeDetails.url && !nodeDetails.doi && (
                      <motion.a
                        href={nodeDetails.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="action-link primary"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <ExternalLink size={16} />
                        View Paper
                      </motion.a>
                    )}
                    
                    {nodeDetails.pdf_url && (
                      <motion.a
                        href={nodeDetails.pdf_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="action-link"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Download size={16} />
                        Download PDF
                      </motion.a>
                    )}
                  </div>
                </div>
              )}

              {/* ===== ✨ ENHANCED DETAILS TAB ===== */}
              {activeTab === 'details' && (
                <div className="tab-content">
                  {/* AI SUMMARY SECTION */}
                  <div className="content-section">
                    <div className="section-header" style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      marginBottom: '1.5rem',
                      padding: '1rem',
                      background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(167, 139, 250, 0.05))',
                      borderRadius: '0.75rem',
                      border: '1px solid rgba(139, 92, 246, 0.3)'
                    }}>
                      <div style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '0.5rem',
                        background: 'linear-gradient(135deg, #8B5CF6, #A78BFA)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white'
                      }}>
                        <Sparkles size={20} />
                      </div>
                      <div style={{ flex: 1 }}>
                        <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '600' }}>
                          AI-Generated Summary
                        </h3>
                        <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.75rem', color: '#94A3B8' }}>
                          Powered by Vertex AI Gemini 2.0 Flash
                        </p>
                      </div>
                    </div>

                    {summaryLoading && (
                      <div className="summary-loading" style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '3rem 2rem',
                        textAlign: 'center'
                      }}>
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                          style={{ marginBottom: '1rem' }}
                        >
                          <Loader2 size={48} style={{ color: '#8B5CF6' }} />
                        </motion.div>
                        <p style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                          Generating AI summary with Vertex AI Gemini...
                        </p>
                        <p className="hint" style={{ fontSize: '0.85rem', color: '#94A3B8' }}>
                          Analyzing paper content • ETA: 5-10 seconds
                        </p>
                      </div>
                    )}

                    {summaryError && !summaryLoading && (
                      <div className="summary-error" style={{
                        padding: '2rem',
                        background: 'rgba(239, 68, 68, 0.1)',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: '0.75rem',
                        textAlign: 'center'
                      }}>
                        <AlertCircle size={32} style={{ color: '#EF4444', marginBottom: '1rem' }} />
                        <p className="error-message" style={{
                          color: '#EF4444',
                          fontSize: '0.95rem',
                          marginBottom: '1rem'
                        }}>
                          {summaryError}
                        </p>
                        <motion.button
                          className="retry-btn"
                          onClick={generateAISummary}
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          style={{
                            padding: '0.75rem 1.5rem',
                            background: 'linear-gradient(135deg, #EF4444, #F87171)',
                            border: 'none',
                            borderRadius: '0.5rem',
                            color: 'white',
                            fontWeight: '600',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            cursor: 'pointer'
                          }}
                        >
                          <RefreshCw size={16} />
                          Try Again
                        </motion.button>
                      </div>
                    )}

                    {aiSummary && !summaryLoading && (
                      <motion.div
                        className="summary-content"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                      >
                        <div className="summary-text" style={{
                          padding: '1.5rem',
                          background: 'rgba(139, 92, 246, 0.05)',
                          borderRadius: '0.75rem',
                          border: '1px solid rgba(139, 92, 246, 0.2)',
                          marginBottom: '1rem'
                        }}>
                          {/* ✅ PARSE BULLET POINTS FROM BACKEND RESPONSE */}
                          {aiSummary.split('\n').map((line, index) => {
                            // Check if line is a bullet point (starts with *, •, or -)
                            const isBullet = /^[\*•\-]\s/.test(line.trim())
                            
                            if (isBullet) {
                              // Remove bullet marker and clean up
                              const cleanText = line.trim().replace(/^[\*•\-]\s*/, '').replace(/^\*\*/, '').replace(/\*\*$/, '')
                              
                              if (cleanText) {
                                return (
                                  <motion.div
                                    key={index}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                    style={{
                                      display: 'flex',
                                      gap: '1rem',
                                      marginBottom: '1rem',
                                      padding: '1rem',
                                      background: 'rgba(255, 255, 255, 0.02)',
                                      borderRadius: '0.5rem',
                                      borderLeft: '3px solid #8B5CF6'
                                    }}
                                  >
                                    <div style={{
                                      width: '24px',
                                      height: '24px',
                                      borderRadius: '50%',
                                      background: 'linear-gradient(135deg, #8B5CF6, #A78BFA)',
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      color: 'white',
                                      fontSize: '0.75rem',
                                      fontWeight: '700',
                                      flexShrink: 0
                                    }}>
                                      {Math.floor(index / 2) + 1}
                                    </div>
                                    <p style={{
                                      lineHeight: '1.7',
                                      fontSize: '0.95rem',
                                      margin: 0,
                                      color: '#E2E8F0'
                                    }}>
                                      {cleanText}
                                    </p>
                                  </motion.div>
                                )
                              }
                            }
                            
                            // Regular paragraph (non-bullet text)
                            if (line.trim() && !isBullet) {
                              return (
                                <p key={index} style={{
                                  lineHeight: '1.8',
                                  fontSize: '0.95rem',
                                  marginBottom: '1rem',
                                  color: '#CBD5E1'
                                }}>
                                  {line.trim()}
                                </p>
                              )
                            }
                            
                            return null
                          })}
                        </div>

                        <motion.button
                          className="regenerate-btn"
                          onClick={generateAISummary}
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          style={{
                            padding: '0.75rem 1.5rem',
                            background: 'rgba(139, 92, 246, 0.1)',
                            border: '1px solid rgba(139, 92, 246, 0.3)',
                            borderRadius: '0.5rem',
                            color: '#8B5CF6',
                            fontWeight: '600',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            cursor: 'pointer',
                            fontSize: '0.875rem'
                          }}
                        >
                          <RefreshCw size={14} />
                          Regenerate Summary
                        </motion.button>
                      </motion.div>
                    )}
                  </div>

                  {/* Key Findings */}
                  {nodeDetails.key_findings && nodeDetails.key_findings.length > 0 && (
                    <div className="content-section">
                      <h3>
                        <Lightbulb size={18} />
                        Key Findings
                      </h3>
                      <div className="findings-list">
                        {nodeDetails.key_findings.map((finding, index) => (
                          <motion.div
                            key={index}
                            className="finding-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <TrendingUp size={16} className="finding-icon" />
                            <p>{finding}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Limitations */}
                  {nodeDetails.limitations && nodeDetails.limitations.length > 0 && (
                    <div className="content-section">
                      <h3>
                        <AlertCircle size={18} />
                        Limitations
                      </h3>
                      <div className="limitations-list">
                        {nodeDetails.limitations.map((limitation, index) => (
                          <motion.div
                            key={index}
                            className="limitation-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <AlertCircle size={16} className="limitation-icon" />
                            <p>{limitation}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Future Directions */}
                  {nodeDetails.future_directions && nodeDetails.future_directions.length > 0 && (
                    <div className="content-section">
                      <h3>
                        <ArrowUpRight size={18} />
                        Future Directions
                      </h3>
                      <div className="future-list">
                        {nodeDetails.future_directions.map((direction, index) => (
                          <motion.div
                            key={index}
                            className="future-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <ArrowUpRight size={16} className="future-icon" />
                            <p>{direction}</p>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Quality Metrics */}
                  {nodeDetails.quality_metrics && Object.keys(nodeDetails.quality_metrics).length > 0 && (
                    <div className="content-section">
                      <h3>Quality Metrics</h3>
                      <div className="quality-grid">
                        {Object.entries(nodeDetails.quality_metrics).map(([key, value]) => (
                          <div key={key} className="quality-item">
                            <span className="metric-label">{key.replace('_', ' ')}</span>
                            <div className="metric-bar">
                              <motion.div
                                className="metric-fill"
                                initial={{ width: 0 }}
                                animate={{ width: `${value * 100}%` }}
                                transition={{ duration: 1, delay: 0.5 }}
                              />
                            </div>
                            <span className="metric-value">{Math.round(value * 100)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ===== 🎨 ENHANCED CONNECTIONS TAB WITH DYNAMIC COLORS ===== */}
              {activeTab === 'connections' && (
                <div className="tab-content">
                  {/* Edge Weights Section - WITH COLORS */}
                  {nodeDetails.edge_weights && nodeDetails.edge_weights.length > 0 && (
                    <div className="content-section">
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                        <GitBranch size={18} />
                        Connected Papers ({nodeDetails.edge_weights.length})
                      </h3>
                      
                      <div className="connections-list">
                        {nodeDetails.edge_weights.map((edge, index) => {
                          const colorData = getColorFromSimilarity(edge.similarity)
                          
                          return (
                            <motion.div
                              key={index}
                              className="connection-item"
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.05 }}
                              onClick={() => onRelatedNodeClick && onRelatedNodeClick(edge.connected_to, 'click')}
                              whileHover={{ scale: 1.02, x: 4 }}
                              whileTap={{ scale: 0.98 }}
                              style={{
                                cursor: 'pointer',
                                padding: '1rem',
                                marginBottom: '0.75rem',
                                background: colorData.bgColor,
                                border: `2px solid ${colorData.borderColor}`,
                                borderRadius: '0.75rem',
                                transition: 'all 0.3s ease'
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                {/* Icon Badge */}
                                <div style={{
                                  width: '48px',
                                  height: '48px',
                                  borderRadius: '0.5rem',
                                  background: colorData.gradient,
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  fontSize: '1.5rem',
                                  flexShrink: 0
                                }}>
                                  {colorData.icon}
                                </div>

                                {/* Connection Info */}
                                <div style={{ flex: 1 }}>
                                  <div style={{
                                    fontSize: '0.875rem',
                                    fontWeight: '600',
                                    color: colorData.color,
                                    marginBottom: '0.25rem'
                                  }}>
                                    {edge.connected_to}
                                  </div>
                                  <div style={{
                                    fontSize: '0.75rem',
                                    color: '#94A3B8',
                                    marginBottom: '0.5rem'
                                  }}>
                                    {edge.label || 'Related Paper'}
                                  </div>

                                  {/* Similarity Bar */}
                                  <div style={{
                                    height: '6px',
                                    background: 'rgba(100, 116, 139, 0.2)',
                                    borderRadius: '1rem',
                                    overflow: 'hidden'
                                  }}>
                                    <motion.div
                                      initial={{ width: 0 }}
                                      animate={{ width: `${edge.similarity * 100}%` }}
                                      transition={{ duration: 0.8, delay: index * 0.05 }}
                                      style={{
                                        height: '100%',
                                        background: colorData.gradient,
                                        borderRadius: '1rem'
                                      }}
                                    />
                                  </div>
                                </div>

                                {/* Strength Badge */}
                                <div style={{
                                  padding: '0.5rem 1rem',
                                  borderRadius: '2rem',
                                  background: colorData.bgColor,
                                  border: `1px solid ${colorData.borderColor}`,
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'center',
                                  minWidth: '80px'
                                }}>
                                  <div style={{
                                    fontSize: '1.25rem',
                                    fontWeight: '700',
                                    background: colorData.gradient,
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    lineHeight: 1
                                  }}>
                                    {Math.round(edge.similarity * 100)}%
                                  </div>
                                  <div style={{
                                    fontSize: '0.65rem',
                                    color: colorData.color,
                                    fontWeight: '600',
                                    marginTop: '0.25rem'
                                  }}>
                                    {colorData.label}
                                  </div>
                                </div>
                              </div>
                            </motion.div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Related Papers - WITH COLORS */}
                  {nodeDetails.related_papers && nodeDetails.related_papers.length > 0 && (
                    <div className="content-section" style={{ marginTop: '2rem' }}>
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                        <Link2 size={18} />
                        Related Papers ({nodeDetails.related_papers.length})
                      </h3>
                      
                      <div className="related-papers">
                        {nodeDetails.related_papers.map((paper, index) => {
                          const similarity = paper.similarity || 0.5
                          const colorData = getColorFromSimilarity(similarity)
                          
                          return (
                            <motion.div
                              key={paper.id || index}
                              className="related-paper"
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: index * 0.1 }}
                              onClick={() => onRelatedNodeClick && onRelatedNodeClick(paper.id, 'click')}
                              whileHover={{ scale: 1.02, y: -4 }}
                              whileTap={{ scale: 0.98 }}
                              style={{
                                cursor: 'pointer',
                                padding: '1.25rem',
                                marginBottom: '1rem',
                                background: colorData.bgColor,
                                border: `2px solid ${colorData.borderColor}`,
                                borderRadius: '0.75rem',
                                borderLeft: `4px solid ${colorData.color}`,
                                transition: 'all 0.3s ease'
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '0.75rem' }}>
                                <h4 style={{ flex: 1, marginRight: '1rem', fontSize: '1rem', fontWeight: '600' }}>
                                  {paper.title}
                                </h4>
                                <div style={{
                                  padding: '0.25rem 0.75rem',
                                  borderRadius: '1rem',
                                  background: colorData.gradient,
                                  color: 'white',
                                  fontSize: '0.75rem',
                                  fontWeight: '600',
                                  whiteSpace: 'nowrap'
                                }}>
                                  {Math.round(similarity * 100)}%
                                </div>
                              </div>

                              <p className="paper-meta" style={{ fontSize: '0.85rem', color: '#94A3B8', marginBottom: '0.75rem' }}>
                                {paper.authors && (
                                  <span>{paper.authors.split(';')[0]} et al.</span>
                                )}
                                {paper.published_date && (
                                  <span> • {paper.published_date}</span>
                                )}
                              </p>

                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <div style={{
                                  flex: 1,
                                  height: '4px',
                                  background: 'rgba(100, 116, 139, 0.2)',
                                  borderRadius: '1rem',
                                  overflow: 'hidden'
                                }}>
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${similarity * 100}%` }}
                                    transition={{ duration: 1, delay: index * 0.1 }}
                                    style={{
                                      height: '100%',
                                      background: colorData.gradient
                                    }}
                                  />
                                </div>
                                <span style={{
                                  fontSize: '0.75rem',
                                  fontWeight: '600',
                                  color: colorData.color
                                }}>
                                  {colorData.label}
                                </span>
                              </div>
                            </motion.div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Co-authors */}
                  {nodeDetails.co_authors && nodeDetails.co_authors.length > 0 && (
                    <div className="content-section" style={{ marginTop: '2rem' }}>
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                        <Users size={18} />
                        Co-authors ({nodeDetails.co_authors.length})
                      </h3>
                      
                      <div className="author-network" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
                        {nodeDetails.co_authors.map((author, index) => (
                          <motion.div
                            key={index}
                            className="author-card"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.05 }}
                            whileHover={{ scale: 1.05 }}
                            style={{
                              padding: '1rem',
                              background: 'rgba(59, 130, 246, 0.1)',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                              borderRadius: '0.75rem',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.75rem'
                            }}
                          >
                            <div className="author-avatar" style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '50%',
                              background: 'linear-gradient(135deg, #3B82F6, #60A5FA)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white'
                            }}>
                              <Users size={20} />
                            </div>
                            <div className="author-info">
                              <h4 style={{ fontSize: '0.9rem', fontWeight: '600', marginBottom: '0.25rem' }}>
                                {author.name}
                              </h4>
                              <p style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
                                {author.papers_count || 0} papers
                              </p>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Empty State */}
                  {(!nodeDetails.edge_weights || nodeDetails.edge_weights.length === 0) &&
                   (!nodeDetails.related_papers || nodeDetails.related_papers.length === 0) &&
                   (!nodeDetails.co_authors || nodeDetails.co_authors.length === 0) && (
                    <div className="empty-state" style={{
                      textAlign: 'center',
                      padding: '4rem 2rem',
                      color: '#64748B'
                    }}>
                      <Users size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                      <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                        No Connections Yet
                      </h3>
                      <p style={{ fontSize: '0.9rem' }}>
                        Double-click this node to find similar papers and build connections.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        ) : (
          <div className="inspector-error">
            <HelpCircle size={48} />
            <h3>No details available</h3>
            <p>Unable to load paper details at this time.</p>
          </div>
        )}
      </div>
    </motion.div>
  )
}

export default NodeInspector
