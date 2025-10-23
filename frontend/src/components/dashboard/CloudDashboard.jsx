/**
 * ==================================================================================
 * CLOUD STORAGE DASHBOARD - SLIDES IN FROM RIGHT (MIRROR OF NODE INSPECTOR)
 * ==================================================================================
 */

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Cloud,
  Database,
  Activity,
  Trash2,
  RefreshCw,
  Zap,
  HardDrive,
  Clock,
  AlertCircle,
  Save,
  Loader2,
  BookOpen
} from 'lucide-react'
import { cloudAPI } from '../../utils/api'
import { toast } from 'react-hot-toast'

const CloudDashboard = ({ isOpen, onClose, refreshTrigger }) => {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [stats, setStats] = useState({
    totalPapers: 0,
    totalNodes: 0,
    storageUsed: 0,
    lastSync: null,
    elasticsearchHealth: 'unknown',
    vertexAIStatus: 'unknown'
  })

  const [papers, setPapers] = useState([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  const isMountedRef = useRef(true)
  const fetchTimeoutRef = useRef(null)
  const autoRefreshIntervalRef = useRef(null)

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'saved', label: 'Saved Papers', icon: BookOpen },
    { id: 'all', label: 'All Papers', icon: Database }
  ]

  // ✅ AUTO-REFRESH EVERY 5 SECONDS
  useEffect(() => {
    isMountedRef.current = true

    if (isOpen) {
      fetchCloudData()
      
      autoRefreshIntervalRef.current = setInterval(() => {
        console.log('🔄 Auto-refreshing cloud data...')
        fetchCloudData()
      }, 5000)
    }

    return () => {
      isMountedRef.current = false
      
      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current)
      }
      
      if (autoRefreshIntervalRef.current) {
        clearInterval(autoRefreshIntervalRef.current)
      }
    }
  }, [isOpen, activeTab, refreshTrigger])

  const fetchCloudData = async () => {
    if (refreshing) {
      console.log('⏭️ Skipping fetch - already in progress')
      return
    }

    try {
      setRefreshing(true)
      setError(null)

      const timeoutPromise = new Promise((_, reject) => {
        fetchTimeoutRef.current = setTimeout(() => {
          reject(new Error('Request timeout'))
        }, 15000)
      })

      const statsPromise = cloudAPI.getStats()

      let papersPromise
      if (activeTab === 'saved') {
        papersPromise = cloudAPI.getSavedPapers({ limit: 50 })
      } else if (activeTab === 'all') {
        papersPromise = cloudAPI.getAllPapers({ limit: 50 })
      } else {
        papersPromise = Promise.resolve({ papers: [] })
      }

      const fetchPromise = Promise.all([statsPromise, papersPromise])
      const [statsResponse, papersResponse] = await Promise.race([
        fetchPromise,
        timeoutPromise
      ])

      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current)
      }

      if (isMountedRef.current) {
        setStats(statsResponse.stats || stats)
        setPapers(papersResponse.papers || [])
        console.log(`✅ Cloud data refreshed (${activeTab} tab)`)
      }
    } catch (error) {
      console.error('❌ Failed to fetch cloud data:', error)
      
      if (isMountedRef.current) {
        setError(error.message)
      }
    } finally {
      if (isMountedRef.current) {
        setRefreshing(false)
      }
      
      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current)
        fetchTimeoutRef.current = null
      }
    }
  }

  const handleDeletePaper = async (paperId) => {
    try {
      setLoading(true)
      await cloudAPI.deletePaper(paperId)
      await fetchCloudData()
    } catch (error) {
      console.error('❌ Delete failed:', error)
      setError('Failed to delete paper')
    } finally {
      setLoading(false)
    }
  }

  const handleSavePaper = async (paper) => {
    try {
      setLoading(true)
      await cloudAPI.savePaper(paper)
      await fetchCloudData()
    } catch (error) {
      console.error('❌ Save failed:', error)
      setError('Failed to save paper')
    } finally {
      setLoading(false)
    }
  }

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  const formatTimeAgo = (isoString) => {
    if (!isoString) return 'Never'
    const seconds = Math.floor((new Date() - new Date(isoString)) / 1000)
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
    return `${Math.floor(seconds / 86400)}d ago`
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* ✅ SAME BACKDROP AS NODE INSPECTOR */}
          <motion.div
            className="inspector-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* ✅ SLIDES FROM RIGHT (400 → 0) - MIRROR OF NODE INSPECTOR */}
          <motion.div
            className="node-inspector cloud-dashboard-right"
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
          >
            {/* Header - SAME AS NODE INSPECTOR */}
            <div className="inspector-header">
              <div className="header-title">
                <Cloud size={20} />
                <h2>Cloud Storage</h2>
                <span 
                  className={`status-dot ${stats.elasticsearchHealth}`}
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    display: 'inline-block',
                    marginLeft: '8px',
                    background: stats.elasticsearchHealth === 'green' ? '#10b981' : 
                                stats.elasticsearchHealth === 'yellow' ? '#f59e0b' : '#ef4444'
                  }}
                />
              </div>
              <div className="header-actions">
                <motion.button
                  className="action-btn"
                  onClick={fetchCloudData}
                  disabled={refreshing}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  title="Refresh"
                >
                  <RefreshCw 
                    size={18} 
                    className={refreshing ? 'spinner' : ''}
                  />
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

            {/* Tab Navigation - SAME AS NODE INSPECTOR */}
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
                  {tab.id === 'saved' && ` (${stats.totalNodes})`}
                  {tab.id === 'all' && ` (${stats.totalPapers})`}
                </motion.button>
              ))}
            </div>

            {/* Content - SAME AS NODE INSPECTOR */}
            <div className="inspector-content">
              {error && (
                <div className="error-banner" style={{
                  padding: '1rem',
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  borderRadius: '0.5rem',
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  color: '#EF4444'
                }}>
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              )}

              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  {/* ===== DASHBOARD TAB ===== */}
                  {activeTab === 'dashboard' && (
                    <div className="tab-content">
                      <div className="content-section">
                        <h3>Storage Statistics</h3>
                        <div className="stats-grid" style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(2, 1fr)',
                          gap: '1rem',
                          marginTop: '1rem'
                        }}>
                          <motion.div 
                            className="stat-card"
                            whileHover={{ scale: 1.02 }}
                            style={{
                              padding: '1.25rem',
                              background: 'rgba(59, 130, 246, 0.1)',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                              borderRadius: '0.75rem',
                              display: 'flex',
                              gap: '1rem',
                              alignItems: 'center'
                            }}
                          >
                            <div className="stat-icon" style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '0.5rem',
                              background: 'linear-gradient(135deg, #3B82F6, #60A5FA)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white'
                            }}>
                              <Database size={20} />
                            </div>
                            <div className="stat-info">
                              <span className="stat-value" style={{
                                display: 'block',
                                fontSize: '1.5rem',
                                fontWeight: '700',
                                color: '#3B82F6'
                              }}>{stats.totalPapers}</span>
                              <span className="stat-label" style={{
                                fontSize: '0.75rem',
                                color: '#94A3B8'
                              }}>Total Papers</span>
                            </div>
                          </motion.div>

                          <motion.div 
                            className="stat-card"
                            whileHover={{ scale: 1.02 }}
                            style={{
                              padding: '1.25rem',
                              background: 'rgba(16, 185, 129, 0.1)',
                              border: '1px solid rgba(16, 185, 129, 0.3)',
                              borderRadius: '0.75rem',
                              display: 'flex',
                              gap: '1rem',
                              alignItems: 'center'
                            }}
                          >
                            <div className="stat-icon" style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '0.5rem',
                              background: 'linear-gradient(135deg, #10B981, #34D399)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white'
                            }}>
                              <BookOpen size={20} />
                            </div>
                            <div className="stat-info">
                              <span className="stat-value" style={{
                                display: 'block',
                                fontSize: '1.5rem',
                                fontWeight: '700',
                                color: '#10B981'
                              }}>{stats.totalNodes}</span>
                              <span className="stat-label" style={{
                                fontSize: '0.75rem',
                                color: '#94A3B8'
                              }}>Saved Papers</span>
                            </div>
                          </motion.div>

                          <motion.div 
                            className="stat-card"
                            whileHover={{ scale: 1.02 }}
                            style={{
                              padding: '1.25rem',
                              background: 'rgba(245, 158, 11, 0.1)',
                              border: '1px solid rgba(245, 158, 11, 0.3)',
                              borderRadius: '0.75rem',
                              display: 'flex',
                              gap: '1rem',
                              alignItems: 'center'
                            }}
                          >
                            <div className="stat-icon" style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '0.5rem',
                              background: 'linear-gradient(135deg, #F59E0B, #FBBF24)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white'
                            }}>
                              <HardDrive size={20} />
                            </div>
                            <div className="stat-info">
                              <span className="stat-value" style={{
                                display: 'block',
                                fontSize: '1.25rem',
                                fontWeight: '700',
                                color: '#F59E0B'
                              }}>{formatBytes(stats.storageUsed)}</span>
                              <span className="stat-label" style={{
                                fontSize: '0.75rem',
                                color: '#94A3B8'
                              }}>Storage Used</span>
                            </div>
                          </motion.div>

                          <motion.div 
                            className="stat-card"
                            whileHover={{ scale: 1.02 }}
                            style={{
                              padding: '1.25rem',
                              background: 'rgba(139, 92, 246, 0.1)',
                              border: '1px solid rgba(139, 92, 246, 0.3)',
                              borderRadius: '0.75rem',
                              display: 'flex',
                              gap: '1rem',
                              alignItems: 'center'
                            }}
                          >
                            <div className="stat-icon" style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '0.5rem',
                              background: 'linear-gradient(135deg, #8B5CF6, #A78BFA)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'white'
                            }}>
                              <Clock size={20} />
                            </div>
                            <div className="stat-info">
                              <span className="stat-value" style={{
                                display: 'block',
                                fontSize: '0.9rem',
                                fontWeight: '700',
                                color: '#8B5CF6'
                              }}>{formatTimeAgo(stats.lastSync)}</span>
                              <span className="stat-label" style={{
                                fontSize: '0.75rem',
                                color: '#94A3B8'
                              }}>Last Sync</span>
                            </div>
                          </motion.div>
                        </div>
                      </div>

                      {/* System Health */}
                      <div className="content-section">
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                          <Activity size={18} />
                          System Health
                        </h3>
                        <div className="health-items" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                          <div className="health-item" style={{
                            padding: '1rem',
                            background: 'rgba(59, 130, 246, 0.05)',
                            border: '1px solid rgba(59, 130, 246, 0.2)',
                            borderRadius: '0.5rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.75rem'
                          }}>
                            <Zap size={16} style={{ color: '#3B82F6' }} />
                            <span style={{ flex: 1, fontWeight: '500' }}>Elasticsearch</span>
                            <span style={{
                              color: stats.elasticsearchHealth === 'green' ? '#10b981' : 
                                     stats.elasticsearchHealth === 'yellow' ? '#f59e0b' : '#ef4444',
                              fontWeight: '600'
                            }}>
                              {stats.elasticsearchHealth === 'green' ? '● Healthy' : 
                               stats.elasticsearchHealth === 'yellow' ? '● Warning' : '● Critical'}
                            </span>
                          </div>
                          <div className="health-item" style={{
                            padding: '1rem',
                            background: 'rgba(139, 92, 246, 0.05)',
                            border: '1px solid rgba(139, 92, 246, 0.2)',
                            borderRadius: '0.5rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.75rem'
                          }}>
                            <Zap size={16} style={{ color: '#8B5CF6' }} />
                            <span style={{ flex: 1, fontWeight: '500' }}>Vertex AI</span>
                            <span style={{
                              color: stats.vertexAIStatus === 'online' ? '#10b981' : '#ef4444',
                              fontWeight: '600'
                            }}>
                              {stats.vertexAIStatus === 'online' ? '● Online' : '● Offline'}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Quick Actions */}
                      <div className="content-section">
                        <h3>Quick Actions</h3>
                        <div className="action-links" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                          <motion.button
                            onClick={() => setActiveTab('saved')}
                            className="action-link primary"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            style={{
                              padding: '1rem',
                              background: 'linear-gradient(135deg, #3B82F6, #60A5FA)',
                              border: 'none',
                              borderRadius: '0.5rem',
                              color: 'white',
                              fontWeight: '600',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              cursor: 'pointer'
                            }}
                          >
                            <BookOpen size={16} />
                            View Saved Papers
                          </motion.button>
                          <motion.button
                            onClick={() => setActiveTab('all')}
                            className="action-link"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            style={{
                              padding: '1rem',
                              background: 'rgba(59, 130, 246, 0.1)',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                              borderRadius: '0.5rem',
                              color: '#3B82F6',
                              fontWeight: '600',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              cursor: 'pointer'
                            }}
                          >
                            <Database size={16} />
                            Browse All Papers
                          </motion.button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ===== SAVED PAPERS TAB ===== */}
                  {activeTab === 'saved' && (
                    <div className="tab-content">
                      <div className="content-section">
                        <h3>Your Saved Papers ({stats.totalNodes})</h3>
                        {refreshing && papers.length === 0 ? (
                          <div className="loading-state" style={{
                            textAlign: 'center',
                            padding: '3rem 1rem',
                            color: '#64748B'
                          }}>
                            <motion.div
                              animate={{ rotate: 360 }}
                              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            >
                              <Loader2 size={32} style={{ marginBottom: '1rem' }} />
                            </motion.div>
                            <p>Loading saved papers...</p>
                          </div>
                        ) : papers.length === 0 ? (
                          <div className="empty-state" style={{
                            textAlign: 'center',
                            padding: '3rem 1rem',
                            color: '#64748B'
                          }}>
                            <Cloud size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                            <h3 style={{ marginBottom: '0.5rem' }}>No Saved Papers</h3>
                            <p>Save papers from the inspector to see them here</p>
                          </div>
                        ) : (
                          <div className="papers-list" style={{ marginTop: '1rem' }}>
                            {papers.map((paper, index) => (
                              <motion.div
                                key={paper.id}
                                className="paper-item"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.02 }}
                                style={{
                                  padding: '1rem',
                                  marginBottom: '0.75rem',
                                  background: 'rgba(59, 130, 246, 0.05)',
                                  border: '1px solid rgba(59, 130, 246, 0.2)',
                                  borderRadius: '0.75rem',
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'start'
                                }}
                              >
                                <div className="paper-info" style={{ flex: 1 }}>
                                  <h4 style={{
                                    fontSize: '0.95rem',
                                    fontWeight: '600',
                                    marginBottom: '0.5rem',
                                    color: '#E2E8F0'
                                  }}>
                                    {paper.title || 'Untitled'}
                                  </h4>
                                  <p className="paper-meta" style={{
                                    fontSize: '0.8rem',
                                    color: '#94A3B8',
                                    marginBottom: '0.25rem'
                                  }}>
                                    {Array.isArray(paper.authors) 
                                      ? (paper.authors[0] || 'Unknown')
                                      : (paper.authors?.split(';')[0] || 'Unknown')
                                    } • {paper.publication_year || paper.year || 'N/A'}
                                  </p>
                                  {paper.saved_at && (
                                    <span className="saved-time" style={{
                                      fontSize: '0.7rem',
                                      color: '#64748B'
                                    }}>
                                      {formatTimeAgo(paper.saved_at)}
                                    </span>
                                  )}
                                </div>
                                <div className="paper-actions">
                                  <motion.button
                                    className="delete-btn"
                                    onClick={() => handleDeletePaper(paper.id)}
                                    disabled={loading}
                                    whileHover={{ scale: 1.1 }}
                                    whileTap={{ scale: 0.9 }}
                                    title="Delete from cloud"
                                    style={{
                                      padding: '0.5rem',
                                      background: 'rgba(239, 68, 68, 0.1)',
                                      border: '1px solid rgba(239, 68, 68, 0.3)',
                                      borderRadius: '0.375rem',
                                      color: '#EF4444',
                                      cursor: 'pointer'
                                    }}
                                  >
                                    <Trash2 size={16} />
                                  </motion.button>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* ===== ALL PAPERS TAB ===== */}
                  {activeTab === 'all' && (
                    <div className="tab-content">
                      <div className="content-section">
                        <h3>All Papers in Database ({stats.totalPapers})</h3>
                        {refreshing && papers.length === 0 ? (
                          <div className="loading-state" style={{
                            textAlign: 'center',
                            padding: '3rem 1rem',
                            color: '#64748B'
                          }}>
                            <motion.div
                              animate={{ rotate: 360 }}
                              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            >
                              <Loader2 size={32} style={{ marginBottom: '1rem' }} />
                            </motion.div>
                            <p>Loading all papers...</p>
                          </div>
                        ) : papers.length === 0 ? (
                          <div className="empty-state" style={{
                            textAlign: 'center',
                            padding: '3rem 1rem',
                            color: '#64748B'
                          }}>
                            <Database size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                            <h3 style={{ marginBottom: '0.5rem' }}>No Papers Found</h3>
                            <p>Search for papers to populate the database</p>
                          </div>
                        ) : (
                          <div className="papers-list" style={{ marginTop: '1rem' }}>
                            {papers.map((paper, index) => (
                              <motion.div
                                key={paper.id}
                                className="paper-item"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.02 }}
                                style={{
                                  padding: '1rem',
                                  marginBottom: '0.75rem',
                                  background: 'rgba(59, 130, 246, 0.05)',
                                  border: '1px solid rgba(59, 130, 246, 0.2)',
                                  borderRadius: '0.75rem',
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'start'
                                }}
                              >
                                <div className="paper-info" style={{ flex: 1 }}>
                                  <h4 style={{
                                    fontSize: '0.95rem',
                                    fontWeight: '600',
                                    marginBottom: '0.5rem',
                                    color: '#E2E8F0'
                                  }}>
                                    {paper.title || 'Untitled'}
                                  </h4>
                                  <p className="paper-meta" style={{
                                    fontSize: '0.8rem',
                                    color: '#94A3B8',
                                    marginBottom: '0.25rem'
                                  }}>
                                    {Array.isArray(paper.authors) 
                                      ? (paper.authors[0] || 'Unknown')
                                      : (paper.authors?.split(';')[0] || 'Unknown')
                                    } • {paper.publication_year || paper.year || 'N/A'}
                                  </p>
                                  {paper.citation_count && (
                                    <span className="citations" style={{
                                      fontSize: '0.7rem',
                                      color: '#64748B'
                                    }}>
                                      {paper.citation_count} citations
                                    </span>
                                  )}
                                </div>
                                <div className="paper-actions">
                                  {!paper.is_saved && (
                                    <motion.button
                                      className="save-btn"
                                      onClick={() => handleSavePaper(paper)}
                                      disabled={loading}
                                      whileHover={{ scale: 1.1 }}
                                      whileTap={{ scale: 0.9 }}
                                      title="Save to cloud"
                                      style={{
                                        padding: '0.5rem',
                                        background: 'rgba(16, 185, 129, 0.1)',
                                        border: '1px solid rgba(16, 185, 129, 0.3)',
                                        borderRadius: '0.375rem',
                                        color: '#10B981',
                                        cursor: 'pointer'
                                      }}
                                    >
                                      <Save size={16} />
                                    </motion.button>
                                  )}
                                  {paper.is_saved && (
                                    <motion.button
                                      className="delete-btn"
                                      onClick={() => handleDeletePaper(paper.id)}
                                      disabled={loading}
                                      whileHover={{ scale: 1.1 }}
                                      whileTap={{ scale: 0.9 }}
                                      title="Delete from cloud"
                                      style={{
                                        padding: '0.5rem',
                                        background: 'rgba(239, 68, 68, 0.1)',
                                        border: '1px solid rgba(239, 68, 68, 0.3)',
                                        borderRadius: '0.375rem',
                                        color: '#EF4444',
                                        cursor: 'pointer'
                                      }}
                                    >
                                      <Trash2 size={16} />
                                    </motion.button>
                                  )}
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.div>

          <style jsx>{`
            /* ✅ RIGHT-SIDE POSITIONING */
            .cloud-dashboard-right {
              right: 0 !important;
              left: auto !important;
            }
            
            @keyframes spin {
              to {
                transform: rotate(360deg);
              }
            }
          `}</style>
        </>
      )}
    </AnimatePresence>
  )
}

export default CloudDashboard
