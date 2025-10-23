import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  Settings,
  Cloud,
  Globe,
  Zap,
  Menu,
  X,
  Play,
  Square,
  RefreshCw
} from 'lucide-react'
import SearchAutocomplete from '../search/SearchAutocomplete'

const TopBar = ({
  onSearch,
  searchQuery,
  onSearchQueryChange,
  searchInputRef,
  loading,
  graphMode,
  onGraphModeChange,
  hasData,
  onToggleControls,
  onToggleCloud,
  showControls,
  showCloud,
  languageSwitcher
}) => {
  const [showMobileMenu, setShowMobileMenu] = useState(false)
  const [showAutocomplete, setShowAutocomplete] = useState(false)
  const searchContainerRef = useRef(null)
  const [suggestions] = useState([
    { text: 'Machine Learning in Healthcare', icon: Zap, category: 'AI' },
    { text: 'Computer Vision Applications', icon: Globe, category: 'CV' },
    { text: 'Natural Language Processing', icon: Zap, category: 'NLP' },
    { text: 'Climate Change Research', icon: Globe, category: 'Environment' },
    { text: 'Quantum Computing', icon: Zap, category: 'Quantum' },
    { text: 'Deep Learning', icon: Zap, category: 'AI' }
  ])

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      onSearch(searchQuery.trim())
      setShowAutocomplete(false)
    }
  }

  const handleSearchChange = (value) => {
    onSearchQueryChange(value)
    setShowAutocomplete(value.length > 0)
  }

  const handleSuggestionSelect = (suggestion) => {
    onSearchQueryChange(suggestion)
    onSearch(suggestion)
    setShowAutocomplete(false)
  }

  // Close autocomplete when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setShowAutocomplete(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <motion.header
      className="top-bar"
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100 }}
    >
      <div className="top-bar-content">
        {/* Logo/Brand */}
        <motion.div
          className="brand"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <div className="brand-icon">
            <motion.div
              className="neural-network-mini"
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <div className="neural-node" />
              <div className="neural-node" />
              <div className="neural-node" />
            </motion.div>
          </div>
          <div className="brand-text">
            <h1>Research<span>Graph</span></h1>
            <span className="brand-subtitle">AI Discovery Platform</span>
          </div>
        </motion.div>

        {/* Search Container */}
        <div className="search-container" ref={searchContainerRef} style={{ position: 'relative' }}>
          <form onSubmit={handleSearchSubmit} className="search-form">
            <div className="search-input-wrapper">
              <Search size={20} className="search-icon" />
              <input
                ref={searchInputRef}
                type="text"
                placeholder="Discover research connections..."
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                onFocus={() => setShowAutocomplete(searchQuery.length > 0)}
                className="search-input"
                disabled={loading}
              />
              {loading && (
                <motion.div
                  className="search-loading"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                >
                  <RefreshCw size={16} />
                </motion.div>
              )}
              <motion.button
                type="submit"
                className="search-btn"
                disabled={loading || !searchQuery.trim()}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Zap size={16} />
              </motion.button>
            </div>
          </form>

          {/* Autocomplete - Positioned Absolutely */}
          <AnimatePresence>
            {showAutocomplete && (
              <div style={{ 
                position: 'absolute', 
                top: '100%', 
                left: 0, 
                right: 0, 
                zIndex: 1000,
                marginTop: '8px'
              }}>
                <SearchAutocomplete
                  query={searchQuery}
                  suggestions={suggestions}
                  onSelect={handleSuggestionSelect}
                />
              </div>
            )}
          </AnimatePresence>
        </div>

        {/* Navigation Controls */}
        <div className="nav-actions">
          {/* ✅ LANGUAGE SWITCHER */}
          {languageSwitcher && (
            <div className="nav-item">
              {languageSwitcher}
            </div>
          )}

          {/* View Mode Toggle */}
          {hasData && (
            <div className="view-toggle">
              <motion.button
                className={`mode-btn ${graphMode === '2d' ? 'active' : ''}`}
                onClick={() => onGraphModeChange('2d')}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="2D View"
              >
                <Square size={16} />
                <span>2D</span>
              </motion.button>
              <motion.button
                className={`mode-btn ${graphMode === '3d' ? 'active' : ''}`}
                onClick={() => onGraphModeChange('3d')}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="3D View"
              >
                <Play size={16} />
                <span>3D</span>
              </motion.button>
            </div>
          )}

          {/* Panel Toggles */}
          {hasData && (
            <>
              {/* ✅ CLOUD BUTTON (REPLACES INSPECTOR) */}
              <motion.button
                className={`nav-btn ${showCloud ? 'active' : ''}`}
                onClick={onToggleCloud}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Toggle Cloud Dashboard"
              >
                <Cloud size={18} />
              </motion.button>

              <motion.button
                className={`nav-btn ${showControls ? 'active' : ''}`}
                onClick={onToggleControls}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Toggle Controls"
              >
                <Settings size={18} />
              </motion.button>
            </>
          )}

          {/* Mobile Menu Toggle */}
          <motion.button
            className="mobile-menu-btn"
            onClick={() => setShowMobileMenu(!showMobileMenu)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {showMobileMenu ? <X size={20} /> : <Menu size={20} />}
          </motion.button>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {showMobileMenu && (
            <motion.div
              className="mobile-menu"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="mobile-menu-content">
                {hasData && (
                  <>
                    <div className="mobile-section">
                      <h4>View Mode</h4>
                      <div className="mobile-buttons">
                        <button
                          className={`mobile-btn ${graphMode === '2d' ? 'active' : ''}`}
                          onClick={() => {
                            onGraphModeChange('2d')
                            setShowMobileMenu(false)
                          }}
                        >
                          2D View
                        </button>
                        <button
                          className={`mobile-btn ${graphMode === '3d' ? 'active' : ''}`}
                          onClick={() => {
                            onGraphModeChange('3d')
                            setShowMobileMenu(false)
                          }}
                        >
                          3D View
                        </button>
                      </div>
                    </div>

                    <div className="mobile-section">
                      <h4>Panels</h4>
                      <div className="mobile-buttons">
                        <button
                          className={`mobile-btn ${showCloud ? 'active' : ''}`}
                          onClick={() => {
                            onToggleCloud()
                            setShowMobileMenu(false)
                          }}
                        >
                          Cloud Dashboard
                        </button>
                        <button
                          className={`mobile-btn ${showControls ? 'active' : ''}`}
                          onClick={() => {
                            onToggleControls()
                            setShowMobileMenu(false)
                          }}
                        >
                          Controls
                        </button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Status Indicator */}
      <div className="status-indicator">
        <motion.div
          className={`status-dot ${loading ? 'loading' : hasData ? 'active' : 'idle'}`}
          animate={loading ? { scale: [1, 1.2, 1] } : {}}
          transition={{ duration: 1, repeat: Infinity }}
        />
        <span className="status-text">
          {loading ? 'Processing...' : hasData ? 'Connected' : 'Ready'}
        </span>
      </div>
    </motion.header>
  )
}

export default TopBar
