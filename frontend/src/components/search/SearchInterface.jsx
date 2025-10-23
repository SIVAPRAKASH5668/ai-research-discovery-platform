import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  Search,
  Zap,
  Globe,
  Brain,
  Microscope,
  Sparkles,
  BookOpen,
  TrendingUp,
  Clock,
  Filter
} from 'lucide-react'

const SearchInterface = ({
  onSearch,
  searchInputRef,
  recentSearches = [],
  popularTopics = []
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const heroRef = useRef(null)

  // ⭐ FORCE SCROLL TO HERO ON MOUNT
  useEffect(() => {
    if (heroRef.current) {
      heroRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [])

  const handleSearch = (query) => {
    if (query.trim()) {
      onSearch(query.trim())
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    handleSearch(searchQuery)
  }

  const quickTopics = [
    { text: 'Machine Learning', icon: Brain, color: 'from-blue-500 to-indigo-600' },
    { text: 'Healthcare AI', icon: BookOpen, color: 'from-green-500 to-emerald-600' },
    { text: 'Computer Vision', icon: Sparkles, color: 'from-purple-500 to-violet-600' },
    { text: 'Climate Science', icon: Globe, color: 'from-orange-500 to-red-600' },
    { text: 'Quantum Computing', icon: Zap, color: 'from-teal-500 to-cyan-600' },
    { text: 'Robotics', icon: Microscope, color: 'from-pink-500 to-rose-600' }
  ]

  return (
    <div style={{
      width: '100%',
      height: '100%',
      overflowY: 'auto',
      overflowX: 'hidden'
    }}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        {/* HERO SECTION - FULL SCREEN */}
        <div 
          ref={heroRef}
          style={{
            position: 'relative',
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '100px 20px 60px',
            overflow: 'hidden'
          }}
        >
          {/* Title & Description - CENTERED */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
            style={{
              position: 'relative',
              zIndex: 10,
              textAlign: 'center',
              maxWidth: '900px',
              width: '100%'
            }}
          >
            {/* BIG TITLE */}
            <h1 style={{
              fontSize: '60px',
              fontWeight: '900',
              lineHeight: '1.1',
              marginBottom: '25px',
              color: '#ffffff',
              textShadow: '0 10px 50px rgba(0, 0, 0, 1)',
              letterSpacing: '-2px'
            }}>
              Discover Research
            </h1>
            
            <h1 style={{
              fontSize: '60px',
              fontWeight: '900',
              lineHeight: '1.1',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              marginBottom: '40px',
              filter: 'drop-shadow(0 0 50px rgba(139, 92, 246, 1))',
              letterSpacing: '-2px'
            }}>
              Connections
            </h1>

            {/* Description */}
            <p style={{
              fontSize: '20px',
              lineHeight: '1.7',
              color: 'rgba(255, 255, 255, 0.95)',
              maxWidth: '750px',
              margin: '0 auto',
              textShadow: '0 4px 25px rgba(0, 0, 0, 1)',
              fontWeight: '400'
            }}>
              Explore the interconnected world of scientific research with AI-powered discovery.
              Uncover hidden patterns, relationships, and insights across millions of papers.
            </p>
          </motion.div>

          {/* Neural Network Background */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '100%',
            maxWidth: '1400px',
            height: '900px',
            zIndex: 0,
            pointerEvents: 'none',
            opacity: 0.05
          }}>
            {[...Array(15)].map((_, i) => (
              <motion.div
                key={i}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{
                  delay: 0.3 + i * 0.08,
                  duration: 0.6
                }}
                style={{
                  position: 'absolute',
                  width: '18px',
                  height: '18px',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  borderRadius: '50%',
                  boxShadow: '0 0 40px rgba(102, 126, 234, 0.9)',
                  top: `${Math.random() * 100}%`,
                  left: `${Math.random() * 100}%`
                }}
              />
            ))}
          </div>
        </div>

        {/* Quick Topics */}
        <motion.div
          className="quick-topics-section"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          style={{
            marginBottom: '80px',
            width: '100%',
            maxWidth: '1200px',
            padding: '0 20px',
            margin: '0 auto 80px'
          }}
        >
          <h3 style={{
            fontSize: '1.5rem',
            fontWeight: '600',
            color: 'rgba(255, 255, 255, 0.9)',
            marginBottom: '40px',
            textAlign: 'center'
          }}>Quick Start Topics</h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '20px',
            width: '100%'
          }}>
            {quickTopics.map((topic, index) => (
              <motion.button
                key={topic.text}
                className={`topic-card bg-gradient-to-r ${topic.color}`}
                onClick={() => handleSearch(topic.text)}
                whileHover={{ scale: 1.05, y: -5 }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9 + index * 0.1 }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '16px 20px',
                  border: 'none',
                  borderRadius: '12px',
                  color: 'white',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '15px',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
                }}
              >
                <topic.icon size={24} />
                <span>{topic.text}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Categories */}
        <motion.div
          className="categories-section"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.1 }}
          style={{
            marginBottom: '80px',
            width: '100%',
            maxWidth: '1200px',
            padding: '0 20px',
            margin: '0 auto 80px'
          }}
        >
          <h3 style={{
            fontSize: '1.5rem',
            fontWeight: '600',
            color: 'rgba(255, 255, 255, 0.9)',
            marginBottom: '40px',
            textAlign: 'center'
          }}>Popular Research Areas</h3>
          <div className="categories-grid">
            {popularTopics.map((category, index) => (
              <motion.div
                key={category}
                className="category-card"
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleSearch(category)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.3 + index * 0.05 }}
              >
                <div className="category-icon">
                  <TrendingUp size={20} />
                </div>
                <h4>{category}</h4>
                <p>Explore cutting-edge research in {category.toLowerCase()}</p>
                <div className="category-stats">
                  <span>{Math.floor(Math.random() * 10000) + 1000}+ papers</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Recent Searches */}
        {recentSearches.length > 0 && (
          <motion.div
            className="recent-searches-section"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.5 }}
            style={{
              width: '100%',
              maxWidth: '1200px',
              padding: '0 20px',
              margin: '0 auto 60px'
            }}
          >
            <h3>Recent Searches</h3>
            <div className="recent-searches">
              {recentSearches.map((search, index) => (
                <motion.button
                  key={index}
                  className="recent-search-item"
                  onClick={() => handleSearch(search)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Clock size={16} />
                  {search}
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  )
}

export default SearchInterface
