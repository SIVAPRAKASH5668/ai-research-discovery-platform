import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Database, Network, Zap, Globe, Search } from 'lucide-react'

const LoadingScreen = ({ message = "Processing...", progress = 0 }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [timeElapsed, setTimeElapsed] = useState(0)

  // ✅ ADJUSTED TO 30 SECONDS TOTAL
  const steps = [
    { 
      icon: Globe, 
      text: "Translating query to 15 languages...", 
      duration: 3,  // Increased from 2s
      color: "#06B6D4"
    },
    { 
      icon: Search, 
      text: "Fetching papers from arXiv, Crossref, Semantic Scholar, DOAJ, and others...", 
      duration: 10,  // Increased from 8s
      color: "#3B82F6"
    },
    { 
      icon: Database, 
      text: "Generating embeddings with Vertex AI...", 
      duration: 6,  // Increased from 5s
      color: "#8B5CF6"
    },
    { 
      icon: Database, 
      text: "Indexing papers to Elasticsearch...", 
      duration: 5,  // Increased from 4s
      color: "#10B981"
    },
    { 
      icon: Brain, 
      text: "Searching Elasticsearch with hybrid KNN...", 
      duration: 2,  // Same
      color: "#F59E0B"
    },
    { 
      icon: Network, 
      text: "Calculating similarity edges between papers...", 
      duration: 3,  // Same
      color: "#EC4899"
    },
    { 
      icon: Zap, 
      text: "Building final knowledge graph...", 
      duration: 1,  // Same
      color: "#EF4444"
    }
  ]

  // Total expected time: 30 seconds (3+10+6+5+2+3+1)
  const TOTAL_DURATION = steps.reduce((sum, step) => sum + step.duration, 0)

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeElapsed(prev => {
        const newElapsed = prev + 1
        
        // Update step based on elapsed time
        let totalDuration = 0
        for (let i = 0; i < steps.length; i++) {
          totalDuration += steps[i].duration
          if (newElapsed <= totalDuration) {
            setCurrentStep(i)
            break
          }
        }
        
        // Cap at last step
        if (newElapsed > TOTAL_DURATION) {
          setCurrentStep(steps.length - 1)
        }
        
        return newElapsed
      })
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const calculateProgress = () => {
    if (progress > 0) return progress
    // Calculate based on real total time (30 seconds = 100%)
    return Math.min(95, (timeElapsed / TOTAL_DURATION) * 100)
  }

  return (
    <motion.div
      className="loading-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="loading-content">
        {/* Neural Network Animation */}
        <motion.div
          className="neural-network-loader"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.8 }}
        >
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="neural-node-loader"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                delay: i * 0.2,
                duration: 0.5,
                repeat: Infinity,
                repeatType: "reverse",
                repeatDelay: 2
              }}
            />
          ))}
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {message}
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          style={{ textAlign: 'center' }}
        >
          🌐 Multilingual Research Discovery with Vertex AI
          <br />
          <strong>Elapsed: {formatTime(timeElapsed)}</strong> | Expected: ~{TOTAL_DURATION}s
        </motion.p>

        {/* Progress Bar */}
        <motion.div
          className="progress-container"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.7 }}
        >
          <div className="progress-bar">
            <motion.div
              className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${calculateProgress()}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <div className="progress-text">
            {Math.round(calculateProgress())}% Complete
            {timeElapsed > 25 && " - Almost done, finalizing graph..."}
          </div>
        </motion.div>

        {/* Loading Steps */}
        <motion.div
          className="loading-steps"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          {steps.map((step, index) => {
            const StepIcon = step.icon
            const isActive = index === currentStep
            const isCompleted = index < currentStep
            
            return (
              <motion.div
                key={index}
                className={`loading-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.2 + index * 0.15 }}
                style={{
                  borderLeftColor: isActive ? step.color : isCompleted ? '#10B981' : '#64748B'
                }}
              >
                <div 
                  className="step-indicator"
                  style={{
                    backgroundColor: isActive ? step.color : isCompleted ? '#10B981' : '#64748B'
                  }}
                >
                  <StepIcon size={16} />
                </div>
                <span>{step.text}</span>
                {isActive && (
                  <motion.div
                    className="step-spinner"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    style={{ color: step.color }}
                  >
                    <Zap size={12} />
                  </motion.div>
                )}
                {isCompleted && (
                  <motion.div
                    className="step-checkmark"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    ✓
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </motion.div>

        {/* Helpful Tips */}
        {timeElapsed > 12 && timeElapsed <= 25 && (
          <motion.div
            className="loading-tip"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <p>💡 <strong>Tip:</strong> We're searching across arXiv, Crossref, Semantic Scholar, 
            PubMed, and DOAJ in 15 languages to find the most relevant papers!</p>
          </motion.div>
        )}

        {timeElapsed > 25 && (
          <motion.div
            className="loading-tip"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <p>🎯 <strong>Almost there!</strong> Building your personalized knowledge graph 
            with {timeElapsed > 27 ? 'AI-powered' : 'intelligent'} similarity connections...</p>
          </motion.div>
        )}

        {/* Backend Info */}
        <motion.div
          className="backend-info"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          transition={{ delay: 2 }}
          style={{ 
            marginTop: '2rem', 
            fontSize: '0.8rem', 
            color: '#94A3B8',
            textAlign: 'center'
          }}
        >
          Powered by: Elasticsearch Serverless • Vertex AI Gemini • Google Cloud
        </motion.div>
      </div>
    </motion.div>
  )
}

export default LoadingScreen
