import React, { forwardRef, useEffect, useRef, useState, useCallback, memo, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ForceGraph2D from 'react-force-graph-2d'

const GraphViewer = memo(forwardRef(({
  data,
  mode = '2d',
  config = {},
  selectedNode,
  hoveredNode,
  onNodeInteraction,
  onConfigChange
}, ref) => {
  const graphRef = useRef()
  const containerRef = useRef()
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, node: null })
  
  const [lastClickTime, setLastClickTime] = useState(0)
  const [lastClickedNode, setLastClickedNode] = useState(null)

  const defaultConfig = {
    nodeSize: 1.0,
    linkWidth: 2,
    linkOpacity: 0.8,
    showLabels: true,
    paused: false,
    ...config
  }

  // ✅ Create immutable graph data - NO VERSION KEY!
  const stableGraphData = useMemo(() => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return { nodes: [], links: [] }
    }

    console.log('🔄 Creating graph data:', {
      nodes: data.nodes.length,
      links: data.links?.length || 0
    })

    // Deep clone nodes
    const nodes = data.nodes.map(node => ({ ...node }))
    
    // Deep clone links with FORCED string IDs
    const links = (data.links || []).map(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source
      const targetId = typeof link.target === 'object' ? link.target.id : link.target
      
      return {
        source: String(sourceId),
        target: String(targetId),
        strength: link.strength || 0.5,
        similarity_score: link.similarity_score || link.strength || 0.5,
        label: link.label,
        color: link.color,
        width: link.width
      }
    })

    return { nodes, links }
  }, [data?.nodes?.length, data?.links?.length, data?._stableVersion])

  // Update dimensions
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setDimensions({ width: rect.width, height: rect.height })
      }
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Expose graph ref
  useEffect(() => {
    if (ref) ref.current = graphRef.current
  }, [ref])

  // ✅ PROPER PHYSICS: Natural spreading with collision
  useEffect(() => {
    if (graphRef.current && stableGraphData.nodes.length > 0) {
      const timer = setTimeout(() => {
        if (graphRef.current?.d3Force) {
          // Gentle repulsion
          graphRef.current.d3Force('charge').strength(-300)
          
          // Moderate link strength
          graphRef.current.d3Force('link').distance(100).strength(0.4)
          
          // Weak centering
          graphRef.current.d3Force('center').strength(0.03)
          
          // ✅ COLLISION: Prevent overlap (25px radius)
          try {
            const d3Force = require('d3-force')
            graphRef.current.d3Force('collide', d3Force.forceCollide().radius(25).strength(0.8))
          } catch (e) {
            console.warn('d3-force not available for collision')
          }
          
          console.log('✅ Physics configured')
        }
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [stableGraphData.nodes.length])

  const handleNodeClick = useCallback((node, event) => {
    const now = Date.now()
    const timeDiff = now - lastClickTime
    
    if (node?.id) {
      if (lastClickedNode === node.id && timeDiff < 300) {
        console.log('⚡ DOUBLE-CLICK:', node.id)
        onNodeInteraction(node.id, 'dblclick')
        setLastClickTime(0)
        setLastClickedNode(null)
      } else {
        onNodeInteraction(node.id, 'click')
        setLastClickTime(now)
        setLastClickedNode(node.id)
      }
    } else {
      onNodeInteraction(null, 'click')
      setLastClickTime(0)
      setLastClickedNode(null)
    }
  }, [onNodeInteraction, lastClickTime, lastClickedNode])

  const handleNodeHover = useCallback((node) => {
    if (node?.id) {
      setTooltip({ show: true, x: 0, y: 0, node })
      onNodeInteraction(node.id, 'hover')
    } else {
      setTooltip({ show: false, x: 0, y: 0, node: null })
      onNodeInteraction(null, 'unhover')
    }
  }, [onNodeInteraction])

  const nodePointerAreaPaint = useCallback((node, color, ctx) => {
    if (!node?.x || !node?.y) return
    const radius = 20
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI)
    ctx.fill()
  }, [])

  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    if (!node?.x || !node?.y || !isFinite(node.x) || !isFinite(node.y)) return

    const label = node.label || node.title
    const fontSize = 10 / globalScale
    const nodeRadius = 8  // ← FIXED SIZE!

    // Node color
    let color = node.color || '#3B82F6'
    if (node.id === selectedNode) color = '#FBBF24'
    else if (node.id === hoveredNode) color = '#60A5FA'
    
    // Draw node
    ctx.beginPath()
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()

    // Border
    ctx.strokeStyle = '#FFF'
    ctx.lineWidth = node.id === selectedNode || node.id === hoveredNode ? 2.5 : 1.5
    ctx.stroke()

    // Label
    if (defaultConfig.showLabels && globalScale > 0.5 && label) {
      ctx.font = `${fontSize}px Inter, sans-serif`
      ctx.fillStyle = '#FFF'
      ctx.textAlign = 'center'
      ctx.shadowColor = 'rgba(0,0,0,0.8)'
      ctx.shadowBlur = 3
      
      const maxLen = 25
      const text = label.length > maxLen ? label.substring(0, maxLen) + '…' : label
      ctx.fillText(text, node.x, node.y + nodeRadius + fontSize + 4)
      
      ctx.shadowBlur = 0
    }
  }, [selectedNode, hoveredNode, defaultConfig.showLabels])

  if (stableGraphData.nodes.length === 0) {
    return (
      <motion.div className="graph-empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="empty-content">
          <h3>No Research Network</h3>
          <p>Search to discover connections</p>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="graph-viewer"
      ref={containerRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      onMouseMove={(e) => {
        if (tooltip.show) {
          setTooltip(prev => ({ ...prev, x: e.clientX, y: e.clientY }))
        }
      }}
    >
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={stableGraphData}
        
        nodeLabel={node => `${node.title || 'Untitled'}\n\nDouble-click for similar`}
        nodeColor={node => {
          if (node.id === selectedNode) return '#FBBF24'
          if (node.id === hoveredNode) return '#60A5FA'
          return node.color || '#3B82F6'
        }}
        nodeRelSize={8}  // ← FIXED NODE SIZE!
        nodePointerAreaPaint={nodePointerAreaPaint}
        nodeCanvasObject={nodeCanvasObject}
        
        linkWidth={link => {
          const s = link.strength || 0.5
          return Math.max(1.5, s * 4)
        }}
        linkColor={link => {
          const s = link.strength || 0.5
          if (s > 0.85) return 'rgba(59,130,246,0.9)'
          if (s > 0.75) return 'rgba(16,185,129,0.9)'
          return 'rgba(100,116,139,0.7)'
        }}
        linkDirectionalParticles={link => (link.strength || 0) > 0.85 ? 2 : 0}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleSpeed={0.004}
        linkCurvature={0}
        
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onBackgroundClick={() => handleNodeClick(null)}
        
        cooldownTicks={Infinity}
        d3AlphaDecay={0.0228}
        d3VelocityDecay={0.4}
        warmupTicks={50}
        
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        
        backgroundColor="rgba(0,0,0,0)"
      />

      <AnimatePresence>
        {tooltip.show && tooltip.node && (
          <motion.div
            className="node-tooltip"
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              left: Math.min(tooltip.x + 15, window.innerWidth - 280),
              top: Math.max(tooltip.y - 60, 10),
              zIndex: 1100,
              pointerEvents: 'none'
            }}
          >
            <div className="tooltip-content">
              <h4>{tooltip.node.title || 'Untitled'}</h4>
              <div className="tooltip-meta">
                <span className="domain-tag">{tooltip.node.research_domain || 'Unknown'}</span>
                {tooltip.node.citation_count && (
                  <span className="citation-count">{tooltip.node.citation_count} cites</span>
                )}
              </div>
              <div className="tooltip-hint">Click: Details • Double: Similar</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="graph-info-overlay">
        <div className="info-item">
          <span className="info-label">Papers:</span>
          <span className="info-value">{stableGraphData.nodes.length}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Links:</span>
          <span className="info-value">{stableGraphData.links.length}</span>
        </div>
      </div>
    </motion.div>
  )
}))

GraphViewer.displayName = 'GraphViewer'

export default GraphViewer
