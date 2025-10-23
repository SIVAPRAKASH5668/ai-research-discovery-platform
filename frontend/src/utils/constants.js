/**
 * Application constants and configuration
 * UPDATED WITH AI ANALYSIS ENDPOINTS
 */

// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.VITE_API_URL || 'http://localhost:8000',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
}

// API Endpoints
export const API_ENDPOINTS = {
  // Research endpoints
  DISCOVER: '/api/research/discover',
  CHAT: '/api/chat',
  FIND_SIMILAR: '/api/research/find-similar',
  PAPER_DETAILS: '/api/papers/details',
  PAPER_SEARCH: '/api/papers/search',
  PAPER_RELATIONSHIPS: '/api/papers/{id}/relationships',
  EXPORT_GRAPH: '/api/research/export',
  
  // 🆕 AI Analysis endpoints
  PAPER_SUMMARIZE: '/api/paper/summarize',
  PAPER_ASK: '/api/paper/ask',
  
  // User endpoints
  USER_PROFILE: '/api/user/profile',
  USER_PREFERENCES: '/api/user/preferences',
  SEARCH_HISTORY: '/api/user/search-history',
  
  // Analytics endpoints
  ANALYTICS_TRACK: '/api/analytics/track',
  ANALYTICS_USAGE: '/api/analytics/usage',
  
  // System endpoints
  HEALTH: '/api/health',
  STATUS: '/api/status'
}

// Graph Visualization Constants
export const GRAPH_CONSTANTS = {
  DEFAULT_NODE_SIZE: 5,
  MIN_NODE_SIZE: 2,
  MAX_NODE_SIZE: 25,
  DEFAULT_LINK_WIDTH: 1,
  MIN_LINK_WIDTH: 0.5,
  MAX_LINK_WIDTH: 8,
  DEFAULT_LINK_OPACITY: 0.6,
  CAMERA_DISTANCE_3D: 1000,
  ZOOM_FACTOR: 1.5,
  ANIMATION_DURATION: 1000
}

// Color Palettes
export const COLOR_PALETTES = {
  research_domains: {
    'Machine Learning': '#3B82F6',
    'Computer Vision': '#8B5CF6',
    'Natural Language Processing': '#06B6D4',
    'Healthcare': '#10B981',
    'Climate Science': '#F59E0B',
    'Physics': '#EF4444',
    'Chemistry': '#06B6D4',
    'Biology': '#84CC16',
    'Mathematics': '#F97316',
    'Materials Science': '#EC4899',
    'Neuroscience': '#8B5CF6',
    'Robotics': '#6366F1',
    'default': '#64748B'
  },
  quality_levels: {
    high: '#10B981',
    medium: '#F59E0B', 
    low: '#EF4444'
  },
  citation_ranges: {
    high_impact: '#EF4444',
    medium_impact: '#F59E0B',
    low_impact: '#10B981',
    no_citations: '#64748B'
  }
}

// UI Constants
export const UI_CONSTANTS = {
  HEADER_HEIGHT: 80,
  SIDEBAR_WIDTH: 480,
  SIDEBAR_WIDTH_COLLAPSED: 60,
  MOBILE_BREAKPOINT: 768,
  TABLET_BREAKPOINT: 1024,
  DESKTOP_BREAKPOINT: 1280,
  ANIMATION_DURATION: {
    FAST: 150,
    NORMAL: 300,
    SLOW: 500
  },
  Z_INDEX: {
    BACKGROUND: 1,
    CONTENT: 10,
    HEADER: 100,
    SIDEBAR: 50,
    MODAL: 1000,
    TOOLTIP: 1100,
    NOTIFICATION: 1200
  }
}

// Search Configuration
export const SEARCH_CONFIG = {
  MIN_QUERY_LENGTH: 2,
  MAX_RESULTS: 100,
  DEBOUNCE_DELAY: 500,
  SUGGESTION_LIMIT: 8,
  RECENT_SEARCHES_LIMIT: 10,
  DEFAULT_FILTERS: {
    minCitations: 0,
    minQuality: 0,
    yearRange: null,
    domains: [],
    sources: []
  }
}

// Paper Sources
export const PAPER_SOURCES = {
  ARXIV: {
    id: 'arxiv',
    name: 'arXiv',
    url: 'https://arxiv.org',
    color: '#B91C1C',
    description: 'Preprint repository for physics, mathematics, computer science'
  },
  PUBMED: {
    id: 'pubmed',
    name: 'PubMed',
    url: 'https://pubmed.ncbi.nlm.nih.gov',
    color: '#059669',
    description: 'Biomedical and life sciences literature'
  },
  SEMANTIC_SCHOLAR: {
    id: 'semantic_scholar',
    name: 'Semantic Scholar',
    url: 'https://www.semanticscholar.org',
    color: '#0891B2',
    description: 'AI-powered research tool'
  },
  CROSSREF: {
    id: 'crossref',
    name: 'Crossref',
    url: 'https://crossref.org',
    color: '#7C3AED',
    description: 'Official DOI registration agency'
  }
}

// Research Domains Configuration
export const RESEARCH_DOMAINS = {
  'AI_ML': {
    id: 'ai_ml',
    label: 'AI & Machine Learning',
    description: 'Artificial Intelligence and Machine Learning research',
    subcategories: [
      'Deep Learning',
      'Neural Networks', 
      'Computer Vision',
      'Natural Language Processing',
      'Reinforcement Learning',
      'Machine Learning Theory',
      'AI Ethics'
    ],
    color: '#3B82F6',
    icon: '🤖'
  },
  'HEALTHCARE': {
    id: 'healthcare',
    label: 'Healthcare & Medicine',
    description: 'Medical and healthcare research',
    subcategories: [
      'Medical Imaging',
      'Drug Discovery',
      'Precision Medicine',
      'Healthcare AI',
      'Bioinformatics',
      'Clinical Trials',
      'Public Health'
    ],
    color: '#10B981',
    icon: '🏥'
  },
  'PHYSICAL_SCIENCES': {
    id: 'physical_sciences', 
    label: 'Physical Sciences',
    description: 'Physics, Chemistry, and Materials Science',
    subcategories: [
      'Quantum Physics',
      'Materials Science',
      'Physical Chemistry',
      'Condensed Matter',
      'Optics',
      'Thermodynamics'
    ],
    color: '#EF4444',
    icon: '⚛️'
  },
  'ENVIRONMENTAL': {
    id: 'environmental',
    label: 'Environmental Sciences',
    description: 'Climate, Ecology, and Environmental research',
    subcategories: [
      'Climate Science',
      'Ecology',
      'Environmental Engineering',
      'Sustainability',
      'Conservation Biology',
      'Atmospheric Science'
    ],
    color: '#84CC16',
    icon: '🌍'
  }
}

// Keyboard Shortcuts
export const KEYBOARD_SHORTCUTS = {
  'cmd+k': 'Open search',
  'cmd+shift+c': 'Toggle controls panel',
  'cmd+shift+i': 'Toggle inspector panel', 
  'escape': 'Close panels/Clear selection',
  'f': 'Toggle fullscreen',
  'r': 'Reset graph view',
  '+': 'Zoom in',
  '-': 'Zoom out',
  'l': 'Toggle labels',
  'p': 'Toggle physics',
  '2': 'Switch to 2D view',
  '3': 'Switch to 3D view',
  's': 'Take screenshot',
  'e': 'Export graph'
}

// Application Messages
export const MESSAGES = {
  SUCCESS: {
    SEARCH_COMPLETED: 'Research discovery completed successfully',
    PAPER_LOADED: 'Paper details loaded',
    EXPORT_COMPLETED: 'Graph exported successfully',
    SETTINGS_SAVED: 'Settings saved successfully',
    SUMMARY_GENERATED: 'AI summary generated successfully', // NEW!
    ANSWER_GENERATED: 'Answer generated successfully' // NEW!
  },
  ERROR: {
    SEARCH_FAILED: 'Failed to discover research papers',
    PAPER_NOT_FOUND: 'Paper details not found',
    NETWORK_ERROR: 'Network error. Please check your connection',
    EXPORT_FAILED: 'Failed to export graph',
    INVALID_QUERY: 'Please enter a valid search query',
    SUMMARY_FAILED: 'Failed to generate AI summary', // NEW!
    QA_FAILED: 'Failed to generate answer' // NEW!
  },
  LOADING: {
    SEARCHING: 'Discovering research connections...',
    LOADING_PAPER: 'Loading paper details...',
    BUILDING_GRAPH: 'Building knowledge graph...',
    EXPORTING: 'Exporting graph data...',
    GENERATING_SUMMARY: 'Generating AI summary...', // NEW!
    THINKING: 'Thinking...' // NEW!
  }
}

// Local Storage Keys
export const STORAGE_KEYS = {
  USER_PREFERENCES: 'research-app-preferences',
  SEARCH_HISTORY: 'research-app-search-history',
  RECENT_PAPERS: 'research-app-recent-papers',
  GRAPH_SETTINGS: 'research-app-graph-settings',
  AUTH_TOKEN: 'research-app-auth-token'
}

// Event Names for Analytics
export const ANALYTICS_EVENTS = {
  SEARCH_PERFORMED: 'search_performed',
  PAPER_VIEWED: 'paper_viewed',
  NODE_CLICKED: 'node_clicked',
  GRAPH_EXPORTED: 'graph_exported',
  VIEW_CHANGED: 'view_changed',
  CONTROLS_TOGGLED: 'controls_toggled',
  PAPER_SHARED: 'paper_shared',
  AI_SUMMARY_GENERATED: 'ai_summary_generated', // NEW!
  QA_ASKED: 'qa_asked' // NEW!
}

// Feature Flags
export const FEATURES = {
  ADVANCED_SEARCH: true,
  EXPORT_FUNCTIONALITY: true,
  USER_ACCOUNTS: false,
  ANALYTICS: true,
  DARK_MODE: true,
  MOBILE_SUPPORT: true,
  OFFLINE_MODE: false,
  COLLABORATIVE_FEATURES: false,
  AI_ANALYSIS: true, // NEW!
  PAPER_QA: true // NEW!
}

// Performance Configuration
export const PERFORMANCE = {
  MAX_NODES_2D: 500,
  MAX_NODES_3D: 200,
  DEBOUNCE_SEARCH: 300,
  CACHE_DURATION: 300000, // 5 minutes
  MAX_CONCURRENT_REQUESTS: 3,
  IMAGE_LAZY_LOADING: true,
  VIRTUAL_SCROLLING_THRESHOLD: 100
}

// Validation Rules
export const VALIDATION = {
  SEARCH_QUERY: {
    MIN_LENGTH: 2,
    MAX_LENGTH: 500,
    PATTERN: /^[a-zA-Z0-9\s\-.,;:'"()\[\]{}!?]+$/
  },
  PAPER_ID: {
    PATTERN: /^[a-zA-Z0-9_\-:.\/]+$/  // UPDATED for DOI/arXiv IDs
  },
  EMAIL: {
    PATTERN: /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  },
  QUESTION: {  // NEW!
    MIN_LENGTH: 5,
    MAX_LENGTH: 500
  }
}

// Default Configuration
export const DEFAULT_CONFIG = {
  graph: {
    nodeSize: 5,
    linkWidth: 1,
    linkOpacity: 0.6,
    showLabels: true,
    physicsEnabled: true,
    view: '3d'
  },
  search: {
    maxResults: 50,
    includeAbstracts: true,
    languageFilter: 'any',
    dateRange: 'any'
  },
  ui: {
    theme: 'dark',
    animations: true,
    notifications: true,
    autoSave: true
  },
  ai: {  // NEW!
    enableSummary: true,
    enableQA: true,
    maxQAHistory: 10
  }
}

export default {
  API_CONFIG,
  API_ENDPOINTS,
  GRAPH_CONSTANTS,
  COLOR_PALETTES,
  UI_CONSTANTS,
  SEARCH_CONFIG,
  PAPER_SOURCES,
  RESEARCH_DOMAINS,
  KEYBOARD_SHORTCUTS,
  MESSAGES,
  STORAGE_KEYS,
  ANALYTICS_EVENTS,
  FEATURES,
  PERFORMANCE,
  VALIDATION,
  DEFAULT_CONFIG
}
