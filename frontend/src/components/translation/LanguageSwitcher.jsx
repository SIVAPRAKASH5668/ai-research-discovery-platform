import React from 'react'
import { useTranslation } from 'react-i18next'
import { Globe } from 'lucide-react'
import { toast } from 'react-hot-toast'

const LANGUAGES = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'zh', name: '中文', flag: '🇨🇳' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'ja', name: '日本語', flag: '🇯🇵' },
  { code: 'ko', name: '한국어', flag: '🇰🇷' },
  { code: 'pt', name: 'Português', flag: '🇧🇷' },
  { code: 'ru', name: 'Русский', flag: '🇷🇺' },
  { code: 'ar', name: 'العربية', flag: '🇸🇦' }
]

const LanguageSwitcher = ({ onLanguageChange }) => {
  const { i18n } = useTranslation()

  const changeLanguage = async (lng) => {
    // Update i18n for UI
    await i18n.changeLanguage(lng)
    
    // Store preference
    localStorage.setItem('preferred_language', lng)
    
    // Trigger data refresh with new language
    if (onLanguageChange) {
      await onLanguageChange(lng)
    }
    
    toast.success(`Language changed to ${LANGUAGES.find(l => l.code === lng)?.name}`, {
      icon: '🌍'
    })
  }

  return (
    <div className="language-switcher">
      <Globe size={20} />
      <select 
        value={i18n.language} 
        onChange={(e) => changeLanguage(e.target.value)}
        className="language-select"
      >
        {LANGUAGES.map(lang => (
          <option key={lang.code} value={lang.code}>
            {lang.flag} {lang.name}
          </option>
        ))}
      </select>
    </div>
  )
}

export default LanguageSwitcher
