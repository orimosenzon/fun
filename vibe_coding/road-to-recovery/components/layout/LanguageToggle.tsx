'use client'

interface LanguageToggleProps {
  lang: 'he' | 'ar'
  onToggle: (lang: 'he' | 'ar') => void
}

export function LanguageToggle({ lang, onToggle }: LanguageToggleProps) {
  return (
    <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
      <button
        onClick={() => onToggle('he')}
        className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
          lang === 'he'
            ? 'bg-white text-green-700 shadow-sm'
            : 'text-gray-500 hover:text-gray-700'
        }`}
      >
        עב
      </button>
      <button
        onClick={() => onToggle('ar')}
        className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
          lang === 'ar'
            ? 'bg-white text-green-700 shadow-sm'
            : 'text-gray-500 hover:text-gray-700'
        }`}
      >
        عر
      </button>
    </div>
  )
}
