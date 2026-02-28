'use client'

import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'
import type { LanguagePreference } from '@/lib/types'

import he from './he.json'
import ar from './ar.json'

const translations = { he, ar }

type TranslationKeys = typeof he
type DeepKeyOf<T> = T extends object
  ? { [K in keyof T]: K extends string ? `${K}` | `${K}.${DeepKeyOf<T[K]>}` : never }[keyof T]
  : never

// context
interface I18nContextValue {
  lang: LanguagePreference
  setLang: (lang: LanguagePreference) => void
  t: (key: string) => string
  dir: 'rtl' | 'ltr'
}

const I18nContext = createContext<I18nContextValue | null>(null)
const I18nContextProvider = I18nContext.Provider

// Provider
export function I18nProvider({
  children,
  defaultLang = 'he',
}: {
  children: ReactNode
  defaultLang?: LanguagePreference
}) {
  const [lang, setLangState] = useState<LanguagePreference>(defaultLang)

  useEffect(() => {
    const saved = localStorage.getItem('lang') as LanguagePreference | null
    if (saved && (saved === 'he' || saved === 'ar')) {
      setLangState(saved)
    }
  }, [])

  const setLang = (newLang: LanguagePreference) => {
    setLangState(newLang)
    localStorage.setItem('lang', newLang)
    // עדכון dir על ה-html element
    document.documentElement.dir = newLang === 'ar' ? 'rtl' : 'rtl'
    document.documentElement.lang = newLang
  }

  const t = (key: string): string => {
    const keys = key.split('.')
    let current: unknown = translations[lang]
    for (const k of keys) {
      if (current && typeof current === 'object' && k in (current as object)) {
        current = (current as Record<string, unknown>)[k]
      } else {
        // fallback להעברית אם המפתח לא קיים בשפה הנוכחית
        let fallback: unknown = translations.he
        for (const fk of keys) {
          if (fallback && typeof fallback === 'object' && fk in (fallback as object)) {
            fallback = (fallback as Record<string, unknown>)[fk]
          } else {
            return key // מחזיר את המפתח עצמו כ-fallback אחרון
          }
        }
        return typeof fallback === 'string' ? fallback : key
      }
    }
    return typeof current === 'string' ? current : key
  }

  return (
    <I18nContextProvider value={{ lang, setLang, t, dir: 'rtl' }}>
      {children}
    </I18nContextProvider>
  )
}

// Hook
export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used within I18nProvider')
  return ctx
}
