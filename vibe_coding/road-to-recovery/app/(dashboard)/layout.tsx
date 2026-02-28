'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Sidebar } from '@/components/layout/Sidebar'
import { Navbar } from '@/components/layout/Navbar'
import { LoadingScreen } from '@/components/ui/Spinner'
import { useCurrentUser } from '@/lib/hooks/useCurrentUser'
import type { LanguagePreference } from '@/lib/types'

const ROLE_LABELS = {
  he: {
    israeli_coordinator: 'רכזת ישראלית',
    palestinian_coordinator: 'רכז פלסטיני',
    driver: 'נהג מתנדב',
  },
  ar: {
    israeli_coordinator: 'منسقة إسرائيلية',
    palestinian_coordinator: 'منسق فلسطيني',
    driver: 'سائق متطوع',
  },
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, loading } = useCurrentUser()
  const router = useRouter()
  const [lang, setLang] = useState<LanguagePreference>('he')

  useEffect(() => {
    const saved = localStorage.getItem('lang') as LanguagePreference | null
    if (saved === 'he' || saved === 'ar') setLang(saved)
  }, [])

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login')
    }
  }, [loading, user, router])

  function handleLangChange(newLang: LanguagePreference) {
    setLang(newLang)
    localStorage.setItem('lang', newLang)
  }

  if (loading || !user) return <LoadingScreen />

  const roleLabel = ROLE_LABELS[lang][user.role]

  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      <Sidebar role={user.role} lang={lang} />
      <Navbar
        user={user}
        lang={lang}
        onLangChange={handleLangChange}
        roleLabel={roleLabel}
      />
      {/* Main content — offset לסרגל הצד (264px) ולנאבבר (56px) */}
      <main className="mr-64 pt-14 min-h-screen">
        <div className="p-6">
          {children}
        </div>
      </main>
    </div>
  )
}
