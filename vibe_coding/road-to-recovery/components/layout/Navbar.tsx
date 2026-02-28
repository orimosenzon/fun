'use client'

import { useRouter } from 'next/navigation'
import { LogOut } from 'lucide-react'
import { createClient } from '@/lib/supabase/client'
import { LanguageToggle } from './LanguageToggle'
import { NotificationBell } from '@/components/notifications/NotificationBell'
import type { User, LanguagePreference } from '@/lib/types'

interface NavbarProps {
  user: User
  lang: LanguagePreference
  onLangChange: (lang: LanguagePreference) => void
  roleLabel: string
}

export function Navbar({ user, lang, onLangChange, roleLabel }: NavbarProps) {
  const router = useRouter()

  async function handleLogout() {
    const supabase = createClient()
    await supabase.auth.signOut()
    router.push('/login')
  }

  const displayName = lang === 'ar' && user.name_ar ? user.name_ar : user.name_he

  return (
    <header className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-4 fixed top-0 right-0 left-64 z-10">
      {/* שם משתמש + תפקיד */}
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-full bg-green-100 border border-green-200 flex items-center justify-center">
          <span className="text-green-700 text-sm font-bold">
            {displayName.charAt(0)}
          </span>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-800">{displayName}</p>
          <p className="text-xs text-gray-400">{roleLabel}</p>
        </div>
      </div>

      {/* פעולות */}
      <div className="flex items-center gap-3">
        <LanguageToggle lang={lang} onToggle={onLangChange} />
        <NotificationBell userId={user.id} lang={lang} />
        <button
          onClick={handleLogout}
          className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700"
          title={lang === 'he' ? 'התנתקות' : 'تسجيل الخروج'}
        >
          <LogOut size={18} />
        </button>
      </div>
    </header>
  )
}
