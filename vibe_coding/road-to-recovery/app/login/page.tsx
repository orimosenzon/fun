'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import type { UserRole } from '@/lib/types'

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [lang, setLang] = useState<'he' | 'ar'>('he')

  const labels = {
    he: {
      title: 'בדרך להחלמה',
      subtitle: 'מערכת תיאום נסיעות',
      email: 'אימייל',
      password: 'סיסמה',
      button: 'כניסה למערכת',
      error: 'אימייל או סיסמה שגויים',
    },
    ar: {
      title: 'طريق التعافي',
      subtitle: 'نظام تنسيق الرحلات',
      email: 'البريد الإلكتروني',
      password: 'كلمة المرور',
      button: 'الدخول إلى النظام',
      error: 'البريد الإلكتروني أو كلمة المرور غير صحيحة',
    },
  }

  const t = labels[lang]

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')

    const supabase = createClient()
    const { error: authError } = await supabase.auth.signInWithPassword({
      email,
      password,
    })

    if (authError) {
      setError(t.error)
      setLoading(false)
      return
    }

    // שליפת role לredirect
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { setLoading(false); return }

    const { data: profile } = await supabase
      .from('users')
      .select('role')
      .eq('id', user.id)
      .single()

    const roleRoutes: Record<UserRole, string> = {
      israeli_coordinator: '/coordinator/rides',
      palestinian_coordinator: '/palestinian-coordinator/rides',
      driver: '/driver/rides',
    }

    const route = profile ? roleRoutes[profile.role as UserRole] : '/'
    router.push(route)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4" dir="rtl">
      <div className="w-full max-w-md">
        {/* מתג שפה */}
        <div className="flex justify-center gap-2 mb-6">
          <button
            onClick={() => setLang('he')}
            className={`px-4 py-1.5 rounded-full text-sm font-medium border transition-colors ${
              lang === 'he'
                ? 'bg-green-700 text-white border-green-700'
                : 'bg-white text-gray-600 border-gray-300 hover:border-green-600'
            }`}
          >
            עברית
          </button>
          <button
            onClick={() => setLang('ar')}
            className={`px-4 py-1.5 rounded-full text-sm font-medium border transition-colors ${
              lang === 'ar'
                ? 'bg-green-700 text-white border-green-700'
                : 'bg-white text-gray-600 border-gray-300 hover:border-green-600'
            }`}
          >
            العربية
          </button>
        </div>

        {/* כרטיס כניסה */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
          {/* לוגו + כותרת */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-50 border-2 border-green-200 mb-4">
              <span className="text-3xl">🏥</span>
            </div>
            <h1 className="text-2xl font-bold text-green-800">{t.title}</h1>
            <p className="text-gray-500 text-sm mt-1">{t.subtitle}</p>
          </div>

          {/* טופס */}
          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {t.email}
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                dir="ltr"
                className="w-full px-4 py-2.5 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent text-left"
                placeholder="user@example.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {t.password}
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                dir="ltr"
                className="w-full px-4 py-2.5 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              />
            </div>

            {error && (
              <div className="bg-red-50 text-red-700 text-sm px-4 py-3 rounded-lg border border-red-200">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-green-700 text-white py-3 rounded-lg font-medium hover:bg-green-800 disabled:opacity-50 disabled:cursor-not-allowed mt-2"
            >
              {loading ? '...' : t.button}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-gray-400 mt-6">
          Road to Recovery &nbsp;•&nbsp; theroadtorecovery.org.il
        </p>
      </div>
    </div>
  )
}
