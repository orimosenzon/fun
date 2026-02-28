'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Car, CreditCard, Users } from 'lucide-react'
import type { UserRole, LanguagePreference } from '@/lib/types'

interface NavItem {
  href: string
  labelHe: string
  labelAr: string
  icon: React.ReactNode
}

const NAV_ITEMS: Record<UserRole, NavItem[]> = {
  israeli_coordinator: [
    { href: '/coordinator/rides', labelHe: 'נסיעות', labelAr: 'الرحلات', icon: <Car size={18} /> },
    { href: '/coordinator/reimbursements', labelHe: 'החזרי נסיעה', labelAr: 'استرداد السفر', icon: <CreditCard size={18} /> },
    { href: '/coordinator/users', labelHe: 'משתמשים', labelAr: 'المستخدمون', icon: <Users size={18} /> },
  ],
  palestinian_coordinator: [
    { href: '/palestinian-coordinator/rides', labelHe: 'הבקשות שלי', labelAr: 'طلباتي', icon: <Car size={18} /> },
  ],
  driver: [
    { href: '/driver/rides', labelHe: 'נסיעות', labelAr: 'الرحلات', icon: <Car size={18} /> },
  ],
}

interface SidebarProps {
  role: UserRole
  lang: LanguagePreference
}

export function Sidebar({ role, lang }: SidebarProps) {
  const pathname = usePathname()
  const navItems = NAV_ITEMS[role]

  return (
    <aside className="fixed top-0 right-0 h-full w-64 bg-white border-l border-gray-200 flex flex-col z-20">
      {/* לוגו */}
      <div className="h-14 flex items-center gap-3 px-4 border-b border-gray-100">
        <div className="w-8 h-8 rounded-lg bg-green-700 flex items-center justify-center">
          <span className="text-white text-lg">🏥</span>
        </div>
        <div>
          <p className="text-sm font-bold text-green-800">
            {lang === 'he' ? 'בדרך להחלמה' : 'طريق التعافي'}
          </p>
          <p className="text-xs text-gray-400">
            {lang === 'he' ? 'תיאום נסיעות' : 'تنسيق الرحلات'}
          </p>
        </div>
      </div>

      {/* ניווט */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(item => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/')
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-green-50 text-green-700'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-800'
              }`}
            >
              <span className={isActive ? 'text-green-600' : 'text-gray-400'}>
                {item.icon}
              </span>
              {lang === 'ar' ? item.labelAr : item.labelHe}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
