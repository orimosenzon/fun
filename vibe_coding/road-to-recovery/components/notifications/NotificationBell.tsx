'use client'

import { useState } from 'react'
import { Bell } from 'lucide-react'
import { useNotifications } from '@/lib/hooks/useNotifications'
import { timeAgo } from '@/lib/utils/formatters'
import type { LanguagePreference } from '@/lib/types'

interface NotificationBellProps {
  userId: string
  lang: LanguagePreference
}

export function NotificationBell({ userId, lang }: NotificationBellProps) {
  const [open, setOpen] = useState(false)
  const { notifications, unreadCount, markAsRead, markAllAsRead } = useNotifications(userId)

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="relative p-2 rounded-lg hover:bg-gray-100 text-gray-600"
        aria-label="התראות"
      >
        <Bell size={20} />
        {unreadCount > 0 && (
          <span className="absolute top-1 right-1 w-4 h-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-medium">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <>
          {/* backdrop */}
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />

          {/* dropdown */}
          <div className="absolute left-0 top-full mt-2 w-80 bg-white rounded-xl shadow-lg border border-gray-200 z-20 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <span className="font-semibold text-gray-800 text-sm">
                {lang === 'he' ? 'התראות' : 'الإشعارات'}
              </span>
              {unreadCount > 0 && (
                <button
                  onClick={markAllAsRead}
                  className="text-xs text-green-700 hover:underline"
                >
                  {lang === 'he' ? 'סמן הכל כנקרא' : 'تعليم الكل كمقروء'}
                </button>
              )}
            </div>

            <div className="max-h-80 overflow-y-auto divide-y divide-gray-50">
              {notifications.length === 0 ? (
                <p className="text-center text-gray-400 text-sm py-6">
                  {lang === 'he' ? 'אין התראות חדשות' : 'لا توجد إشعارات جديدة'}
                </p>
              ) : (
                notifications.map(n => (
                  <button
                    key={n.id}
                    onClick={() => markAsRead(n.id)}
                    className={`w-full text-right px-4 py-3 hover:bg-gray-50 transition-colors ${
                      !n.read ? 'bg-green-50' : ''
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {!n.read && (
                        <span className="w-2 h-2 rounded-full bg-green-500 mt-1.5 flex-shrink-0" />
                      )}
                      <div className={!n.read ? '' : 'mr-5'}>
                        <p className="text-sm font-medium text-gray-800">
                          {lang === 'he' ? n.title_he : n.title_ar}
                        </p>
                        <p className="text-xs text-gray-500 mt-0.5">
                          {lang === 'he' ? n.message_he : n.message_ar}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          {timeAgo(n.created_at)}
                        </p>
                      </div>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
