'use client'

import { useEffect, useState, useCallback } from 'react'
import { createClient } from '@/lib/supabase/client'
import type { Notification } from '@/lib/types'

interface NotificationsState {
  notifications: Notification[]
  unreadCount: number
  loading: boolean
}

export function useNotifications(userId: string | null) {
  const [state, setState] = useState<NotificationsState>({
    notifications: [],
    unreadCount: 0,
    loading: true,
  })

  const fetchNotifications = useCallback(async () => {
    if (!userId) return

    const supabase = createClient()
    const { data } = await supabase
      .from('notifications')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(50)

    const notifications = (data as Notification[]) ?? []
    setState({
      notifications,
      unreadCount: notifications.filter(n => !n.read).length,
      loading: false,
    })
  }, [userId])

  useEffect(() => {
    if (!userId) return

    fetchNotifications()

    const supabase = createClient()

    // Supabase Realtime — האזנה לשינויים בטבלת notifications של המשתמש
    const channel = supabase
      .channel(`notifications:${userId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'notifications',
          filter: `user_id=eq.${userId}`,
        },
        () => {
          // כל שינוי — refresh
          fetchNotifications()
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [userId, fetchNotifications])

  const markAsRead = useCallback(async (notificationId: string) => {
    const supabase = createClient()
    await supabase
      .from('notifications')
      .update({ read: true, read_at: new Date().toISOString() })
      .eq('id', notificationId)

    setState(prev => ({
      ...prev,
      notifications: prev.notifications.map(n =>
        n.id === notificationId ? { ...n, read: true } : n
      ),
      unreadCount: Math.max(0, prev.unreadCount - 1),
    }))
  }, [])

  const markAllAsRead = useCallback(async () => {
    if (!userId) return
    const supabase = createClient()
    await supabase
      .from('notifications')
      .update({ read: true, read_at: new Date().toISOString() })
      .eq('user_id', userId)
      .eq('read', false)

    setState(prev => ({
      ...prev,
      notifications: prev.notifications.map(n => ({ ...n, read: true })),
      unreadCount: 0,
    }))
  }, [userId])

  return { ...state, markAsRead, markAllAsRead, refresh: fetchNotifications }
}
