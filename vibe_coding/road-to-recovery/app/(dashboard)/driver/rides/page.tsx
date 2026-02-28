'use client'

import { useEffect, useState, useCallback } from 'react'
import Link from 'next/link'
import { MapPin, RefreshCw } from 'lucide-react'
import { createClient } from '@/lib/supabase/client'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { RideStatusBadge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatDateTime } from '@/lib/utils/formatters'
import { formatDistance } from '@/lib/utils/reimbursement'
import { useCurrentUser } from '@/lib/hooks/useCurrentUser'
import type { RideWithDetails, RideStatus } from '@/lib/types'

const STATUS_LABELS_HE: Record<RideStatus, string> = {
  pending: 'פנויה לאיסוף',
  assigned: 'שובצת אליך',
  in_progress: 'בביצוע',
  completed: 'הושלמה',
  cancelled: 'בוטלה',
  return_needed: 'נדרשת חזרה',
}
const STATUS_LABELS_AR: Record<RideStatus, string> = {
  pending: 'متاحة للاستلام',
  assigned: 'معين لك',
  in_progress: 'جارية',
  completed: 'مكتملة',
  cancelled: 'ملغاة',
  return_needed: 'يلزم عودة',
}

type TabType = 'open' | 'mine'

export default function DriverRidesPage() {
  const { user } = useCurrentUser()
  const [rides, setRides] = useState<RideWithDetails[]>([])
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<TabType>('open')
  const [lang, setLang] = useState<'he' | 'ar'>('he')
  const [taking, setTaking] = useState<string | null>(null)

  useEffect(() => {
    const saved = localStorage.getItem('lang') as 'he' | 'ar' | null
    if (saved) setLang(saved)
  }, [])

  const fetchRides = useCallback(async () => {
    setLoading(true)
    const params = new URLSearchParams()
    if (tab === 'open') params.set('status', 'pending')
    const res = await fetch(`/api/rides?${params}`)
    const { data } = await res.json()

    if (tab === 'mine') {
      // סינון נסיעות של המשתמש הנוכחי
      setRides((data ?? []).filter((r: RideWithDetails) => r.driver_id === user?.id))
    } else {
      setRides(data ?? [])
    }
    setLoading(false)
  }, [tab, user?.id])

  useEffect(() => {
    if (user) fetchRides()

    const supabase = createClient()
    const channel = supabase
      .channel('driver-rides')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'rides' }, () => {
        if (user) fetchRides()
      })
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [fetchRides, user])

  async function handleTakeRide(rideId: string, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    setTaking(rideId)

    const res = await fetch(`/api/rides/${rideId}/take`, { method: 'POST' })

    if (res.status === 409) {
      alert(lang === 'ar' ? 'تم استلام هذه الرحلة من قبل سائق آخر' : 'הנסיעה כבר נלקחה על ידי נהג אחר')
    } else if (res.ok) {
      fetchRides()
    }
    setTaking(null)
  }

  const t = (he: string, ar: string) => lang === 'ar' ? ar : he

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <h1 className="text-xl font-bold text-gray-900">
          {t('נסיעות', 'الرحلات')}
        </h1>
        <Button variant="ghost" size="sm" onClick={fetchRides}>
          <RefreshCw size={16} />
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 mb-4">
        {([
          { id: 'open' as TabType, he: 'נסיעות פתוחות', ar: 'رحلات متاحة' },
          { id: 'mine' as TabType, he: 'הנסיעות שלי', ar: 'رحلاتي' },
        ] as const).map(t_item => (
          <button
            key={t_item.id}
            onClick={() => setTab(t_item.id)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              tab === t_item.id
                ? 'border-green-600 text-green-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {lang === 'ar' ? t_item.ar : t_item.he}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : rides.length === 0 ? (
        <Card className="text-center py-12">
          <p className="text-gray-400">
            {tab === 'open'
              ? t('אין נסיעות פתוחות כרגע', 'لا توجد رحلات متاحة الآن')
              : t('לא קיבלת נסיעות עדיין', 'لم تستلم أي رحلات بعد')}
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {rides.map(ride => (
            <Link key={ride.id} href={`/driver/rides/${ride.id}`}>
              <Card className="hover:border-green-300 hover:shadow-md transition-all cursor-pointer">
                <div className="flex items-start gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <RideStatusBadge
                        status={ride.status}
                        label={lang === 'ar' ? STATUS_LABELS_AR[ride.status] : STATUS_LABELS_HE[ride.status]}
                      />
                    </div>
                    <div className="flex items-center gap-1.5 text-sm font-medium text-gray-900 mb-1">
                      <MapPin size={14} className="text-green-600 shrink-0" />
                      <span className="truncate">{ride.pickup_address}</span>
                    </div>
                    <p className="text-sm text-gray-600">
                      → {lang === 'ar' ? ride.hospital?.name_ar : ride.hospital?.name_he}
                    </p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-gray-400">
                      <span>📅 {formatDateTime(ride.scheduled_at)}</span>
                      {ride.distance_km && (
                        <span>📍 {formatDistance(ride.distance_km)}</span>
                      )}
                      {ride.distance_km && (
                        <span className="text-green-600 font-medium">
                          💳 ₪{(ride.distance_km * 1.5).toFixed(0)}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* כפתור לקיחה — רק בנסיעות פתוחות */}
                  {tab === 'open' && ride.status === 'pending' && (
                    <div className="shrink-0">
                      <Button
                        size="sm"
                        onClick={(e) => handleTakeRide(ride.id, e)}
                        disabled={taking === ride.id}
                      >
                        {taking === ride.id
                          ? <Spinner size="sm" />
                          : t('קח נסיעה', 'استلام الرحلة')}
                      </Button>
                    </div>
                  )}
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
