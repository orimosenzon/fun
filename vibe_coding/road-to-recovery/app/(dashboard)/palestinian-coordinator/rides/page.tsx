'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Plus, RefreshCw } from 'lucide-react'
import { createClient } from '@/lib/supabase/client'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { RideStatusBadge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatDateTime } from '@/lib/utils/formatters'
import type { RideWithDetails } from '@/lib/types'

const STATUS_LABELS_HE: Record<string, string> = {
  pending: 'ממתינה לשיבוץ',
  assigned: 'שובץ נהג',
  in_progress: 'בביצוע',
  completed: 'הושלמה',
  cancelled: 'בוטלה',
  return_needed: 'נדרשת חזרה',
}
const STATUS_LABELS_AR: Record<string, string> = {
  pending: 'بانتظار التعيين',
  assigned: 'تم تعيين سائق',
  in_progress: 'جارية',
  completed: 'مكتملة',
  cancelled: 'ملغاة',
  return_needed: 'يلزم رحلة عودة',
}

export default function PalestinianCoordinatorRidesPage() {
  const [rides, setRides] = useState<RideWithDetails[]>([])
  const [loading, setLoading] = useState(true)
  const [lang, setLang] = useState<'he' | 'ar'>('he')

  useEffect(() => {
    const saved = localStorage.getItem('lang') as 'he' | 'ar' | null
    if (saved) setLang(saved)
  }, [])

  async function fetchRides() {
    setLoading(true)
    const res = await fetch('/api/rides')
    const { data } = await res.json()
    setRides(data ?? [])
    setLoading(false)
  }

  useEffect(() => {
    fetchRides()

    // Realtime subscription
    const supabase = createClient()
    const channel = supabase
      .channel('pal-coordinator-rides')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'rides' }, fetchRides)
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [])

  const t = (he: string, ar: string) => lang === 'ar' ? ar : he

  return (
    <div>
      {/* כותרת */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-gray-900">
            {t('הבקשות שלי', 'طلباتي')}
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {t('נסיעות שהגשתי', 'الرحلات التي قدمتها')}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={fetchRides}>
            <RefreshCw size={16} />
          </Button>
          <Link href="/palestinian-coordinator/rides/new">
            <Button size="sm">
              <Plus size={16} />
              {t('בקשה חדשה', 'طلب جديد')}
            </Button>
          </Link>
        </div>
      </div>

      {/* רשימת נסיעות */}
      {loading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : rides.length === 0 ? (
        <Card className="text-center py-12">
          <p className="text-gray-400 mb-4">{t('לא הוגשו בקשות עדיין', 'لم يتم تقديم أي طلبات بعد')}</p>
          <Link href="/palestinian-coordinator/rides/new">
            <Button size="sm">
              <Plus size={16} />
              {t('הגש בקשה ראשונה', 'تقديم الطلب الأول')}
            </Button>
          </Link>
        </Card>
      ) : (
        <div className="space-y-3">
          {rides.map(ride => (
            <Link key={ride.id} href={`/palestinian-coordinator/rides/${ride.id}`}>
              <Card className="hover:border-green-300 hover:shadow-md transition-all cursor-pointer">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <RideStatusBadge
                        status={ride.status}
                        label={lang === 'ar' ? STATUS_LABELS_AR[ride.status] : STATUS_LABELS_HE[ride.status]}
                      />
                      {ride.is_return_ride && (
                        <span className="text-xs text-purple-600 bg-purple-50 border border-purple-200 px-2 py-0.5 rounded-full">
                          {t('חזרה', 'عودة')}
                        </span>
                      )}
                    </div>
                    <h3 className="font-semibold text-gray-900 truncate">{ride.patient_name}</h3>
                    <p className="text-sm text-gray-500 mt-0.5">
                      {lang === 'ar' ? ride.hospital?.name_ar : ride.hospital?.name_he}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {formatDateTime(ride.scheduled_at)}
                    </p>
                  </div>
                  {ride.driver && (
                    <div className="text-left shrink-0">
                      <p className="text-xs text-gray-400">{t('נהג', 'السائق')}</p>
                      <p className="text-sm font-medium text-gray-700">
                        {lang === 'ar' ? (ride.driver.name_ar ?? ride.driver.name_he) : ride.driver.name_he}
                      </p>
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
