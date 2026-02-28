'use client'

import { useEffect, useState, useCallback } from 'react'
import Link from 'next/link'
import { Calendar, RefreshCw, ChevronRight, ChevronLeft } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { RideStatusBadge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatTime, toDateString } from '@/lib/utils/formatters'
import { createClient } from '@/lib/supabase/client'
import type { RideWithDetails, RideStatus } from '@/lib/types'

const STATUS_LABELS: Record<RideStatus, string> = {
  pending: 'ממתינה לשיבוץ',
  assigned: 'שובץ נהג',
  in_progress: 'בביצוע',
  completed: 'הושלמה',
  cancelled: 'בוטלה',
  return_needed: 'נדרשת חזרה',
}

const STATUS_FILTERS: { value: RideStatus | 'all'; label: string }[] = [
  { value: 'all', label: 'הכל' },
  { value: 'pending', label: 'ממתינות' },
  { value: 'assigned', label: 'משובצות' },
  { value: 'in_progress', label: 'בדרך' },
  { value: 'return_needed', label: 'נדרשת חזרה' },
  { value: 'completed', label: 'הושלמו' },
]

export default function CoordinatorRidesPage() {
  const [rides, setRides] = useState<RideWithDetails[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedDate, setSelectedDate] = useState(toDateString())
  const [statusFilter, setStatusFilter] = useState<RideStatus | 'all'>('all')

  const fetchRides = useCallback(async () => {
    setLoading(true)
    const params = new URLSearchParams({ date: selectedDate })
    if (statusFilter !== 'all') params.set('status', statusFilter)

    const res = await fetch(`/api/rides?${params}`)
    const { data } = await res.json()
    setRides(data ?? [])
    setLoading(false)
  }, [selectedDate, statusFilter])

  useEffect(() => {
    fetchRides()

    const supabase = createClient()
    const channel = supabase
      .channel('coordinator-rides')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'rides' }, fetchRides)
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [fetchRides])

  // קיצורי תאריך
  function changeDate(days: number) {
    const d = new Date(selectedDate)
    d.setDate(d.getDate() + days)
    setSelectedDate(toDateString(d))
  }

  const today = toDateString()
  const isToday = selectedDate === today
  const displayDate = isToday ? 'היום' : selectedDate

  // סטטיסטיקות מהירות
  const stats = {
    total: rides.length,
    pending: rides.filter(r => r.status === 'pending').length,
    return_needed: rides.filter(r => r.status === 'return_needed').length,
    completed: rides.filter(r => r.status === 'completed').length,
  }

  return (
    <div>
      {/* כותרת + ניווט תאריך */}
      <div className="flex items-center justify-between mb-5">
        <h1 className="text-xl font-bold text-gray-900">לוח נסיעות יומי</h1>
        <Button variant="ghost" size="sm" onClick={fetchRides}>
          <RefreshCw size={16} />
        </Button>
      </div>

      {/* בחירת תאריך */}
      <div className="flex items-center gap-3 mb-4">
        <button onClick={() => changeDate(-1)} className="p-1.5 rounded-lg hover:bg-gray-100">
          <ChevronRight size={18} className="text-gray-500" />
        </button>
        <div className="flex items-center gap-2">
          <Calendar size={16} className="text-gray-400" />
          <input
            type="date"
            value={selectedDate}
            onChange={e => setSelectedDate(e.target.value)}
            className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-green-500"
          />
          <span className="text-sm font-medium text-gray-700">{isToday && '(היום)'}</span>
        </div>
        <button onClick={() => changeDate(1)} className="p-1.5 rounded-lg hover:bg-gray-100">
          <ChevronLeft size={18} className="text-gray-500" />
        </button>
        {!isToday && (
          <Button variant="ghost" size="sm" onClick={() => setSelectedDate(today)}>
            היום
          </Button>
        )}
      </div>

      {/* סטטיסטיקות מהירות */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        {[
          { label: 'סה"כ', value: stats.total, color: 'text-gray-700' },
          { label: 'ממתינות', value: stats.pending, color: 'text-yellow-700' },
          { label: 'נדרשת חזרה', value: stats.return_needed, color: 'text-purple-700' },
          { label: 'הושלמו', value: stats.completed, color: 'text-green-700' },
        ].map(s => (
          <Card key={s.label} padding="sm" className="text-center">
            <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-xs text-gray-500 mt-0.5">{s.label}</p>
          </Card>
        ))}
      </div>

      {/* פילטר סטטוס */}
      <div className="flex gap-2 mb-4 overflow-x-auto pb-1">
        {STATUS_FILTERS.map(f => (
          <button
            key={f.value}
            onClick={() => setStatusFilter(f.value)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap border transition-colors ${
              statusFilter === f.value
                ? 'bg-green-700 text-white border-green-700'
                : 'bg-white text-gray-600 border-gray-300 hover:border-green-400'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* רשימת נסיעות */}
      {loading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : rides.length === 0 ? (
        <Card className="text-center py-10">
          <p className="text-gray-400">אין נסיעות ל-{displayDate}</p>
        </Card>
      ) : (
        <div className="space-y-2">
          {rides.map(ride => (
            <Link key={ride.id} href={`/coordinator/rides/${ride.id}`}>
              <Card padding="sm" className="hover:border-green-300 hover:shadow-md transition-all cursor-pointer">
                <div className="flex items-center gap-4">
                  {/* שעה */}
                  <div className="text-center w-12 shrink-0">
                    <p className="text-sm font-bold text-gray-800">{formatTime(ride.scheduled_at)}</p>
                  </div>

                  {/* פרטים */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-gray-900 text-sm">{ride.patient_name}</span>
                      <RideStatusBadge status={ride.status} label={STATUS_LABELS[ride.status]} />
                      {ride.is_return_ride && (
                        <span className="text-xs text-purple-600 bg-purple-50 border border-purple-200 px-2 py-0.5 rounded-full">חזרה</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5 truncate">
                      {ride.hospital?.name_he} • {ride.pickup_address}
                    </p>
                  </div>

                  {/* נהג */}
                  <div className="text-left shrink-0">
                    {ride.driver ? (
                      <p className="text-sm font-medium text-gray-700">{ride.driver.name_he}</p>
                    ) : (
                      <span className="text-xs text-yellow-600 bg-yellow-50 border border-yellow-200 px-2 py-1 rounded-full">
                        לא שובץ
                      </span>
                    )}
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
