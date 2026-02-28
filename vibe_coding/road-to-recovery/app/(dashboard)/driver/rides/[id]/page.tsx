'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ArrowRight, MapPin, Clock, CheckCircle } from 'lucide-react'
import { Card, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { RideStatusBadge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatDateTime } from '@/lib/utils/formatters'
import { formatAmount, formatDistance, calculateReimbursement } from '@/lib/utils/reimbursement'
import type { RideWithDetails, RideStatus } from '@/lib/types'

const STATUS_LABELS: Record<RideStatus, string> = {
  pending: 'ממתינה',
  assigned: 'שובצת אליך',
  in_progress: 'בביצוע',
  completed: 'הושלמה',
  cancelled: 'בוטלה',
  return_needed: 'הסתיימה',
}

// מדרגות לנהג
const STEPS = [
  { status: 'assigned' as RideStatus, icon: <Clock size={16} />, label: 'מוכן' },
  { status: 'in_progress' as RideStatus, icon: <MapPin size={16} />, label: 'בדרך' },
  { status: 'completed' as RideStatus, icon: <CheckCircle size={16} />, label: 'הגעתי' },
]

export default function DriverRideDetailPage() {
  const { id } = useParams() as { id: string }
  const router = useRouter()
  const [ride, setRide] = useState<RideWithDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [working, setWorking] = useState(false)
  const [reimbursementSubmitted, setReimbursementSubmitted] = useState(false)
  const [lang, setLang] = useState<'he' | 'ar'>('he')

  useEffect(() => {
    const saved = localStorage.getItem('lang') as 'he' | 'ar' | null
    if (saved) setLang(saved)
  }, [])

  async function fetchRide() {
    const res = await fetch(`/api/rides/${id}`)
    const { data } = await res.json()
    setRide(data)
    setLoading(false)
  }

  useEffect(() => { fetchRide() }, [id])

  const t = (he: string, ar: string) => lang === 'ar' ? ar : he

  async function handleStatusChange(newStatus: RideStatus) {
    setWorking(true)
    await fetch(`/api/rides/${id}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus }),
    })
    setWorking(false)
    fetchRide()
  }

  async function handleSubmitReimbursement() {
    setWorking(true)
    const res = await fetch('/api/reimbursements', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ride_id: id }),
    })
    setWorking(false)
    if (res.ok || res.status === 409) {
      setReimbursementSubmitted(true)
    }
  }

  if (loading) return <div className="flex justify-center py-12"><Spinner /></div>
  if (!ride) return <p className="text-center text-gray-400">{t('נסיעה לא נמצאה', 'الرحلة غير موجودة')}</p>

  const reimbursementAmount = ride.distance_km ? calculateReimbursement(ride.distance_km) : null
  const currentStepIndex = STEPS.findIndex(s => s.status === ride.status)

  // הפעולה הבאה לנהג
  const nextActions: Partial<Record<RideStatus, { label: string; labelAr: string; nextStatus: RideStatus }>> = {
    assigned: { label: 'התחל נסיעה', labelAr: 'بدء الرحلة', nextStatus: 'in_progress' },
    in_progress: { label: 'הגעתי לבית חולים', labelAr: 'وصلت إلى المستشفى', nextStatus: 'completed' },
  }
  const nextAction = nextActions[ride.status]

  return (
    <div className="max-w-xl mx-auto">
      <button onClick={() => router.back()} className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 mb-4">
        <ArrowRight size={16} />
        {t('חזרה', 'رجوع')}
      </button>

      {/* Stepper */}
      {['assigned', 'in_progress', 'completed'].includes(ride.status) && (
        <Card className="mb-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-bold text-gray-900">{t('מצב הנסיעה', 'حالة الرحلة')}</h2>
            <RideStatusBadge status={ride.status} label={STATUS_LABELS[ride.status]} />
          </div>
          <div className="flex items-center gap-0">
            {STEPS.map((step, i) => {
              const isCompleted = i < currentStepIndex
              const isCurrent = i === currentStepIndex
              return (
                <div key={step.status} className="flex items-center flex-1">
                  <div className="flex flex-col items-center">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center
                      ${isCompleted ? 'bg-green-600 text-white' : isCurrent ? 'bg-green-100 border-2 border-green-600 text-green-700' : 'bg-gray-100 text-gray-400'}`}>
                      {step.icon}
                    </div>
                    <p className={`text-xs mt-1 ${isCurrent ? 'text-green-700 font-medium' : 'text-gray-400'}`}>
                      {step.label}
                    </p>
                  </div>
                  {i < STEPS.length - 1 && (
                    <div className={`flex-1 h-0.5 mx-1 mb-4 ${i < currentStepIndex ? 'bg-green-500' : 'bg-gray-200'}`} />
                  )}
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {/* פרטי נסיעה */}
      <Card className="mb-4">
        <CardHeader>
          <h2 className="font-bold text-gray-900">{t('פרטי הנסיעה', 'تفاصيل الرحلة')}</h2>
        </CardHeader>
        <div className="space-y-3 text-sm">
          <div className="flex items-start gap-2">
            <MapPin size={16} className="text-green-600 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs text-gray-400">{t('כתובת איסוף', 'عنوان الاستلام')}</p>
              <p className="font-medium text-gray-800">{ride.pickup_address}</p>
            </div>
          </div>
          <div className="border-t border-gray-50 pt-3">
            <p className="text-xs text-gray-400">{t('יעד', 'الوجهة')}</p>
            <p className="font-medium text-gray-800">
              {lang === 'ar' ? ride.hospital?.name_ar : ride.hospital?.name_he}
            </p>
          </div>
          <div className="border-t border-gray-50 pt-3 flex justify-between">
            <div>
              <p className="text-xs text-gray-400">{t('שעת נסיעה', 'وقت الرحلة')}</p>
              <p className="font-medium text-gray-800">{formatDateTime(ride.scheduled_at)}</p>
            </div>
            {ride.distance_km && (
              <div className="text-left">
                <p className="text-xs text-gray-400">{t('מרחק', 'المسافة')}</p>
                <p className="font-medium text-gray-800">{formatDistance(ride.distance_km)}</p>
              </div>
            )}
          </div>
          {ride.driver_notes && (
            <div className="border-t border-gray-50 pt-3">
              <p className="text-xs text-gray-400 mb-1">{t('הערות', 'ملاحظات')}</p>
              <p className="text-sm text-gray-700 bg-blue-50 rounded-lg p-2">{ride.driver_notes}</p>
            </div>
          )}
        </div>
      </Card>

      {/* כפתור פעולה הבאה */}
      {nextAction && (
        <Button
          className="w-full mb-3"
          size="lg"
          onClick={() => handleStatusChange(nextAction.nextStatus)}
          disabled={working}
        >
          {working ? <Spinner size="sm" /> : null}
          {lang === 'ar' ? nextAction.labelAr : nextAction.label}
        </Button>
      )}

      {/* החזר נסיעה — מוצג כשנסיעה הושלמה */}
      {ride.status === 'completed' && reimbursementAmount && (
        <Card className="bg-green-50 border-green-200">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-bold text-green-800">
              {t('החזר נסיעה', 'استرداد السفر')}
            </h2>
            <span className="text-2xl font-bold text-green-700">
              {formatAmount(reimbursementAmount)}
            </span>
          </div>
          <p className="text-sm text-green-600 mb-4">
            {t(
              `${formatDistance(ride.distance_km!)} × ₪1.50 לק"מ`,
              `${formatDistance(ride.distance_km!)} × ₪1.50 للكيلومتر`
            )}
          </p>

          {reimbursementSubmitted ? (
            <div className="bg-white rounded-lg p-3 border border-green-200 text-center">
              <CheckCircle size={20} className="text-green-600 mx-auto mb-1" />
              <p className="text-sm font-medium text-green-700">
                {t('בקשת ההחזר הוגשה בהצלחה', 'تم تقديم طلب الاسترداد بنجاح')}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                {t('הרכזת תאשר בקרוב', 'ستوافق المنسقة قريباً')}
              </p>
            </div>
          ) : (
            <Button
              className="w-full"
              onClick={handleSubmitReimbursement}
              disabled={working}
            >
              {working ? <Spinner size="sm" /> : '💳'}
              {t('הגש בקשת החזר', 'تقديم طلب الاسترداد')}
            </Button>
          )}
        </Card>
      )}
    </div>
  )
}
