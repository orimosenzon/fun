'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ArrowRight } from 'lucide-react'
import { Card, CardHeader } from '@/components/ui/Card'
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

// מדרגות סטטוס ויזואלי
const STATUS_STEPS = ['pending', 'assigned', 'in_progress', 'completed']

export default function RideDetailPage() {
  const { id } = useParams() as { id: string }
  const router = useRouter()
  const [ride, setRide] = useState<RideWithDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [lang, setLang] = useState<'he' | 'ar'>('he')

  useEffect(() => {
    const saved = localStorage.getItem('lang') as 'he' | 'ar' | null
    if (saved) setLang(saved)
  }, [])

  useEffect(() => {
    fetch(`/api/rides/${id}`)
      .then(r => r.json())
      .then(({ data }) => { setRide(data); setLoading(false) })
  }, [id])

  const t = (he: string, ar: string) => lang === 'ar' ? ar : he

  if (loading) return <div className="flex justify-center py-12"><Spinner /></div>
  if (!ride) return <p className="text-center text-gray-400">{t('נסיעה לא נמצאה', 'الرحلة غير موجودة')}</p>

  const currentStepIndex = STATUS_STEPS.indexOf(ride.status)
  const statusLabel = lang === 'ar' ? STATUS_LABELS_AR[ride.status] : STATUS_LABELS_HE[ride.status]

  return (
    <div className="max-w-2xl mx-auto">
      {/* חזרה */}
      <button
        onClick={() => router.back()}
        className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 mb-4"
      >
        <ArrowRight size={16} />
        {t('חזרה לרשימה', 'العودة إلى القائمة')}
      </button>

      {/* סטטוס ויזואלי */}
      <Card className="mb-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-bold text-gray-900">{t('מצב הנסיעה', 'حالة الرحلة')}</h2>
          <RideStatusBadge status={ride.status} label={statusLabel} />
        </div>

        {/* stepper */}
        {!['cancelled', 'return_needed'].includes(ride.status) && (
          <div className="flex items-center gap-0">
            {STATUS_STEPS.map((step, i) => {
              const isCompleted = i < currentStepIndex
              const isCurrent = i === currentStepIndex
              const label = lang === 'ar' ? STATUS_LABELS_AR[step] : STATUS_LABELS_HE[step]
              return (
                <div key={step} className="flex items-center flex-1">
                  <div className="flex flex-col items-center">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                      ${isCompleted ? 'bg-green-600 text-white' : isCurrent ? 'bg-green-100 border-2 border-green-600 text-green-700' : 'bg-gray-100 text-gray-400'}`}>
                      {isCompleted ? '✓' : i + 1}
                    </div>
                    <p className={`text-xs mt-1 text-center ${isCurrent ? 'text-green-700 font-medium' : 'text-gray-400'}`}>
                      {label}
                    </p>
                  </div>
                  {i < STATUS_STEPS.length - 1 && (
                    <div className={`flex-1 h-0.5 mx-1 mb-5 ${i < currentStepIndex ? 'bg-green-500' : 'bg-gray-200'}`} />
                  )}
                </div>
              )
            })}
          </div>
        )}
      </Card>

      {/* פרטי נסיעה */}
      <Card>
        <CardHeader>
          <h2 className="font-bold text-gray-900">{t('פרטי הנסיעה', 'تفاصيل الرحلة')}</h2>
        </CardHeader>

        <div className="space-y-3 text-sm">
          <Row label={t('שם מטופל', 'اسم المريض')} value={ride.patient_name} />
          {ride.patient_phone && <Row label={t('טלפון', 'الهاتف')} value={ride.patient_phone} />}
          <Row label={t('כתובת איסוף', 'عنوان الاستلام')} value={ride.pickup_address} />
          <Row
            label={t('בית חולים', 'المستشفى')}
            value={lang === 'ar' ? ride.hospital?.name_ar : ride.hospital?.name_he}
          />
          <Row label={t('תאריך ושעה', 'التاريخ والوقت')} value={formatDateTime(ride.scheduled_at)} />
          {ride.distance_km && (
            <Row label={t('מרחק מוערך', 'المسافة التقريبية')} value={`${ride.distance_km} ק"מ`} />
          )}
        </div>
      </Card>

      {/* נהג שובץ */}
      {ride.driver && (
        <Card className="mt-4">
          <h2 className="font-bold text-gray-900 mb-3">{t('פרטי הנהג', 'تفاصيل السائق')}</h2>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-blue-100 border border-blue-200 flex items-center justify-center">
              <span className="text-blue-700 font-bold text-sm">
                {ride.driver.name_he.charAt(0)}
              </span>
            </div>
            <div>
              <p className="font-medium text-gray-800">
                {lang === 'ar' ? (ride.driver.name_ar ?? ride.driver.name_he) : ride.driver.name_he}
              </p>
              <p className="text-sm text-gray-500">{ride.driver.phone}</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null
  return (
    <div className="flex justify-between py-2 border-b border-gray-50 last:border-0">
      <span className="text-gray-500">{label}</span>
      <span className="font-medium text-gray-800 text-left max-w-xs">{value}</span>
    </div>
  )
}
