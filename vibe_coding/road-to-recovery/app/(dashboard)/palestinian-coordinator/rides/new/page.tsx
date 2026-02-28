'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import type { Hospital } from '@/lib/types'

export default function NewRidePage() {
  const router = useRouter()
  const [lang, setLang] = useState<'he' | 'ar'>('he')
  const [hospitals, setHospitals] = useState<Hospital[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [form, setForm] = useState({
    patient_name: '',
    patient_phone: '',
    pickup_address: '',
    hospital_id: '',
    scheduled_at: '',
    medical_notes: '',
    driver_notes: '',
  })

  useEffect(() => {
    const saved = localStorage.getItem('lang') as 'he' | 'ar' | null
    if (saved) setLang(saved)

    fetch('/api/hospitals')
      .then(r => r.json())
      .then(({ data }) => setHospitals(data ?? []))
  }, [])

  const t = (he: string, ar: string) => lang === 'ar' ? ar : he

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')

    const res = await fetch('/api/rides', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    })

    const { data, error: apiError } = await res.json()

    if (apiError || !res.ok) {
      setError(apiError ?? t('שגיאה בהגשת הבקשה', 'خطأ في تقديم الطلب'))
      setLoading(false)
      return
    }

    router.push(`/palestinian-coordinator/rides/${data.id}`)
  }

  function set(field: string, value: string) {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  const labelClass = 'block text-sm font-medium text-gray-700 mb-1'
  const inputClass = 'w-full px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent text-sm'

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-gray-900">
          {t('בקשת נסיעה חדשה', 'طلب رحلة جديدة')}
        </h1>
        <p className="text-sm text-gray-500 mt-0.5">
          {t('מלא את הפרטים — נמצא נהג לנסיעה', 'أكمل التفاصيل — سنجد سائقاً للرحلة')}
        </p>
      </div>

      <Card>
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* שם מטופל */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={labelClass}>{t('שם המטופל', 'اسم المريض')} *</label>
              <input
                type="text"
                required
                value={form.patient_name}
                onChange={e => set('patient_name', e.target.value)}
                className={inputClass}
                placeholder={t('שם מלא', 'الاسم الكامل')}
              />
            </div>
            <div>
              <label className={labelClass}>{t('טלפון מטופל', 'هاتف المريض')}</label>
              <input
                type="tel"
                value={form.patient_phone}
                onChange={e => set('patient_phone', e.target.value)}
                className={inputClass}
                dir="ltr"
                placeholder="05X-XXXXXXX"
              />
            </div>
          </div>

          {/* כתובת איסוף */}
          <div>
            <label className={labelClass}>{t('כתובת איסוף', 'عنوان الاستلام')} *</label>
            <input
              type="text"
              required
              value={form.pickup_address}
              onChange={e => set('pickup_address', e.target.value)}
              className={inputClass}
              placeholder={t('רחוב, מספר, עיר', 'الشارع، الرقم، المدينة')}
            />
            <p className="text-xs text-gray-400 mt-1">
              {t('המרחק לבית החולים יחושב אוטומטית', 'سيتم احتساب المسافة إلى المستشفى تلقائياً')}
            </p>
          </div>

          {/* בית חולים */}
          <div>
            <label className={labelClass}>{t('בית חולים יעד', 'المستشفى المقصود')} *</label>
            <select
              required
              value={form.hospital_id}
              onChange={e => set('hospital_id', e.target.value)}
              className={inputClass}
            >
              <option value="">{t('בחר בית חולים', 'اختر مستشفى')}</option>
              {hospitals.map(h => (
                <option key={h.id} value={h.id}>
                  {lang === 'ar' ? h.name_ar : h.name_he} — {h.city}
                </option>
              ))}
            </select>
          </div>

          {/* תאריך ושעה */}
          <div>
            <label className={labelClass}>{t('תאריך ושעת הנסיעה', 'تاريخ ووقت الرحلة')} *</label>
            <input
              type="datetime-local"
              required
              value={form.scheduled_at}
              onChange={e => set('scheduled_at', e.target.value)}
              className={inputClass}
              dir="ltr"
              min={new Date().toISOString().slice(0, 16)}
            />
          </div>

          {/* הערות רפואיות */}
          <div>
            <label className={labelClass}>{t('הערות רפואיות', 'ملاحظات طبية')}</label>
            <textarea
              value={form.medical_notes}
              onChange={e => set('medical_notes', e.target.value)}
              rows={3}
              className={inputClass}
              placeholder={t('אבחנה, מצב רפואי, צרכים מיוחדים...', 'التشخيص، الحالة الطبية، الاحتياجات الخاصة...')}
            />
            <p className="text-xs text-gray-400 mt-1">
              {t('לא יוצג לנהג', 'لن يظهر للسائق')}
            </p>
          </div>

          {/* הערות לנהג */}
          <div>
            <label className={labelClass}>{t('הערות לנהג', 'ملاحظات للسائق')}</label>
            <textarea
              value={form.driver_notes}
              onChange={e => set('driver_notes', e.target.value)}
              rows={2}
              className={inputClass}
              placeholder={t('כניסה ראשית, צריך עזרה בכניסה...', 'المدخل الرئيسي، يحتاج مساعدة في الدخول...')}
            />
          </div>

          {error && (
            <div className="bg-red-50 text-red-700 text-sm px-4 py-3 rounded-lg border border-red-200">
              {error}
            </div>
          )}

          <div className="flex gap-3 pt-2">
            <Button type="submit" disabled={loading} className="flex-1">
              {loading ? <Spinner size="sm" /> : null}
              {t('הגש בקשה', 'تقديم الطلب')}
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={() => router.back()}
            >
              {t('ביטול', 'إلغاء')}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  )
}
