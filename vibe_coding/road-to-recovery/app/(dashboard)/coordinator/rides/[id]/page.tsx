'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ArrowRight, UserPlus } from 'lucide-react'
import { Card, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { RideStatusBadge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatDateTime } from '@/lib/utils/formatters'
import { formatAmount, formatDistance } from '@/lib/utils/reimbursement'
import type { RideWithDetails, PublicUser, RideStatus } from '@/lib/types'

const STATUS_LABELS: Record<RideStatus, string> = {
  pending: 'ממתינה לשיבוץ',
  assigned: 'שובץ נהג',
  in_progress: 'בביצוע',
  completed: 'הושלמה',
  cancelled: 'בוטלה',
  return_needed: 'נדרשת חזרה',
}

export default function CoordinatorRideDetailPage() {
  const { id } = useParams() as { id: string }
  const router = useRouter()
  const [ride, setRide] = useState<RideWithDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [showAssignModal, setShowAssignModal] = useState(false)
  const [drivers, setDrivers] = useState<PublicUser[]>([])
  const [selectedDriver, setSelectedDriver] = useState('')
  const [assigning, setAssigning] = useState(false)
  const [cancelMode, setCancelMode] = useState(false)
  const [cancelReason, setCancelReason] = useState('')
  const [working, setWorking] = useState(false)

  async function fetchRide() {
    const res = await fetch(`/api/rides/${id}`)
    const { data } = await res.json()
    setRide(data)
    setLoading(false)
  }

  useEffect(() => {
    fetchRide()
  }, [id])

  async function fetchDrivers() {
    const res = await fetch('/api/users?role=driver')
    const { data } = await res.json()
    setDrivers(data ?? [])
  }

  async function handleAssign() {
    if (!selectedDriver) return
    setAssigning(true)
    await fetch(`/api/rides/${id}/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ driver_id: selectedDriver }),
    })
    setShowAssignModal(false)
    setAssigning(false)
    fetchRide()
  }

  async function handleStatusChange(status: RideStatus, reason?: string) {
    setWorking(true)
    await fetch(`/api/rides/${id}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, cancellation_reason: reason }),
    })
    setWorking(false)
    setCancelMode(false)
    fetchRide()
  }

  async function handleCreateReturnRide() {
    if (!ride) return
    router.push(`/coordinator/rides/new?outbound=${id}&patient=${encodeURIComponent(ride.patient_name)}&hospital=${ride.hospital_id}&address=${encodeURIComponent(ride.pickup_address)}`)
  }

  if (loading) return <div className="flex justify-center py-12"><Spinner /></div>
  if (!ride) return <p className="text-center text-gray-400">נסיעה לא נמצאה</p>

  const canCancel = ['pending', 'assigned'].includes(ride.status)
  const canCreateReturn = ride.status === 'return_needed'

  return (
    <div className="max-w-2xl mx-auto">
      <button onClick={() => router.back()} className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 mb-4">
        <ArrowRight size={16} />
        חזרה ללוח
      </button>

      {/* כותרת */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold text-gray-900">{ride.patient_name}</h1>
          <p className="text-sm text-gray-500">{formatDateTime(ride.scheduled_at)}</p>
        </div>
        <RideStatusBadge status={ride.status} label={STATUS_LABELS[ride.status]} />
      </div>

      {/* פרטי נסיעה */}
      <Card className="mb-4">
        <CardHeader>
          <h2 className="font-bold text-gray-900">פרטי נסיעה</h2>
        </CardHeader>
        <div className="space-y-3 text-sm">
          <Row label="שם מטופל" value={ride.patient_name} />
          <Row label="טלפון מטופל" value={ride.patient_phone} />
          <Row label="כתובת איסוף" value={ride.pickup_address} />
          <Row label="בית חולים" value={ride.hospital?.name_he} />
          {ride.distance_km && (
            <Row label="מרחק" value={formatDistance(ride.distance_km)} />
          )}
          {ride.distance_km && (
            <Row label="החזר נסיעה מוערך" value={formatAmount(ride.distance_km * 1.5)} />
          )}
          <Row label="הגיש" value={ride.creator?.name_he} />
        </div>
      </Card>

      {/* הערות */}
      {(ride.medical_notes || ride.driver_notes) && (
        <Card className="mb-4">
          <CardHeader><h2 className="font-bold text-gray-900">הערות</h2></CardHeader>
          {ride.medical_notes && (
            <div className="mb-3">
              <p className="text-xs font-medium text-red-600 mb-1">🔒 הערות רפואיות (לא נחשפות לנהג)</p>
              <p className="text-sm text-gray-700 bg-red-50 rounded-lg p-3">{ride.medical_notes}</p>
            </div>
          )}
          {ride.driver_notes && (
            <div>
              <p className="text-xs font-medium text-gray-500 mb-1">הערות לנהג</p>
              <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">{ride.driver_notes}</p>
            </div>
          )}
        </Card>
      )}

      {/* נהג */}
      <Card className="mb-4">
        <CardHeader>
          <h2 className="font-bold text-gray-900">נהג משובץ</h2>
          {['pending', 'assigned'].includes(ride.status) && (
            <Button size="sm" variant="secondary" onClick={() => { setShowAssignModal(true); fetchDrivers() }}>
              <UserPlus size={14} />
              {ride.driver ? 'החלף נהג' : 'שבץ נהג'}
            </Button>
          )}
        </CardHeader>
        {ride.driver ? (
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-blue-100 border border-blue-200 flex items-center justify-center">
              <span className="text-blue-700 font-bold">{ride.driver.name_he.charAt(0)}</span>
            </div>
            <div>
              <p className="font-medium text-gray-800">{ride.driver.name_he}</p>
              <p className="text-sm text-gray-500">{ride.driver.phone}</p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-400">לא שובץ נהג עדיין</p>
        )}
      </Card>

      {/* פעולות */}
      <div className="flex flex-wrap gap-3">
        {canCreateReturn && (
          <Button onClick={handleCreateReturnRide} disabled={working}>
            🔄 צור נסיעת חזרה
          </Button>
        )}
        {canCancel && !cancelMode && (
          <Button variant="danger" onClick={() => setCancelMode(true)} disabled={working}>
            ביטול נסיעה
          </Button>
        )}
      </div>

      {/* ביטול */}
      {cancelMode && (
        <Card className="mt-4">
          <h3 className="font-bold text-gray-900 mb-3">סיבת ביטול</h3>
          <textarea
            value={cancelReason}
            onChange={e => setCancelReason(e.target.value)}
            rows={3}
            className="w-full px-3 py-2 rounded-lg border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
            placeholder="הסבר מדוע הנסיעה מבוטלת..."
          />
          <div className="flex gap-3 mt-3">
            <Button
              variant="danger"
              disabled={!cancelReason || working}
              onClick={() => handleStatusChange('cancelled', cancelReason)}
            >
              {working ? <Spinner size="sm" /> : null}
              אשר ביטול
            </Button>
            <Button variant="secondary" onClick={() => setCancelMode(false)}>חזרה</Button>
          </div>
        </Card>
      )}

      {/* Modal שיבוץ נהג */}
      {showAssignModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-md shadow-xl">
            <div className="p-6">
              <h2 className="font-bold text-gray-900 mb-4">בחר נהג</h2>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {drivers.map(d => (
                  <label
                    key={d.id}
                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedDriver === d.id ? 'border-green-500 bg-green-50' : 'border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <input
                      type="radio"
                      name="driver"
                      value={d.id}
                      checked={selectedDriver === d.id}
                      onChange={() => setSelectedDriver(d.id)}
                      className="sr-only"
                    />
                    <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                      <span className="text-blue-700 font-bold text-sm">{d.name_he.charAt(0)}</span>
                    </div>
                    <div>
                      <p className="font-medium text-sm text-gray-800">{d.name_he}</p>
                      <p className="text-xs text-gray-500">{d.phone}</p>
                    </div>
                  </label>
                ))}
              </div>
              <div className="flex gap-3 mt-4">
                <Button disabled={!selectedDriver || assigning} onClick={handleAssign} className="flex-1">
                  {assigning ? <Spinner size="sm" /> : null}
                  שבץ
                </Button>
                <Button variant="secondary" onClick={() => setShowAssignModal(false)}>ביטול</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null
  return (
    <div className="flex justify-between py-2 border-b border-gray-50 last:border-0">
      <span className="text-gray-500">{label}</span>
      <span className="font-medium text-gray-800">{value}</span>
    </div>
  )
}
