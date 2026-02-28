'use client'

import { useEffect, useState } from 'react'
import { RefreshCw, Check, X } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ReimbursementStatusBadge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatDate } from '@/lib/utils/formatters'
import { formatAmount, formatDistance } from '@/lib/utils/reimbursement'
import type { ReimbursementStatus } from '@/lib/types'

interface ReimbursementWithDetails {
  id: string
  status: ReimbursementStatus
  amount_ils: number
  distance_km: number
  created_at: string
  rejection_reason?: string
  ride: { patient_name: string; scheduled_at: string; hospital: { name_he: string } }
  driver: { name_he: string; phone: string }
}

const STATUS_LABELS: Record<ReimbursementStatus, string> = {
  pending: 'ממתין לאישור',
  approved: 'אושר',
  rejected: 'נדחה',
  paid: 'שולם',
}

export default function ReimbursementsPage() {
  const [items, setItems] = useState<ReimbursementWithDetails[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<ReimbursementStatus | 'all'>('pending')
  const [rejectModal, setRejectModal] = useState<string | null>(null)
  const [rejectReason, setRejectReason] = useState('')
  const [working, setWorking] = useState(false)

  async function fetchItems() {
    setLoading(true)
    const res = await fetch('/api/reimbursements')
    const { data } = await res.json()
    setItems(data ?? [])
    setLoading(false)
  }

  useEffect(() => { fetchItems() }, [])

  async function handleApprove(id: string) {
    setWorking(true)
    await fetch(`/api/reimbursements/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'approve' }),
    })
    setWorking(false)
    fetchItems()
  }

  async function handleReject(id: string) {
    if (!rejectReason) return
    setWorking(true)
    await fetch(`/api/reimbursements/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'reject', rejection_reason: rejectReason }),
    })
    setWorking(false)
    setRejectModal(null)
    setRejectReason('')
    fetchItems()
  }

  const filtered = filter === 'all' ? items : items.filter(i => i.status === filter)

  const totalPending = items.filter(i => i.status === 'pending').reduce((s, i) => s + i.amount_ils, 0)

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-gray-900">החזרי נסיעה</h1>
          {items.filter(i => i.status === 'pending').length > 0 && (
            <p className="text-sm text-yellow-700 mt-0.5">
              {items.filter(i => i.status === 'pending').length} בקשות ממתינות •{' '}
              סה"כ {formatAmount(totalPending)}
            </p>
          )}
        </div>
        <Button variant="ghost" size="sm" onClick={fetchItems}><RefreshCw size={16} /></Button>
      </div>

      {/* פילטר */}
      <div className="flex gap-2 mb-4">
        {([
          { value: 'pending' as const, label: 'ממתינות' },
          { value: 'approved' as const, label: 'אושרו' },
          { value: 'rejected' as const, label: 'נדחו' },
          { value: 'all' as const, label: 'הכל' },
        ]).map(f => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
              filter === f.value
                ? 'bg-green-700 text-white border-green-700'
                : 'bg-white text-gray-600 border-gray-300 hover:border-green-400'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-12"><Spinner /></div>
      ) : filtered.length === 0 ? (
        <Card className="text-center py-10">
          <p className="text-gray-400">אין בקשות החזר להצגה</p>
        </Card>
      ) : (
        <div className="space-y-3">
          {filtered.map(item => (
            <Card key={item.id} padding="sm">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <ReimbursementStatusBadge status={item.status} label={STATUS_LABELS[item.status]} />
                    <span className="text-xs text-gray-400">{formatDate(item.created_at)}</span>
                  </div>
                  <p className="font-semibold text-gray-900">{item.driver?.name_he}</p>
                  <p className="text-sm text-gray-500">{item.ride?.patient_name} → {item.ride?.hospital?.name_he}</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {formatDistance(item.distance_km)} • {formatDate(item.ride?.scheduled_at)}
                  </p>
                  {item.rejection_reason && (
                    <p className="text-xs text-red-600 mt-1 bg-red-50 px-2 py-1 rounded">{item.rejection_reason}</p>
                  )}
                </div>

                <div className="text-left shrink-0">
                  <p className="text-lg font-bold text-green-700">{formatAmount(item.amount_ils)}</p>
                  {item.status === 'pending' && (
                    <div className="flex gap-1.5 mt-2">
                      <Button
                        size="sm"
                        onClick={() => handleApprove(item.id)}
                        disabled={working}
                        className="px-2"
                      >
                        <Check size={14} />
                      </Button>
                      <Button
                        size="sm"
                        variant="danger"
                        onClick={() => { setRejectModal(item.id); setRejectReason('') }}
                        disabled={working}
                        className="px-2"
                      >
                        <X size={14} />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Modal דחייה */}
      {rejectModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-sm shadow-xl p-6">
            <h2 className="font-bold text-gray-900 mb-3">סיבת דחייה</h2>
            <textarea
              value={rejectReason}
              onChange={e => setRejectReason(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-red-400 mb-4"
              placeholder="הסבר מדוע הבקשה נדחתה..."
            />
            <div className="flex gap-3">
              <Button
                variant="danger"
                disabled={!rejectReason || working}
                onClick={() => handleReject(rejectModal)}
                className="flex-1"
              >
                {working ? <Spinner size="sm" /> : null}
                דחה
              </Button>
              <Button variant="secondary" onClick={() => setRejectModal(null)}>ביטול</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
