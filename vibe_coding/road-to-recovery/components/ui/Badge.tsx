import type { RideStatus, ReimbursementStatus } from '@/lib/types'
import { STATUS_COLORS } from '@/lib/utils/rideStatus'

interface RideStatusBadgeProps {
  status: RideStatus
  label: string
}

export function RideStatusBadge({ status, label }: RideStatusBadgeProps) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${STATUS_COLORS[status]}`}>
      {label}
    </span>
  )
}

const REIMBURSEMENT_COLORS: Record<ReimbursementStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  approved: 'bg-green-100 text-green-800 border-green-200',
  rejected: 'bg-red-100 text-red-800 border-red-200',
  paid: 'bg-blue-100 text-blue-800 border-blue-200',
}

interface ReimbursementStatusBadgeProps {
  status: ReimbursementStatus
  label: string
}

export function ReimbursementStatusBadge({ status, label }: ReimbursementStatusBadgeProps) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${REIMBURSEMENT_COLORS[status]}`}>
      {label}
    </span>
  )
}
