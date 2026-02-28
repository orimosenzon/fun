import type { RideStatus, UserRole } from '@/lib/types'
import { VALID_STATUS_TRANSITIONS } from '@/lib/types'

// בדיקה האם מעבר סטטוס חוקי
export function isValidTransition(from: RideStatus, to: RideStatus): boolean {
  return VALID_STATUS_TRANSITIONS[from].includes(to)
}

// מה המעברים האפשריים מסטטוס נוכחי לפי role
export function getAllowedTransitions(
  currentStatus: RideStatus,
  role: UserRole
): RideStatus[] {
  const all = VALID_STATUS_TRANSITIONS[currentStatus]

  // נהג יכול רק להתקדם בנסיעות שלו
  if (role === 'driver') {
    const driverAllowed: Partial<Record<RideStatus, RideStatus[]>> = {
      assigned: ['in_progress'],
      in_progress: ['completed'],
    }
    return driverAllowed[currentStatus] ?? []
  }

  // רכז פלסטיני לא יכול לשנות סטטוס
  if (role === 'palestinian_coordinator') {
    return []
  }

  // רכזת ישראלית — הכל (כולל ביטול)
  return all
}

// צבע badge לפי סטטוס
export const STATUS_COLORS: Record<RideStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  assigned: 'bg-blue-100 text-blue-800 border-blue-200',
  in_progress: 'bg-orange-100 text-orange-800 border-orange-200',
  completed: 'bg-green-100 text-green-800 border-green-200',
  cancelled: 'bg-gray-100 text-gray-600 border-gray-200',
  return_needed: 'bg-purple-100 text-purple-800 border-purple-200',
}

// האם נסיעה פתוחה לאיסוף ע"י נהג
export function isRideAvailableForDriver(status: RideStatus): boolean {
  return status === 'pending'
}

// האם הנסיעה פעילה (לא סופית)
export function isRideActive(status: RideStatus): boolean {
  return !['completed', 'cancelled'].includes(status)
}
