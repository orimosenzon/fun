import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { isValidTransition, getAllowedTransitions } from '@/lib/utils/rideStatus'
import type { RideStatus, UserRole } from '@/lib/types'

// PATCH /api/rides/[id]/status
// עדכון סטטוס נסיעה — state machine מאכף מעברים חוקיים
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { data: profile } = await supabase
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  if (!profile) return NextResponse.json({ error: 'User not found' }, { status: 404 })

  const body = await request.json()
  const { status: newStatus, cancellation_reason } = body as {
    status: RideStatus
    cancellation_reason?: string
  }

  // שליפת הנסיעה הנוכחית
  const { data: ride, error: rideError } = await supabase
    .from('rides')
    .select('id, status, driver_id, is_return_ride')
    .eq('id', id)
    .single()

  if (rideError || !ride) {
    return NextResponse.json({ error: 'Ride not found' }, { status: 404 })
  }

  // נהג יכול לעדכן רק נסיעות שלו
  if (profile.role === 'driver' && ride.driver_id !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  // בדיקת חוקיות המעבר
  const allowedTransitions = getAllowedTransitions(ride.status as RideStatus, profile.role as UserRole)
  if (!allowedTransitions.includes(newStatus)) {
    return NextResponse.json(
      { error: `Invalid transition from ${ride.status} to ${newStatus}` },
      { status: 400 }
    )
  }

  // שדות נוספים לפי הסטטוס החדש
  const timestampField: Partial<Record<RideStatus, string>> = {
    in_progress: 'started_at',
    completed: 'completed_at',
    cancelled: 'cancelled_at',
  }

  const updates: Record<string, unknown> = { status: newStatus }
  const tsField = timestampField[newStatus]
  if (tsField) updates[tsField] = new Date().toISOString()
  if (newStatus === 'cancelled' && cancellation_reason) {
    updates.cancellation_reason = cancellation_reason
  }

  const { data, error } = await supabase
    .from('rides')
    .update(updates)
    .eq('id', id)
    .select()
    .single()

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  // כשנסיעה מסתיימת (completed) — בדוק אם צריך נסיעת חזרה
  if (newStatus === 'completed' && !ride.is_return_ride) {
    await handleReturnRideCheck(supabase, id, ride.driver_id as string)
  }

  return NextResponse.json({ data })
}

async function handleReturnRideCheck(
  supabase: Awaited<ReturnType<typeof createClient>>,
  rideId: string,
  driverId: string
) {
  // בדיקה: האם כבר קיימת נסיעת חזרה?
  const { data: existingReturn } = await supabase
    .from('rides')
    .select('id')
    .eq('outbound_ride_id', rideId)
    .single()

  if (existingReturn) return // כבר קיימת נסיעת חזרה

  // עדכון סטטוס ל-return_needed
  await supabase
    .from('rides')
    .update({ status: 'return_needed' })
    .eq('id', rideId)

  // שליחת notification לרכזות הישראליות
  const { data: coordinators } = await supabase
    .from('users')
    .select('id')
    .eq('role', 'israeli_coordinator')
    .eq('is_active', true)

  if (!coordinators) return

  const notifications = coordinators.map(c => ({
    user_id: c.id,
    type: 'return_ride_needed' as const,
    title_he: 'נדרשת נסיעת חזרה',
    title_ar: 'يلزم رحلة عودة',
    message_he: 'נסיעת הלוך הסתיימה — יש לתאם נסיעת חזרה למטופל',
    message_ar: 'انتهت رحلة الذهاب — يرجى ترتيب رحلة عودة للمريض',
    ride_id: rideId,
  }))

  await supabase.from('notifications').insert(notifications)
}
