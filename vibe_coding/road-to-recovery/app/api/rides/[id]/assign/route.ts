import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

// POST /api/rides/[id]/assign
// רכזת ישראלית משבצת נהג ספציפי לנסיעה
export async function POST(
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

  if (profile?.role !== 'israeli_coordinator') {
    return NextResponse.json({ error: 'Only coordinators can assign drivers' }, { status: 403 })
  }

  const { driver_id } = await request.json()
  if (!driver_id) return NextResponse.json({ error: 'driver_id required' }, { status: 400 })

  // וידוא שהנהג קיים ופעיל
  const { data: driver } = await supabase
    .from('users')
    .select('id, name_he, name_ar, role')
    .eq('id', driver_id)
    .eq('role', 'driver')
    .eq('is_active', true)
    .single()

  if (!driver) return NextResponse.json({ error: 'Driver not found' }, { status: 404 })

  const { data, error } = await supabase
    .from('rides')
    .update({
      driver_id,
      status: 'assigned',
      assigned_at: new Date().toISOString(),
    })
    .eq('id', id)
    .in('status', ['pending', 'assigned'])   // אפשר לשנות נהג גם אם כבר שובץ אחד
    .select('*, hospital:hospitals(name_he, name_ar)')
    .single()

  if (error || !data) {
    return NextResponse.json({ error: 'Failed to assign driver' }, { status: 500 })
  }

  // notification לנהג
  const rideData = data as typeof data & { hospital: { name_he: string; name_ar: string } }
  await supabase.from('notifications').insert({
    user_id: driver_id,
    type: 'ride_assigned',
    title_he: 'שובצת לנסיעה',
    title_ar: 'تم تعيينك لرحلة',
    message_he: `שובצת לנסיעה לבית חולים ${rideData.hospital?.name_he ?? ''}`,
    message_ar: `تم تعيينك لرحلة إلى مستشفى ${rideData.hospital?.name_ar ?? ''}`,
    ride_id: id,
  })

  return NextResponse.json({ data })
}
