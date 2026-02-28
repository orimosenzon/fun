import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

// POST /api/rides/[id]/take
// נהג לוקח נסיעה פתוחה
// מימוש atomic UPDATE למניעת race condition
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

  if (profile?.role !== 'driver') {
    return NextResponse.json({ error: 'Only drivers can take rides' }, { status: 403 })
  }

  // UPDATE אטומי — מצליח רק אם status='pending'
  // אם שני נהגים לוחצים בו-זמנית: רק אחד יקבל את השורה בחזרה
  const { data, error } = await supabase
    .from('rides')
    .update({
      driver_id: user.id,
      status: 'assigned',
      assigned_at: new Date().toISOString(),
    })
    .eq('id', id)
    .eq('status', 'pending')   // << הגנה מפני race condition
    .select()
    .single()

  if (error || !data) {
    // הנסיעה כבר נלקחה ע"י נהג אחר (או לא קיימת)
    return NextResponse.json(
      { error: 'Ride is no longer available' },
      { status: 409 }   // 409 Conflict
    )
  }

  // יצירת notification לרכזת על שיבוץ
  await createCoordinatorNotification(supabase, id, user.id, data)

  return NextResponse.json({ data })
}

async function createCoordinatorNotification(
  supabase: Awaited<ReturnType<typeof createClient>>,
  rideId: string,
  driverId: string,
  ride: Record<string, unknown>
) {
  // שליפת נתוני הנהג
  const { data: driver } = await supabase
    .from('users')
    .select('name_he, name_ar')
    .eq('id', driverId)
    .single()

  // שליפת כל הרכזות הישראליות
  const { data: coordinators } = await supabase
    .from('users')
    .select('id')
    .eq('role', 'israeli_coordinator')
    .eq('is_active', true)

  if (!coordinators || !driver) return

  const notifications = coordinators.map(c => ({
    user_id: c.id,
    type: 'ride_assigned' as const,
    title_he: 'נהג שובץ לנסיעה',
    title_ar: 'تم تعيين سائق للرحلة',
    message_he: `הנהג ${driver.name_he} קיבל את הנסיעה`,
    message_ar: `السائق ${driver.name_ar ?? driver.name_he} استلم الرحلة`,
    ride_id: rideId,
  }))

  await supabase.from('notifications').insert(notifications)
}
