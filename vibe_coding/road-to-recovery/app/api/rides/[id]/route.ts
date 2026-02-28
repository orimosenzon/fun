import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

// GET /api/rides/[id]
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  // שליפת role לסינון שדות רגישים
  const { data: profile } = await supabase
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  const { data, error } = await supabase
    .from('rides')
    .select(`
      *,
      hospital:hospitals(id, name_he, name_ar, address, city),
      creator:users!rides_created_by_fkey(id, name_he, name_ar, phone),
      driver:users!rides_driver_id_fkey(id, name_he, name_ar, phone)
    `)
    .eq('id', id)
    .single()

  if (error || !data) return NextResponse.json({ error: 'Ride not found' }, { status: 404 })

  // נהג לא רואה medical_notes
  if (profile?.role === 'driver') {
    const { medical_notes: _, ...safeData } = data as typeof data & { medical_notes: unknown }
    return NextResponse.json({ data: safeData })
  }

  return NextResponse.json({ data })
}

// PATCH /api/rides/[id] — עריכת פרטי נסיעה (רכזת בלבד)
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

  if (profile?.role !== 'israeli_coordinator') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  const body = await request.json()
  // רק שדות מותרים לעדכון
  const allowedFields = ['patient_name', 'patient_phone', 'pickup_address', 'scheduled_at', 'medical_notes', 'driver_notes']
  const updates = Object.fromEntries(
    Object.entries(body).filter(([key]) => allowedFields.includes(key))
  )

  const { data, error } = await supabase
    .from('rides')
    .update(updates)
    .eq('id', id)
    .select()
    .single()

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  return NextResponse.json({ data })
}
