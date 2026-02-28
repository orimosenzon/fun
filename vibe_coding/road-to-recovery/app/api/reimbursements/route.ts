import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { calculateReimbursement, DEFAULT_RATE_PER_KM } from '@/lib/utils/reimbursement'

// GET /api/reimbursements
// נהג — רק שלו; רכזת — הכל (RLS מגבילה)
export async function GET() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { data, error } = await supabase
    .from('reimbursement_requests')
    .select(`
      *,
      ride:rides(id, patient_name, scheduled_at, pickup_address, hospital:hospitals(name_he, name_ar)),
      driver:users!reimbursement_requests_driver_id_fkey(id, name_he, name_ar, phone),
      approver:users!reimbursement_requests_approved_by_fkey(id, name_he)
    `)
    .order('created_at', { ascending: false })

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  return NextResponse.json({ data })
}

// POST /api/reimbursements
// נהג מגיש בקשת החזר (רק לנסיעות שסיים)
export async function POST(request: Request) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { data: profile } = await supabase
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'driver') {
    return NextResponse.json({ error: 'Only drivers can submit reimbursements' }, { status: 403 })
  }

  const { ride_id } = await request.json()
  if (!ride_id) return NextResponse.json({ error: 'ride_id required' }, { status: 400 })

  // שליפת נתוני הנסיעה (RLS מוודאת שהנהג הוא זה שביצע)
  const { data: ride, error: rideError } = await supabase
    .from('rides')
    .select('id, distance_km, status, driver_id')
    .eq('id', ride_id)
    .eq('driver_id', user.id)
    .eq('status', 'completed')
    .single()

  if (rideError || !ride) {
    return NextResponse.json(
      { error: 'Ride not found or not eligible for reimbursement' },
      { status: 404 }
    )
  }

  if (!ride.distance_km) {
    return NextResponse.json(
      { error: 'Distance not available for this ride' },
      { status: 400 }
    )
  }

  const amount = calculateReimbursement(ride.distance_km)

  const { data, error } = await supabase
    .from('reimbursement_requests')
    .insert({
      ride_id,
      driver_id: user.id,
      distance_km: ride.distance_km,
      rate_per_km: DEFAULT_RATE_PER_KM,
      amount_ils: amount,
    })
    .select()
    .single()

  if (error) {
    // UNIQUE violation = כבר הוגשה בקשה לנסיעה זו
    if (error.code === '23505') {
      return NextResponse.json(
        { error: 'Reimbursement already submitted for this ride' },
        { status: 409 }
      )
    }
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ data }, { status: 201 })
}
