import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import type { CreateRideRequest } from '@/lib/types'

// GET /api/rides
// query params: date (YYYY-MM-DD), status, hospital_id, driver_id
// RLS מגבילה מה כל role רואה
export async function GET(request: Request) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { searchParams } = new URL(request.url)
  const date = searchParams.get('date')
  const status = searchParams.get('status')
  const hospital_id = searchParams.get('hospital_id')

  let query = supabase
    .from('rides')
    .select(`
      *,
      hospital:hospitals(id, name_he, name_ar, address, city),
      creator:users!rides_created_by_fkey(id, name_he, name_ar, phone),
      driver:users!rides_driver_id_fkey(id, name_he, name_ar, phone)
    `)
    .order('scheduled_at', { ascending: true })

  if (date) {
    // נסיעות שתוכננו לתאריך ספציפי
    const startOfDay = `${date}T00:00:00.000Z`
    const endOfDay = `${date}T23:59:59.999Z`
    query = query.gte('scheduled_at', startOfDay).lte('scheduled_at', endOfDay)
  }

  if (status && status !== 'all') {
    query = query.eq('status', status)
  }

  if (hospital_id) {
    query = query.eq('hospital_id', hospital_id)
  }

  const { data, error } = await query

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  return NextResponse.json({ data })
}

// POST /api/rides
// יצירת נסיעה חדשה (רכז פלסטיני בלבד — RLS מאכפת)
export async function POST(request: Request) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const body: CreateRideRequest = await request.json()
  const { patient_name, pickup_address, hospital_id, scheduled_at } = body

  if (!patient_name || !pickup_address || !hospital_id || !scheduled_at) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
  }

  // חישוב מרחק — פעם אחת בלבד (נשמר ב-DB)
  let distanceKm: number | null = null
  let pickupLat: number | null = null
  let pickupLng: number | null = null

  try {
    const { data: hospital } = await supabase
      .from('hospitals')
      .select('lat, lng')
      .eq('id', hospital_id)
      .single()

    if (hospital) {
      const { calculateDistance } = await import('@/lib/maps')
      const distResult = await calculateDistance(pickup_address, hospital.lat, hospital.lng)
      distanceKm = Math.round(distResult.distanceKm * 10) / 10
      pickupLat = distResult.originLat
      pickupLng = distResult.originLng
    }
  } catch (e) {
    // אם חישוב מרחק נכשל — ממשיכים בלי מרחק (לא חוסמים יצירת נסיעה)
    console.warn('Distance calculation failed, creating ride without distance:', e)
  }

  const { data, error } = await supabase
    .from('rides')
    .insert({
      patient_name: body.patient_name,
      patient_phone: body.patient_phone ?? null,
      pickup_address: body.pickup_address,
      pickup_lat: pickupLat,
      pickup_lng: pickupLng,
      hospital_id: body.hospital_id,
      scheduled_at: body.scheduled_at,
      medical_notes: body.medical_notes ?? null,
      driver_notes: body.driver_notes ?? null,
      is_return_ride: body.is_return_ride ?? false,
      outbound_ride_id: body.outbound_ride_id ?? null,
      created_by: user.id,
      distance_km: distanceKm,
    })
    .select('*')
    .single()

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  // אם זו נסיעת חזרה, עדכן את נסיעת ה"הלוך" עם ה-return_ride_id
  if (body.is_return_ride && body.outbound_ride_id && data) {
    await supabase
      .from('rides')
      .update({ return_ride_id: data.id, status: 'assigned' })
      .eq('id', body.outbound_ride_id)
  }

  return NextResponse.json({ data }, { status: 201 })
}
