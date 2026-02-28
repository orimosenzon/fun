import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { calculateDistance } from '@/lib/maps'

// POST /api/geocode
// קלט: { pickup_address, hospital_id }
// פלט: { distanceKm, durationMinutes, pickupLat, pickupLng }
export async function POST(request: Request) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const body = await request.json()
  const { pickup_address, hospital_id } = body

  if (!pickup_address || !hospital_id) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
  }

  // שליפת קואורדינטות בית החולים מה-DB
  const { data: hospital, error: hospitalError } = await supabase
    .from('hospitals')
    .select('lat, lng, name_he')
    .eq('id', hospital_id)
    .single()

  if (hospitalError || !hospital) {
    return NextResponse.json({ error: 'Hospital not found' }, { status: 404 })
  }

  try {
    const result = await calculateDistance(
      pickup_address,
      hospital.lat,
      hospital.lng
    )

    return NextResponse.json({
      distanceKm: Math.round(result.distanceKm * 10) / 10,
      durationMinutes: Math.round(result.durationMinutes),
      pickupLat: result.originLat,
      pickupLng: result.originLng,
    })
  } catch (err) {
    console.error('Geocoding error:', err)
    // במקרה של כשל ב-API (ל-dev: נחזיר ערך mock)
    if (process.env.NODE_ENV === 'development') {
      return NextResponse.json({
        distanceKm: 45.0,
        durationMinutes: 55,
        pickupLat: 31.5,
        pickupLng: 34.9,
        _mock: true,
      })
    }
    return NextResponse.json({ error: 'Failed to calculate distance' }, { status: 500 })
  }
}
